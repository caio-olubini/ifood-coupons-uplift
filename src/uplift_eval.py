"""Avaliação de uplift por Qini/AUUC (REQ-203) e teste de placebo (REQ-212).

Seleção e comparação de modelos de uplift nunca usa AUC/F1 (métrica de
classificação): um modelo pode classificar `converted` bem e ainda ordenar mal
o *efeito incremental* — Qini mede a segunda coisa. Reusa `sklift.metrics`
(implementação testada) em vez de reimplementar a curva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklift.metrics import qini_auc_score, qini_curve

from src import uplift, viz
from src.config import PipelineConfig


def qini(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> float:
    """Qini AUC: quanto a ordenação por uplift concentra o efeito incremental
    real nos clientes top-ranqueados, contra a curva de ganho aleatório.
    """
    return float(qini_auc_score(y_true, uplift, treatment))


def qini_curve_points(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> pd.DataFrame:
    """Pontos `(n_treated, uplift_gain)` da curva Qini, para a figura."""
    n_treated, gain = qini_curve(y_true, uplift, treatment)
    return pd.DataFrame({"n_treated": n_treated, "gain": gain})


# --- Teste de placebo por permutação (REQ-212) ---------------------------------
#
# A mesma distribuição nula serve dois propósitos: o percentil que o Qini real
# precisa superar (significância) e a dispersão da nula (intervalo de confiança
# do número reportado). Não são dois cálculos — é um só lido de duas formas.


def _permute_treatment_within_offer_type(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Embaralha `treatment` **dentro de cada `offer_type`**, preservando a
    proporção tratado/controle do grupo.

    Embaralhar globalmente mudaria essa proporção por tipo (os três `offer_type`
    têm taxas de view bem diferentes no dado real) e o Qini nulo cairia por
    composição de grupo, não pela ausência de efeito causal — exatamente o que
    este teste não quer medir. `X` e `y` (`converted`) ficam fixos; só o rótulo
    de tratamento embaralha.
    """
    permutado = df["treatment"].copy()
    for _, grupo in df.groupby("offer_type"):
        permutado.loc[grupo.index] = rng.permutation(grupo["treatment"].to_numpy())
    return permutado


def placebo_qini_distribution(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, cfg: PipelineConfig
) -> np.ndarray:
    """Distribuição nula do Qini: refita o X-learner `cfg.placebo_n_permutations`
    vezes com `treatment` embaralhado no treino, prevê no holdout real.

    Cada réplica usa uma seed derivada de `cfg.seed` — determinístico dado o
    config, mas distinto entre réplicas. Reusa `uplift.fit_xlearner`/`predict`
    (a mesma infraestrutura do modelo real), não uma reimplementação paralela.
    """
    scores = np.empty(cfg.placebo_n_permutations)
    for i in range(cfg.placebo_n_permutations):
        rng = np.random.default_rng(cfg.seed + i)
        placebo_train = train_df.copy()
        placebo_train["treatment"] = _permute_treatment_within_offer_type(placebo_train, rng)

        modelos = uplift.fit_xlearner(placebo_train, cfg)
        pred = uplift.predict(modelos, holdout_df)
        scores[i] = qini(holdout_df["converted"], pred["uplift"], holdout_df["treatment"])
    return scores


def placebo_test(
    qini_score: float, null_distribution: np.ndarray, cfg: PipelineConfig
) -> dict[str, float | bool]:
    """Compara o Qini real ao percentil `cfg.placebo_confidence_level` da nula.

    `p_value` é a fração de réplicas nulas que igualam ou superam o Qini real —
    o p-valor empírico da mesma distribuição, de graça (REQ-212).
    """
    limiar = float(np.quantile(null_distribution, cfg.placebo_confidence_level))
    p_value = float((null_distribution >= qini_score).mean())
    return {
        "qini_real": qini_score,
        "limiar_percentil": limiar,
        "passou": qini_score > limiar,
        "p_value": p_value,
        "null_mean": float(null_distribution.mean()),
        "null_std": float(null_distribution.std()),
    }


def fig_placebo_distribution(
    null_distribution: np.ndarray, qini_score: float, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Histograma da distribuição nula com o Qini real e o limiar marcados —
    a leitura visual do mesmo cálculo de `placebo_test`.
    """
    limiar = float(np.quantile(null_distribution, cfg.placebo_confidence_level))
    fig = viz.figure(
        f"Qini real ({qini_score:.3f}) supera o placebo — limiar p{int(100*cfg.placebo_confidence_level)} = {limiar:.3f}",
        f"Distribuição nula de {len(null_distribution)} permutações de `treatment` dentro de cada offer_type.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    ink_primary, secondary, _ = viz.ink(theme)
    fig.add_trace(go.Histogram(x=null_distribution, name="Qini sob placebo", marker_color=secondary))
    fig.add_vline(x=qini_score, line=dict(color=cor, width=2.5, dash="solid"))
    fig.add_vline(x=limiar, line=dict(color=ink_primary, width=1.5, dash="dot"))
    return fig


def fig_qini_curve(
    curve: pd.DataFrame, qini_score: float, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Curva Qini no tema executivo: rótulo direto no fim da série (a paleta
    validada não deixa a cor sozinha carregar identidade — ver `src/viz.py`).
    """
    fig = viz.figure(
        f"Ordenar por uplift concentra o ganho incremental — Qini AUC = {qini_score:.3f}",
        "Ganho acumulado real vs. nº de clientes tratados, ordenados por uplift previsto.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    fig.add_trace(go.Scatter(
        x=curve["n_treated"], y=curve["gain"], name="modelo de uplift",
        mode="lines", line=dict(color=cor, width=2.5),
    ))
    ink_primary, secondary, _ = viz.ink(theme)
    fig.add_trace(go.Scatter(
        x=[curve["n_treated"].iloc[0], curve["n_treated"].iloc[-1]],
        y=[curve["gain"].iloc[0], curve["gain"].iloc[-1]],
        name="aleatório", mode="lines", line=dict(color=secondary, width=1.5, dash="dot"),
    ))
    return viz.add_end_labels(fig, theme=theme)
