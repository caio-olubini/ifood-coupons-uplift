"""Avaliação de uplift: Qini/AUUC (REQ-203) e placebo (REQ-212).

Seleção e comparação de modelos de uplift nunca usa AUC/F1 (métrica de
classificação): um modelo pode classificar `converted` bem e ainda ordenar mal
o *efeito incremental* — Qini mede a segunda coisa. Reusa `sklift.metrics`
(implementação testada) em vez de reimplementar a curva.

Dois olhares complementares sobre o mesmo modelo: Qini mede **ordenação**
(concentra o efeito nos top-ranqueados?), placebo mede **significância** (a
ordenação é real ou ruído?).

`qini`/`auuc` não exigem que o score venha do X-learner — `qini_by_strategy`/
`qini_curves_by_strategy` reusam a mesma métrica para comparar o modelo de
uplift contra conversão crua (P(converte) previsto) e ranking aleatório, a
mesma pergunta que `gaincurve` responde em R$, aqui na métrica de ordenação
(REQ-203).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklift.metrics import qini_auc_score, qini_curve, uplift_auc_score

from src import uplift, viz
from src.config import PipelineConfig


def qini(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> float:
    """Qini AUC: quanto a ordenação por uplift concentra o efeito incremental
    real nos clientes top-ranqueados, contra a curva de ganho aleatório.
    """
    return float(qini_auc_score(y_true, uplift, treatment))


def auuc(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> float:
    """AUUC (Area Under the Uplift Curve): mesma ideia do Qini, mas contra a
    curva de ganho aleatório em vez da curva Qini ótima — normalizações
    diferentes do mesmo ganho incremental acumulado (REQ-203).
    """
    return float(uplift_auc_score(y_true, uplift, treatment))


def qini_curve_points(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> pd.DataFrame:
    """Pontos `(n_treated, uplift_gain)` da curva Qini, para a figura."""
    n_treated, gain = qini_curve(y_true, uplift, treatment)
    return pd.DataFrame({"n_treated": n_treated, "gain": gain})


# --- Qini/AUUC por estratégia (REQ-203) -----------------------------------------
#
# `qini_auc_score`/`uplift_auc_score` só pedem um *score* para ordenar por —
# não precisa vir do X-learner. Ranquear por P(converte) ou por ordem aleatória
# e medir o mesmo Qini/AUUC responde "esse ranking também concentra efeito
# incremental, ou só o modelo de uplift faz isso?" (a mesma pergunta de
# `gaincurve`, mas na métrica de ordenação, não em R$).


def qini_by_strategy(
    y_true: pd.Series, treatment: pd.Series, scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """Qini AUC e AUUC de cada estratégia nomeada, no mesmo holdout.

    `scores` é `{nome_estrategia: score_por_linha}` — score maior é mais
    prioritário para tratar (mesma convenção de `gaincurve.uplift_ranking`/
    `completion_ranking`/`random_ranking`, mas aqui é o valor que ordena, não
    já um índice permutado). Ranking aleatório precisa de um score contínuo
    (ex.: `np.random.default_rng(cfg.seed).random(len(y_true))`), não da
    permutação de índice que `gaincurve.random_ranking` devolve — os dois
    servem propósitos diferentes (reordenar linhas vs. pontuar cada uma).

    Retorna `[strategy, qini, auuc]`, uma linha por estratégia.
    """
    linhas = []
    for nome, score in scores.items():
        linhas.append({
            "strategy": nome,
            "qini": qini(y_true, score, treatment),
            "auuc": auuc(y_true, score, treatment),
        })
    return pd.DataFrame(linhas)


def qini_curves_by_strategy(
    y_true: pd.Series, treatment: pd.Series, scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """`qini_curve_points` de cada estratégia nomeada, numa tabela longa.

    Mesmo `y_true`/`treatment` para todas — a comparação exige o mesmo
    holdout. Retorna `[strategy, n_treated, gain]`.
    """
    partes = []
    for nome, score in scores.items():
        curva = qini_curve_points(y_true, score, treatment)
        partes.append(curva.assign(strategy=nome))
    return pd.concat(partes, ignore_index=True)[["strategy", "n_treated", "gain"]]


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


def fig_qini_curves_by_strategy(curves: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Curva Qini de várias estratégias sobrepostas — modelo de uplift vs.
    conversão crua vs. aleatório, na mesma pergunta do Qini isolado
    (`fig_qini_curve`): ordenar por este score concentra o ganho incremental
    real, ou só parece concentrar? Rótulo direto no fim de cada série.
    """
    fig = viz.figure(
        "Qini por estratégia: quem concentra o ganho incremental real",
        "Ganho acumulado real vs. nº de clientes tratados, uma curva por estratégia de ranking.",
        theme=theme,
    )
    cores = viz.palette(theme)
    for i, (nome, grupo) in enumerate(curves.groupby("strategy")):
        cor = cores[i % len(cores)]
        fig.add_trace(go.Scatter(
            x=grupo["n_treated"], y=grupo["gain"], name=nome,
            mode="lines", line=dict(color=cor, width=2.5),
        ))
    return viz.add_end_labels(fig, theme=theme)


def fig_blend_importance(importance: pd.DataFrame, top: int = 15, theme: str = "light") -> go.Figure:
    """Importância de features do blend: as três séries em barras horizontais.

    `importance` é a saída de `BlendedUpliftModel.feature_importance` — colunas
    `uplift` (causal, o que dirige o τ), `conversion` (preditiva, o que dirige a
    conversão) e `combined` (a mistura que ranqueia). Mostra as `top` features por
    `combined`; barras agrupadas deixam ler onde as duas fontes concordam e onde a
    causal destaca uma feature que a preditiva ignora (ou vice-versa). Cada série
    leva rótulo de legenda **e** cor distinta — a paleta validada não deixa a cor
    sozinha identificar (ver `src/viz.py`).

    Ordenado ascendente porque a barra horizontal do Plotly empilha de baixo para
    cima: a feature mais importante fica no topo.
    """
    dados = importance.sort_values("combined", ascending=False).head(top).iloc[::-1]
    fig = viz.figure(
        "O que dirige o score do blend: efeito causal vs. propensão a converter",
        f"Top {top} features por importância combinada. Causal = dirige o τ; conversão = dirige quem converte.",
        theme=theme,
        barmode="group",
    )
    cores = viz.palette(theme)
    series = [("combined", "combinada"), ("uplift", "causal (uplift)"), ("conversion", "conversão")]
    for i, (col, nome) in enumerate(series):
        fig.add_trace(go.Bar(
            y=dados.index, x=dados[col], name=nome, orientation="h",
            marker_color=cores[i % len(cores)], marker_line_width=0,
        ))
    return fig
