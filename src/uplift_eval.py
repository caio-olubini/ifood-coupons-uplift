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
from sklift.metrics import qini_auc_score, qini_curve, uplift_auc_score

from src import uplift
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
