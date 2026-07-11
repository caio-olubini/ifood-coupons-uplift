"""Classificação de quadrante de uplift.

Diagnóstico exploratório, não um requisito formal da spec: de que tipo de
cliente o holdout é composto, usando os quatro quadrantes clássicos de uplift
sobre μ₀/μ₁ previstos (`uplift.predict_stages`):

- **persuadable** (μ₀ baixo, μ₁ alto): converte só se tratado — o alvo ideal,
  τ real positivo.
- **sure thing** (μ₀ alto, μ₁ alto): converte de qualquer forma — enviar
  oferta paga desconto sem causar nada.
- **lost cause** (μ₀ baixo, μ₁ baixo): não converte de qualquer forma —
  enviar não muda o resultado, só teria custo se pagasse desconto.
- **sleeping dog** (μ₀ alto, μ₁ baixo): tratamento **atrapalha** — τ real
  negativo, "não enviar" é estritamente melhor.

O limiar (`cfg.quadrant_probability_threshold`, default 0,5) separa "alto" de
"baixo" em cada estágio — natural para μ₀/μ₁ serem probabilidades previstas de
um outcome binário.
"""

from __future__ import annotations

import pandas as pd

from src.config import PipelineConfig

PERSUADABLE = "persuadable"
SURE_THING = "sure_thing"
LOST_CAUSE = "lost_cause"
SLEEPING_DOG = "sleeping_dog"


def classify_quadrant(stages: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    """Quadrante de cada linha de `stages` (saída de `uplift.predict_stages`),
    a partir de `mu0`/`mu1` previstos e `cfg.quadrant_probability_threshold`.
    """
    threshold = cfg.quadrant_probability_threshold
    mu0_alto = stages["mu0"] >= threshold
    mu1_alto = stages["mu1"] >= threshold

    quadrante = pd.Series(index=stages.index, dtype=object)
    quadrante[~mu0_alto & mu1_alto] = PERSUADABLE
    quadrante[mu0_alto & mu1_alto] = SURE_THING
    quadrante[~mu0_alto & ~mu1_alto] = LOST_CAUSE
    quadrante[mu0_alto & ~mu1_alto] = SLEEPING_DOG
    return quadrante


def quadrant_distribution(stages: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Distribuição de quadrantes por `offer_type` — visão geral antes de olhar
    por política: o quanto de cada tipo de cliente existe no holdout, período.
    """
    quadrante = classify_quadrant(stages, cfg)
    contagem = (
        stages.assign(quadrante=quadrante)
        .groupby(["offer_type", "quadrante"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_por_tipo = contagem.groupby("offer_type")["n"].transform("sum")
    contagem["pct"] = contagem["n"] / total_por_tipo
    return contagem
