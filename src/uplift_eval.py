"""Avaliação de uplift por Qini/AUUC (REQ-203).

Seleção e comparação de modelos de uplift nunca usa AUC/F1 (métrica de
classificação): um modelo pode classificar `converted` bem e ainda ordenar mal
o *efeito incremental* — Qini mede a segunda coisa. Reusa `sklift.metrics`
(implementação testada) em vez de reimplementar a curva.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from sklift.metrics import qini_auc_score, qini_curve

from src import viz
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
