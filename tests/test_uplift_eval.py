"""T-205 — Qini/AUUC reportado; curva renderiza no tema executivo."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import load
from src.uplift_eval import fig_qini_curve, qini, qini_curve_points


def _synthetic_uplift_eval(n=500, seed=0):
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, size=n)
    uplift = rng.normal(0, 1, size=n)
    # y correlaciona com uplift previsto sob tratamento — sinal real a capturar.
    p = 1 / (1 + np.exp(-(uplift * treatment)))
    y = rng.binomial(1, p)
    return pd.Series(y), pd.Series(uplift), pd.Series(treatment)


def test_qini_score_beats_random_with_planted_signal():
    y, uplift, treatment = _synthetic_uplift_eval()
    score = qini(y, uplift, treatment)
    assert score > 0  # sinal plantado deve ordenar melhor que aleatório (score 0)


def test_qini_curve_renders_in_theme():
    cfg = load()
    y, uplift, treatment = _synthetic_uplift_eval()
    score = qini(y, uplift, treatment)
    curve = qini_curve_points(y, uplift, treatment)

    fig = fig_qini_curve(curve, score, cfg, theme="light")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # curva do modelo + referência aleatória
    assert fig.layout.template is not None
