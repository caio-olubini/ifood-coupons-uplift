"""Tema Plotly único para figuras estatísticas.

No repositório, delega a `src/viz.py` (fonte canônica do projeto).
Fora do repo, registra um template mínimo compatível.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import viz  # noqa: E402

LIGHT = viz.LIGHT
DARK = viz.DARK
CATEGORICAL_LIGHT = viz.CATEGORICAL_LIGHT
CATEGORICAL_DARK = viz.CATEGORICAL_DARK
SEQUENTIAL = viz.SEQUENTIAL
DIVERGING_LIGHT = viz.DIVERGING_LIGHT
DIVERGING_DARK = viz.DIVERGING_DARK


def apply_theme(theme: str = "light") -> None:
    """Define o template Plotly padrão da sessão."""
    pio.templates.default = LIGHT if theme == "light" else DARK


def palette(theme: str = "light") -> tuple[str, ...]:
    return viz.palette(theme)


def ink(theme: str = "light") -> tuple[str, str, str]:
    return viz.ink(theme)


def figure(title: str, subtitle: str | None = None, *, theme: str = "light", **layout) -> go.Figure:
    return viz.figure(title, subtitle, theme=theme, **layout)


def add_bar_labels(fig: go.Figure, template: str = "%{y:,.0f}", theme: str = "light") -> go.Figure:
    return viz.add_bar_labels(fig, template=template, theme=theme)


def add_end_labels(fig: go.Figure, theme: str = "light", xshift: int = 8) -> go.Figure:
    return viz.add_end_labels(fig, theme=theme, xshift=xshift)


def save_figure(fig: go.Figure, path: str | Path, *, scale: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lstrip(".").lower() or "html"
    if fmt == "html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out), format=fmt, scale=scale)
    return out
