"""Tema executivo único para todas as figuras (REQ-108, T-109).

Nenhuma figura define estilo ad hoc: todas nascem de `figure()`, que aplica o
template registrado aqui. A paleta é a instância de referência do método de
dataviz, **validada por script** (banda de luminosidade OKLCH, piso de croma,
separação CVD entre pares adjacentes e contraste contra a superfície):

- light — pior ΔE CVD entre adjacentes 21,6 (alvo ≥ 12); porém `#1baf7a` (2,74:1)
  e `#eda100` (2,11:1) ficam abaixo de 3:1 contra a superfície clara;
- dark — pior ΔE CVD 10,3, no piso (8–12), legal apenas com codificação secundária.

Nos dois casos a cor **não pode carregar identidade sozinha**. Daí a regra desta
casa, que coincide com o padrão executivo da spec: toda série leva **rótulo
direto** além da legenda. `add_end_labels` e `add_bar_labels` existem para isso.

Rótulos e valores usam tinta de texto, nunca a cor da série — a marca colorida ao
lado é que carrega a identidade.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Ordem fixa dos slots categóricos: é o mecanismo de segurança CVD, não estética.
# Nunca cicle as cores; uma 5ª série vira "Outros" ou small multiples.
CATEGORICAL_LIGHT = ("#2a78d6", "#1baf7a", "#eda100", "#008300")
CATEGORICAL_DARK = ("#3987e5", "#199e70", "#c98500", "#008300")

SURFACE_LIGHT, SURFACE_DARK = "#fcfcfb", "#1a1a19"
INK_LIGHT = ("#0b0b0b", "#52514e", "#8a8983")  # primary, secondary, muted
INK_DARK = ("#ffffff", "#c3c2b7", "#8a8983")
GRID_LIGHT, GRID_DARK = "#eceae5", "#333330"

# Cores de estado são reservadas: nunca viram "série 4", e nunca aparecem sem
# rótulo/ícone ao lado (a cor jamais carrega o significado sozinha).
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Escalas contínuas. Sequencial para magnitude (zero na base), divergente para
# grandeza com zero significativo (correlação, z-score) — usar a sequencial num
# eixo divergente esconde o sinal do desvio.
SEQUENTIAL = ((0.0, "#cde2fb"), (1.0, "#0d366b"))
DIVERGING_LIGHT = ((0.0, "#0d366b"), (0.5, "#f2f1ee"), (1.0, "#8f2626"))
DIVERGING_DARK = ((0.0, "#7fb2ee"), (0.5, "#2e2e2c"), (1.0, "#e07a7a"))

LIGHT, DARK = "ifood_light", "ifood_dark"
_THEMES = {"light": LIGHT, "dark": DARK}


def _build_template(surface: str, ink: tuple[str, str, str], grid: str, colorway) -> go.layout.Template:
    primary, secondary, muted = ink
    axis = dict(
        showgrid=False,          # grade vertical some; a horizontal fica recessiva
        zeroline=False,
        showline=True,
        linecolor=grid,
        ticks="outside",
        tickcolor=grid,
        tickfont=dict(color=secondary, size=12),
        title=dict(font=dict(color=muted, size=12)),
    )
    return go.layout.Template(
        layout=go.Layout(
            colorway=list(colorway),
            paper_bgcolor=surface,
            plot_bgcolor=surface,
            font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=primary),
            title=dict(font=dict(size=18, color=primary), x=0, xanchor="left", y=0.95, yanchor="top"),
            margin=dict(l=72, r=132, t=96, b=56),  # folga à direita para rótulo direto
            xaxis={**axis},
            yaxis={**axis, "showgrid": True, "gridcolor": grid, "gridwidth": 1},
            legend=dict(
                orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(color=secondary, size=12), bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(font_size=12),
            colorscale=dict(sequential=[[0, "#cde2fb"], [1, "#0d366b"]]),
        )
    )


pio.templates[LIGHT] = _build_template(SURFACE_LIGHT, INK_LIGHT, GRID_LIGHT, CATEGORICAL_LIGHT)
pio.templates[DARK] = _build_template(SURFACE_DARK, INK_DARK, GRID_DARK, CATEGORICAL_DARK)
pio.templates.default = LIGHT


def palette(theme: str = "light") -> tuple[str, ...]:
    return CATEGORICAL_LIGHT if theme == "light" else CATEGORICAL_DARK


def ink(theme: str = "light") -> tuple[str, str, str]:
    return INK_LIGHT if theme == "light" else INK_DARK


def figure(title: str, subtitle: str | None = None, theme: str = "light", **layout) -> go.Figure:
    """Figura no tema. `title` deve **afirmar a conclusão**, não nomear os eixos."""
    _, secondary, _ = ink(theme)
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:{secondary}'>{subtitle}</span>"
    fig = go.Figure()
    fig.update_layout(template=_THEMES[theme], title_text=text, **layout)
    return fig


def add_end_labels(fig: go.Figure, theme: str = "light", xshift: int = 8) -> go.Figure:
    """Rótulo direto no fim de cada série (a cor sozinha não identifica — ver docstring)."""
    _, secondary, _ = ink(theme)
    for trace in fig.data:
        if not len(trace.x):
            continue
        fig.add_annotation(
            x=trace.x[-1], y=trace.y[-1], text=trace.name, showarrow=False,
            xanchor="left", xshift=xshift, font=dict(color=secondary, size=12),
        )
    return fig


def add_bar_labels(fig: go.Figure, template: str = "%{y:,.0f}", theme: str = "light") -> go.Figure:
    """Valor direto na ponta da barra, em tinta de texto."""
    _, secondary, _ = ink(theme)
    fig.update_traces(
        texttemplate=template, textposition="outside", cliponaxis=False,
        textfont=dict(color=secondary, size=12),
        marker_line_width=0,
    )
    return fig
