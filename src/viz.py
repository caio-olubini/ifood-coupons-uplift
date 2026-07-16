"""Tema executivo único e primitivas de plot para todas as figuras (REQ-108, T-109).

Um só módulo, dois níveis. **Tema** (topo): none figura define estilo ad hoc —
todas nascem de `figure()`, que aplica o template registrado aqui. A paleta é a
instância de referência do método de dataviz, **validada por script** (banda de
luminosidade OKLCH, piso de croma, separação CVD entre pares adjacentes e
contraste contra a superfície):

- light — pior ΔE CVD entre adjacentes 21,6 (alvo ≥ 12); porém `#1baf7a` (2,74:1)
  e `#eda100` (2,11:1) ficam abaixo de 3:1 contra a superfície clara;
- dark — pior ΔE CVD 10,3, no piso (8–12), legal apenas com codificação secundária.

Nos dois casos a cor **não pode carregar identidade sozinha**. Daí a regra desta
casa, que coincide com o padrão executivo da spec: toda série leva **rótulo
direto** além da legenda. `add_end_labels` e `add_bar_labels` existem para isso.
Rótulos e valores usam tinta de texto, nunca a cor da série — a marca colorida ao
lado é que carrega a identidade.

**Primitivas** (base): barras, linhas, heatmap, histograma, facetas — funções
parametrizadas que escondem o Plotly cru e padronizam o resultado. O notebook
agrega o dado (Spark → pandas pequeno) e chama a primitiva certa passando
título/subtítulo/rótulos próprios (o conteúdo é da célula; só o estilo é daqui).
Cor e tinta saem sempre das funções de tema deste módulo, nunca hex hardcoded.
Não há função `fig_*` por gráfico: um gráfico específico é a *chamada* de uma
primitiva com os argumentos daquele gráfico, não uma função nova.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
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


# ==============================================================================
# Primitivas de plot — barras, linhas, distribuições, facetas.
# ==============================================================================

def setup_theme(*, theme: str = "light") -> None:
    """Aplica o template do projeto uma vez por sessão de notebook."""
    pio.templates.default = LIGHT if theme == "light" else DARK


def format_pt_br(n: int | float) -> str:
    return _format_pt_br(n)


def _format_pt_br(n: int | float) -> str:
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, int):
        return f"{n:,}".replace(",", ".")
    return f"{n:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_count(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def _side_margin_for_labels(labels: Sequence[str], *, px_per_char: int = 8, floor: int = 120) -> int:
    """Piso de espaço lateral proporcional ao rótulo mais longo do eixo de categorias.

    `automargin` mede a largura real e expande além disto se preciso; 8px/char
    (Inter 12px) cobre glifos largos (`m`, `_`) para o piso já não cortar sozinho.
    """
    longest = max((len(str(label)) for label in labels), default=0)
    return max(floor, longest * px_per_char + 40)


def _bar_layout(
    fig: go.Figure,
    *,
    categories: Sequence[str] | None = None,
    horizontal: bool = False,
    height: int | None = None,
    has_outside_labels: bool = True,
    has_legend: bool = False,
    log_value_axis: bool = False,
    value_max: float | None = None,
) -> go.Figure:
    """Margens e automargin para barras — evita rótulo colado ou cortado na borda.

    Em barras horizontais o `l` é só um **piso**: `automargin=True` mede a
    largura real do rótulo mais longo e expande além dele se preciso (o palpite
    de `_side_margin_for_labels` subestima glifos largos, e um `l` fixo curto
    corta o rótulo em vez de crescer). O piso evita a figura colapsar quando os
    rótulos são curtos."""
    margins = dict(l=72, r=132, t=110, b=72)
    if horizontal:
        if categories is not None:
            margins["l"] = _side_margin_for_labels(categories)
        if has_outside_labels:
            margins["r"] = 180 if log_value_axis else 148
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if log_value_axis and value_max is not None and value_max > 0:
            fig.update_xaxes(type="log", range=[None, value_max * 2.5])
    else:
        if has_outside_labels:
            margins["t"] = 118
            margins["r"] = 72
        if categories is not None:
            margins["b"] = max(margins["b"], 88)
        fig.update_xaxes(automargin=True, tickangle=-18)
        fig.update_yaxes(automargin=True)
        if has_outside_labels and value_max is not None:
            fig.update_yaxes(range=[0, value_max * 1.18])

    if has_legend:
        margins["r"] = max(margins["r"], 196)

    fig.update_layout(margin=margins, height=height)
    return fig


# --- Barras -------------------------------------------------------------------

def horizontal_bars(
    data: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    subtitle: str | None = None,
    xlabel: str = "Contagem",
    color: str | None = None,
    log_scale: bool = False,
    labels: bool = True,
    hovertemplate: str = "%{y}<br>%{x:,.0f}<extra></extra>",
    theme: str = "light",
) -> go.Figure:
    """Barras horizontais de série única — comparar magnitudes entre categorias.

    Ordena por `value` (mais alto no topo). `labels=True` escreve a contagem
    formatada na ponta de cada barra (bom para volumetria); `labels=False` deixa
    a barra sem rótulo (bom para valores fracionários como importância
    normalizada, onde a contagem inteira mentiria).
    """
    ordered = data.sort_values(value, ascending=True)
    n = len(ordered)
    _, secondary, _ = ink(theme)
    bar_color = color or palette(theme)[0]

    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(
        go.Bar(
            y=ordered[category],
            x=ordered[value],
            orientation="h",
            marker_color=bar_color,
            marker_line_width=0,
            text=[_format_count(v) for v in ordered[value]] if labels else None,
            textposition="outside",
            textfont=dict(color=secondary, size=12),
            cliponaxis=False,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title="",
        showlegend=False,
    )
    if log_scale:
        fig.update_xaxes(title=f"{xlabel} (escala log)")
    # ~38px por barra: confortável numa volumetria de 3–4 e sem esticar num
    # ranking from 15 (72px/barra dava uma figura from 1200px, barras enormes).
    _bar_layout(
        fig,
        categories=ordered[category].tolist(),
        horizontal=True,
        height=max(300, 38 * n + 120),
        log_value_axis=log_scale,
        value_max=float(ordered[value].max()) if len(ordered) else None,
    )
    return fig


def vertical_bars(
    data: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "Contagem",
    label_template: str = "%{y:,.0f}",
    tickformat: str | None = None,
    xticktext: Sequence[str] | None = None,
    hovertemplate: str | None = None,
    customdata: np.ndarray | None = None,
    color: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Barras verticais de série única, com valor direto na ponta.

    `xticktext` troca o rótulo do eixo x sem reordenar os dados (ex.: "wave 1
    <br>t=0"); `tickformat` (`.0%`, `.2f`) formata o eixo y; `customdata`/
    `hovertemplate` customizam o tooltip. O rótulo na barra usa `label_template`
    (`%{y:.1%}` para taxa, `%{y:,.0f}` para contagem).
    """
    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(go.Bar(
        x=xticktext if xticktext is not None else data[category],
        y=data[value],
        marker_color=color or palette(theme)[0],
        customdata=customdata,
        hovertemplate=hovertemplate or "%{x}<br>%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=ylabel, showlegend=False)
    if tickformat is not None:
        fig.update_layout(yaxis_tickformat=tickformat)
    fig = add_bar_labels(fig, label_template, theme)
    _bar_layout(
        fig,
        categories=(xticktext if xticktext is not None else data[category].tolist()),
        horizontal=False,
        height=380,
        value_max=float(data[value].max()) if len(data) else None,
    )
    return fig


def vertical_bars_ci(
    data: pd.DataFrame,
    *,
    category: str,
    value: str,
    value_lo: str,
    value_hi: str,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "Valor",
    label_template: str = "R$ %{y:,.0f}",
    tickformat: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Barras verticais com intervalo de confiança assimétrico (`value_lo`/`value_hi`)."""
    cores = palette(theme)
    _, secondary, _ = ink(theme)

    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(go.Bar(
        x=data[category],
        y=data[value],
        marker_color=[cores[i % len(cores)] for i in range(len(data))],
        marker_line_width=0,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(data[value_hi] - data[value]).to_numpy(),
            arrayminus=(data[value] - data[value_lo]).to_numpy(),
            color=secondary,
            thickness=1.5,
            width=6,
        ),
        text=data[value],
        texttemplate=label_template,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=secondary, size=11),
        hovertemplate=(
            "%{x}<br>R$ %{y:,.2f}<br>IC [%{customdata[0]:,.2f}, %{customdata[1]:,.2f}]"
            "<extra></extra>"
        ),
        customdata=np.column_stack([data[value_lo], data[value_hi]]),
    ))
    fig.update_layout(yaxis_title=ylabel, showlegend=False)
    if tickformat is not None:
        fig.update_layout(yaxis_tickformat=tickformat)
    ymax = float(data[value_hi].max()) if len(data) else None
    _bar_layout(
        fig,
        categories=data[category].tolist(),
        horizontal=False,
        height=400,
        value_max=ymax,
    )
    return fig


def vertical_share_bars(
    data: pd.DataFrame,
    *,
    category: str,
    value: str,
    share: str,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "Contagem",
    theme: str = "light",
) -> go.Figure:
    """Barras verticais com rótulo de participação — quebras de composição."""
    ordered = data.sort_values(value, ascending=False)
    _, secondary, _ = ink(theme)

    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(
        go.Bar(
            x=ordered[category],
            y=ordered[value],
            marker_color=palette(theme)[0],
            text=[f"{row[share]:.1%}" for _, row in ordered.iterrows()],
            textposition="outside",
            textfont=dict(color=secondary, size=12),
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.0f}<br>%{text}<extra></extra>",
        )
    )
    ymax = ordered[value].max()
    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title="",
        showlegend=False,
    )
    _bar_layout(
        fig,
        categories=ordered[category].tolist(),
        horizontal=False,
        height=400,
        value_max=float(ymax) if len(ordered) else None,
    )
    return fig


def grouped_bars(
    data: pd.DataFrame,
    *,
    category: str,
    series: Sequence[str],
    title: str,
    subtitle: str | None = None,
    orientation: str = "v",
    value_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    series_names: Sequence[str] | None = None,
    height: int | None = None,
    theme: str = "light",
) -> go.Figure:
    """Barras agrupadas: uma categoria no eixo, uma barra por série ao lado.

    `data` no formato *wide* — uma linha por categoria, uma coluna por série de
    `series`. Cada série ganha uma cor da paleta (na ordem) e sua entrada na
    legenda. `value_label` (ex.: `%{y:.2f}`, `%{x:.3f}`) escreve o valor na ponta
    de cada barra; `orientation="h"` deita as barras (a categoria vira o eixo y).
    `series_names` renomeia as séries na legenda sem tocar as colunas.
    """
    names = list(series_names) if series_names is not None else list(series)
    _, secondary, _ = ink(theme)
    cores = palette(theme)
    horizontal = orientation == "h"

    fig = figure(title, subtitle, theme=theme, barmode="group")
    for i, (col, nome) in enumerate(zip(series, names)):
        eixo = {"y" if horizontal else "x": data[category],
                "x" if horizontal else "y": data[col]}
        trace = go.Bar(
            name=nome, orientation=orientation,
            marker_color=cores[i % len(cores)], marker_line_width=0, **eixo,
        )
        if value_label:
            trace.update(
                text=data[col], texttemplate=value_label,
                textposition="outside", cliponaxis=False,
                textfont=dict(color=secondary, size=10),
            )
        fig.add_trace(trace)
    if height is None:
        height = max(400, 56 * len(data) + 160) if horizontal else 400
    has_legend = len(series) > 1
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, showlegend=has_legend)
    value_max = float(data[list(series)].max().max()) if len(data) else None
    _bar_layout(
        fig,
        categories=data[category].tolist(),
        horizontal=horizontal,
        height=height,
        has_legend=has_legend,
        value_max=value_max,
    )
    return fig


def stacked_bars(
    data: pd.DataFrame,
    *,
    category: str,
    segment: str,
    value: str,
    title: str,
    subtitle: str | None = None,
    order: Sequence[str] | None = None,
    theme: str = "light",
) -> go.Figure:
    """Barras empilhadas: uma barra por `category`, um segmento por `segment`.

    `data` no formato *long* (`[category, segment, value]`); é pivotado aqui. `order`
    fixa a ordem de empilhamento e a cor de cada segmento entre figuras (ex.:
    passar `quadrant.QUADRANT_ORDER` para que composição e revenue usem a mesma cor
    por quadrante). Segmentos ausentes de `order` são ignorados; presentes mas sem
    ordem entram depois, na ordem de aparição.
    """
    pivot = data.pivot(index=category, columns=segment, values=value)
    if order is not None:
        segmentos = [s for s in order if s in pivot.columns]
        segmentos += [s for s in pivot.columns if s not in segmentos]
    else:
        segmentos = list(pivot.columns)

    fig = figure(title, subtitle, theme=theme, barmode="stack")
    cores = palette(theme)
    for i, seg in enumerate(segmentos):
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[seg], name=seg,
            marker_color=cores[i % len(cores)], marker_line_width=0,
        ))
    return fig


# --- Linhas -------------------------------------------------------------------

def line_series(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    mode: str = "lines",
    end_labels: bool = True,
    tickformat: str | None = None,
    log_y: bool = False,
    category_order: Sequence[str] | None = None,
    discrete_when: Callable[[str], bool] | None = None,
    vlines: Sequence[float] | None = None,
    reference: tuple[Sequence, Sequence, str] | None = None,
    ci: tuple[str, str] | None = None,
    hover_unit: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Uma linha por grupo ao longo de `x` — curvas, séries temporais, tendências.

    `group=None` desenha uma série só. `mode` controla a marca padrão
    (`"lines"`, `"lines+markers"`, `"markers"`); `discrete_when(nome)` força
    marcadores por série (ex.: eventos de campanha são disparos discretos, só a
    compra é contínua). `category_order` reindexa o eixo x categórico. `vlines`
    marca instantes (ondas, k escolhido) com linha pontilhada; `reference` desenha
    uma reta tracejada nomeada (baseline random do Qini). `ci=(lo, hi)` sombreia
    a banda de confiança de cada série. Rótulo direto no fim de cada série
    (`end_labels`) — a cor sozinha não identifica (ver docstring do módulo).
    """
    cores = palette(theme)
    grupos = [None] if group is None else list(dict.fromkeys(data[group]))

    fig = figure(title, subtitle, theme=theme)
    for i, nome in enumerate(grupos):
        serie = data if nome is None else data[data[group] == nome]
        if category_order is not None:
            serie = serie.set_index(x).reindex(category_order).reset_index()
        cor = cores[i % len(cores)]
        discreto = discrete_when(nome) if (discrete_when and nome is not None) else False
        modo = "markers" if discreto else mode
        unidade = f" {hover_unit}" if hover_unit else ""
        rotulo = "" if nome is None else f"{nome}<br>"
        fig.add_trace(go.Scatter(
            x=serie[x], y=serie[y], name=(nome if nome is not None else y),
            mode=modo,
            line=dict(color=cor, width=2.5) if "lines" in modo else None,
            marker=dict(color=cor, size=8) if "markers" in modo else None,
            hovertemplate=f"{rotulo}%{{x}}<br>%{{y}}{unidade}<extra></extra>",
        ))
        if ci is not None:
            lo, hi = ci
            fig.add_trace(go.Scatter(
                x=list(serie[x]) + list(serie[x][::-1]),
                y=list(serie[hi]) + list(serie[lo][::-1]),
                fill="toself", fillcolor=_rgba(cor, 0.15),
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))

    if reference is not None:
        ref_x, ref_y, ref_name = reference
        fig.add_trace(go.Scatter(
            x=ref_x, y=ref_y, name=ref_name, mode="lines",
            line=dict(color=ink(theme)[1], width=1.5, dash="dot"),
        ))
    for vx in vlines or ():
        fig.add_vline(x=vx, line=dict(color=ink(theme)[2], width=1, dash="dot"))

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    if tickformat is not None:
        fig.update_layout(yaxis_tickformat=tickformat)
    if log_y:
        fig.update_layout(yaxis_type="log")
        # end_labels em escala log distorce o autorange (ex.: eixo até 10^147).
    elif end_labels:
        add_end_labels(fig, theme)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --- Distribuições ------------------------------------------------------------

def histogram(
    data: pd.DataFrame,
    *,
    x: str,
    y: str | None = None,
    title: str,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "contagem",
    markers: Sequence[tuple[float, str]] | None = None,
    color: str | None = None,
    hovertemplate: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Distribuição em barras. `y=None` conta as ocorrências de `x` (`go.Histogram`);
    `y` desenha um histograma pré-agregado (centro do bin em `x`, contagem em `y`).

    `markers=[(x, rótulo), ...]` desenha linhas verticais de referência (ex.: o Qini
    real e o limiar do placebo sobre a distribuição nula) — a primeira em cor de
    série, as demais em tinta pontilhada.
    """
    bar_color = color or palette(theme)[0]
    fig = figure(title, subtitle, theme=theme)
    if y is None:
        fig.add_trace(go.Histogram(
            x=data[x], marker_color=bar_color,
            hovertemplate=hovertemplate or "%{x}<br>%{y:,}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Bar(
            x=data[x], y=data[y], marker_color=bar_color,
            hovertemplate=hovertemplate or "%{x}<br>%{y:,}<extra></extra>",
        ))

    ink_primary, _, _ = ink(theme)
    for i, (mx, _label) in enumerate(markers or ()):
        estilo = (dict(color=bar_color, width=2.5)
                  if i == 0 else dict(color=ink_primary, width=1.5, dash="dot"))
        fig.add_vline(x=mx, line=estilo)

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False, bargap=0.05)
    return fig


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    subtitle: str | None = None,
    diverging: bool = True,
    zmin: float | None = None,
    zmax: float | None = None,
    colorbar_title: str = "",
    annotate: bool | str = "auto",
    text: np.ndarray | None = None,
    text_template: str = "%{text:.2f}",
    hovertemplate: str | None = None,
    reverse_y: bool = True,
    theme: str = "light",
) -> go.Figure:
    """Heatmap com a escala do tema. `diverging=True` usa `DIVERGING_*` (zero
    significativo: correlação, z-score); `False` usa a sequencial de magnitude.

    `annotate="auto"` escreve o valor em cada célula quando a matriz é pequena
    (≤ 8×8); `annotate=True/False` força. `text` sobrepõe outra matriz de valores
    para anotar (ex.: média bruta sobre o z-score). Dimensiona a figura ao número
    de linhas/colunas e inverte o eixo y (reading de matriz).
    """
    escala = (DIVERGING_LIGHT if theme == "light" else DIVERGING_DARK) if diverging \
        else SEQUENTIAL
    z = matrix.to_numpy()
    if annotate == "auto":
        annotate = max(matrix.shape) <= 8
    z_text = text if text is not None else (matrix.round(2).to_numpy() if annotate else None)

    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(go.Heatmap(
        z=z, x=list(matrix.columns), y=list(matrix.index),
        zmin=zmin, zmax=zmax, zmid=0 if diverging else None,
        colorscale=[list(p) for p in escala],
        colorbar=dict(title=colorbar_title, tickfont=dict(color=ink(theme)[1], size=11)),
        text=z_text, texttemplate=(text_template if z_text is not None else None),
        textfont=dict(size=10, color=ink(theme)[0]),
        hovertemplate=hovertemplate or "%{y} × %{x}<br>%{z:.2f}<extra></extra>",
    ))
    n_rows, n_cols = matrix.shape
    lado = 22 * max(n_rows, n_cols) + 200
    fig.update_layout(
        height=90 + 46 * n_rows if annotate else lado,
        xaxis_tickangle=-45,
        margin=dict(l=160, r=48, t=130, b=120),
    )
    if reverse_y:
        fig.update_layout(yaxis_autorange="reversed")
    return fig


def markers_with_thresholds(
    data: pd.DataFrame,
    *,
    value: str,
    label: str,
    flag: str | None = None,
    thresholds: Sequence[float] = (),
    zero_line: bool = True,
    flag_note: str = " acima do limiar",
    title: str,
    subtitle: str | None = None,
    xlabel: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Marcadores num eixo contínuo com linhas de limiar — dumbbell/lollipop de
    desvio (ex.: SMD por covariável no balanço).

    `flag` é a coluna booleana que sinaliza "fora do limiar": marca esses pontos
    com cor de estado (`STATUS.critical`), símbolo losango e a anotação `flag_note`
    ao lado — a cor de estado nunca vem sozinha. `thresholds` desenha linhas
    pontilhadas (ex.: ±SMD); `zero_line` marca o zero.
    """
    ordenado = data.sort_values(value)
    _, secondary, muted = ink(theme)
    surface = SURFACE_LIGHT if theme == "light" else SURFACE_DARK
    base = palette(theme)[0]

    if flag is not None:
        cores = [STATUS["critical"] if f else base for f in ordenado[flag]]
        simbolos = ["diamond" if f else "circle" for f in ordenado[flag]]
    else:
        cores, simbolos = base, "circle"

    fig = figure(title, subtitle, theme=theme)
    fig.add_trace(go.Scatter(
        x=ordenado[value], y=ordenado[label], mode="markers",
        marker=dict(color=cores, size=12, symbol=simbolos,
                    line=dict(color=surface, width=2)),
        hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>", showlegend=False,
    ))
    for t in thresholds:
        fig.add_vline(x=t, line=dict(color=muted, width=1, dash="dot"))
    if zero_line:
        fig.add_vline(x=0, line=dict(color=secondary, width=1))

    if flag is not None:
        for _, linha in ordenado.iterrows():
            if linha[flag]:
                fig.add_annotation(x=linha[value], y=linha[label], text=flag_note,
                                   showarrow=False, xanchor="left", xshift=10,
                                   font=dict(color=secondary, size=11))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=None)
    return fig


# --- Facetas (small multiples) ------------------------------------------------

def faceted(
    panels: Sequence[dict],
    *,
    title: str,
    subtitle: str | None = None,
    kind: str = "bar",
    cols: int = 2,
    shared_yaxes: bool = False,
    vline: float | None = None,
    vlines: Sequence[float] | None = None,
    xlabel: str | None = None,
    x_range: tuple[float, float] | None = None,
    x_dtick: float | None = None,
    y_tickformat: str | None = None,
    y_pad: float = 0.12,
    row_height: int = 260,
    theme: str = "light",
) -> go.Figure:
    """Small multiples: um painel por elemento de `panels`, cada um com sua escala.

    Cada painel é um dict `{"title", "x", "y"}`; opcionalmente `"kind"` por painel
    (`"bar"` ou `"line"`), senão usa o `kind` global. `cols` define a grade (linhas
    derivadas). `shared_yaxes` compartilha a escala y entre painéis.
    `vline`/`vlines` desenham referências verticais em todos os painéis (ex.: ondas
    de campanha). `x_range` e `x_dtick` alinham o eixo temporal entre painéis.
    `y_tickformat` e `y_pad` formatam o eixo y e reservam margem acima do pico.
    """
    from plotly.subplots import make_subplots

    n = len(panels)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols, shared_yaxes=shared_yaxes, shared_xaxes=True,
        subplot_titles=[p["title"] for p in panels],
        horizontal_spacing=0.10, vertical_spacing=0.22,
    )
    cores = palette(theme)
    marcas = vlines if vlines is not None else ([vline] if vline is not None else [])
    for i, p in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        cor = cores[i % len(cores)]
        marca = p.get("kind", kind)
        if marca == "line":
            trace = go.Scatter(
                x=p["x"], y=p["y"], mode="lines+markers",
                line=dict(color=cor, width=2), marker=dict(color=cor, size=7),
                hovertemplate="dia %{x}<br>%{y:,.0f}<extra></extra>",
            )
        else:
            trace = go.Bar(
                x=p["x"], y=p["y"], marker_color=cor,
                hovertemplate="dia %{x}<br>%{y:,.0f}<extra></extra>",
            )
        fig.add_trace(trace, row=r, col=c)
        for vx in marcas:
            fig.add_vline(
                x=vx, line=dict(color=ink(theme)[2], width=1, dash="dot"),
                row=r, col=c,
            )
        if not shared_yaxes and len(p["y"]) > 0:
            topo = max(p["y"]) * (1 + y_pad)
            eixo_y = {"range": [0, topo]}
            if y_tickformat is not None:
                eixo_y["tickformat"] = y_tickformat
            fig.update_yaxes(**eixo_y, row=r, col=c)

    base = figure(title, subtitle, theme=theme)
    fig.update_layout(
        template=base.layout.template, title=base.layout.title,
        showlegend=False, height=row_height * rows + 80,
        margin=dict(l=72, r=40, t=160, b=64),
    )
    eixo_x = {}
    if x_range is not None:
        eixo_x["range"] = list(x_range)
    if x_dtick is not None:
        eixo_x["dtick"] = x_dtick
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                title_text=xlabel if (r == rows and c == 1) else "",
                row=r, col=c,
                **eixo_x,
            )
    fig.update_annotations(font=dict(size=11, color=ink(theme)[1]), yshift=4)
    return fig


def timeline_ranges(
    data: pd.DataFrame,
    *,
    label: str,
    start: str,
    end: str,
    observed_end: str,
    censored: str,
    title: str,
    subtitle: str | None = None,
    xlabel: str | None = None,
    vline: float | None = None,
    theme: str = "light",
) -> go.Figure:
    """Barras horizontais de intervalo — janela de validade e right censoring.

    Cada linha desenha a parte observável em cor sólida e o trecho além do fim
    dos dados em tom mais claro. `vline` marca o instante em que a coleta termina.
    """
    cores = palette(theme)
    _, secondary, muted = ink(theme)
    fig = figure(title, subtitle, theme=theme, barmode="overlay")

    for i, row in data.iterrows():
        cor = cores[i % len(cores)]
        ini = float(row[start])
        obs_fim = float(row[observed_end])
        val_fim = float(row[end])
        fig.add_trace(go.Bar(
            y=[row[label]], x=[obs_fim - ini], base=ini, orientation="h",
            marker_color=cor, marker_line_width=0, showlegend=False,
            hovertemplate=(
                f"{row[label]}<br>observable: {ini:g}–{obs_fim:g} "
                f"({obs_fim - ini:g}d)<extra></extra>"
            ),
        ))
        if row[censored]:
            fig.add_trace(go.Bar(
                y=[row[label]], x=[val_fim - obs_fim], base=obs_fim, orientation="h",
                marker_color=_rgba(cor, 0.22), marker_line_width=0, showlegend=False,
                hovertemplate=(
                    f"{row[label]}<br>censored: {obs_fim:g}–{val_fim:g} "
                    f"({val_fim - obs_fim:g}d)<extra></extra>"
                ),
            ))

    fig.add_trace(go.Bar(
        x=[None], y=[None], name="observable window",
        marker_color=cores[0], orientation="h",
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None], name="right censoring",
        marker_color=_rgba(cores[0], 0.22), orientation="h",
    ))
    if vline is not None:
        fig.add_vline(
            x=vline, line=dict(color=muted, width=1.5, dash="dash"),
            annotation_text=f"end of data (t={vline:g})",
            annotation_position="top right",
            annotation_font=dict(color=secondary, size=11),
        )

    n = len(data)
    fig.update_layout(
        xaxis_title=xlabel,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=max(320, 56 * n + 180),
        margin=dict(l=200, r=48, t=150, b=64),
    )
    fig.update_yaxes(automargin=True, categoryorder="array", categoryarray=data[label].tolist())
    fig.update_xaxes(automargin=True)
    return fig
