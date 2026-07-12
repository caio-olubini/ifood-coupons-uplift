# Receitas Plotly — statistical plots

Todas assumem `setup_theme()` já chamado e DataFrame pandas pequeno.

---

## Barras horizontais (magnitudes)

```python
from src.viz import horizontal_bars

fig = horizontal_bars(
    df,
    category="fonte",
    value="linhas",
    title="Volume de linhas por fonte",
    subtitle="events: 306.534 · processed: 76.277",
    xlabel="Linhas",
    log_scale=True,
)
fig.show()
```

---

## Barras verticais com participação

```python
from src.viz import vertical_share_bars

fig = vertical_share_bars(
    df,
    category="event",
    value="eventos",
    share="fracao",
    title="Transações dominam o fluxo bruto",
    subtitle="n=306.534 eventos",
)
fig.show()
```

---

## Histograma

```python
import plotly.graph_objects as go
from src import viz

fig = viz.figure("Distribuição de valor", "n=20.412", theme="light")
fig.add_trace(go.Histogram(x=df["valor"], marker_color=viz.palette()[0]))
fig.update_layout(xaxis_title="Valor (R$)", yaxis_title="Contagem", bargap=0.05)
fig.show()
```

---

## Scatter com hue

```python
import plotly.graph_objects as go
from src import viz

colors = viz.palette()
fig = viz.figure("Relação x–y", theme="light")
for color, grupo, parte in zip(colors, grupos, partes):
    fig.add_trace(go.Scatter(
        x=parte["x"], y=parte["y"], mode="markers", name=grupo,
        marker=dict(color=color, size=7, opacity=0.7),
    ))
fig.update_layout(xaxis_title="x", yaxis_title="y")
viz.add_end_labels(fig)
fig.show()
```

---

## Heatmap de correlação

```python
import plotly.graph_objects as go
from src import viz

corr = df[numeric_cols].corr()
fig = viz.figure("Correlação entre features", theme="light")
fig.add_trace(go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale=viz.DIVERGING_LIGHT,
    zmid=0,
    text=corr.round(2).values if len(numeric_cols) <= 8 else None,
    texttemplate="%{text}",
))
fig.update_layout(height=420)
fig.show()
```

---

## Linha temporal

```python
import plotly.graph_objects as go
from src import viz

fig = viz.figure("Taxa por onda de campanha", "Taxa sobre recebidos", theme="light")
for color, tipo, parte in zip(viz.palette(), tipos, partes):
    fig.add_trace(go.Scatter(
        x=parte["onda"], y=parte["taxa"], mode="lines+markers", name=tipo,
        line=dict(color=color), marker=dict(size=7),
    ))
fig.update_layout(xaxis_title="Onda", yaxis_title="Taxa de conversão (%)")
viz.add_end_labels(fig)
fig.show()
```

---

## Import no notebook

```python
from src.viz import setup_theme, horizontal_bars, vertical_share_bars, volumetry_figures

setup_theme()
```
