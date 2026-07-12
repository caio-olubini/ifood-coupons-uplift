---
name: statistical-plots
description: >-
  Creates publication-quality statistical visualizations with Plotly and the
  project theme in src/viz.py — executive style, validated palette, high
  legibility, direct labels. Use when writing or editing plots, figures, charts,
  EDA visualizations, notebooks with graphs, or when the user mentions Plotly,
  statistical graphics, or figure styling.
---

# Statistical Plots (Plotly)

> Figuras estatísticas nascem de **Plotly + `src/viz.py`**. Legibilidade e
> honestidade visual vêm antes de decoração.

**Local canônico:** `.claude/skills/statistical-plots/` (Cursor lê via symlink em
`.cursor/skills/statistical-plots/`).

---

## Regra de ouro

1. **Sempre** use `viz.figure()` ou `theme.figure()` — nunca `go.Figure()` cru.
2. **Nunca** defina estilo ad hoc (paletas default, cores saturadas, layout solto).
3. **Cor não carrega identidade sozinha** — rótulo direto (`add_end_labels`,
   `add_bar_labels`) ou legenda enxuta.
4. **Escolha o gráfico certo**; se a mensagem exige tabela, use tabela.

---

## Workflow

```
1. apply_theme()                  # theme.py — uma vez por sessão/notebook
2. Agregar no Spark → pandas pequeno   # nunca plotar milhões de linhas
3. go.Bar / go.Scatter / go.Heatmap    # traces no fig do tema
4. add_bar_labels / add_end_labels     # rótulo direto
5. fig.show() ou save_figure(fig, path)
```

No projeto, prefira **`src/viz.py`** para padrões repetidos; o notebook
só agrega e chama helpers.

---

## Paleta e tema

```python
from src import viz
from src.viz import setup_theme, horizontal_bars

setup_theme()  # aplica template ifood_light
```

| Papel | Onde |
|-------|------|
| `viz.palette()` | Séries discretas (≤4 slots; 5ª vira "Outros" ou faceta) |
| `viz.SEQUENTIAL` | Magnitude unidirecional |
| `viz.DIVERGING_*` | Desvio em torno de zero |
| `viz.ink()` | Texto e eixos — nunca cor de série |

Detalhes em [palette.md](palette.md).

---

## Escolha do gráfico

| Pergunta | Plotly | Evitar |
|----------|--------|--------|
| Distribuição contínua | `histogram` + opcional KDE trace | Pizza, 3D |
| Comparar grupos (poucos) | `go.Bar` com `error_y` | Barras sem contexto de n |
| Comparar magnitudes (categorias) | `go.Bar(orientation='h')` | Barras verticais com rótulos longos |
| Relação x–y | `go.Scatter` + `scattermode` | Regressão em nuvem sem sentido |
| Matriz de correlação | `go.Heatmap` (annot só se p≤8 vars) | Heatmap gigante |
| Série temporal | `go.Scatter(mode='lines+markers')` | Área empilhada confusa |
| Composição categórica | `go.Bar` + % no `text` | Pie chart >3 fatias |

Sempre declare **denominador** no subtítulo quando a taxa depende dele.

---

## Boas práticas

- Título afirma a **conclusão**; subtítulo traz `n=` e denominador.
- Eixos com unidade (`Taxa de conversão (%)`, `Valor (R$)`).
- Máximo **4** cores distintas por figura (paleta validada do projeto).
- Escala log: avisar no eixo ou subtítulo.
- `height` proporcional ao nº de categorias em barras horizontais.

---

## Anti-padrões

| Não fazer | Por quê |
|-----------|---------|
| `plotly.express` sem passar pelo template | Perde tema executivo |
| Seaborn/matplotlib em EDA nova do projeto | Fora do padrão atual |
| Pie chart >3 fatias | Ângulos humanos são ruins |
| Cor como única chave | Falha CVD |
| Função `fig_*` por gráfico em `src/eda.py` | Anti-padrão; usar as primitivas de `src/viz.py` |

---

## Exemplo mínimo

```python
import plotly.graph_objects as go
from src import viz
from src.viz import setup_theme, horizontal_bars

setup_theme()

fig = horizontal_bars(
    df,
    category="fonte",
    value="linhas",
    title="Volume condensa o bruto no grão analítico",
    subtitle="n=306.534 eventos · escala log",
    log_scale=True,
)
fig.show()
```

Mais receitas: [reference.md](reference.md).

---

## Checklist antes de entregar

- [ ] `setup_theme()` ou `apply_theme()` aplicado
- [ ] Paleta de `src/viz.py`, não default Plotly
- [ ] Eixos rotulados; taxa nomeia denominador
- [ ] `n` visível quando amostra não é óbvia
- [ ] Rótulos diretos nas barras/séries
- [ ] ≤4 categorias coloridas ou agrupadas
