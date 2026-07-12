# Paleta — statistical plots (Plotly)

Fonte canônica: **`src/viz.py`**. A skill espelha os tokens para referência rápida.

## Categórica (ordem fixa)

Não ciclar. Séries 5+ → agrupar "Outros" ou facetas.

| Slot | Light | Dark | Uso típico |
|------|-------|------|------------|
| 1 | `#2a78d6` | `#3987e5` | Referência / baseline |
| 2 | `#1baf7a` | `#199e70` | Tratamento / positivo |
| 3 | `#eda100` | `#c98500` | Segunda série / alerta |
| 4 | `#008300` | `#008300` | Terceira série |

## Sequencial

Magnitude unidirecional (`viz.SEQUENTIAL`):

```
#cde2fb → #0d366b
```

## Divergente

Desvio em torno de zero (`viz.DIVERGING_LIGHT` / `viz.DIVERGING_DARK`):

```
negativo → superfície → positivo
```

## Tinta (texto, não dados)

| Papel | Light | Dark |
|-------|-------|------|
| Primária | `#0b0b0b` | `#ffffff` |
| Secundária | `#52514e` | `#c3c2b7` |
| Muted | `#8a8983` | `#8a8983` |
| Grid | `#eceae5` | `#333330` |
| Superfície | `#fcfcfb` | `#1a1a19` |

## Estados (uso raro)

`viz.STATUS`: good, warning, serious, critical — nunca como "série 4".

## Contraste e CVD

Paleta validada por script no projeto (OKLCH, ΔE CVD, contraste). Cor **nunca**
carrega identidade sozinha — sempre rótulo direto ou legenda.
