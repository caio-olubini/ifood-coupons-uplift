"""Simulador de alocação de cupons (spec 03).

Camada de apresentação estática por cima do projeto: `export.py` congela a matriz
pontuada (clientes × ofertas ativas) em JSON, e `index.html` refaz a seleção do
`serve.recommend` no browser. Importa livremente de `src/`; nunca é importado por
`src/` — não é uma etapa do pipeline.
"""
