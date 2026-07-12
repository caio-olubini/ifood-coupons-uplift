# Simulador de Cupons — iFood

Interface estática que mostra, para um orçamento de N cupons, **a quem enviar,
com qual oferta e quanto isso deve render** — com projeção da campanha escolhida
e relatório comparando as três estratégias no histórico de validação.

## Pré-requisitos

- [UV](https://docs.astral.sh/uv/) instalado
- Dados processados e modelo treinado no repositório

## Passo a passo (4 comandos)

### 1. Processar os dados brutos

```bash
uv run python -m src.pipeline
```

Escreve `data/processed/` a partir dos JSONs em `data/raw/`.

### 2. Treinar o modelo de produção

```bash
uv run python -m src.cli train
```

Serializa o `BlendedUpliftModel` em `models/` (padrão: `models/blended_uplift_model.pkl`).

### 3. Exportar os artefatos do simulador

```bash
uv run python -m simulator.export
```

Gera em `simulator/data/`:

| Arquivo | Conteúdo |
|---------|----------|
| `matrix.json` | Matriz de serving pontuada (clientes × ofertas) |
| `holdout.json` | Holdout rotulado para o relatório comparativo |
| `offers.json` | Catálogo de ofertas |
| `metadata.json` | Defaults, lucro/receita médios, budgets do analytics, rótulos de UI |

Reexecute este comando sempre que retreinar o modelo ou alterar a config.

### 4. Abrir localmente

```bash
cd simulator
python -m http.server 8080
```

Abra [http://localhost:8080](http://localhost:8080) no navegador.

> Em produção o servidor é o **GitHub Pages** — não há backend; tudo roda no browser
> a partir dos JSONs estáticos.

## Uso da interface

1. Escolha a **estratégia**: Distribuir ao acaso, Priorizar conversão ou Priorizar uplift.
2. Marque os **tipos de cupom** e os **quadrantes de público** (Persuadables, Sure things, Lost causes, Sleeping dogs).
3. Ajuste o **orçamento**. A **exploração** só vale para Priorizar uplift.
4. Clique em **Gerar campanha** para ver a lista da estratégia escolhida e o **comparativo** das três abordagens com os mesmos filtros e orçamento.
5. **Baixar campanha (CSV)** exporta a audiência gerada (disponível após Gerar).

## Publicação (GitHub Pages)

**URL:** https://caio-olubini.github.io/ifood-coupons-uplift/

O deploy é automático via GitHub Actions (`.github/workflows/deploy-simulator-pages.yml`)
sempre que `main` recebe alterações em `simulator/`. O workflow publica o conteúdo de
`simulator/` como raiz do site — paths relativos (`./data/*.json`, `./assets/plotly.min.js`).

### Pré-requisitos no repositório

1. **Settings → Pages → Build and deployment → Source:** GitHub Actions.
2. Commite `simulator/data/` após cada `simulator.export` (artefatos estáticos).
3. Nenhum arquivo em `simulator/data/` deve ultrapassar 50 MB.

### Teste local

## Testes de paridade

```bash
uv run pytest tests/test_simulator_export.py -q
```

Verifica paridade da alocação com `serve.recommend`, fórmula do lucro projetado e
curva de ganho com `gaincurve.incremental_gain_curve`.
