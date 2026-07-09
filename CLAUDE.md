# CLAUDE.md

Guia para trabalhar neste repositório. Leia as specs em `specification/` antes
de mexer no pipeline — elas são a fonte da verdade, não este arquivo.

## O que é o projeto

Uplift de cupons iFood: decidir **qual oferta enviar (ou não enviar)** a cada
cliente para maximizar lucro líquido, e provar o ganho por avaliação offline,
sem rodar A/B. O problema é **incrementalidade (uplift)**, não classificação de
completion. Ver `specification/00-clarify.md` para as premissas numeradas.

## Comandos

```bash
uv sync                        # instala dependências (inclui grupo dev)
uv run pytest -q               # roda toda a suíte de testes de integridade
uv run pytest tests/test_leakage.py -q   # um arquivo só

uv run python -m src.pipeline   # bruto → data/processed/ (valida o contrato antes de escrever)
uv run python -m src.pipeline --config outra.yaml

# Executar um notebook de ponta a ponta (nbclient já está no grupo dev):
uv run python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('notebooks/0_pipeline_audit.ipynb',as_version=4); \
NotebookClient(nb,timeout=5400,kernel_name='python3',resources={'metadata':{'path':'.'}}).execute()"
```

Ambiente é gerenciado por **UV**, tudo roda local (sem nuvem/Databricks).
Processamento em **PySpark local**.

## Arquitetura

Fluxo em estágios, cada um uma função pura testável em `src/`. Notebooks só
**importam de `src/` e exibem** — nenhuma lógica de transformação vive neles.

| Módulo | Papel |
|---|---|
| `src/config.py` | `PipelineConfig` (Pydantic) lido de `config.yaml`. Todo parâmetro de comportamento vive na config — nada hardcoded em `src/` (REQ-110). `load(config_path=..., **overrides)`. |
| `src/io.py` | Lê os 3 JSONs; `parse_events` desempacota `value` e coalesce `offer id`/`offer_id` numa `offer_ref` única (REQ-101). |
| `src/clean.py` | `normalize_profile`: sentinela `age=118` → `identity_missing=1` + `age=null`; `gender` ausente → `unknown`; `tenure_days` (REQ-102). |
| `src/attribution.py` | `attribute`: grão `(account_id, offer_id, received_time)`, uma linha por `offer received`, com view e transações agregadas dentro da janela de validade; resolve sobreposição por `AttributionPriority` (REQ-103). `build_label`: `converted`/`conversion_value` influence-aware (REQ-104). |
| `src/features.py` | `build`: features `hist_*` sem leakage (só `time < received_time`, G2) + features de oferta/contexto (REQ-105). |
| `src/cost.py` | `add_reward_cost`: `reward_cost` = `discount_value` do catálogo em conversões bogo/discount; 0 caso contrário (REQ-106, G6). |
| `src/contract.py` | Encarnação executável do contrato (REQ-107): `StructType` **e** modelo Pydantic gerados de uma única lista `_COLUMNS` — divergir é impossível. `enforce_schema`, `assert_schema`, `assert_no_unexpected_nulls` (G8), `validate_sample`. |
| `src/pipeline.py` | Orquestra bruto→processado: `assemble_processed` (junta perfil, deriva `treatment` e `campaign_wave`, projeta no contrato), `validate`, `run` (escreve `data/processed/`), `build_spark(cfg)` e o entrypoint CLI. |

Ordem de chamada: `io.parse_events` → `clean.normalize_profile` (paralelo) →
`attribution.attribute` → `attribution.build_label` → `features.build` →
`cost.add_reward_cost` → junção do perfil + `treatment`/`campaign_wave` →
`contract.enforce_schema`. Tudo isso é `pipeline.assemble_processed`; `pipeline.run`
valida e escreve. `notebooks/0_pipeline_audit.ipynb` prova as garantias sobre o dado real.

`treatment` = a oferta foi **vista**. `campaign_wave` = rank do `received_time` distinto
(os disparos são discretos: t=0, 7, 14, 17, 21, 24) — **não** um bucket de largura fixa.

## Contrato e garantias

`specification/schema-processed.md` é **o contrato** entre pipeline (spec 01) e
modelagem (spec 02). O grão é `(account_id, offer_id, received_time)`, único.
Mudança no contrato é mudança de interface — atualize as specs, não só o código.

As garantias **G1–G9** são invariantes testados; se violados, quebram o projeto
em silêncio. Cada uma tem teste dedicado em `tests/`:

- **G1** grão único · **G2** sem leakage temporal · **G3** label exige view ·
  **G4** conversão pós-view e dentro da validade · **G5** informational sem
  `offer completed` · **G6** custo coerente · **G7** sentinela tratada ·
  **G8** sem nulo em coluna não-nullable · **G9** exposição exclusiva (uma view
  física marca no máximo um recebimento).

Testes usam **fixtures sintéticas minúsculas e determinísticas** montadas para
exercitar a falha específica — não amostras do dataset real. Rodar o dado real
já pegou bugs que as fixtures não pegaram (ex.: duplicação de grão por múltiplas
transações na janela); ao mexer no pipeline, valide também no notebook.

## Convenções

- **Configurabilidade é lei**: um valor mágico (janela, limiar, caminho, seed)
  dentro de uma função é um defeito. Vai em `config.yaml` / `PipelineConfig`.
- **Anti-leakage é estrutural**: features históricas filtram `event_time < received_time`
  *antes* de agregar, e re-anexam ao grão por left-join (linha sem histórico
  sobrevive com zeros). Nunca filtre depois de agregar.
- **Pydantic nas bordas, não no caminho quente do Spark** (Premissa 7): valida
  config, schema e amostra — nunca linha a linha em UDF.
- **Figuras** (a partir de T-109): Plotly, padrão executivo, tema único em
  `src/viz.py`; nenhuma figura define estilo ad hoc.

## Estado atual

Implementado e testado (43 testes verdes): T-101 a T-108 (config, io, clean,
attribution, label, features, cost, contrato + escrita). Ver `specification/tasks.md`
para o board. Próximo: T-109 (`src/viz.py`) e T-110 (EDA).

`notebooks/0_pipeline_audit.ipynb` **prova** as garantias G1–G9 e os REQ-101…110 sobre o
dado real: 54 verificações, cada uma um `assert` sobre o DataFrame completo — nenhuma
amostra, nenhum `try/except`. Roda de ponta a ponta com "Run All". Se uma garantia
regredir, o notebook fica vermelho na célula que a provou.

`notebooks/00_pipeline_review.ipynb` é **temporário** — foi superado pelo de auditoria;
descartar ou promover a `notebooks/1_eda.ipynb` em T-110.

Duas divergências entre spec e dado, levantadas pela auditoria (não são bugs do pipeline):
`completed` sem view precedente mede **25,8%**, não os 28,4% da Premissa 2; e o empate de
`received_time` (mesmo cliente, duas ofertas no mesmo instante) **não ocorre** no dado real
— o desempate estável por `offer_id` é defensivo e coberto só por fixture sintética.
