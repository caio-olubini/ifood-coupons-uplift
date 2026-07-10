# Plan: Pipeline de dados & EDA

> Implementa: 01-pipeline-eda/spec.md · Produz: schema-processed.md

## Tech stack & key decisions

| Decisão | Escolha | Rationale |
|---|---|---|
| Ambiente | UV | Reprodutível, offline, sem dependência de nuvem; comandos via `uv run`. |
| Processamento | PySpark local | Exigência do case; 306k eventos com joins temporais idiomáticos. |
| Config | Pydantic `BaseSettings` | Parâmetros validados na carga; elimina hardcode (REQ-110). |
| Contrato | Spark `StructType` + modelo Pydantic | Schema imposto na escrita; Pydantic valida schema+amostra (REQ-107). |
| Figuras | Plotly | Exportável, interativo em notebook, controle fino do estilo executivo. |
| Testes | pytest + `chispa` (ou asserções sobre DataFrame) | Cobre G1–G10 com dados sintéticos minúsculos e determinísticos. |
| Notebooks | só análise | Transformação em `src/`; notebook importa e exibe (spec, NFR). |

## Architecture

Fluxo em estágios, cada um uma função pura testável em `src/`, orquestrados por um entrypoint CLI.

- **`src/config.py`** — modelo Pydantic da config (janelas, limiar SMD, prioridade de
  atribuição, caminhos, seeds). Carregado uma vez, passado adiante.
- **`src/io.py`** — leitura dos três JSONs; parsing de `value` com coalescência
  `offer id`/`offer_id` (REQ-101).
- **`src/clean.py`** — normalização de perfil, sentinela de identidade, `tenure` (REQ-102).
- **`src/attribution.py`** — atribuição temporal oferta→transação sob a regra de uma-ativa
  (REQ-103) e construção do label influence-aware (REQ-104).
- **`src/features.py`** — features `hist_*` com janela estritamente anterior ao recebimento
  (REQ-105) e features de oferta/contexto.
- **`src/cost.py`** — `reward_cost` coerente (REQ-106).
- **`src/contract.py`** — `StructType` do contrato + validação Pydantic de schema e amostra
  (REQ-107).
- **`src/eda.py`** — visões e figuras Plotly (REQ-108), SMD do balanço (REQ-109) e segmentação
  K-Means de clientes (REQ-111).
- **`src/pipeline.py`** — orquestra bruto→processado; entrypoint CLI.

Notebooks em `notebooks/` importam de `src/` para exibir EDA e balanço; não contêm lógica.

## Data model

Saída = o grão e as colunas de `schema-processed.md`. O `StructType` em `src/contract.py` é a
fonte da verdade do schema físico; o modelo Pydantic espelha-o para validação de borda.

## Interfaces & contracts

```
config.load(path) -> PipelineConfig            # falha se inválida (REQ-110)
io.parse_events(spark, cfg) -> DataFrame        # value desempacotado (REQ-101)
clean.normalize_profile(df, cfg) -> DataFrame   # sentinela + tenure (REQ-102)
attribution.attribute(events, offers, cfg) -> DataFrame   # (REQ-103)
attribution.build_label(df, cfg) -> DataFrame   # converted influence-aware (REQ-104)
features.build(events, base, cfg) -> DataFrame  # hist_* anti-leakage (REQ-105)
contract.validate(df) -> None                   # levanta se G1–G10 falham (REQ-107)
eda.covariate_balance(df, cfg) -> DataFrame     # SMD viu/não-viu por covariável (REQ-109)
eda.assignment_balance(df, cfg) -> DataFrame    # SMD entre ofertas recebidas — verifica a Premissa 4
```

A regra de prioridade de atribuição (usada quando há sobreposição apesar da Premissa 1) é um
enum na config, não um `if` embutido — mantém a decisão visível e ajustável.

## Dependencies

- **Externas:** `pyspark`, `pydantic`, `plotly`, `pytest`; opcional `chispa` para asserções
  de DataFrame. Versões fixadas em `requirements.txt` / lock do UV.
- **Internas:** nenhuma — este é o primeiro estágio.

## Risks & mitigations

- **Risco:** parser ler só `offer_id` e perder 100% dos received/viewed. → **Mitigação:**
  REQ-101 com teste que falha se a coalescência não cobrir os dois nomes.
- **Risco:** leakage temporal sutil numa feature de janela. → **Mitigação:** teste G2 com
  cliente sintético cujo evento pós-recebimento envenenaria a feature se incluído.
- **Risco:** Pydantic por-linha estrangular o Spark. → **Mitigação:** contrato valida schema
  e amostra; proibição explícita de validação em UDF (Premissa 7).
- **Risco:** figuras poluídas fugindo do padrão executivo. → **Mitigação:** um módulo único de
  tema Plotly aplicado a todas as figuras; nenhuma figura define estilo ad hoc.

## Testes (crítico — a validação do pipeline é inegociável)

A suíte não é decorativa; ela guarda os invariantes que quebram o projeto em silêncio. Cada
teste é load-bearing: se removido, uma mina volta a passar despercebida.

- **T-G1 grão único** — dois recebimentos distintos não colapsam; duplicata proposital é detectada.
- **T-G2 anti-leakage** — cliente com transação após o recebimento; a feature `hist_*` não a inclui.
- **T-G3 label exige view** — completed-sem-view resulta em `converted=0`.
- **T-G4 dentro da validade** — transação um dia após o fim não converte.
- **T-G5 informational** — conversão vem da janela pós-view, não de `offer completed`.
- **T-G6 custo coerente** — `reward_cost>0` implica conversão não-informational.
- **T-G7 sentinela** — os sentinela e só eles recebem `identity_missing=1` e `age` null.
- **T-G10 gasto mínimo** — compra abaixo do `min_value` não é atribuída, não converte e não custa;
  a oferta inelegível não rouba a transação da elegível.
- **T-parsing** — received/viewed leem `offer id`; completed lê `offer_id`.
- **T-config** — config inválida (limiar negativo, janela ≤ 0) falha na carga, antes do dado.

Dados de teste são fixtures sintéticas minúsculas e determinísticas, montadas para exercitar a
falha específica — não amostras do dataset real.

## Traceability

| Requirement | Satisfeito por |
|---|---|
| REQ-101 | `io.parse_events` + T-parsing |
| REQ-102 | `clean.normalize_profile` + T-G7 |
| REQ-103 | `attribution.attribute` + T-G10 |
| REQ-104 | `attribution.build_label` + T-G3, T-G4, T-G5, T-G10 |
| REQ-105 | `features.build` + T-G2 |
| REQ-106 | `cost` + T-G6, T-G10 |
| REQ-107 | `contract.validate` + T-G1 |
| REQ-108 | `eda` + tema Plotly |
| REQ-109 | `eda.covariate_balance` |
| REQ-110 | `config` + T-config |
