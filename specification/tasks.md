# Tasks: Pipeline de dados & EDA

> Implementa: 01-pipeline-eda/spec.md + plan.md
> Legenda: `[ ]` todo · `[~]` andamento · `[x]` feito · `[!]` bloqueado

---

## T-101 — Config tipada
- **Status:** [x]
- **Satisfies:** REQ-110
- **Depends on:** —
- **Files:** `src/config.py`, `tests/test_config.py`
- **Do:** Modelo Pydantic com janelas, limiar SMD (default 0.1), enum de prioridade de
  atribuição, caminhos, seeds. Loader que falha na validação.
- **Accept:** T-config passa — config com limiar negativo ou janela ≤ 0 levanta erro na carga.

## T-102 — Leitura e parsing de `value`
- **Status:** [x]
- **Satisfies:** REQ-101
- **Depends on:** T-101
- **Files:** `src/io.py`, `tests/test_parsing.py`
- **Do:** Ler os três JSONs; desempacotar `value`; coalescer `offer id` e `offer_id` numa
  referência única.
- **Accept:** T-parsing passa — received/viewed vêm de `offer id`, completed de `offer_id`,
  transaction tem `amount` e sem referência.

## T-103 — Normalização de perfil
- **Status:** [x]
- **Satisfies:** REQ-102
- **Depends on:** T-101
- **Files:** `src/clean.py`, `tests/test_profile.py`
- **Do:** Sentinela 118 → `identity_missing=1` e `age` null; `gender` ausente → `unknown`;
  derivar `tenure_days` de `registered_on`.
- **Accept:** T-G7 passa — exatamente os sentinela recebem a flag e `age` null.

## T-104 — Atribuição temporal
- **Status:** [x]
- **Satisfies:** REQ-103
- **Depends on:** T-102
- **Files:** `src/attribution.py`, `tests/test_attribution.py`
- **Do:** Atribuir transação à oferta vista cuja janela de validade a contém, sob regra de
  uma-ativa; aplicar prioridade configurada em sobreposição e logar a ocorrência.
- **Accept:** transação fora de janela não é atribuída; sobreposição usa a regra da config.

## T-105 — Label influence-aware
- **Status:** [x]
- **Satisfies:** REQ-104
- **Depends on:** T-104
- **Files:** `src/attribution.py`, `tests/test_label.py`
- **Do:** `converted=1` sse vista E transação atribuída dentro da validade; informational via
  janela pós-view.
- **Accept:** T-G3, T-G4, T-G5 passam.

## T-106 — Features anti-leakage
- **Status:** [x]
- **Satisfies:** REQ-105
- **Depends on:** T-104
- **Files:** `src/features.py`, `tests/test_leakage.py`
- **Do:** Construir `hist_*` (transacionais, resposta a ofertas) usando só eventos com
  `time < received_time`; features de oferta/contexto.
- **Accept:** T-G2 passa — evento pós-recebimento não entra em nenhuma feature.

## T-107 — Custo do desconto
- **Status:** [x]
- **Satisfies:** REQ-106
- **Depends on:** T-105
- **Files:** `src/cost.py`, `tests/test_cost.py`
- **Do:** Preencher `reward_cost` para conversões bogo/discount; 0 caso contrário.
- **Accept:** T-G6 passa.

## T-108 — Contrato e escrita
- **Status:** [ ]
- **Satisfies:** REQ-107
- **Depends on:** T-103, T-105, T-106, T-107
- **Files:** `src/contract.py`, `src/pipeline.py`, `tests/test_contract.py`
- **Do:** `StructType` do contrato; validação Pydantic de schema e amostra; escrever
  `data/processed/`; orquestrar bruto→processado no entrypoint CLI.
- **Accept:** T-G1 passa; G1–G8 verdes; validação de amostra sem erro; dataset escrito por CLI.

## T-109 — Tema de figuras
- **Status:** [ ]
- **Satisfies:** REQ-108 (parte visual)
- **Depends on:** T-101
- **Files:** `src/viz.py`
- **Do:** Módulo único de tema Plotly executivo: paleta sóbria limitada, sem gridlines densas,
  rótulos diretos, título-conclusão. Toda figura usa este tema.
- **Accept:** uma figura de exemplo renderiza no padrão; nenhuma figura define estilo ad hoc.

## T-110 — EDA
- **Status:** [ ]
- **Satisfies:** REQ-108
- **Depends on:** T-108, T-109
- **Files:** `notebooks/1_eda.ipynb`, `src/eda.py`
- **Do:** Distribuição dos eventos no tempo, seis ondas, completou-sem-ver por tipo,
  sobreposição de nulos, distribuições de features-chave. Notebook importa de `src/`.
- **Accept:** cada visão tem figura no tema; completou-sem-ver reportado por tipo.

## T-111 — Check de balanço
- **Status:** [ ]
- **Satisfies:** REQ-109
- **Depends on:** T-108, T-109
- **Files:** `src/eda.py`, `notebooks/1_eda.ipynb`
- **Do:** SMD por covariável entre tratado (viu) e controle (não viu); sinalizar acima do
  limiar da config como diagnóstico.
- **Accept:** SMD reportado por covariável; acima do limiar listado, sem alterar estimador.

---

## Execution notes

- Executar em ordem; ler os REQs e seções do plan antes de cada task.
- Rodar o **Accept** ao fim de cada task; só então marcar `[x]`.
- Task inexecutável como escrita → `[!]`, dizer o porquê, parar. Atualizar spec/plan antes de
  retomar; não improvisar escopo.
