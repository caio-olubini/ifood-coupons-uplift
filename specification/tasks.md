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
- **Do:** Atribuir transação à oferta vista cuja janela de validade a contém **e cujo `min_value`
  a transação atinge**, sob regra de uma-ativa; aplicar prioridade configurada em sobreposição e
  logar a ocorrência. O gasto mínimo é filtrado antes do desempate de posse.
- **Accept:** transação fora de janela não é atribuída; sobreposição usa a regra da config;
  T-G10 passa — compra abaixo do `min_value` não é atribuída e a oferta inelegível não a rouba.

## T-105 — Label influence-aware
- **Status:** [x]
- **Satisfies:** REQ-104
- **Depends on:** T-104
- **Files:** `src/attribution.py`, `tests/test_label.py`
- **Do:** `converted=1` sse vista E transação atribuída dentro da validade; informational via
  janela pós-view. Como T-104 só atribui transações ≥ `min_value`, o label herda G10.
- **Accept:** T-G3, T-G4, T-G5, T-G10 passam.

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
- **Do:** Preencher `reward_cost` para conversões bogo/discount; 0 caso contrário. Correto porque
  G10 garante que toda conversão atingiu o `min_value` — nenhum desconto indevido é debitado.
- **Accept:** T-G6 e T-G10 passam.

## T-108 — Contrato e escrita
- **Status:** [x]
- **Satisfies:** REQ-107
- **Depends on:** T-103, T-105, T-106, T-107
- **Files:** `src/contract.py`, `src/pipeline.py`, `tests/test_contract.py`
- **Do:** `StructType` do contrato; validação Pydantic de schema e amostra; escrever
  `data/processed/`; orquestrar bruto→processado no entrypoint CLI.
- **Accept:** T-G1 passa; G1–G10 verdes; validação de amostra sem erro; dataset escrito por CLI.

## T-109 — Tema de figuras
- **Status:** [x]
- **Satisfies:** REQ-108 (parte visual)
- **Depends on:** T-101
- **Files:** `src/viz.py`
- **Do:** Módulo único de tema Plotly executivo: paleta sóbria limitada, sem gridlines densas,
  rótulos diretos, título-conclusão. Toda figura usa este tema.
- **Accept:** uma figura de exemplo renderiza no padrão; nenhuma figura define estilo ad hoc.

## T-110 — EDA
- **Status:** [x]
- **Satisfies:** REQ-108
- **Depends on:** T-108, T-109
- **Files:** `notebooks/1_eda.ipynb`, `src/eda.py`
- **Do:** Notebook seccionado (11 seções, ver REQ-108): panorama, eventos no tempo,
  qualidade do dado, distribuição das features, correlação, funil de conversão,
  segmentação (T-112), resposta por segmento, diagnóstico causal, por que uplift, síntese.
  Números e gráficos primeiro, leitura curta depois. Máximo 12 figuras no corpo; toda
  transformação em `src/eda.py`, notebook só importa e exibe. Referencia
  `0_pipeline_audit.ipynb` para garantia; não repete provas. Divergência entre premissa e
  dado medido é registrada com o número, nunca corrigida em código.
- **Accept:** roda de ponta a ponta com "Run All" a partir do dado processado; ≤ 12
  figuras no tema; completou-sem-ver reportado por tipo; toda taxa de conversão com
  denominador explícito; síntese final lista toda decisão de projeto nascida de achado
  exploratório com número computado na célula.

## T-111 — Check de balanço
- **Status:** [x]
- **Satisfies:** REQ-109
- **Depends on:** T-108, T-109
- **Files:** `src/eda.py`, `notebooks/1_eda.ipynb`
- **Do:** SMD por covariável entre tratado (viu) e controle (não viu); sinalizar acima do
  limiar da config como diagnóstico.
- **Accept:** SMD reportado por covariável; acima do limiar listado, sem alterar estimador.

## T-112 — Segmentação de clientes (K-Means)
- **Status:** [x]
- **Satisfies:** REQ-111
- **Depends on:** T-108, T-110
- **Files:** `src/eda.py`, `tests/test_eda.py`, `notebooks/1_eda.ipynb`, `config.yaml`
- **Do:** `client_features` (grão de cliente), `cluster_matrix` (sem imputação, `log1p` nas
  caudas, z-score), `cluster_scan`/`choose_k` (inércia + silhouette na faixa da config),
  `fit_clusters`, `assign_segments` (nomeia o segmento sentinela), `segment_profile`,
  `segment_response`, `window_spend` + `naive_spend_lift` (diferença bruta visto × não-visto,
  rotulada como confundida). Figuras: varredura de `k`, heatmap de perfil, margem por segmento.
- **Accept:** matriz padronizada (média 0, desvio 1, sem nulo); sentinela fora do ajuste e
  nomeado na leitura; nulo em perfil completo falha alto; rótulos determinísticos dada a seed;
  varredura completa reportada; resposta por segmento × tipo com os dois denominadores.

---

## Execution notes

- Executar em ordem; ler os REQs e seções do plan antes de cada task.
- Rodar o **Accept** ao fim de cada task; só então marcar `[x]`.
- Task inexecutável como escrita → `[!]`, dizer o porquê, parar. Atualizar spec/plan antes de
  retomar; não improvisar escopo.
