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
| `src/attribution.py` | `attribute`: grão `(account_id, offer_id, received_time)`, uma linha por `offer received`, com view e transações agregadas dentro da janela de validade; só atribui transação com `amount ≥ min_value` (G10), filtrado **antes** do desempate de posse; resolve sobreposição por `AttributionPriority` (REQ-103). `build_label`: `converted`/`conversion_value` influence-aware (REQ-104). |
| `src/features.py` | `build`: features `hist_*` sem leakage (só `time < received_time`, G2) + features de oferta/contexto (REQ-105). |
| `src/cost.py` | `add_reward_cost`: `reward_cost` = `discount_value` do catálogo em conversões bogo/discount; 0 caso contrário (REQ-106, G6). Correto porque G10 garante que toda conversão atingiu o `min_value`. |
| `src/contract.py` | Encarnação executável do contrato (REQ-107): `StructType` **e** modelo Pydantic gerados de uma única lista `_COLUMNS` — divergir é impossível. `enforce_schema`, `assert_schema`, `assert_no_unexpected_nulls` (G8), `validate_sample`. |
| `src/pipeline.py` | Orquestra bruto→processado: `assemble_processed` (junta perfil, deriva `treatment` e `campaign_wave`, projeta no contrato), `validate`, `run` (escreve `data/processed/`), `build_spark(cfg)` e o entrypoint CLI. |
| `src/viz.py` | Tema Plotly executivo único (REQ-108): paleta categórica validada por script (banda OKLCH, piso de croma, separação CVD, contraste — ver docstring), light/dark, mais `SEQUENTIAL`/`DIVERGING_*` para heatmap. `figure()`, `add_end_labels`, `add_bar_labels`. Nenhuma figura define estilo ad hoc. |
| `src/eda.py` | Funções da EDA (REQ-108), balanço de covariáveis (REQ-109) e segmentação K-Means (REQ-111): cada uma agrega no Spark e devolve pandas pequeno, mais o construtor de figura correspondente. `covariate_balance` (viu/não-viu, o que o REQ-109 pede) e `assignment_balance` (entre ofertas recebidas, o que de fato verifica a Premissa 4); `numeric_profile`/`categorical_profile`/`correlation_matrix`/`sanity_checks` são o olhar univariado; `response_funnel` e `segment_response` sempre com os dois denominadores; `cluster_matrix` → `cluster_scan` → `fit_clusters` → `assign_segments` é a segmentação; `window_spend`+`naive_spend_lift` dão a diferença bruta (confundida) visto × não-visto. |

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

As garantias **G1–G10** são invariantes testados; se violados, quebram o projeto
em silêncio. Cada uma tem teste dedicado em `tests/`:

- **G1** grão único · **G2** sem leakage temporal · **G3** label exige view ·
  **G4** conversão pós-view e dentro da validade · **G5** informational sem
  `offer completed` · **G6** custo coerente · **G7** sentinela tratada ·
  **G8** sem nulo em coluna não-nullable · **G9** exposição exclusiva (uma view
  física marca no máximo um recebimento) · **G10** conversão atinge o gasto
  mínimo (`converted=1` ⇒ `conversion_value ≥ min_value`).

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
- **Figuras**: Plotly, padrão executivo, tema único em `src/viz.py`; nenhuma
  figura define estilo ad hoc. A cor nunca carrega identidade sozinha — toda
  série leva rótulo direto (`add_end_labels`/`add_bar_labels`), porque a
  paleta validada tem pares abaixo de 3:1 de contraste e ΔE CVD no piso em
  modo escuro (ver docstring de `src/viz.py`).
- **Balanço é diagnóstico, não gate** (Premissas 4 e 5): SMD acima do limiar
  qualifica a leitura causal, nunca altera o estimador de uplift.
- **Toda taxa nomeia seu denominador**: `taxa_conversao` (sobre recebidos) e
  `taxa_conversao_vistos` (sobre vistos) são números diferentes e vivem lado a
  lado. Vale `taxa_conversao = taxa_view × taxa_conversao_vistos` (G3).
- **Divergência entre premissa e dado se registra, não se conserta em código**:
  o número medido vai para o notebook e para a spec; o código só muda por
  decisão de contrato.

## Estado atual

Implementado e testado (82 testes verdes): T-101 a T-112 — pipeline completo
(config, io, clean, attribution, label, features, cost, contrato + escrita),
tema de figuras, EDA, balanço de covariáveis e segmentação K-Means. Ver
`specification/tasks.md` para o board.

Spec 02 (modelagem, `specification/02-modeling/`) em andamento — T-201 a T-205
implementados (103 testes verdes no total): config de modelagem estendida, split
temporal por `campaign_wave` (`src/split.py`), baseline preditivo logística+LGBM
com tracking MLflow (`src/model_baseline.py`, `src/tracking.py`), X-learner por
`offer_type` (`src/uplift.py`) e avaliação Qini/AUUC (`src/uplift_eval.py`, via
`sklift`). `notebooks/2_modeling.ipynb` roda tudo de ponta a ponta sobre o dado
real e cresce seção a seção com as próximas tasks. `auc_lgbm=0.82` supera
`auc_logit=0.75` (T-203 ok).

**T-204 está BLOQUEADA `[!]` — o uplift atual não é causal.** `G3` define
`converted=1` ⇒ houve view, e `treatment=1` ⇔ viu. Logo `treatment=0 ⇒
converted=0` **por construção do label**: o controle tem zero outcomes positivos
em 21.623 linhas. Portanto μ₀ ≡ 0 e τ = μ₁ − μ₀ degenera em **τ ≡ μ₁** — o
"uplift" é a taxa de conversão prevista dos tratados. Medido: 45,8 p.p. contra
uma diferença bruta *já confundida* de 8,8 p.p. (`eda.window_spend`), isto é
5,2× o teto que a docstring de `eda.naive_spend_lift` já fixava como limite
superior. Qini AUC 0,548 é alto **pelo** defeito, não apesar dele. Diagnóstico
em `notebooks/2_modeling.ipynb` §3.2–3.5 via `uplift.label_by_arm` e
`uplift.stage_diagnostics`; travado por
`test_label_impossible_in_control_degenerates_uplift_into_mu1`.

Cuidado com `test_uplift_surething_tends_to_zero`: passa num mundo que o
pipeline não produz (a fixture gera `converted` independente de `treatment`,
permitindo μ₀ > 0). Não é evidência de que REQ-202 está satisfeito. **Não siga
para T-206 antes da decisão de contrato** (tratamento = recebeu, ou outcome =
`window_spend`) — a nota completa está em `specification/02-modeling/spec.md`
REQ-202. Uma política sobre τ ≡ μ₁ aloca por propensão a converter, que é
exatamente o baseline *top-completion* que REQ-205 manda **bater**.

O X-learner exige propensity fixa explícita (taxa de view observada por
`offer_type`, não estimada) porque o `LogisticRegressionCV` default do CausalML
não tolera os nulos legítimos de G8 — bug que só apareceu ao rodar sobre o dado
real, pois a fixture sintética original não tinha nulo (ver
`test_nullable_contract_columns_do_not_break_fit_or_predict`).

**Os notebooks precisam ser re-executados após G10**: os números impressos em
`1_eda.ipynb` (funil, conversão por onda/segmento, custo, `paid_below_minimum`) são
pré-G10. `paid_below_minimum` agora é auditoria — deve dar zero, não um achado.

`notebooks/0_pipeline_audit.ipynb` **prova** as garantias G1–G10 e os REQ-101…110 sobre o
dado real: 57 verificações, cada uma um `assert` sobre o DataFrame completo — nenhuma
amostra, nenhum `try/except`. `notebooks/1_eda.ipynb` é o EDA **entregável** do case:
onze seções (panorama, eventos no tempo, qualidade, distribuições, correlação, funil,
segmentação, resposta por segmento, diagnóstico causal, por que uplift, síntese), números e
gráficos primeiro e leitura curta depois. Referencia o audit para garantia, não repete provas;
máximo 12 figuras, tema de `src/viz.py` em todas. Ambos rodam de ponta a ponta com "Run All",
só importam de `src/` e leem `data/processed/` (escrito por `python -m src.pipeline`).

A segmentação (REQ-111) tem geometria explícita porque K-Means só significa algo com ela:
`identity_missing` fica **fora** do ajuste (não se imputa um segmento), `log1p` nas caudas de
gasto, z-score depois, `k` por silhouette medida na mesma matriz padronizada do ajuste. No dado
real dá k=2 com silhouette 0,278 numa curva rasa — os segmentos são cortes num gradiente de
gasto, não espécies. O rótulo é descritivo e **não pode** virar feature do X-learner (usa a
janela inteira; seria leakage).

Cinco divergências entre spec e dado, levantadas pela auditoria e pela EDA. Quatro seguem
registradas sem correção em código; a #3 virou **decisão de contrato** e foi implementada (G10):

1. `completed` sem view precedente mede **25,8%**, não os 28,4% da Premissa 2.
2. A **Premissa 1** ("uma oferta ativa por vez") é falsa em **56,7%** dos recebimentos; a disputa
   é resolvida por `AttributionPriority` e logada, mas a premissa descreve outro dataset.
3. **RESOLVIDA (G10).** O pipeline cobrava `reward_cost` em 25,9% de conversões cujo
   `conversion_value < min_value` — desconto que nunca teria sido concedido. A atribuição passou a
   exigir `txn_amount ≥ min_value`. No dado real: conversões 37.899 → **26.246**, custo
   R$ 177.618 → **R$ 107.779**. Das 12.714 conversões perdidas, 8.545 não atingiam o mínimo com
   nenhuma compra e 4.169 só o atingiam **somando** compras pequenas (o limiar é por transação,
   não acumulado). Outras 1.061 foram **ganhas**: transações que uma oferta inelegível vencia no
   desempate e descartava.
4. **Censura à direita**: 13,4% dos recebimentos têm validade além do fim dos dados (t=29,75).
   A conversão das ondas 4–5 é subestimada por construção; `campaign_wave` como feature aprenderia
   o artefato de coleta.
5. O empate de `received_time` (mesmo cliente, duas ofertas no mesmo instante) **não ocorre** no
   dado real — o desempate estável por `offer_id` é defensivo e coberto só por fixture sintética.

O balanço de covariáveis tem duas leituras (`src/eda.py`): `covariate_balance` compara
viu/não-viu (o que o REQ-109 pede, mas é pós-tratamento) e `assignment_balance` compara
entre ofertas recebidas (o que de fato verifica a Premissa 4 — envio aleatório). No dado
real, nenhuma covariável passa do limiar em nenhuma das duas leituras.
