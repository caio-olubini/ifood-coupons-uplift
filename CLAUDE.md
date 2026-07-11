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

- **G1** grão único · **G2** sem leakage temporal · **G3** label **não** exige view
  (o controle converte — ver é o tratamento, não o rótulo) · **G4** conversão dentro
  da validade · **G5** informational sem
  `offer completed` · **G6** custo coerente · **G7** sentinela tratada ·
  **G8** sem nulo em coluna não-nullable · **G9** exposição exclusiva (uma view
  física marca no máximo um recebimento) · **G10** conversão atinge o gasto
  mínimo (`converted=1` ⇒ `conversion_value ≥ min_value`).

Testes usam **fixtures sintéticas minúsculas e determinísticas** montadas para
exercitar a falha específica — não amostras do dataset real. Rodar o dado real
já pegou bugs que as fixtures não pegaram (ex.: duplicação de grão por múltiplas
transações na janela); ao mexer no pipeline, valide também no notebook.

**O objetivo da suíte é garantir o comportamento estrutural e end-to-end da
solução — não documentar cada passo da construção.** ~36 testes cobrem: as
garantias formais G1–G10, os requisitos numerados da spec de modelagem
(REQ-201…214), e os invariantes de fronteira (config, contrato, split
temporal). Um teste que existe só porque uma função foi escrita (sem um
requisito ou garantia por trás) é candidato a corte, não a manutenção. Fora de
escopo dos testes automatizados: notebooks, EDA/`src/eda.py`, gráficos/`src/viz.py`,
e qualquer módulo de diagnóstico exploratório (ex.: `src/quadrant.py`) — essas
peças são analytics/apresentação, verificadas ao rodar o notebook, não pela
suíte. Ao adicionar uma função nova, pergunte "que garantia formal ou
comportamento de contrato isso quebra se estiver errado?" antes de escrever o
teste; se a resposta for "nenhuma, é só mais uma view sobre o dado", não
escreva o teste.

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
  lado. A identidade `taxa_conversao = taxa_view × taxa_conversao_vistos`
  **não vale mais** desde a mudança de G3 (0,4458 contra 0,3522): quem não viu
  também converte, e essa massa é exatamente o grupo de controle.
- **Divergência entre premissa e dado se registra, não se conserta em código**:
  o número medido vai para o notebook e para a spec; o código só muda por
  decisão de contrato.
- **Notebook não é ensaio**: markdown de célula é título de seção + uma frase
  objetiva do que a célula de código faz. Nunca "leitura real", achado
  extenso, ressalva de parágrafo ou discussão de trade-off — isso vai para
  `specification/`/CLAUDE.md (que persistem) ou para fora do notebook por
  pedido explícito, nunca por iniciativa própria dentro da célula.

## Estado atual

Implementado: T-101 a T-112 — pipeline completo (config, io, clean, attribution,
label, features, cost, contrato + escrita), tema de figuras, EDA, balanço de
covariáveis e segmentação K-Means. Ver `specification/tasks.md` para o board.
A suíte de testes automatizados (`tests/`, ~36 testes) cobre o comportamento
estrutural — ver a nota de filosofia de testes acima; EDA e figuras são
verificadas rodando os notebooks, não pela suíte.

Spec 02 (modelagem, `specification/02-modeling/`) em andamento — T-201 a T-208,
T-212, T-213 e T-214 implementados: config de
modelagem estendida, split temporal por `campaign_wave` (`src/split.py`), baseline
preditivo logística+LGBM com tracking MLflow (`src/model_baseline.py`,
`src/tracking.py`), X-learner por `offer_type` (`src/uplift.py`), avaliação
Qini/AUUC (`src/uplift_eval.py`, via `sklift`), teste de placebo por permutação
(REQ-212), calibração da magnitude do uplift (REQ-213) e correção isotônica
pós-hoc por cross-fitting (REQ-214) — todos em `src/uplift_eval.py` — política
sensível a custo + três baselines (`src/policy.py`) e a avaliação offline por
**curva de ganho incremental por budget top-N** (`src/gaincurve.py`, REQ-206/T-208).
`notebooks/2_modeling.ipynb` roda tudo de ponta a ponta sobre o dado real e cresce
seção a seção com as próximas tasks. `auc_lgbm=0.85` supera `auc_logit=0.80` (T-203 ok).

**`informational` saiu da modelagem por decisão do usuário (2026-07-10).**
`split.exclude_informational` filtra `offer_type == informational` logo após
`temporal_split`, antes de `toPandas()` — o único ponto de entrada de
`train_df`/`holdout_df` no notebook. Nenhum modelo a jusante (baseline
preditivo, X-learner, Qini/placebo/calibração, curva de ganho) o vê mais;
antes, `fit_xlearner` ajustava um terceiro braço para `informational` e as
métricas agregadas (Qini, holdout de 25.469 linhas) o misturavam com
bogo/discount. `informational` já estava fora do escopo de `src/policy.py`
(`ELIGIBLE_OFFER_TYPES`) desde 2026-07-10; agora o mesmo escopo vale para a
modelagem inteira, não só a alocação. Todos os números abaixo (Qini, placebo,
calibração, curva de ganho) foram medidos **sem** informational — holdout
20.412 linhas, não 25.469.

**`reward_cost` só existe em conversão real, controle incluso — não é uma
assimetria a impor no lucro líquido.** `cost.add_reward_cost` (G6) já zera
`reward_cost` sempre que `converted != 1`; e quando converte, o desconto é
concedido **view ou não** — o controle pode converter e pagar desconto de
verdade (`test_unviewed_conversion_still_costs`, `src/attribution.py`: "o
custo segue a conversão, não a exposição"). `gaincurve.add_net_profit`
(`conversion_value − reward_cost`, por linha, igual para tratado e controle)
já estava correto — o bug era só no docstring, que alegava "o controle não tem
desconto a pagar" e foi corrigido (2026-07-10); `L_controle(N)` na fórmula do
contrafactual escalado sempre incluiu qualquer `reward_cost` real do
controle. `test_net_profit_desconta_reward_cost_tambem_no_controle` e
`test_net_profit_e_zero_quando_nao_converteu_sem_reward_cost_solto` guardam os
dois lados do invariante. A assimetria de REQ-204 (custo total × receita
incremental) é da política **prospectiva** (`policy.expected_net_profit`, que
ainda não observou quem converte); sobre dado já realizado, cada linha carrega
seu próprio custo.

**Qini/AUUC por estratégia (2026-07-10, por pedido do usuário) mostra a
inversão que a curva de ganho em R$ escondia.** `uplift_eval.qini_by_strategy`/
`qini_curves_by_strategy` aplicam a mesma métrica de ordenação (REQ-203) ao
modelo de uplift, à conversão crua (P(converte) do baseline) e a um score
aleatório — não só ao X-learner. No holdout real: **Qini 0,034 / AUUC 0,040**
(modelo de uplift) contra **0,009 / −0,006** (conversão crua) e **−0,012 /
−0,012** (aleatório). O modelo de uplift é a única estratégia que concentra
efeito incremental real — a conversão crua mal supera o aleatório em Qini e
fica **pior** que ele em AUUC. Isso contradiz a curva de ganho em R$ (§8, onde
conversão crua domina em lucro): os dois olhares medem coisas diferentes — Qini
mede se a ordenação captura o *efeito causal*, a curva de ganho mede *lucro*, e
lucro pesa ticket médio, não incrementalidade. A conversão crua ganha em R$
porque manda para quem tem ticket alto, não porque identifica quem a oferta de
fato move.

**Estratégia híbrida X-learner + λ·conversão crua (2026-07-10, por pedido do
usuário) recupera parte do lucro em R$ sem abandonar a ordenação causal.**
`gaincurve.hybrid_score`/`hybrid_ranking`: `score = uplift_x_learner + λ ·
p_convert_cru`, soma direta sem normalizar (os dois termos já vivem em escalas
parecidas). Grid `cfg.hybrid_lambda_grid = [0, 0,1, 0,3, 0,5]`; λ=0 é o modelo
de uplift puro, o ponto de controle do grid, não um caso à parte. Exposto no
notebook em §4.2 (Qini/AUUC do grid) e em §8 (curva de ganho em R$, um
ranking por λ ao lado das três estratégias originais).

No holdout real, **λ=0,3 tem o melhor Qini/AUUC do grid inteiro — melhor que o
modelo de uplift puro**: Qini 0,051 / AUUC 0,048 (λ=0,3) contra 0,034 / 0,040
(λ=0, uplift puro); λ=0,1 e λ=0,5 ficam no meio (0,047/0,050 e 0,047/0,041). Um
pouco de sinal de conversão crua **melhora** a ordenação causal em vez de
dilui-la — provavelmente porque `p_convert` carrega informação preditiva que o
X-learner, treinado com menos dado por braço (μ₀/μ₁ separados), não captura
tão bem sozinho. Na curva de ganho em R$ (§8), λ=0,3/0,5 se aproximam ou
superam conversão crua nos budgets maiores (budget 10.000: híbrido λ=0,3 =
R$78.208 supera conversão crua R$68.015; uplift puro fica em R$55.348) — o
híbrido não é só um meio-termo entre os dois extremos, é estritamente melhor
que ambos em alguns pontos do grid. `test_hybrid_score_e_soma_direta_sem_normalizar`,
`test_hybrid_com_lambda_zero_degenera_no_uplift_puro` e
`test_hybrid_lambda_maior_puxa_ranking_em_direcao_a_conversao_crua` guardam a
fórmula e o caso de controle λ=0.

**Híbrido dinâmico (λ local por incerteza do X-learner, 2026-07-10, por pedido
do usuário) não bate o λ=0,3 fixo — testado e descartado como estritamente
melhor.** `gaincurve.dynamic_hybrid_score`: peso `lambda_local = (|mu1−mu0| /
max|mu1−mu0|) ** γ` por cliente, em vez de um λ constante — a ideia é emprestar
mais peso ao prior de conversão exatamente onde os dois sub-modelos do
X-learner (μ₀/μ₁, `uplift.predict_stages`) discordam mais entre si. Testado com
`cfg.dynamic_hybrid_gamma_grid = [0,5, 1,0, 2,0]` contra uplift puro, conversão
crua e híbrido λ=0,3 fixo (notebook §4.3), nas mesmas duas métricas.

**Qini/AUUC: nenhum γ bate λ=0,3 fixo.** λ=0,3 fixo = 0,051/0,048; o melhor
dinâmico (γ=1,0) fica em 0,048/0,048, e γ=0,5/γ=2,0 pioram mais (0,041/0,033 e
0,039/0,044). **Curva de ganho em R$: resultado misto, não estritamente
melhor.** No budget 1.000, γ=0,5 supera λ=0,3 fixo (R$12.002 vs R$7.048) — o
peso dinâmico ajuda quando o budget é pequeno e concentrado nos casos de maior
incerteza. Mas no budget 10.000, λ=0,3 fixo volta a dominar todos os γs
(R$78.208 vs o melhor dinâmico, γ=0,5, em R$69.233). **Veredito:** o híbrido
dinâmico não é uma estratégia estritamente dominante — perde em ordenação
causal (Qini/AUUC) e só ganha em R$ no budget mais restrito. λ=0,3 fixo
continua sendo o melhor ponto do espaço de estratégias testado até aqui.
`test_dynamic_hybrid_score_pondera_por_incerteza_local`,
`test_dynamic_hybrid_gamma_alto_e_mais_conservador` e
`test_dynamic_hybrid_ranking_e_deterministico_e_ordena_decrescente` guardam a
fórmula, não o resultado do grid — o resultado é um achado do dado real, não
um invariante a testar.

**Qini e a curva de ganho em R$ são apples-to-apples — verificado (2026-07-10,
por pedido do usuário).** O híbrido domina conversão crua com folga em Qini/AUUC
mas ainda perde em R$ nos budgets pequenos (N=1000); investigado se os dois
cálculos mediam a mesma coisa. `sklift.qini_curve` e
`gaincurve._scaled_counterfactual_gain` usam a **mesma fórmula** (contrafactual
do tratado menos o controle escalado pela razão tratado/controle do prefixo) e
o mesmo N (todos os clientes no prefixo, tratados e controle juntos) — conferido
numericamente: em N=1000, a conversão incremental do Qini (184,26) bate exatamente
com `conversions` de `gaincurve.incremental_gain_curve` para o mesmo ranking. O
gap em R$ não é bug de cálculo: o top-1000 de conversão crua tem ticket médio
R$62,22 por conversão contra R$46,49 do híbrido λ=0,3 — conversão crua está
literalmente escolhendo, entre os que convertem, quem gasta mais por compra, e
a curva de lucro pesa ticket médio (`conversion_value`), enquanto Qini só conta
o evento 0/1. Dois modelos podem concordar em *quem* a oferta faz converter e
discordar em *quanto* essa conversão vale — é o mesmo achado já registrado
acima ("conversão crua ganha em R$ porque manda para quem tem ticket alto"),
agora confirmado numericamente ponto a ponto, não só em agregado.

**Avaliação offline = curva de ganho incremental por budget (T-208, REQ-206), não
IPW.** `src/gaincurve.py` compara estratégias (modelo de uplift × conversão crua ×
aleatório) pela pergunta "dado um budget de N clientes, quanto lucro líquido
incremental os top-N de cada estratégia entregam?". Cada estratégia é um *ranking*;
o ganho de cada prefixo top-N sai do **contrafactual observado no dado real** (estilo
Qini: controle acumulado escalado pela razão tratado/controle do prefixo), não de
predição. Eixo Y = lucro líquido incremental: receita incremental (diferença escalada)
menos o desconto pago **só nos tratados reais** — a assimetria de `policy.expected_net_profit`.
**IPW e Direct Method saíram de escopo por decisão do usuário (2026-07-10):** avaliavam
sobre receita bruta realizada (`conversion_value − reward_cost` ocorrido), que soma a
conversão causada com a espontânea — não isolam o incremental. Removidos junto com
`src/offpolicy.py`, `src/impact.py`, seus testes, e os campos de config `ipw_*`/`ab_test_*`;
REQ-207/208/211 e T-209 descontinuados nas specs (ficam como `~~riscado~~`, com o porquê).
**Achado real, registrado:** no dado real a **conversão crua domina** o modelo de uplift em
todos os budgets na curva de *lucro* (o eixo premia ticket alto, não incrementalidade pura —
Qini honesto é só 0,034); o uplift supera a aleatória nos budgets maiores. O valor da política
de uplift está tanto no envio quanto na **recusa** dos ~36% que dariam prejuízo — a curva mede
só o envio; a economia da recusa é a composição de §8.1 do notebook. As duas métricas convivem.

**A curva de ganho também devolve conversão incremental e IC (2026-07-10, por pedido do
usuário).** `gaincurve.incremental_gain_curve`/`gain_curves` trazem `conversions` ao lado de
`gain`: mesmo contrafactual escalado estilo Qini, mas sobre `converted` (0/1) em vez de lucro
— "quantas conversões a mais os top-N tiveram por causa da oferta", sem a assimetria de custo
do lucro (não há desconto a debitar numa contagem). `gaincurve.gain_curves_with_ci` envolve
isso com IC por **bootstrap não paramétrico**: reamostra o holdout inteiro com reposição
(`cfg.gain_curve_n_bootstrap=200`, seed da config) e recomputa a curva completa por réplica; o
IC em cada N é o percentil `cfg.gain_curve_confidence_level=0,95` das réplicas — mesmo padrão
de reamostragem do placebo (T-212), mas medindo incerteza amostral, não uma nula causal.
`fig_conversion_curves` espelha `fig_gain_curves`; ambas sombreiam a banda de IC quando as
colunas `_lo`/`_hi` estão presentes. **Achado do IC no dado real:** no budget 1000 o lucro
incremental do modelo de uplift tem IC `[1.158, 8.384]` — não cruza zero neste corte
(sem informational); nos budgets maiores (5.000/10.000) o IC também não contém zero. A curva
sem IC escondia essa incerteza.

**T-204 foi desbloqueada por decisão de contrato em G3.** O label deixou de
exigir view: `converted` mede compra na validade atingindo o `min_value`, tenha o
cliente visto ou não (`attribution.build_label`). Ver é o **tratamento**, não o
rótulo. Com isso o controle converte — **33,0% (7.140/21.623)** contra 49,2% no
tratado — μ₀ ∈ [0,36; 0,44] por tipo, e τ = μ₁ − μ₀ voltou a ser um efeito:
**4,7 p.p.** (bogo), 6,4 (discount) — `informational` fora da modelagem (ver acima).
Qini AUC honesto: **0,034**, sobre 20.412 linhas (bogo+discount, sem informational).

Antes disso, μ₀ ≡ 0 forçava τ ≡ μ₁ (uplift "médio" 45,8 p.p., Qini 0,548 — altos
**pelo** defeito). Se você achar o Qini de 0,034 baixo demais e suspeitar de bug:
não é — o **teste de placebo** (T-212, REQ-212) confirma que não é ruído. Embaralhando
`treatment` dentro de cada `offer_type` (preservando a proporção tratado/controle) e
refitando o X-learner 20 vezes, a nula tem média ≈ 0 e desvio 0,015; o Qini real (0,0335)
fura o percentil 95 (0,0186) com p-valor empírico 0/20. Baixo, mas real.
`test_pipeline_label_admits_conversion_in_control` guarda o invariante e
`test_label_impossible_in_control_degenerates_uplift_into_mu1` fixa a assinatura
numérica da regressão.

**Ordenação e magnitude são coisas diferentes (T-213, REQ-213).** Qini mede se o
modelo *ordena* o efeito; `uplift_eval.calibration_by_bin` mede se ele acerta o
*tamanho*: por bin de τ previsto, compara o uplift previsto médio ao observado
(taxa tratado − controle no bin). No holdout real o modelo ordena de forma útil
(uplift observado sobe ~0,02 → ~0,16 do bin baixo ao alto) mas a **magnitude é mal
calibrada**: MAE 0,116, bias −0,098 (subestima na média), e erra o sinal nos bins de
τ negativo. Bin sem tratado ou sem controle é inavaliável (positividade por bin),
nunca zero.

**A correção isotônica (T-214, REQ-214) resolve a magnitude, não a ordenação.**
`uplift_eval.isotonic_calibrate_cross_fitted` ajusta `IsotonicRegression` (τ
previsto → τ calibrado) por cross-fitting dentro do holdout: cada fold de
`cfg.calibration_n_folds` é previsto por uma isotônica ajustada nos bins dos
*outros* folds, nunca no próprio — senão o "depois" aprenderia no mesmo dado que
avalia. No holdout real: MAE **0,116 → 0,020** (5,8×), bias **−0,098 → 0,001**
(praticamente zero). A ordenação (Qini) não muda — a isotônica só reescala. O R$
da política deixa de ser puramente direcional; ainda carrega a incerteza do
confundimento residual abaixo, mas a magnitude prevista agora bate com o
observado no próprio holdout.

**Confundimento residual, registrado:** `treatment` = viu continua sendo escolha
do cliente, não braço aleatorizado (o randomizado foi o *envio*, Premissa 4). O
uplift é causal **sob ignorabilidade condicional às features**, não por desenho.
A mudança de G3 não resolveu isso, e nenhum dos caminhos que a spec listava
resolvia.

`uplift.predict` devolve o grão completo `(account_id, offer_id, received_time)`
na ordem da entrada. Sem `received_time` a chave não é única (mesma oferta em
duas ondas) e o join do notebook inflava o holdout de 25.469 para 27.365 linhas —
todo Qini reportado antes desta correção estava contaminado.

A política (`src/policy.py`, REQ-204) trata receita como **incremental**
(`uplift × receita_por_conversão`) e custo como **total**
(`P(converte|tratado) × discount_value`): o desconto é debitado em toda conversão
da oferta, causada ou não. O custo é por `offer_id` (catálogo), não por
`offer_type` — o `plan.md` dizia o contrário e foi corrigido. `policy_send_all`
carrega lucro negativo de propósito — é o custo do status quo.

A seção 7 do notebook (que chamava `policy.allocate`/`policy_send_all`/
`policy_top_completion`/`policy_random` e comparava recusa/lucro por cliente)
foi removida por decisão do usuário (2026-07-10); `notebooks/2_modeling.ipynb`
hoje só chama `offer_economics`/`expected_net_profit` (score, sem alocar). As
tabelas de composição (§8.1) usam `stages`/`scored`, não a saída de `allocate`.

O X-learner exige propensity fixa explícita (taxa de view observada por
`offer_type`, não estimada) porque o `LogisticRegressionCV` default do CausalML
não tolera os nulos legítimos de G8 — bug que só apareceu ao rodar sobre o dado
real, pois a fixture sintética original não tinha nulo (ver
`test_nullable_contract_columns_do_not_break_fit_or_predict`).

**Os notebooks precisam ser re-executados após G10 e a mudança de G3**: os números
impressos em `1_eda.ipynb` (funil, conversão por onda/segmento, custo,
`paid_below_minimum`) são pré-G10. `paid_below_minimum` agora é auditoria — deve dar
zero, não um achado. Com o novo G3 o funil também muda: `converted` não é mais um
subconjunto de quem viu, então "conversão sobre vistos" deixou de conter toda a
conversão, e `0_pipeline_audit.ipynb` precisa reprovar G3/G4 na forma nova.

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
