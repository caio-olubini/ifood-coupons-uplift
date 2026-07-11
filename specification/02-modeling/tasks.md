# Tasks: Modelagem, uplift & avaliação offline

> Implementa: 02-modeling/spec.md + plan.md
> Legenda: `[ ]` todo · `[~]` andamento · `[x]` feito · `[!]` bloqueado

---

## T-201 — Config de modelagem
- **Status:** [x]
- **Satisfies:** REQ-210
- **Depends on:** T-101 (config base da spec 01)
- **Files:** `src/config.py`, `tests/test_config_model.py`
- **Do:** Estender a config com custos por tipo de oferta, corte temporal de validação,
  hiperparâmetros, seeds. Validação na carga.
- **Accept:** T-config-modelagem passa — custo/hiperparâmetro inválido falha antes do treino.

## T-202 — Split temporal
- **Status:** [x]
- **Satisfies:** REQ-201 (NFR de validação)
- **Depends on:** T-201
- **Files:** `src/split.py`, `tests/test_split.py`
- **Do:** Split treino/teste por ordem temporal e ondas de campanha; sem vazamento de cliente.
- **Accept:** T-split-temporal passa — split que vaza cliente ou inverte ordem é rejeitado.

## T-203 — Baseline preditivo
- **Status:** [x]
- **Satisfies:** REQ-201
- **Depends on:** T-202
- **Files:** `src/model_baseline.py`, `src/tracking.py`
- **Do:** Treinar logística e LGBM sob o split temporal; logar em MLflow.
- **Accept:** LGBM ≥ logística na métrica reportada; ambos com run MLflow.

## T-204 — X-learner
- **Status:** [x] — desbloqueada pela decisão de contrato em G3.
- **Satisfies:** REQ-202
- **Depends on:** T-202
- **Files:** `src/uplift.py`, `tests/test_uplift.py`
- **Do:** X-learner (CausalML; fallback scikit-uplift) com `treatment`=viu, `converted`;
  saída uplift por cliente × tipo.
- **Accept:** T-uplift-surething e T-xlearner-grupos passam.
- **Resolução:** o label deixou de exigir view (`attribution.build_label`), então o
  controle converte (33,0%) e μ₀ > 0. τ = μ₁ − μ₀ voltou a ser um efeito: 4,7 p.p. (bogo),
  6,4 (discount), 10,2 (informational). `test_pipeline_label_admits_conversion_in_control`
  guarda o invariante. Confundimento residual (view é auto-selecionado) registrado em
  `spec.md` REQ-202, não corrigido — exige premissa de ignorabilidade condicional.
- **Correção junto:** `uplift.predict` passou a devolver o grão completo
  `(account_id, offer_id, received_time)` na ordem da entrada. Sem `received_time` a chave
  não é única (mesma oferta em duas ondas) e o join do notebook inflava o holdout de
  25.469 para 27.365 linhas, contaminando o Qini.

## T-205 — Avaliação de uplift
- **Status:** [x]
- **Satisfies:** REQ-203
- **Depends on:** T-204
- **Files:** `src/uplift_eval.py`, `src/viz.py`
- **Do:** Qini/AUUC; curva Qini no tema executivo.
- **Accept:** Qini/AUUC reportado; curva renderiza no tema.
- **Nota:** Qini AUC = 0,038 no holdout, sobre as 25.469 linhas sem inflar. O 0,548
  anterior refletia μ₀ ≡ 0 somado ao join duplicado — era artefato, não desempenho.

## T-206 — Política sensível a custo
- **Status:** [x]
- **Satisfies:** REQ-204
- **Depends on:** T-204
- **Files:** `src/policy.py`, `tests/test_policy.py`
- **Do:** `argmax(uplift_receita − custo)` por cliente, incluindo "não enviar"; saída tipada.
- **Accept:** T-policy-noturno passa — custo > ganho ⇒ não enviar.
- **Economia do lucro:** receita esperada é **incremental** (`uplift × receita_por_conversão`);
  custo é **total** (`P(converte|tratado) × discount_value`), porque o desconto é debitado em
  toda conversão da oferta, causada ou não. Custo é por `offer_id` (o `discount_value` do
  catálogo), não por `offer_type` — o `plan.md` diz o contrário e está errado; `config.yaml`
  já registrava o porquê. No holdout: a política recusa envio a 4.636 de 15.919 clientes
  (29,1%) e sobe o lucro esperado de R$ 2,52 para R$ 3,50 por cliente.
- **Escopo restrito a cupom/promoção (2026-07-10):** `informational` saiu do universo de
  ofertas candidatas (`ELIGIBLE_OFFER_TYPES = ("bogo", "discount")` em `src/policy.py`,
  filtrado em `offer_economics`/`expected_net_profit`). O caso ("qualquer uplift positivo em
  `informational` é lucrativo, custo zero") deixou de ser uma virtude do design e passou a
  ser exatamente o que a política evita escolher — decisão de produto: o escopo aqui é
  distribuição de cupons e promoções, ação informacional é estudo à parte. Números do
  holdout acima medidos **antes** do corte; precisam remedição depois de T-208/T-209 se o
  R$ final for reportado.

## T-207 — Baselines de política
- **Status:** [x]
- **Satisfies:** REQ-205
- **Depends on:** T-203, T-206
- **Files:** `src/policy.py`, `tests/test_policy.py`
- **Do:** Aleatória, enviar-a-todos, top-completion (usa o baseline preditivo).
- **Accept:** cada baseline produz recomendação por cliente no conjunto de avaliação.
- **Nota:** `policy_send_all` **carrega lucro negativo** quando toda oferta do cliente dá
  prejuízo — é justamente o custo do status quo que a política evita ao poder não enviar.
  Clipar em zero apagaria a comparação que REQ-206 vai fazer.

## T-212 — Teste de placebo por permutação
- **Status:** [x]
- **Satisfies:** REQ-212
- **Depends on:** T-204, T-205
- **Files:** `src/uplift_eval.py`, `src/config.py`, `tests/test_uplift_eval.py`
- **Do:** Embaralhar `treatment` dentro de cada `offer_type`, preservando a proporção
  tratado/controle do grupo; refitar o X-learner `cfg.placebo_n_permutations` vezes;
  comparar o Qini real ao percentil `cfg.placebo_confidence_level` da distribuição
  nula resultante.
- **Accept:** T-placebo-permutacao passa — permutação preserva a proporção por tipo;
  Qini real supera o limiar quando há efeito heterogêneo real plantado.
- **Nota:** a mesma distribuição nula é o intervalo de confiança do Qini de T-205 — um
  cálculo, duas leituras (significância e incerteza). No holdout real, ver
  `notebooks/2_modeling.ipynb` §5 para os números medidos.

## T-213 — Calibração da magnitude do uplift
- **Status:** [x]
- **Satisfies:** REQ-213
- **Depends on:** T-204, T-205
- **Files:** `src/uplift_eval.py`, `src/config.py`, `tests/test_uplift_eval.py`
- **Do:** Binnar o holdout por τ previsto (`cfg.calibration_n_bins`); por bin, comparar o
  uplift previsto médio contra o observado (taxa de conversão tratado − controle no bin);
  resumir o erro (MAE, bias). Bin sem tratado ou sem controle é inavaliável, não zero.
- **Accept:** T-calibracao passa — previsto ≈ observado dá MAE pequeno; modelo que infla a
  magnitude tem bias > 0 com Qini idêntico; bin sem contrafactual fica NaN/inavaliável.
- **Nota:** Qini (ordenação) e calibração (magnitude) são ortogonais — um modelo pode
  ordenar bem e mentir sobre o tamanho, e é o tamanho que a política (REQ-204) multiplica
  por receita. No holdout real: MAE 0,102, bias −0,069 (subestima, erra o sinal nos bins
  de τ negativo). Números em `notebooks/2_modeling.ipynb` §6.

## T-214 — Calibração isotônica pós-hoc
- **Status:** [x]
- **Satisfies:** REQ-214
- **Depends on:** T-213
- **Files:** `src/uplift_eval.py`, `src/config.py`, `tests/test_uplift_eval.py`
- **Do:** Ajustar `IsotonicRegression` (τ previsto → τ calibrado) por cross-fitting dentro
  do holdout (`cfg.calibration_n_folds`): cada fold é calibrado por uma isotônica ajustada
  nos bins dos outros folds, nunca no próprio. Reportar MAE/bias antes e depois lado a
  lado.
- **Accept:** T-calibracao-isotonica passa — MAE cai depois da correção quando a magnitude
  estava inflada; monotonicidade vale por fold (garantia local do `IsotonicRegression`, não
  global entre folds); nenhum ponto é calibrado por uma isotônica ajustada com ele mesmo;
  fold sem bins avaliáveis suficientes sai sem correção, não com um valor inventado.
- **Nota:** no holdout real, a correção reduz o MAE de **0,102 para 0,019** (5,3×) e o bias
  de **−0,069 para −0,001** — o previsto pós-calibração acompanha de perto o observado em
  9 dos 10 bins (um colapsou por `duplicates="drop"` num fold específico). A ordenação
  (Qini) não muda: a isotônica só reescala a magnitude.

## T-208 — Curva de ganho incremental por budget top-N
- **Status:** [x]
- **Satisfies:** REQ-206
- **Depends on:** T-205
- **Files:** `src/gaincurve.py`, `tests/test_gaincurve.py`
- **Do:** Curva `(budget N, lucro líquido incremental)` por estratégia (uplift / conversão crua /
  aleatória) sobre o retido, contrafactual observado no dado real (estilo Qini). `gain_at_budget`
  para a leitura "se meu budget for N, quanto ganho cada estratégia?".
- **Accept:** T-gaincurve-contrafactual passa; as três estratégias têm curva na mesma base; a de
  uplift domina a aleatória (ou a diferença é reportada); prefixo sem controle não inventa ganho.
- **Nota (decisão de escopo, 2026-07-10):** IPW e Direct Method **saíram** (com `src/offpolicy.py`,
  `src/impact.py` e seus testes). Ambos avaliavam sobre receita bruta realizada
  (`conversion_value − reward_cost` de fato ocorrido), que soma a conversão causada com a
  espontânea; a avaliação passou a medir **só o incremental causal**. A curva reusa a mecânica do
  contrafactual Qini (controle acumulado escalado pela razão tratado/controle do prefixo), agora
  sobre lucro líquido em vez de conversão: a **receita** entra como incremental (diferença
  escalada), o **custo do desconto** só nos tratados reais (o desconto só é pago em conversão de
  tratado, não no contrafactual) — a mesma assimetria de `policy.expected_net_profit`, herdada da
  subtração escalada sem regra especial. `NO_SEND` deixa de ser um caso à parte: a curva compara
  rankings, não estima o valor absoluto de "não enviar" (o que dispensa REQ-207).
- **Composição por quadrante e % lucro negativo (2026-07-10):** `src/quadrant.py`
  (`uplift.predict_stages` — μ₀/μ₁/τ por linha, extraído de `uplift.stage_diagnostics` —
  e `classify_quadrant`/`quadrant_distribution`/`policy_composition`/
  `negative_profit_share`) explica **de quem** vem a divergência acima. Achado principal:
  `pct_lucro_negativo` (fração de clientes enviados com `net_profit<0`, §7) é **1,1%** na
  política de uplift contra 36,7% (enviar-a-todos), 43,8% (top-completion) e 44,4%
  (aleatória) — a política quase elimina o envio com prejuízo esperado. E o tamanho bate
  com precisão: os 36,05% de clientes em `nao_enviar` são quase idênticos aos 36,74% de
  `pct_lucro_negativo` de `enviar-a-todos` — a recusa está isolando quase exatamente o
  conjunto que teria dado prejuízo, **não** os quadrantes causais "ruins" isolados
  (`sleeping_dog`+`sure_thing` com lucro negativo somam só 0,36% do holdout). Ou seja: a
  recusa segue o **custo do desconto por oferta** (`discount_value`), não a classificação
  causal pura. Limiar de quadrante configurável (`cfg.quadrant_probability_threshold`, default
  0,5). Ver `notebooks/2_modeling.ipynb` §8.2.

## ~~T-209 — Impacto em R$~~ (descontinuado, 2026-07-10)
- **Status:** [~]
- **Era:** REQ-208 (R$ com intervalo) + REQ-211 (dimensionamento do A/B), ambos sobre as
  contribuições IPW de T-208. Sai com o IPW: o ganho já é expresso em R$ incremental pela própria
  curva de T-208, e a variância que dimensionava o A/B não existe mais. `src/impact.py` removido.

## T-210 — Explicabilidade
- **Status:** [ ]
- **Satisfies:** REQ-202 (leitura), NFR de visualização
- **Depends on:** T-204
- **Files:** `src/explain.py`, `src/viz.py`
- **Do:** SHAP global direcional e force plot individual para o slide.
- **Accept:** figura global e um force plot individual renderizam no tema.

## T-211 — Notebook de modelagem
- **Status:** [ ]
- **Satisfies:** REQ-203, REQ-206 (exibição)
- **Depends on:** T-205, T-208, T-210
- **Files:** `notebooks/2_modeling.ipynb`
- **Do:** Exibir Qini, curva de ganho incremental por budget (uplift × conversão crua ×
  aleatória) e a leitura em budgets destacados, SHAP. Importa de `src/`; sem lógica.
- **Accept:** notebook roda ponta a ponta e exibe as visões no tema executivo.

---

## Execution notes

- Executar em ordem; ler REQs e seções do plan antes de cada task.
- Rodar o **Accept** ao fim; só então marcar `[x]`.
- Task inexecutável como escrita → `[!]`, dizer o porquê, parar. Atualizar spec/plan antes de
  retomar; não improvisar escopo.
