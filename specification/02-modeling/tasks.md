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

## ~~T-206 — Política sensível a custo~~ (descontinuada)
- **Status:** removida (2026-07-10, decisão do usuário — ver REQ-204 riscada em `spec.md`).
- **Motivo:** o projeto deixou de ter uma etapa de alocação por cliente; a avaliação de
  modelos de uplift passou a ser só Qini/AUUC e a curva de ganho por budget (REQ-206).
  `src/policy.py` e `tests/test_policy.py` foram removidos inteiros.

## ~~T-207 — Baselines de política~~ (descontinuada)
- **Status:** removida junto com T-206 — sem política de alocação, não há o que os
  baselines de alocação comparem. Os baselines de **ranking** (uplift/conversão
  crua/aleatório) continuam em `uplift_eval.qini_by_strategy`/`gaincurve`.

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

## ~~T-213 — Calibração da magnitude do uplift~~ (descontinuada)
- **Status:** removida (2026-07-10, decisão do usuário — ver REQ-213 riscada em `spec.md`).
- **Motivo:** dependia da política (REQ-204), também removida — sem alocação, não há R$
  a jusante que a magnitude calibrada precisasse acertar. `calibration_by_bin`,
  `calibration_error`, `fig_calibration`, `cfg.calibration_n_bins` e seus testes saíram.

## ~~T-214 — Calibração isotônica pós-hoc~~ (descontinuada)
- **Status:** removida (2026-07-10, decisão do usuário — ver REQ-214 riscada em `spec.md`).
- **Motivo:** dependia de T-213, também removida.

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
  tratado, não no contrafactual) — assimetria que dependia de `policy.expected_net_profit`,
  removida com REQ-204/T-206.
- **Composição por quadrante (`src/quadrant.py`, `classify_quadrant`/`quadrant_distribution`,
  a partir de `uplift.predict_stages`):** segue vivo como diagnóstico de composição do
  holdout por μ₀/μ₁. `policy_composition`/`negative_profit_share` — que cruzavam essa
  composição com a saída da política — foram removidos junto com `src/policy.py`
  (2026-07-10); o achado histórico que comparavam (percentual de lucro negativo por
  estratégia de alocação) não se aplica mais, pois a política não existe. Limiar de
  quadrante configurável (`cfg.quadrant_probability_threshold`, default 0,5).

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
