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
  `notebooks/2_modeling.ipynb` §4 para os números medidos.

## T-208 — IPW e positividade
- **Status:** [ ]
- **Satisfies:** REQ-206, REQ-207
- **Depends on:** T-206, T-207
- **Files:** `src/offpolicy.py`, `tests/test_offpolicy.py`
- **Do:** Valor por IPW com propensity fixa sobre o retido; `check_positivity` antes de reportar.
- **Accept:** T-ipw-positividade passa; política de uplift e os três baselines têm valor na
  mesma base; ação sem suporte marcada inavaliável.

## T-209 — Impacto em R$
- **Status:** [ ]
- **Satisfies:** REQ-208, REQ-211
- **Depends on:** T-208
- **Files:** `src/impact.py`, `src/viz.py`
- **Do:** Converter valor em R$ com intervalo, liderando pela economia de desconto; dimensionar
  A/B (N por braço) da variância observada; tabela de lucro por política no tema.
- **Accept:** R$ com intervalo derivado da avaliação; N por braço com a variância que o originou.

## T-210 — Explicabilidade
- **Status:** [ ]
- **Satisfies:** REQ-202 (leitura), NFR de visualização
- **Depends on:** T-204
- **Files:** `src/explain.py`, `src/viz.py`
- **Do:** SHAP global direcional e force plot individual para o slide.
- **Accept:** figura global e um force plot individual renderizam no tema.

## T-211 — Notebook de modelagem
- **Status:** [ ]
- **Satisfies:** REQ-203, REQ-208 (exibição)
- **Depends on:** T-205, T-209, T-210
- **Files:** `notebooks/2_modeling.ipynb`
- **Do:** Exibir Qini, tabela de lucro por política, R$, SHAP. Importa de `src/`; sem lógica.
- **Accept:** notebook roda ponta a ponta e exibe as visões no tema executivo.

---

## Execution notes

- Executar em ordem; ler REQs e seções do plan antes de cada task.
- Rodar o **Accept** ao fim; só então marcar `[x]`.
- Task inexecutável como escrita → `[!]`, dizer o porquê, parar. Atualizar spec/plan antes de
  retomar; não improvisar escopo.
