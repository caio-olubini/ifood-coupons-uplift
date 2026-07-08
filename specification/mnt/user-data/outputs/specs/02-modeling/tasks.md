# Tasks: Modelagem, uplift & avaliação offline

> Implementa: 02-modeling/spec.md + plan.md
> Legenda: `[ ]` todo · `[~]` andamento · `[x]` feito · `[!]` bloqueado

---

## T-201 — Config de modelagem
- **Status:** [ ]
- **Satisfies:** REQ-210
- **Depends on:** T-101 (config base da spec 01)
- **Files:** `src/config.py`, `tests/test_config_model.py`
- **Do:** Estender a config com custos por tipo de oferta, corte temporal de validação,
  hiperparâmetros, seeds. Validação na carga.
- **Accept:** T-config-modelagem passa — custo/hiperparâmetro inválido falha antes do treino.

## T-202 — Split temporal
- **Status:** [ ]
- **Satisfies:** REQ-201 (NFR de validação)
- **Depends on:** T-201
- **Files:** `src/split.py`, `tests/test_split.py`
- **Do:** Split treino/teste por ordem temporal e ondas de campanha; sem vazamento de cliente.
- **Accept:** T-split-temporal passa — split que vaza cliente ou inverte ordem é rejeitado.

## T-203 — Baseline preditivo
- **Status:** [ ]
- **Satisfies:** REQ-201
- **Depends on:** T-202
- **Files:** `src/model_baseline.py`, `src/tracking.py`
- **Do:** Treinar logística e LGBM sob o split temporal; logar em MLflow.
- **Accept:** LGBM ≥ logística na métrica reportada; ambos com run MLflow.

## T-204 — X-learner
- **Status:** [ ]
- **Satisfies:** REQ-202
- **Depends on:** T-202
- **Files:** `src/uplift.py`, `tests/test_uplift.py`
- **Do:** X-learner (CausalML; fallback scikit-uplift) com `treatment`=viu, `converted`;
  saída uplift por cliente × tipo.
- **Accept:** T-uplift-surething e T-xlearner-grupos passam.

## T-205 — Avaliação de uplift
- **Status:** [ ]
- **Satisfies:** REQ-203
- **Depends on:** T-204
- **Files:** `src/uplift_eval.py`, `src/viz.py`
- **Do:** Qini/AUUC; curva Qini no tema executivo.
- **Accept:** Qini/AUUC reportado; curva renderiza no tema.

## T-206 — Política sensível a custo
- **Status:** [ ]
- **Satisfies:** REQ-204
- **Depends on:** T-204
- **Files:** `src/policy.py`, `tests/test_policy.py`
- **Do:** `argmax(uplift_receita − custo)` por cliente, incluindo "não enviar"; saída tipada.
- **Accept:** T-policy-noturno passa — custo > ganho ⇒ não enviar.

## T-207 — Baselines de política
- **Status:** [ ]
- **Satisfies:** REQ-205
- **Depends on:** T-203, T-206
- **Files:** `src/policy.py`
- **Do:** Aleatória, enviar-a-todos, top-completion (usa o baseline preditivo).
- **Accept:** cada baseline produz recomendação por cliente no conjunto de avaliação.

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
