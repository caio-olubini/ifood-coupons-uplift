# Plan: Modelagem, uplift & avaliação offline

> Implementa: 02-modeling/spec.md · Consome: schema-processed.md

## Tech stack & key decisions

| Decisão | Escolha | Rationale |
|---|---|---|
| Baseline preditivo | Logística + LGBM | Âncora direcional + modelo que trata nulos e interações; valida sinal. |
| Estimador de uplift | X-learner (CausalML) | Robusto a grupos desiguais e μ₀ mal-estimado — sem a fraqueza do T (Premissa 5). |
| Lib de fallback | scikit-uplift | Plano garantido, pip puro; CausalML é o upgrade já testado. |
| Métrica | Qini/AUUC | Uplift exige métrica de uplift; AUC/F1 selecionam o modelo errado. |
| Avaliação offline | IPW, propensity fixa | RCT: propensity conhecida, não estimada; simplifica e mantém honesto. |
| Explicabilidade | SHAP | Global direcional + force plot individual para o slide. |
| Tracking | MLflow | Params, métricas, artefatos por run; comparação defensável. |
| Config | Pydantic (mesmo objeto da spec 01, estendido) | Custos, cortes, hiperparâmetros validados; sem hardcode (REQ-210). |

## Architecture

- **`src/model_baseline.py`** — logística e LGBM sob validação temporal (REQ-201).
- **`src/uplift.py`** — X-learner via CausalML; saída uplift por cliente × tipo (REQ-202).
- **`src/uplift_eval.py`** — Qini/AUUC (REQ-203).
- **`src/policy.py`** — decisão `argmax(uplift_receita − custo)` incluindo nula (REQ-204);
  os três baselines (REQ-205).
- **`src/offpolicy.py`** — IPW com propensity fixa; checagem de positividade (REQ-206, 207).
- **`src/impact.py`** — conversão para R$ com intervalo (REQ-208); dimensionamento do A/B (REQ-211).
- **`src/tracking.py`** — wrappers MLflow: nomes de run, o que logar (REQ-209).
- **`src/explain.py`** — SHAP global e force plot.

Notebooks em `notebooks/` importam de `src/` e exibem; sem lógica.

## Data model

Entrada: dataset processado (schema-processed.md). Saídas tipadas (Pydantic):
`UpliftEstimate`, `PolicyRecommendation`, `EvaluationResult` — validadas antes de virar
tabela/artefato MLflow.

## Interfaces & contracts

```
model_baseline.train(df, cfg) -> (logit, lgbm, metrics)     # validação temporal (REQ-201)
uplift.fit_xlearner(df, cfg) -> model
uplift.predict(model, df) -> DataFrame[account_id, offer_type, uplift]   # (REQ-202)
uplift_eval.qini(pred, df) -> float                          # (REQ-203)
policy.allocate(uplift_df, costs, cfg) -> DataFrame[account_id, chosen_action, net_profit]  # (REQ-204)
policy.baselines(df, cfg) -> dict[name -> DataFrame]         # aleatória, todos, top-completion (REQ-205)
offpolicy.ipw_value(policy_df, holdout, cfg) -> EvaluationResult   # propensity fixa (REQ-206)
offpolicy.check_positivity(policy_df, holdout) -> bool       # (REQ-207)
impact.to_currency(eval_result, cfg) -> (reais, interval)   # (REQ-208)
impact.size_ab_test(variance, cfg) -> n_per_arm             # (REQ-211)
```

Custos por tipo de oferta são campos da config, não constantes no `policy` — mudar custo é
mudar config, não código.

## Dependencies

- **Externas:** `causalml`, `scikit-uplift`, `lightgbm`, `scikit-learn`, `shap`, `mlflow`,
  `plotly`, `pydantic`. Versões no lock do UV.
- **Internas:** dataset processado da spec 01; objeto de config compartilhado; tema Plotly (`src/viz.py`).

## Risks & mitigations

- **Risco:** CausalML não instalar no caminho crítico. → **Mitigação:** scikit-uplift como
  fallback garantido; contrato de saída idêntico para a política não perceber a troca.
- **Risco:** selecionar por AUC por hábito. → **Mitigação:** REQ-203 fixa Qini/AUUC; baseline
  preditivo e uplift têm métricas separadas e nomeadas.
- **Risco:** IPW reportar valor para ação sem suporte. → **Mitigação:** `check_positivity`
  antes de reportar; REQ-207 exige marcar inavaliável.
- **Risco:** vazamento de cliente entre treino e teste no split temporal. → **Mitigação:**
  teste que rejeita split onde um cliente aparece nos dois lados fora da ordem temporal.

## Testes (load-bearing)

- **T-uplift-surething** — cliente com μ₀ alto recebe uplift ~0 (guarda a detecção de canibalização).
- **T-policy-noturno** — quando custo > ganho, "não enviar" é escolhido (guarda REQ-204).
- **T-ipw-positividade** — política sem sobreposição retorna `evaluable=False`, nunca um número.
- **T-split-temporal** — split que vaza cliente ou inverte ordem é rejeitado.
- **T-xlearner-grupos** — μ₁ e μ₂ estágios treinam nos subconjuntos corretos (guarda a
  construção do X-learner, não só do T).
- **T-config-modelagem** — custo/hiperparâmetro inválido na config falha antes do treino.

Fixtures sintéticas determinísticas desenhadas para a falha específica.

## Traceability

| Requirement | Satisfeito por |
|---|---|
| REQ-201 | `model_baseline` |
| REQ-202 | `uplift` + T-uplift-surething, T-xlearner-grupos |
| REQ-203 | `uplift_eval` |
| REQ-204 | `policy.allocate` + T-policy-noturno |
| REQ-205 | `policy.baselines` |
| REQ-206 | `offpolicy.ipw_value` |
| REQ-207 | `offpolicy.check_positivity` + T-ipw-positividade |
| REQ-208 | `impact.to_currency` |
| REQ-209 | `tracking` |
| REQ-210 | `config` + T-config-modelagem |
| REQ-211 | `impact.size_ab_test` |
