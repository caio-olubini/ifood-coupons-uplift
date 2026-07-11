# Plan: Modelagem, uplift & avaliação offline

> Implementa: 02-modeling/spec.md · Consome: schema-processed.md

## Tech stack & key decisions

| Decisão | Escolha | Rationale |
|---|---|---|
| Baseline preditivo | Logística + LGBM | Âncora direcional + modelo que trata nulos e interações; valida sinal. |
| Estimador de uplift | X-learner (CausalML) | Robusto a grupos desiguais e μ₀ mal-estimado — sem a fraqueza do T (Premissa 5). |
| Lib de fallback | scikit-uplift | Plano garantido, pip puro; CausalML é o upgrade já testado. |
| Métrica | Qini/AUUC | Uplift exige métrica de uplift; AUC/F1 selecionam o modelo errado. |
| Avaliação offline | Curva de ganho incremental por budget top-N, contrafactual observado | Isola o incremental causal (tratado − controle do RCT); IPW/DM avaliariam receita bruta, que não separa a conversão causada da espontânea. |
| Explicabilidade | SHAP | Global direcional + force plot individual para o slide. |
| Tracking | MLflow | Params, métricas, artefatos por run; comparação defensável. |
| Config | Pydantic (mesmo objeto da spec 01, estendido) | Custos, cortes, hiperparâmetros validados; sem hardcode (REQ-210). |

## Architecture

- **`src/model_baseline.py`** — logística e LGBM sob validação temporal (REQ-201).
- **`src/uplift.py`** — X-learner via CausalML; saída uplift por cliente × tipo (REQ-202).
- **`src/uplift_eval.py`** — Qini/AUUC (REQ-203); teste de placebo por permutação
  estratificada por `offer_type` (REQ-212) — mesma infraestrutura gera o intervalo de
  confiança do Qini reportado. (~~Calibração REQ-213~~ e ~~correção isotônica REQ-214~~
  descontinuadas junto com a política, REQ-204 — ver abaixo.)
- **`src/gaincurve.py`** — curva de lucro líquido incremental por budget top-N, contrafactual
  observado (estilo Qini), para as estratégias uplift / conversão crua / aleatória / híbridos
  de λ fixo e dinâmico (REQ-206).
- **`src/tracking.py`** — wrappers MLflow: nomes de run, o que logar (REQ-209).
- **`src/explain.py`** — SHAP global e force plot.

(~~`src/policy.py`~~ removido por decisão do usuário em 2026-07-10, junto com REQ-204/REQ-205 —
o projeto deixou de ter uma etapa de alocação por cliente.)

Notebooks em `notebooks/` importam de `src/` e exibem; sem lógica.

## Data model

Entrada: dataset processado (schema-processed.md). Saída tipada (Pydantic): `UpliftEstimate`. A
curva de ganho é uma tabela longa `[strategy, n, gain]`, sem objeto tipado próprio.

## Interfaces & contracts

```
model_baseline.train(df, cfg) -> (logit, lgbm, metrics)     # validação temporal (REQ-201)
uplift.fit_xlearner(df, cfg) -> model
uplift.predict(model, df) -> DataFrame[account_id, offer_id, received_time, offer_type, uplift]  # grão do contrato (REQ-202)
uplift_eval.qini(pred, df) -> float                          # (REQ-203)
uplift_eval.placebo_qini_distribution(train_df, holdout_df, cfg) -> np.ndarray  # nula, N réplicas (REQ-212)
uplift_eval.placebo_test(qini_score, null_distribution, cfg) -> dict            # limiar, passou, p_value (REQ-212)
gaincurve.uplift_ranking(uplift_pred) | completion_ranking(p_convert) | random_ranking(holdout, cfg) -> np.ndarray  # ranking por estratégia (REQ-206)
gaincurve.hybrid_ranking(uplift_pred, p_convert, lambda_) -> np.ndarray        # blend λ fixo (REQ-206)
gaincurve.dynamic_hybrid_ranking(uncertainty, uplift_pred, p_convert, gamma) -> np.ndarray  # blend λ dinâmico por incerteza (REQ-206)
gaincurve.incremental_gain_curve(ranking, holdout) -> DataFrame[n, gain]   # contrafactual observado, estilo Qini (REQ-206)
gaincurve.gain_curves(rankings, holdout) -> DataFrame[strategy, n, gain]   # mesma base, todas as estratégias (REQ-206)
gaincurve.gain_at_budget(curves, budget) -> DataFrame[strategy, n, gain]   # leitura "se meu budget for N" (REQ-206)
```

## Dependencies

- **Externas:** `causalml`, `scikit-uplift`, `lightgbm`, `scikit-learn`, `shap`, `mlflow`,
  `plotly`, `pydantic`. Versões no lock do UV.
- **Internas:** dataset processado da spec 01; objeto de config compartilhado; tema Plotly (`src/viz.py`).

## Risks & mitigations

- **Risco:** CausalML não instalar no caminho crítico. → **Mitigação:** scikit-uplift como
  fallback garantido; contrato de saída idêntico para a política não perceber a troca.
- **Risco:** selecionar por AUC por hábito. → **Mitigação:** REQ-203 fixa Qini/AUUC; baseline
  preditivo e uplift têm métricas separadas e nomeadas.
- **Risco:** prefixo top-N sem controle observado dar contrafactual indefinido. → **Mitigação:**
  `incremental_gain_curve` não desconta contrafactual onde `N_controle=0` (não inventa ganho nem
  produz NaN); coberto por T-gaincurve-sem-controle.
- **Risco:** vazamento de cliente entre treino e teste no split temporal. → **Mitigação:**
  teste que rejeita split onde um cliente aparece nos dois lados fora da ordem temporal.

## Testes (load-bearing)

- **T-uplift-surething** — cliente com μ₀ alto recebe uplift ~0 (guarda a detecção de canibalização).
- **T-gaincurve-contrafactual** — no prefixo completo o ganho é `L_t − L_c·(N_t/N_c)` (a fórmula
  Qini), e um prefixo sem controle não desconta contrafactual inexistente nem produz NaN; uma
  estratégia com sinal plantado supera a aleatória no ganho por budget.
- **T-split-temporal** — split que vaza cliente ou inverte ordem é rejeitado.
- **T-xlearner-grupos** — μ₁ e μ₂ estágios treinam nos subconjuntos corretos (guarda a
  construção do X-learner, não só do T).
- **T-config-modelagem** — custo/hiperparâmetro inválido na config falha antes do treino.
- **T-placebo-permutacao** — embaralhamento preserva a proporção tratado/controle por
  `offer_type`; Qini real supera o percentil da nula quando há efeito heterogêneo real.
Fixtures sintéticas determinísticas desenhadas para a falha específica.

## Traceability

| Requirement | Satisfeito por |
|---|---|
| REQ-201 | `model_baseline` |
| REQ-202 | `uplift` + T-uplift-surething, T-xlearner-grupos |
| REQ-203 | `uplift_eval` |
| ~~REQ-204~~ | descontinuado (política de alocação removida, `src/policy.py`) |
| ~~REQ-205~~ | descontinuado junto com REQ-204 |
| REQ-206 | `gaincurve.gain_curves` / `incremental_gain_curve` + T-gaincurve-contrafactual |
| ~~REQ-207~~ | descontinuado (positividade do IPW; a curva não estima valor absoluto de ação) |
| ~~REQ-208~~ | absorvido por REQ-206 (ganho já é R$ incremental) |
| REQ-209 | `tracking` |
| REQ-210 | `config` + T-config-modelagem |
| ~~REQ-211~~ | descontinuado (dimensionamento de A/B via variância IPW) |
| REQ-212 | `uplift_eval.placebo_qini_distribution` + `placebo_test` + T-placebo-permutacao |
| ~~REQ-213~~ | descontinuado (calibração de magnitude, dependia da política) |
| ~~REQ-214~~ | descontinuado (correção isotônica dependia de REQ-213) |
