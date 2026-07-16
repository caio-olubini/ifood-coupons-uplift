# Post-delivery notes

Análises feitas **após a entrega**, fora do escopo do pipeline/modelagem
principal. Cada nota resume o que foi estudado, o achado e a decisão. Os
notebooks correspondentes vivem em `notebooks/appendix/` e rodam de ponta a
ponta sobre `data/processed/`.

---

## Propensity do X-learner: estimada vs. fixa — é variância, não confounding (2026-07-16)

**Notebook:** [`notebooks/appendix/propensity_models_exploration.ipynb`](notebooks/appendix/propensity_models_exploration.ipynb)

### Contexto

Houve uma tentativa de trocar a propensity do X-learner de **fixa** (taxa de view
constante por `offer_type`) para **estimada por X** (`model_p` = LGBMClassifier
sobre `treatment`). O Qini AUC do modelo de uplift **despencou** (0,0335 → 0,0132
no holdout). A pergunta: bug ou comportamento esperado?

### O experimento

Rodei **cinco variantes de propensity** sobre o **mesmo** X-learner (só o peso
`g(x)` muda) e medi o Qini junto com a distribuição de `g(x)`:

| variante | Qini AUC | g(x) mean | g(x) std | nº valores distintos |
|---|---|---|---|---|
| **fixa global** (produção) | **0,0335** | 0,70 | 0,072 | 2 |
| fixa por conjunto de canais | 0,0195 | 0,70 | 0,218 | 6 |
| estimada — clipada [0,1; 0,9] | 0,0180 | 0,68 | 0,255 | 14.294 |
| estimada — sem features suspeitas | 0,0147 | 0,70 | 0,266 | 19.894 |
| estimada — todas (a tentativa) | 0,0132 | 0,69 | 0,266 | 20.037 |

**A média bate em todas (~0,70)** — todo mundo captura a taxa real de view. O que
diverge é o **desvio padrão**: a fixa tem std 0,07 com só 2 valores possíveis;
qualquer versão calculada salta para std ~0,27 com dezenas de milhares de valores
distintos. E o Qini cai na mesma ordem — **quase monotônico com o std de `g(x)`**.

### O desempate: variância vs. confounding

Duas hipóteses explicariam a queda:

1. **Confounding de feature** — a propensity estimada carrega features que também
   dirigem o outcome, injetando sinal preditivo (não-causal) no ranking.
2. **Variância de estimação** — um `g(x)` estimado por linha é ruidoso, e esse
   ruído se propaga para o efeito estimado.

O teste desempatou: **remover as features suspeitas quase não mudou o resultado**
(0,0132 → 0,0147, ainda muito abaixo da fixa). Se o mecanismo fosse confounding,
tirar as features compartilhadas recuperaria o Qini — e não recupera. **O
mecanismo dominante é variância, não confounding.**

### Por que variância derruba o Qini

A fórmula do X-learner pondera linearmente por `g(x)`:
`τ(x) = g(x)·τ_c(x) + (1−g(x))·τ_t(x)`. Propensity **fixa** dá o mesmo peso suave
para todo mundo; propensity **calculada** dá um peso ruidoso e diferente por
linha, e essa variância se propaga direto para o efeito estimado — degradando a
ordenação. O **clip [0,1; 0,9] ajudou pouco** porque reduz só os extremos, não a
variância do corpo da distribuição (std cai de 0,266 para 0,255 apenas).

### Decisão

**Mantida a propensity fixa em produção** (`src/uplift.py`, byte a byte como
antes: `fixed_propensity`, `model.fit(..., p=p)`). Neste regime —
**view auto-selecionada, sem propensity vinda de um desenho experimental real** —
o ganho teórico de estimar `g(x)` com precisão **não paga o custo de variância**
que a precisão introduz. Fica documentado como **divergência testada, não decisão
arbitrária**.

---

## Meta-learner benchmark: the S-learner wins, and it is not a disguised conversion model (2026-07-16)

**Notebook:** [`notebooks/appendix/model_benchmarking.ipynb`](notebooks/appendix/model_benchmarking.ipynb)

### Context

With the delivery deadline pushed by a week, the extra time went into
benchmarking alternatives to the **X-learner** — the original production choice,
justified a priori by the arm imbalance — on an exploratory branch. All learners
share the **same** base learner (the config's LGBM), the **same** fixed
propensity and the **same** temporal split, so the comparison isolates the
*meta-learner*, not the base learner.

### The benchmark (Qini/AUUC, holdout)

| strategy | Qini | AUUC | placebo p |
|---|---|---|---|
| **S-learner** | **0,0897** [0,078; 0,101] | **0,1074** | 0,0 |
| T-learner | 0,0419 [0,029; 0,055] | 0,0499 | 0,0 |
| CausalForest | 0,0401 [0,028; 0,053] | 0,0456 | 0,2 |
| X-learner (produção) | 0,0335 [0,019; 0,047] | 0,0403 | 0,0 |

The S-learner wins by a wide margin — its CI **does not overlap** any of the
other three. CausalForest is the only case where the placebo nearly fails
(p=0,2), which is reassuring in itself: **the test discriminates.**

### Why the number was not trusted on sight — three robustness checks

1. **The estimated CATE is genuinely heterogeneous**, not a constant dressed up
   as conversion. τ̂ has std (0,071) **larger** than its mean (0,058), is skewed,
   and carries a negative tail consistent with the ~27% of **sleeping dogs**
   already mapped in the case.
2. **Correlation with the propensity score is moderate, not high.** Pearson 0,37
   / Spearman 0,40 — this **rejects** the "S-learner = disguised conversion
   model" hypothesis, which would predict 0,8+.
3. **The `treatment` marginal importance is low (3%, isolated)** — expected, and
   *not* in contradiction with the two checks above: the effect is captured
   through **interaction** with `hist_avg_ticket` and `tenure_days` (the two
   dominant features), not as a direct additive shift.

### Mechanistic explanation for the win

The coupon effect here is **small and concentrated in a few strong variables** —
not diffuse and complex. T/X/CausalForest estimate the effect as the *difference
between two separately trained models*, and that subtraction **adds up the error
of both**; with a small signal, the summed noise overwhelms it. The S-learner
trains everything jointly and does not pay that variance toll — a more stable
result in exactly this regime of subtle effect plus arm imbalance.

### Explicit limit of what was tested

The three checks and the placebo rule out **noise, leakage and the trivial
disguised-propensity explanation**. **None of them addresses confounding**: every
estimator runs on the same observational holdout, under the same non-randomized
view. The causal question stays **open**, and only the randomized A/B already
proposed in the case settles it.

### Scope decision

Deliberate stop after this investigation. **Not** tested: other base learners
inside the S-learner (explicit interactions, XGBoost) and other meta-learners
(DR-learner). Judged as **diminishing marginal return** against this phase's
objective.

### Status

Finding registered as a **post-delivery exploration candidate**. The delivered
case's production model **remains the X-learner**, with its original rationale
preserved.
