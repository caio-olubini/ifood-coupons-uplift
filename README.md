# iFood Coupons Uplift

Decide **who gets which coupon** to maximize incremental profit — not who would convert anyway.

X-learner (CausalML) + LGBM conversion prior, blended by CATE uncertainty. Local PySpark, offline evaluation on holdout.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-local-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Demo](https://img.shields.io/badge/simulator-live-EA1D2C)](https://caio-olubini.github.io/ifood-coupons-uplift/)

## Simulator

> **[caio-olubini.github.io/ifood-coupons-uplift](https://caio-olubini.github.io/ifood-coupons-uplift/)**
>
> Set a budget, filter the audience, compare three ranking strategies, export a campaign CSV.
> Scores pre-computed offline — the browser only ranks and aggregates.

Details: [`simulator/README.md`](simulator/README.md)

## Results (holdout)

20,412 sends the model never trained on · **BlendedUpliftModel** (dynamic λ, γ = 1.0).

| Budget | Random | Raw conversion | **Blend γ = 1.0** |
|---:|---:|---:|---:|
| 1,000 | 1,557 | **4,010** | 2,814 |
| 5,000 | 9,667 | 10,983 | **12,036** |
| 10,000 | 17,820 | 19,613 | **24,244** |
| 15,000 | 25,934 | 23,929 | **32,079** |

**At 15k coupons:** R$ 32,079 incremental profit · **+24%** vs random · **+34%** vs raw conversion

| Qini / AUUC | Placebo |
|---:|---|
| **0.069 / 0.073** | **p = 0/20** |

## What was built

- [x] **Spark pipeline** — extraction, cleaning, attribution, leakage-free features, G1–G10 contract
- [x] **EDA** — funnels, covariate balance, causal diagnostics
- [x] **X-learner** — CATE per offer type; placebo confirms real signal
- [x] **Uncertainty blend** — uplift + conversion prior; dynamic λ by CATE confidence
- [x] **Exploration** — softmax temperature on ranking (CLI + simulator)
- [x] **Holdout eval** — Qini/AUUC + incremental profit curves by budget
- [x] **Feature importance** — causal, predictive, combined
- [x] **Test suite** — ~36 tests encoding structural guarantees
- [x] **Product** — `pipeline` / `train` / `predict` CLI + allocation simulator

## Quick start

```bash
git clone https://github.com/caio-olubini/ifood-coupons-uplift.git && cd ifood-coupons-uplift
uv sync
uv run coupons-uplift pipeline
uv run coupons-uplift train
uv run coupons-uplift predict --budget 15000 --out campanha.csv
```

Requirements: [UV](https://docs.astral.sh/uv/) · Python ≥ 3.12 · JDK 11+ · Git

Raw data in `data/raw/` · processed Parquet generated locally · model in `models/`

## Notebooks

| Notebook | Content |
|---|---|
| [`1_data_processing`](notebooks/1_data_processing.ipynb) | Pipeline audit, G1–G10 |
| [`1_1_exploratory_analysis`](notebooks/1_1_exploratory_analysis.ipynb) | EDA, funnels, balance |
| [`1_2_clustering`](notebooks/1_2_clustering.ipynb) | Segmentation *(in progress)* |
| [`2_modeling`](notebooks/2_modeling.ipynb) | X-learner, placebo, blends, gain curves |

## Assumptions

[`specification/00-clarify.md`](specification/00-clarify.md)

| Topic | Choice |
|---|---|
| Send | RCT — propensity known |
| Treatment | Viewed offer vs received-but-not-viewed (pseudo-control) |
| Label | Conversion in validity window, independent of view (G3) |
| Eval | Observed counterfactual on holdout (Qini-style) |

Divergences logged: 56.7% overlapping receipts · 13.4% right-censored · 25.8% completed without view.

## Limitations

- Promo-window outcome only — no LTV or long-term behavior
- Last wave censored; blend λ/γ tuned on holdout (optimistic)
- Simulator projects from model scores, not realized counterfactuals
- Viewing not randomized — confounding possible
- Offline proof ≠ live A/B

## Roadmap

- [ ] Persona clustering + simulator filter by segment
- [ ] MLflow tracking hardening
- [ ] Databricks integration (deferred — local PySpark for now)
- [ ] Hyperparameter tuning at production scale

---

## Specification-driven design

This is a case study, but it was built like a product: **decisions live in specs before code**, contracts are explicit, and divergences between premise and data are logged — not patched silently.

The [`specification/`](specification/) tree is the backbone of the repo. Each major deliverable has its own spec, plan, and task board; shared foundations and schemas sit at the root.

| Area | What the spec covers |
|---|---|
| [`00-clarify.md`](specification/00-clarify.md) | Premises, scope, causal framing — what the problem is and is not |
| [`spec.md`](specification/spec.md) + [`plan.md`](specification/plan.md) | Data pipeline & EDA: ingestion, attribution, anti-leakage, exploratory analysis |
| [`schema-raw.md`](specification/schema-raw.md) · [`schema-processed.md`](specification/schema-processed.md) | Executable contracts between raw events, processed grain, and downstream modeling |
| [`02-modeling/`](specification/02-modeling/) | X-learner, blends, Qini/placebo, incremental gain curves, serving |
| [`03-simulator/`](specification/03-simulator/) | Offline export, browser ranking, parity with CLI scoring |
| [`tasks.md`](specification/tasks.md) | Cross-cutting task board — what shipped and what was deliberately cut |

That structure is what makes the work reviewable: a reviewer can read the clarify doc and schemas, then trace guarantees G1–G10 and the holdout numbers back to named decisions. Notebooks and the simulator are **consumers** of `src/` — they do not define behavior.

---

## Engineering

What you need to know to work on or review this codebase.

**Principles**

- All transformation logic lives in `src/` — notebooks only import and display.
- Every behavior parameter is in `config.yaml` — nothing hardcoded in `src/`.
- Pipeline output contract: [`specification/schema-processed.md`](specification/schema-processed.md) — grain `(account_id, offer_id, received_time)`.
- Guarantees G1–G10 are structural invariants with dedicated tests; breaking one fails silently in production.
- Historical features filter `event_time < received_time` *before* aggregating (G2) — anti-leakage is structural, not conventional.

### CLI

```bash
uv run coupons-uplift <command> [--config path/to.yaml]
```

| Command | What it does |
|---|---|
| `pipeline` | Raw JSONs → `data/processed/` — validates contract before writing |
| `train` | Fit `BlendedUpliftModel` on training split → `models/` |
| `predict` | Score customers × active offers, return top-N (one offer per customer) |
| `export` | Freeze simulator JSON artifacts → `simulator/data/` |

`predict` flags:

| Flag | Default | Purpose |
|---|---|---|
| `--budget N` | `cfg.predict_budget` | Number of actions to recommend |
| `--decision-time T` | end of history | As-of instant for scoring features |
| `--out path.csv` | stdout | Write recommendations to CSV |

```bash
uv run pytest -q                              # full integrity suite (~36 tests)
uv run pytest tests/test_leakage.py -q        # single file
```

### Directory tree

```
ifood-coupons-uplift/
├── config.yaml                 # pipeline + modeling + simulator params
├── pyproject.toml              # UV deps, coupons-uplift entrypoint
├── data/
│   ├── raw/                    # offers.json, profile.json, transactions.json
│   └── processed/              # pipeline output (generated, gitignored)
├── models/                     # blended_uplift_model.pkl (committed)
├── src/                        # all business logic
├── tests/                      # G1–G10, modeling, serving invariants
├── notebooks/                  # audit, EDA, modeling (no logic here)
├── simulator/                  # static UI + offline export
└── specification/              # SDD: clarify, per-product specs, schemas, tasks
```

### Modules (`src/`)

| Module | Responsibility |
|---|---|
| `config.py` | `PipelineConfig` (Pydantic) loaded from `config.yaml` |
| `io.py` | Read raw JSONs; parse events, normalize `offer_ref` |
| `clean.py` | Profile normalization (`age=118` sentinel, `tenure_days`) |
| `attribution.py` | Offer→transaction attribution, label, recurrence flag |
| `features.py` | Leakage-free `hist_*` + offer/context features |
| `cost.py` | `reward_cost` on real conversions only (G6) |
| `contract.py` | Executable schema: Spark `StructType` + Pydantic from one `_COLUMNS` list |
| `pipeline.py` | Orchestrate raw→processed; `build_spark`, `run` |
| `split.py` | Temporal split by `campaign_wave`; exclude `informational` |
| `model_baseline.py` | Predictive baseline (logistic + LGBM) with MLflow |
| `uplift.py` | X-learner per `offer_type`, CATE uncertainty, causal importance |
| `uplift_eval.py` | Qini/AUUC, placebo permutation test |
| `gaincurve.py` | Incremental gain curves, hybrid/dynamic blend scoring, bootstrap CIs |
| `models.py` | `UpliftModel`, `ConversionModel`, `BlendedUpliftModel` — `save`/`load` |
| `serve.py` | `build_scoring_frame` (clients × offers), `recommend` (top-N) |
| `cli.py` | `train` / `predict` orchestration |
| `main.py` | Unified CLI entrypoint |
| `quadrant.py` | Uplift quadrants, gain/recurrence by quadrant |
| `eda.py` | EDA aggregations, covariate balance, segmentation helpers |
| `viz.py` | Single Plotly theme for all figures |
| `clustering.py` | Customer-level K-Means personas *(in progress)* |
| `tracking.py` | MLflow experiment tracking |

### Other directories

| Path | Responsibility |
|---|---|
| `tests/` | Structural guarantees (G1–G10), modeling invariants, simulator export parity |
| `notebooks/` | End-to-end demos over real data — import `src/`, never define transforms |
| `simulator/` | `export.py` freezes scores to JSON; `index.html` ranks in-browser |
| `simulator/data/` | Pre-computed `matrix.json`, `holdout.json`, `metadata.json` — committed |
| `specification/` | SDD backbone — see [Specification-driven design](#specification-driven-design) |
| `models/` | Serialized production model — boundary between `train` and `predict` |

Further detail: [`CLAUDE.md`](CLAUDE.md) · [`simulator/README.md`](simulator/README.md)

