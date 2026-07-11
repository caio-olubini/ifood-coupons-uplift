# iFood Coupons Uplift

Decide **which offer to send (or not send)** to each customer to maximize net
profit, and prove the gain through offline evaluation — no live A/B. The problem
is **incrementality (uplift)**, not completion classification: sending a coupon
to someone who would have bought anyway costs margin without adding revenue.

Everything runs locally (no cloud / Databricks) on **PySpark local**, managed by
**UV**. Read the specs in `specification/` before touching the pipeline — they
are the source of truth, not this file.

---

## CLI commands

### Setup

```bash
uv sync                        # install deps (includes the dev group)
```

### Data pipeline — raw → processed

```bash
uv run python -m src.pipeline                    # raw JSONs → data/processed/ (validates the contract before writing)
uv run python -m src.pipeline --config other.yaml
```

Reads the 3 raw JSONs, runs the full staged transform (parse → clean →
attribution → label → features → cost → contract), validates against
`specification/schema-processed.md`, and writes the partitioned Parquet.

### Product CLI — train / predict

```bash
uv run python -m src.cli train                       # fit BlendedUpliftModel on the training split, serialize into models_dir
uv run python -m src.cli predict --budget 5000       # recommend the top-N actions (one offer per customer)
uv run python -m src.cli predict --out recs.csv      # write the CSV instead of printing
uv run python -m src.cli predict --decision-time 24  # score as-of a decision instant (default: end of history)
```

- **`train`** fits the `BlendedUpliftModel` from the config on the training side
  of the temporal split (no `informational`) and serializes it into
  `cfg.models_dir` — same data prep as the notebook, no new numbers.
- **`predict`** is **real serving, not prediction over the historical base**: it
  builds the scoring matrix **customers × active offers** as-of a decision
  instant, scores it with the saved model, and returns the top-N actions. It
  applies **one offer per customer** (best offer per `account_id`), then top-N
  by budget (`cfg.predict_budget` / `--budget`).

### Tests

```bash
uv run pytest -q                          # full integrity suite (~36 tests)
uv run pytest tests/test_leakage.py -q    # a single file
```

### Notebooks (end-to-end)

```bash
uv run python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('notebooks/2_modeling.ipynb',as_version=4); \
NotebookClient(nb,timeout=5400,kernel_name='python3',resources={'metadata':{'path':'.'}}).execute()"
```

---

## Where each artifact lives

| Path | What it is |
|---|---|
| `config.yaml` | Every behavior parameter of the pipeline & modeling (REQ-110). Nothing is hardcoded in `src/`. |
| `data/raw/` | The 3 input JSONs: `offers.json`, `profile.json`, `transactions.json`. |
| `data/processed/` | Output of `python -m src.pipeline` — partitioned Parquet at grain `(account_id, offer_id, received_time)`. The contract between pipeline and modeling. |
| `models/` | Serialized fitted model (`blended_uplift_model.pkl`) — where `cli train` writes and `cli predict` reads. |
| `mlflow.db` | Local SQLite MLflow tracking store (no server). |
| `src/` | All logic — pure, testable functions. Notebooks only import from here and display. |
| `tests/` | ~36 structural integrity tests (guarantees G1–G10, REQ-2xx, boundary invariants). |
| `notebooks/1_eda.ipynb` | Deliverable EDA: overview, events over time, quality, distributions, correlation, funnel, segmentation, causal diagnostics. |
| `notebooks/2_modeling.ipynb` | Full modeling run over real data: split, baselines, X-learner, Qini/AUUC, blends, gain curve. |
| `specification/` | Source of truth. `schema-processed.md` is the contract; `spec.md`, `tasks.md`, `00-clarify.md`, `02-modeling/`. |

### `src/` modules

**Pipeline (raw → processed):**

| Module | Role |
|---|---|
| `config.py` | `PipelineConfig` (Pydantic) loaded from `config.yaml`. `load(config_path=..., **overrides)`. |
| `io.py` | Reads the 3 JSONs; `parse_events` unpacks `value` and coalesces `offer id`/`offer_id` into a single `offer_ref`. |
| `clean.py` | `normalize_profile`: `age=118` sentinel → `identity_missing`, missing `gender` → `unknown`, `tenure_days`. |
| `attribution.py` | `attribute` (grain, validity window, `min_value` filter, overlap resolution), `build_label` (influence-aware `converted`/`conversion_value`), `add_recurrence_flag`. |
| `features.py` | `build`: leakage-free `hist_*` features + offer/context features. |
| `cost.py` | `add_reward_cost`: discount cost on real conversions only. |
| `contract.py` | Executable contract: `StructType` + Pydantic from one `_COLUMNS` list. `enforce_schema`, `assert_schema`, guards. |
| `pipeline.py` | Orchestrates raw→processed: `assemble_processed`, `validate`, `run`, `build_spark`, CLI entrypoint. |

**Modeling & serving:**

| Module | Role |
|---|---|
| `split.py` | Temporal split by `campaign_wave`; `exclude_informational`; `MODELED_OFFER_TYPES`. |
| `model_baseline.py` | Predictive baseline (logistic + LGBM) with MLflow tracking. |
| `uplift.py` | X-learner per `offer_type` (fixed propensity), CATE uncertainty, causal importance. |
| `uplift_eval.py` | Qini/AUUC (via `sklift`), placebo test, importance figures. |
| `gaincurve.py` | Offline eval: incremental gain curve per budget top-N; hybrid/dynamic blend scoring; bootstrap CIs. |
| `models.py` | Model wrappers: `UpliftModel`, `ConversionModel`, `BlendedUpliftModel` (production model). `from_config`, `save`/`load`, `feature_importance`. |
| `serve.py` | `build_scoring_frame` (customers × active offers as-of decision), `recommend` (one offer/customer + top-N by budget). |
| `cli.py` | Product CLI `train`/`predict` — orchestrates `models` + `serve` + `split`. |
| `quadrant.py` | Uplift-quadrant classification, gain-by-quadrant, recurrence-by-quadrant. |
| `tracking.py` | MLflow experiment tracking. |
| `eda.py` | EDA / covariate balance / K-Means segmentation functions (Spark → small pandas + figures). |
| `viz.py` | Single executive Plotly theme (validated palette, light/dark). |

---

## What's implemented

**Data pipeline (spec 01, T-101…T-112).** Full staged transform behind
`python -m src.pipeline`: config, io, clean, attribution + influence-aware label,
leakage-free features, reward cost, executable contract + write. The grain is
`(account_id, offer_id, received_time)`, unique.

- **Guarantees G1–G10** are tested invariants: G1 unique grain · G2 no temporal
  leakage · G3 label does **not** require view (control converts — view is the
  treatment, not the label) · G4 conversion within validity · G5 informational
  has no `offer completed` · G6 coherent cost · G7 sentinel handled · G8 no null
  in non-nullable column · G9 exclusive exposure · G10 conversion reaches
  `min_value`.
- **`is_recurrent`** in the contract: a converted receipt whose customer
  converts again (any offer) within `recurrence_window_days`. Derived from the
  target — never a feature.

**EDA & segmentation (T-108…T-111).** Descriptive EDA, covariate balance
(view/no-view **and** across received offers), K-Means segmentation with
explicit geometry. All figures on the single `viz.py` theme.

**Modeling (spec 02).** Implemented: T-201…T-203, T-205, T-208, T-212.

- **Baseline** predictive model (logistic + LGBM), `auc_lgbm=0.85 > auc_logit=0.80`.
- **X-learner** per `offer_type` with fixed propensity; `informational` excluded
  from modeling.
- **Qini/AUUC evaluation** (via `sklift`) + **placebo permutation test**: real
  Qini 0.0335 clears the null's 95th percentile (0.0186), empirical p = 0/20.
- **Blends**: hybrid `X-learner + λ·raw-conversion` (best fixed λ=0.3) and a
  **dynamic** version weighted by the X-learner's internal CATE disagreement
  (best γ=1.0, Qini/AUUC 0.069/0.073 — best of the whole tested space).
- **Offline evaluation** = incremental gain curve per budget top-N (not IPW):
  incremental conversions (Qini-style scaled counterfactual) × mean profit per
  treated conversion, monotone envelope, bootstrap CIs.

**Product CLI (`cli.py`, `serve.py`, `models.py`).** Model wrappers encapsulate
train + predict; `train` fits and serializes the `BlendedUpliftModel`, `predict`
serves the top-N actions as-of a decision instant.

**Removed by user decision** (recorded in the specs as `~~struck~~`):
cost-sensitive policy + allocation baselines (REQ-204/205), IPW / Direct Method
(REQ-207/208/211), magnitude calibration + isotonic correction (REQ-213/214),
and `informational` from modeling. The project evaluates uplift models by
Qini/AUUC and the per-budget gain curve — there is no longer a "decide who to
send to" allocation step.

---

## Conventions

- **Configurability is law**: a magic value (window, threshold, path, seed)
  inside a function is a defect → `config.yaml`.
- **Anti-leakage is structural**: historical features filter
  `event_time < received_time` *before* aggregating, then re-join to the grain.
- **Pydantic at the edges**, never row-by-row in the Spark hot path.
- **Every rate names its denominator**: `taxa_conversao` (over received) and
  `taxa_conversao_vistos` (over viewers) are different numbers and live side by
  side.
- **Balance is diagnostic, not a gate**: SMD above threshold qualifies the
  causal reading, never changes the estimator.
- **Spec vs. data divergence is recorded, not patched in code** — the measured
  number goes to the notebook and the spec; code changes only by contract
  decision.
