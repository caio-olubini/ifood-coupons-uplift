"""X-learner de uplift: efeito de ver a oferta sobre `converted` (REQ-202).

Um X-learner por `offer_type` — cada tipo tem mecanismo de recompensa distinto
(bogo/discount pagam, informational não) e a saída precisa ser comparável por
tipo (REQ-202: "uplift por cliente × tipo de oferta"). `treatment` do contrato
(view=1/0) é o tratamento causal; `converted` é o outcome. Só as features
`hist_*`/cliente/oferta entram em X — nunca `converted`, `conversion_value`,
`reward_cost` (seriam leakage do próprio rótulo) nem `treatment` (é o
tratamento, não uma feature).

Escolhido por robustez a grupos de tamanhos desiguais e μ₀ mal-estimado
(Premissa 5) — a fraqueza exata do T-learner: um cliente "sure thing" (que já
converteria de qualquer forma, μ₀ alto) deve receber uplift ≈ 0, não um
artefato de dois modelos mal calibrados subtraídos.

Propensity é passada explicitamente (taxa de view observada, constante por
`offer_type`) em vez de deixar o CausalML estimar a sua internamente: o
projeto não estima propensity para de-confounding (RCT, Premissa 4/Não-objetivo
em `00-clarify.md`), e o estimador default do CausalML (`LogisticRegressionCV`)
não tolera os nulos legítimos de `age`/`credit_card_limit`/`hist_*` (G8) —
usar propensity fixa evita alimentar esse modelo auxiliar de qualquer forma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from lightgbm import LGBMRegressor

from src.config import PipelineConfig
from src.model_baseline import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMN = "converted"
OFFER_TYPE_COLUMN = "offer_type"

# offer_type é o eixo de estratificação (um modelo por tipo); não entra como
# feature dentro de cada modelo — seria constante no grupo.
_XLEARNER_FEATURES = [c for c in FEATURE_COLUMNS if c != OFFER_TYPE_COLUMN]
_XLEARNER_CATEGORICAL = [c for c in CATEGORICAL_COLUMNS if c != OFFER_TYPE_COLUMN]


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[_XLEARNER_FEATURES].copy()
    for col in _XLEARNER_CATEGORICAL:
        X[col] = X[col].astype("category").cat.codes
    return X


def _fixed_propensity(treatment: np.ndarray) -> np.ndarray:
    """Propensity constante = taxa de view observada no grupo (Premissa 4: RCT,
    propensity conhecida, não estimada). Mesmo valor repetido por linha —
    CausalML espera um vetor do tamanho de `treatment`, não um escalar.
    """
    rate = treatment.mean()
    return np.full_like(treatment, fill_value=rate, dtype=float)


def _make_learner(cfg: PipelineConfig) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=cfg.xlearner_n_estimators,
        max_depth=cfg.xlearner_max_depth,
        learning_rate=cfg.xlearner_learning_rate,
        random_state=cfg.seed,
        verbose=-1,
    )


def fit_xlearner(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, BaseXRegressor]:
    """Ajusta um X-learner por `offer_type` presente em `df`.

    Retorna `{offer_type: modelo_ajustado}`. Cada modelo vê só as linhas do
    seu tipo — `informational` não tem `reward_cost`, misturar os tipos no
    mesmo modelo confundiria mecanismos de recompensa diferentes.
    """
    models: dict[str, BaseXRegressor] = {}
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        X = _design_matrix(group)
        treatment = group[TREATMENT_COLUMN].to_numpy()
        y = group[OUTCOME_COLUMN].to_numpy()
        p = _fixed_propensity(treatment)

        model = BaseXRegressor(learner=_make_learner(cfg), control_name=0)
        model.fit(X=X, treatment=treatment, y=y, p=p)
        models[offer_type] = model
    return models


def predict(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """Uplift por linha, usando o modelo do `offer_type` daquela linha.

    Retorna `[account_id, offer_id, offer_type, uplift]`, alinhado ao índice
    de `df` (uma estimativa por par cliente × oferta recebida — REQ-202).
    """
    parts = []
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        X = _design_matrix(group)
        treatment = group[TREATMENT_COLUMN].to_numpy()
        p = _fixed_propensity(treatment)
        te = model.predict(X=X, p=p)
        uplift = te[:, 0] if te.ndim == 2 else te.ravel()
        parts.append(pd.DataFrame({
            "account_id": group["account_id"].to_numpy(),
            "offer_id": group["offer_id"].to_numpy(),
            "offer_type": offer_type,
            "uplift": uplift,
        }))
    return pd.concat(parts, ignore_index=True)


# --- Diagnóstico: de onde vem o número de uplift -------------------------------
#
# O X-learner é τ(x) = p·τ_c(x) + (1−p)·τ_t(x), construído sobre dois modelos de
# resultado: μ₀ (treinado só no controle) e μ₁ (só no tratado). Um uplift alto
# demais quase sempre é μ₀ degenerado, não efeito causal grande — e μ₀ degenera
# quando o label é impossível no controle. Estas funções abrem a caixa.


def label_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Estrutura do outcome dentro de cada braço, por `offer_type`.

    Se `taxa_outcome` no controle for exatamente 0, μ₀ ≡ 0 e o "uplift" degenera
    em μ₁ — o modelo devolve a taxa de conversão dos tratados, não um efeito.
    """
    linhas = []
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        for arm in (0, 1):
            braco = group[group[TREATMENT_COLUMN] == arm]
            linhas.append({
                "offer_type": offer_type,
                "braco": "controle (não viu)" if arm == 0 else "tratado (viu)",
                "n": len(braco),
                "outcome_positivo": int(braco[OUTCOME_COLUMN].sum()),
                "taxa_outcome": braco[OUTCOME_COLUMN].mean() if len(braco) else float("nan"),
            })
    return pd.DataFrame(linhas)


def stage_diagnostics(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """μ₀, μ₁ e τ previstos em `df`, por `offer_type` — os estágios do X-learner.

    `mu0_*` vem de `model_mu_c` (ajustado só no controle) e `mu1_*` de
    `models_mu_t` (só no tratado). `tau_medio` é o uplift final. A identidade
    `tau ≈ mu1 − mu0` deve valer; quando `mu0` é constante zero, `tau ≈ mu1` e
    o número não é causal.
    """
    linhas = []
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        X = _design_matrix(group)
        arm = model.t_groups[0]

        mu0 = model.model_mu_c.predict(X)
        mu1 = model.models_mu_t[arm].predict(X)
        tau = model.predict(X=X, p=_fixed_propensity(group[TREATMENT_COLUMN].to_numpy()))
        tau = tau[:, 0] if tau.ndim == 2 else tau.ravel()

        linhas.append({
            "offer_type": offer_type,
            "mu0_medio": mu0.mean(), "mu0_desvio": mu0.std(),
            "mu1_medio": mu1.mean(), "mu1_desvio": mu1.std(),
            "mu1_menos_mu0": mu1.mean() - mu0.mean(),
            "tau_medio": tau.mean(),
        })
    return pd.DataFrame(linhas)
