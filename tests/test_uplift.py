"""T-204 — X-learner: T-uplift-surething e T-xlearner-grupos."""

import numpy as np

from src.config import load
from src.uplift import fit_xlearner, label_by_arm, predict, stage_diagnostics
from tests.modeling_fixtures import synthetic_processed


def test_uplift_surething_tends_to_zero():
    """Cliente sure thing (converte com ou sem view, μ0 já alto) deve ter
    uplift perto de zero — a fraqueza exata do T-learner que o X-learner
    corrige (Premissa 5): não pode confundir "já ia converter" com "a view causou".
    """
    rng = np.random.default_rng(0)
    n = 800
    df = synthetic_processed(n=n, seed=1)
    df = df[df["offer_type"] == "bogo"].reset_index(drop=True)
    if len(df) < 100:
        df = synthetic_processed(n=2000, seed=1)
        df = df[df["offer_type"] == "bogo"].reset_index(drop=True)

    is_sure_thing = rng.binomial(1, 0.5, size=len(df)).astype(bool)
    p_convert = np.where(
        is_sure_thing, 0.95, np.where(df["treatment"] == 1, 0.9, 0.05)
    )
    df["converted"] = rng.binomial(1, p_convert)
    df["hist_spend_total"] = np.where(is_sure_thing, 500.0, df["hist_spend_total"])

    cfg = load(xlearner_n_estimators=50)
    models = fit_xlearner(df, cfg)
    preds = predict(models, df)

    sure_thing_uplift = preds.loc[is_sure_thing, "uplift"].abs().mean()
    responsive_uplift = preds.loc[~is_sure_thing, "uplift"].abs().mean()
    assert sure_thing_uplift < responsive_uplift


def test_label_impossible_in_control_degenerates_uplift_into_mu1():
    """Guard contra a reintrodução do defeito: label que exige view não é causal.

    Se a atribuição voltar a exigir view para converter, o controle fica sem
    nenhum outcome positivo, μ₀ (treinado só no controle, por offer_type) vira
    a função constante zero e τ = μ₁ − μ₀ degenera em τ ≡ μ₁.
    `test_pipeline_label_admits_conversion_in_control` garante que o pipeline
    real não a produz.
    """
    df = synthetic_processed(n=400, seed=4)
    df.loc[df["treatment"] == 0, "converted"] = 0  # G3 no dado real

    cfg = load(xlearner_n_estimators=30)
    models = fit_xlearner(df, cfg)
    for offer_type, group in df.groupby("offer_type"):
        assert list(models[offer_type].t_groups) == [1]  # μ1 treina só no tratado

    diag = stage_diagnostics(models, df)
    assert (diag["mu0_medio"] == 0).all(), "μ₀ deveria colapsar em zero"
    assert np.allclose(diag["tau_medio"], diag["mu1_medio"], atol=0.02)

    controle_sem_outcome = label_by_arm(df).query("braco.str.startswith('controle')")
    assert (controle_sem_outcome["outcome_positivo"] == 0).all()
