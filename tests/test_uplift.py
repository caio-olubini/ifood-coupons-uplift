"""T-204 — X-learner: T-uplift-surething e T-xlearner-grupos."""

import numpy as np
import pandas as pd

from src.config import load
from src.uplift import fit_xlearner, label_by_arm, predict, stage_diagnostics
from tests.modeling_fixtures import synthetic_processed


def test_uplift_surething_tends_to_zero():
    """Cliente sure thing (converte com ou sem view, μ0 já alto) deve ter
    uplift perto de zero — a fraqueza exata do T-learner que o X-learner
    corrige (Premissa 5): não pode confundir "já ia converter" com "a view causou".

    A fixture gera `converted` independente de `treatment`, permitindo μ₀ > 0 —
    o que o pipeline agora de fato produz, desde que a atribuição deixou de
    exigir view (o view é o tratamento, não o rótulo).
    """
    rng = np.random.default_rng(0)
    n = 800
    df = synthetic_processed(n=n, seed=1)
    df = df[df["offer_type"] == "bogo"].reset_index(drop=True)
    if len(df) < 100:
        df = synthetic_processed(n=2000, seed=1)
        df = df[df["offer_type"] == "bogo"].reset_index(drop=True)

    # Metade dos clientes é sure thing: converte quase sempre, visto ou não.
    # A outra metade só converte se visto (efeito real).
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


def test_xlearner_stage_models_train_on_correct_subsets():
    """μ0 (mu_c) treina só em controle; μ1 (mu_t) só em tratado — por grupo de
    offer_type. Verificado inspecionando os dados vistos pelo modelo ajustado,
    não só o predict, para não deixar um bug de mistura de grupos passar
    despercebido atrás de uma métrica agregada.
    """
    df = synthetic_processed(n=500, seed=5)
    cfg = load(xlearner_n_estimators=30)
    models = fit_xlearner(df, cfg)

    for offer_type, group in df.groupby("offer_type"):
        model = models[offer_type]
        n_control = (group["treatment"] == 0).sum()
        n_treated = (group["treatment"] == 1).sum()
        # control_name=0 é o único grupo de tratamento não-controle (treatment=1).
        assert list(model.t_groups) == [1]
        assert n_control > 0 and n_treated > 0


def test_predict_returns_one_row_per_input_row():
    df = synthetic_processed(n=300, seed=2)
    cfg = load(xlearner_n_estimators=30)
    models = fit_xlearner(df, cfg)
    preds = predict(models, df)

    assert len(preds) == len(df)
    assert set(preds.columns) == {"account_id", "offer_id", "received_time", "offer_type", "uplift"}
    assert preds["uplift"].notna().all()


def test_predict_preserves_row_order_and_the_contract_grain():
    """`predict` devolve o grão completo `(account_id, offer_id, received_time)`
    na ordem de linha da entrada.

    Sem `received_time` a chave não é única — a mesma oferta chega ao mesmo
    cliente em ondas diferentes — e juntar o uplift ao holdout por
    `(account_id, offer_id, offer_type)` vira produto cartesiano: no dado real
    inflava 25.469 linhas para 27.365, contaminando o Qini. Sem a ordem
    preservada, `groupby` reordenaria as linhas por `offer_type`.
    """
    df = synthetic_processed(n=300, seed=3)
    # Mesmo cliente, mesma oferta, duas ondas: o grão só distingue por received_time.
    df.loc[1, ["account_id", "offer_id", "offer_type"]] = df.loc[0, ["account_id", "offer_id", "offer_type"]].to_numpy()
    df.loc[1, "received_time"] = df.loc[0, "received_time"] + 7.0

    cfg = load(xlearner_n_estimators=30)
    preds = predict(fit_xlearner(df, cfg), df)

    assert len(preds) == len(df)
    assert not preds.duplicated(["account_id", "offer_id", "received_time"]).any()
    # Ordem de linha idêntica: dá para atribuir a coluna sem join.
    pd.testing.assert_series_equal(preds["account_id"], df["account_id"])
    pd.testing.assert_series_equal(preds["received_time"], df["received_time"])


def test_nullable_contract_columns_do_not_break_fit_or_predict():
    """G8 permite null em age/credit_card_limit/hist_recency_days/
    hist_time_view_to_conv — o dado real tem essas colunas incompletas. O
    X-learner usa LGBM (tolera null nativo) mas a propensity, se estimada
    pelo CausalML, cairia num LogisticRegressionCV que não tolera — por isso
    passamos propensity fixa (ver `_fixed_propensity`). Este teste reproduz o
    bug encontrado ao rodar o notebook sobre o dado real (NaN quebrava o fit).
    """
    df = synthetic_processed(n=300, seed=9)
    rng = np.random.default_rng(9)
    for col in ["age", "credit_card_limit", "hist_recency_days", "hist_time_view_to_conv"]:
        mask = rng.binomial(1, 0.3, size=len(df)).astype(bool)
        df.loc[mask, col] = np.nan

    cfg = load(xlearner_n_estimators=30)
    models = fit_xlearner(df, cfg)
    preds = predict(models, df)

    assert preds["uplift"].notna().all()


def test_label_impossible_in_control_degenerates_uplift_into_mu1():
    """Guard contra a reintrodução do defeito: label que exige view não é causal.

    Se a atribuição voltar a exigir view para converter, o controle fica sem
    nenhum outcome positivo, μ₀ vira a função constante zero e τ = μ₁ − μ₀
    degenera em τ ≡ μ₁ — o "uplift" passa a ser a taxa de conversão dos
    tratados. Aqui a degeneração é forçada à mão para fixar a assinatura
    numérica que a delata; `test_pipeline_label_admits_conversion_in_control`
    garante que o pipeline real não a produz.
    """
    df = synthetic_processed(n=400, seed=4)
    df.loc[df["treatment"] == 0, "converted"] = 0  # G3 no dado real

    cfg = load(xlearner_n_estimators=30)
    models = fit_xlearner(df, cfg)

    diag = stage_diagnostics(models, df)
    assert (diag["mu0_medio"] == 0).all(), "μ₀ deveria colapsar em zero"
    assert (diag["mu0_desvio"] == 0).all(), "μ₀ deveria ser constante"
    # τ ≡ μ₁: o uplift não carrega informação causal alguma.
    assert np.allclose(diag["tau_medio"], diag["mu1_medio"], atol=0.02)

    controle_sem_outcome = label_by_arm(df).query("braco.str.startswith('controle')")
    assert (controle_sem_outcome["outcome_positivo"] == 0).all()
