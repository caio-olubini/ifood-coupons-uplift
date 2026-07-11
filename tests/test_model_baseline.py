"""T-203 — baseline preditivo: logística vs LGBM sob validação temporal."""

from src.config import load
from src.model_baseline import train
from tests.modeling_fixtures import synthetic_processed


def _train_holdout_split(df, cutoff=4):
    return df[df["campaign_wave"] < cutoff], df[df["campaign_wave"] >= cutoff]


def test_lgbm_meets_or_beats_logit_auc():
    cfg = load(validation_wave_cutoff=4, n_campaign_waves=6, lgbm_n_estimators=100)
    df = synthetic_processed(n=600, seed=7)
    train_df, holdout_df = _train_holdout_split(df, cutoff=cfg.validation_wave_cutoff)

    _, _, metrics = train(train_df, holdout_df, cfg)

    assert metrics["auc_lgbm"] >= metrics["auc_logit"] - 0.05  # tolerância a ruído de amostra pequena
    assert 0.5 < metrics["auc_logit"] <= 1.0
    assert 0.5 < metrics["auc_lgbm"] <= 1.0


def test_predict_conversion_probability_is_mu1_aligned_to_the_input_index():
    """μ₁ por linha, no índice de entrada — a política o consome no termo de custo
    e o baseline top-completion aloca por ele (REQ-205).
    """
    from src.model_baseline import predict_conversion_probability

    train_df = synthetic_processed(n=300, seed=11)
    holdout_df = synthetic_processed(n=120, seed=12)
    holdout_df.index = range(1000, 1000 + len(holdout_df))  # índice não trivial

    cfg = load(lgbm_n_estimators=30)
    _, lgbm, _ = train(train_df, holdout_df, cfg)
    p = predict_conversion_probability(lgbm, holdout_df)

    assert len(p) == len(holdout_df)
    assert list(p.index) == list(holdout_df.index)
    assert ((p >= 0.0) & (p <= 1.0)).all()
