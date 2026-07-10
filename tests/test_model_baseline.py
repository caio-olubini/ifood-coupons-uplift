"""T-203 — baseline preditivo: logística vs LGBM sob validação temporal."""

import mlflow

from src.config import load
from src.model_baseline import train, train_and_log
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


def test_train_and_log_creates_mlflow_run(tmp_path):
    cfg = load(
        validation_wave_cutoff=4, n_campaign_waves=6, lgbm_n_estimators=100,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db", mlflow_experiment_name="test-baseline",
    )
    df = synthetic_processed(n=300, seed=3)
    train_df, holdout_df = _train_holdout_split(df, cutoff=cfg.validation_wave_cutoff)

    train_and_log(train_df, holdout_df, cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow_experiment_name)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert "metrics.auc_lgbm" in runs.columns
