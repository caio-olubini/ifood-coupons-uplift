"""Baseline preditivo: logística (âncora) e LGBM sob validação temporal (REQ-201).

Não é o modelo de uplift — é o degrau anterior que prova que há sinal aprendível
em `converted` antes de gastar a complexidade extra do X-learner (Premissa 5).
Nunca split aleatório: treina no lado `train` do split temporal (`src.split`),
avalia no `holdout`.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow

from src.config import PipelineConfig
from src.contract import CONTRACT_COLUMNS
from src.tracking import start_run

NON_FEATURE_COLUMNS = frozenset({
    "account_id", "offer_id", "received_time", "campaign_wave",
    "treatment", "converted", "conversion_value", "reward_cost",
})

FEATURE_COLUMNS: list[str] = [c for c in CONTRACT_COLUMNS if c not in NON_FEATURE_COLUMNS]
CATEGORICAL_COLUMNS: list[str] = ["gender", "offer_type"]
TARGET_COLUMN = "converted"


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Features do contrato, categóricas como `category`; nulos ficam como estão
    (LGBM trata nativamente; a logística exige imputação — feita ali, não aqui).
    """
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")
    return X


def _design_matrix_logit(df: pd.DataFrame) -> pd.DataFrame:
    """Versão da matriz para a logística: categóricas viram dummy, nulos viram 0
    (a logística não trata null nativamente; LGBM sim — por isso as matrizes divergem).
    """
    X = _design_matrix(df)
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return X.fillna(0.0)


def train(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, cfg: PipelineConfig
) -> tuple[Pipeline, LGBMClassifier, dict[str, float]]:
    """Treina logística e LGBM no `train_df`, avalia por AUC no `holdout_df`.

    Retorna os dois modelos ajustados e as métricas (`auc_logit`, `auc_lgbm`).
    """
    y_train, y_holdout = train_df[TARGET_COLUMN], holdout_df[TARGET_COLUMN]

    X_train_logit = _design_matrix_logit(train_df)
    X_holdout_logit = _design_matrix_logit(holdout_df).reindex(columns=X_train_logit.columns, fill_value=0.0)
    logit = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=cfg.logit_max_iter, random_state=cfg.seed)
    )
    logit.fit(X_train_logit, y_train)
    auc_logit = roc_auc_score(y_holdout, logit.predict_proba(X_holdout_logit)[:, 1])

    X_train_lgbm = _design_matrix(train_df)
    X_holdout_lgbm = _design_matrix(holdout_df)
    lgbm = LGBMClassifier(
        n_estimators=cfg.lgbm_n_estimators,
        max_depth=cfg.lgbm_max_depth,
        learning_rate=cfg.lgbm_learning_rate,
        random_state=cfg.seed,
        verbose=-1,
    )
    lgbm.fit(X_train_lgbm, y_train, categorical_feature=CATEGORICAL_COLUMNS)
    auc_lgbm = roc_auc_score(y_holdout, lgbm.predict_proba(X_holdout_lgbm)[:, 1])

    return logit, lgbm, {"auc_logit": auc_logit, "auc_lgbm": auc_lgbm}


def train_and_log(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, cfg: PipelineConfig
) -> tuple[Pipeline, LGBMClassifier, dict[str, float]]:
    """`train` com o run MLflow: params do baseline e as duas AUCs (REQ-209)."""
    with start_run("baseline_predictivo", cfg):
        mlflow.log_params({
            "logit_max_iter": cfg.logit_max_iter,
            "lgbm_n_estimators": cfg.lgbm_n_estimators,
            "lgbm_max_depth": cfg.lgbm_max_depth,
            "lgbm_learning_rate": cfg.lgbm_learning_rate,
            "validation_wave_cutoff": cfg.validation_wave_cutoff,
            "seed": cfg.seed,
        })
        logit, lgbm, metrics = train(train_df, holdout_df, cfg)
        mlflow.log_metrics(metrics)
    return logit, lgbm, metrics
