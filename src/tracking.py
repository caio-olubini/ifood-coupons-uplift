"""Wrappers MLflow: nomes de run estáveis, o que logar por run (REQ-209).

Tracking local por arquivo (`cfg.mlflow_tracking_uri`), sem servidor. Todo
treino chama `start_run` como contexto; params/métricas/artefatos entram por
`log_params`/`log_metrics`/`log_artifact` do próprio `mlflow` dentro do bloco.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow

from src.config import PipelineConfig


@contextmanager
def start_run(run_name: str, cfg: PipelineConfig) -> Iterator[None]:
    """Abre um run MLflow com nome estável, no experimento e tracking URI da config."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield
