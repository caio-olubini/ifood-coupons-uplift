"""Split temporal treino/validação por onda de campanha (REQ-201 NFR, T-202).

`campaign_wave` é o rank do `received_time` distinto (ver `pipeline._campaign_wave`),
não uma data — ondas antes do corte treinam, ondas a partir do corte validam. Nunca
split aleatório: um cliente pode aparecer nos dois lados legitimamente (recebeu
ofertas em ondas diferentes), mas nenhuma linha de treino pode ter `received_time`
posterior a uma linha de holdout — a ordem temporal é o invariante, não a exclusividade
de cliente.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PipelineConfig
from src.policy import ELIGIBLE_OFFER_TYPES


def exclude_informational(df: DataFrame) -> DataFrame:
    """Remove `offer_type == informational` antes de qualquer ajuste de modelo.

    `informational` não paga desconto (`cost.add_reward_cost`) e está fora do
    escopo da política (`policy.ELIGIBLE_OFFER_TYPES`); deixá-lo influenciar o
    baseline preditivo ou o X-learner misturaria um mecanismo de recompensa que
    a política nunca vai usar. Chamar logo após `temporal_split`, antes de
    `toPandas()`, garante que nenhum modelo — baseline, X-learner, avaliação ou
    curva de ganho — jamais o veja.
    """
    return df.filter(F.col("offer_type").isin(list(ELIGIBLE_OFFER_TYPES)))


def temporal_split(df: DataFrame, cfg: PipelineConfig) -> tuple[DataFrame, DataFrame]:
    """Divide `df` em (treino, holdout) por `campaign_wave < cfg.validation_wave_cutoff`.

    Ondas `[0, cutoff)` treinam; ondas `[cutoff, n_campaign_waves)` validam.
    """
    train = df.filter(F.col("campaign_wave") < cfg.validation_wave_cutoff)
    holdout = df.filter(F.col("campaign_wave") >= cfg.validation_wave_cutoff)
    return train, holdout


def assert_temporal_order(train: DataFrame, holdout: DataFrame) -> None:
    """Guarda contra split que inverte a ordem temporal (T-split-temporal).

    Rejeita qualquer split onde a linha de treino mais recente é posterior à
    linha de holdout mais antiga — cliente repetido entre os dois lados é
    esperado (recebeu ofertas em ondas diferentes); ordem invertida não é.
    """
    max_train = train.agg(F.max("received_time")).first()[0]
    min_holdout = holdout.agg(F.min("received_time")).first()[0]
    if max_train is None or min_holdout is None:
        raise ValueError("split temporal produziu um lado vazio — ajuste validation_wave_cutoff")
    if max_train >= min_holdout:
        raise ValueError(
            f"split temporal inválido: max(received_time) do treino ({max_train}) "
            f">= min(received_time) do holdout ({min_holdout})"
        )
