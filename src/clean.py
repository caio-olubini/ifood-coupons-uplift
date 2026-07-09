"""Normalização do profile: sentinela de identidade e tenure (REQ-102)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PipelineConfig


def normalize_profile(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Marca `identity_missing`, anula a idade sentinela e deriva `tenure_days`.

    `registered_on` é `YYYYMMDD`; `tenure_days` é a distância até
    `cfg.test_start_date` (t=0 do teste), configurável (REQ-110).
    """
    registered_date = F.to_date(F.col("registered_on").cast("string"), "yyyyMMdd")
    test_start_date = F.to_date(F.lit(cfg.test_start_date), "yyyyMMdd")

    return df.select(
        F.col("id").alias("account_id"),
        F.when(F.col("age") == cfg.age_sentinel, None).otherwise(F.col("age")).alias("age"),
        F.coalesce(F.col("gender"), F.lit("unknown")).alias("gender"),
        F.col("credit_card_limit").alias("credit_card_limit"),
        F.when(F.col("age") == cfg.age_sentinel, 1).otherwise(0).alias("identity_missing"),
        F.datediff(test_start_date, registered_date).alias("tenure_days"),
    )
