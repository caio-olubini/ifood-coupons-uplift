"""Leitura dos três JSONs brutos e parsing do campo `value` (REQ-101).

Os eventos `offer received`/`offer viewed` carregam a referência de oferta em
`value.offer id` (com espaço); `offer completed` carrega em `value.offer_id`
(com underscore). `transaction` não tem referência de oferta, só `amount`.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import PipelineConfig


def read_offers(spark: SparkSession, cfg: PipelineConfig) -> DataFrame:
    return spark.read.option("multiLine", True).json(str(cfg.offers_path))


def read_profile(spark: SparkSession, cfg: PipelineConfig) -> DataFrame:
    return spark.read.option("multiLine", True).json(str(cfg.profile_path))


def read_events(spark: SparkSession, cfg: PipelineConfig) -> DataFrame:
    return spark.read.option("multiLine", True).json(str(cfg.transactions_path))


def parse_events(spark: SparkSession, cfg: PipelineConfig) -> DataFrame:
    """Lê `transactions.json` e desempacota `value` numa referência de oferta única.

    Coalesce `offer id` (received/viewed) e `offer_id` (completed) em `offer_ref`,
    sem perder nenhum dos dois nomes de campo (REQ-101).
    """
    raw = read_events(spark, cfg)
    return raw.select(
        F.col("account_id"),
        F.col("event"),
        F.col("time_since_test_start").alias("time"),
        F.coalesce(F.col("value.offer id"), F.col("value.offer_id")).alias("offer_ref"),
        F.col("value.amount").alias("amount"),
        F.col("value.reward").alias("reward"),
    )
