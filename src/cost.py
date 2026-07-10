"""Custo do desconto concedido (REQ-106, G6).

`reward_cost` é o que a empresa paga quando a oferta converte: o `discount_value`
do catálogo, para conversões de `bogo`/`discount`. Para `informational` (sem
recompensa) e para não-convertidos, o custo é 0 — garantindo G6:
`reward_cost > 0` ⇒ `converted=1` e `offer_type ≠ informational`.

Cobrar o custo em toda conversão só é correto porque `converted=1` implica
`conversion_value ≥ min_value` (G10, imposto lá atrás na atribuição): a compra
de fato disparou a recompensa. Enquanto a atribuição aceitava qualquer compra
pós-view, este módulo debitava desconto que nunca teria sido concedido.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PipelineConfig


def add_reward_cost(df: DataFrame, offers: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Anexa `reward_cost` ao grão rotulado.

    Requer que `df` já traga `converted` e as colunas de oferta. `offer_type` e
    `discount_value` vêm do catálogo; o custo só é não-nulo quando a conversão
    ocorre numa oferta com recompensa (bogo/discount).
    """
    catalog = offers.select(
        F.col("id").alias("offer_id"),
        F.col("offer_type"),
        F.col("discount_value").alias("_catalog_discount_value"),
    )

    joined = df.join(catalog, on="offer_id", how="left")

    is_rewardable = (F.col("converted") == 1) & (F.col("offer_type") != "informational")

    return joined.withColumn(
        "reward_cost",
        F.when(is_rewardable, F.col("_catalog_discount_value").cast("double")).otherwise(F.lit(0.0)),
    ).drop("_catalog_discount_value")
