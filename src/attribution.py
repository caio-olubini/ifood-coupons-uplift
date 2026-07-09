"""Atribuição temporal oferta→transação e label influence-aware (REQ-103, REQ-104).

Grão de saída: uma linha por `offer received` (account_id, offer_id, received_time).
Sob a Premissa 1 (uma oferta ativa por vez, por cliente), no máximo uma oferta
compete pela mesma transação; quando isso falha, a `AttributionPriority` da
config decide e a ocorrência é logada — nunca um `if` silencioso.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import AttributionPriority, PipelineConfig

logger = logging.getLogger(__name__)


def _received(events: DataFrame) -> DataFrame:
    return events.filter(F.col("event") == "offer received").select(
        F.col("account_id"),
        F.col("offer_ref").alias("offer_id"),
        F.col("time").alias("received_time"),
    )


def _viewed(events: DataFrame) -> DataFrame:
    return events.filter(F.col("event") == "offer viewed").select(
        F.col("account_id"),
        F.col("offer_ref").alias("offer_id"),
        F.col("time").alias("view_time"),
    )


def _transactions(events: DataFrame) -> DataFrame:
    return events.filter(F.col("event") == "transaction").select(
        F.col("account_id"),
        F.col("time").alias("txn_time"),
        F.col("amount").alias("txn_amount"),
    )


def attribute(events: DataFrame, offers: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Constrói o grão (account_id, offer_id, received_time) com view e transações atribuídas.

    Uma linha `offer received` recebe, no máximo, uma `view_time` (a primeira
    view do cliente para aquela oferta após o recebimento) e a agregação das
    transações que caem em `[received_time, received_time + duration]`:
    `assigned_txn_count`, `assigned_txn_amount_sum` e `first_assigned_txn_time`.

    Quando uma transação cai na janela de mais de uma oferta recebida pelo
    mesmo cliente (violação da Premissa 1), a `AttributionPriority` decide a
    qual oferta ela é atribuída, e o caso é logado.
    """
    received = _received(events)
    viewed = _viewed(events)
    txns = _transactions(events).withColumn("txn_id", F.monotonically_increasing_id())

    offers_meta = offers.select(F.col("id").alias("offer_id"), F.col("duration"))

    base = received.join(offers_meta, on="offer_id", how="left").withColumn(
        "valid_until", F.col("received_time") + F.col("duration")
    )

    # O view só pode ser atribuído a um recebimento cuja janela de validade o
    # contenha. Isso importa quando a MESMA oferta é recebida em ondas distintas:
    # um view pertence à onda cuja [received_time, valid_until] o cobre, não a
    # qualquer recebimento anterior daquela oferta.
    view_window = Window.partitionBy("account_id", "offer_id", "received_time").orderBy("view_time")
    views_in_window = (
        base.join(viewed, on=["account_id", "offer_id"], how="inner")
        .filter(
            (F.col("view_time") >= F.col("received_time"))
            & (F.col("view_time") <= F.col("valid_until"))
        )
        .withColumn("rn", F.row_number().over(view_window))
        .filter(F.col("rn") == 1)
        .select("account_id", "offer_id", "received_time", "view_time")
    )
    base_with_view = base.join(
        views_in_window, on=["account_id", "offer_id", "received_time"], how="left"
    )

    # Regra influence-aware estrita: a transação só é atribuída como conversão
    # se ocorre DEPOIS do view e dentro da validade. Sem view (view_time nulo),
    # `txn_time >= view_time` é nulo → nada é atribuído, como deve ser.
    in_window_candidates = base_with_view.join(txns, on="account_id", how="inner").filter(
        (F.col("txn_time") >= F.col("view_time")) & (F.col("txn_time") <= F.col("valid_until"))
    )

    priority_order = (
        F.col("received_time").asc()
        if cfg.attribution_priority == AttributionPriority.EARLIEST_RECEIVED
        else F.col("received_time").desc()
    )
    txn_owner_window = Window.partitionBy("account_id", "txn_id").orderBy(priority_order)

    n_overlaps = (
        in_window_candidates.withColumn(
            "n_competing_offers", F.count("offer_id").over(Window.partitionBy("account_id", "txn_id"))
        )
        .filter(F.col("n_competing_offers") > 1)
        .select("account_id", "txn_id")
        .distinct()
        .count()
    )
    if n_overlaps > 0:
        logger.warning(
            "Premissa 1 violada em %d transação(ões): mais de uma oferta ativa no intervalo; "
            "prioridade '%s' aplicada.",
            n_overlaps,
            cfg.attribution_priority.value,
        )

    owned_txns = in_window_candidates.withColumn(
        "txn_owner_rank", F.row_number().over(txn_owner_window)
    ).filter(F.col("txn_owner_rank") == 1)

    assigned = owned_txns.groupBy("account_id", "offer_id", "received_time").agg(
        F.count("txn_id").alias("assigned_txn_count"),
        F.sum("txn_amount").alias("assigned_txn_amount_sum"),
        F.min("txn_time").alias("first_assigned_txn_time"),
    )

    return (
        base_with_view.join(assigned, on=["account_id", "offer_id", "received_time"], how="left")
        .withColumn("assigned_txn_count", F.coalesce(F.col("assigned_txn_count"), F.lit(0)))
        .withColumn("assigned_txn_amount_sum", F.coalesce(F.col("assigned_txn_amount_sum"), F.lit(0.0)))
        .select(
            "account_id",
            "offer_id",
            "received_time",
            "valid_until",
            "view_time",
            "assigned_txn_count",
            "assigned_txn_amount_sum",
            "first_assigned_txn_time",
        )
    )


def build_label(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Deriva `converted` e `conversion_value` (REQ-104, Premissa 2).

    `converted=1` sse há ao menos uma transação atribuída — e a atribuição em
    `attribute` já exige que a transação ocorra DEPOIS do view e dentro da
    validade (influence-aware estrito). Logo `assigned_txn_count > 0` já implica
    view precedente; nunca deriva de `offer completed` (cobre informational, G5).
    `conversion_value` soma as transações atribuídas.
    """
    converted = (F.col("assigned_txn_count") > 0).cast("int")

    return df.withColumn("converted", converted).withColumn(
        "conversion_value",
        F.when(converted == 1, F.col("assigned_txn_amount_sum")).otherwise(F.lit(0.0)),
    )
