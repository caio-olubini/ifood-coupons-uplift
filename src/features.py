"""Features pré-recebimento, sem leakage temporal (REQ-105).

Toda feature `hist_*` de uma linha usa exclusivamente eventos com
`time < received_time` daquela linha (garantia G2). As features de oferta e
contexto vêm do catálogo e da contagem de ofertas concorrentes no recebimento.

O anti-leakage é estrutural: as agregações históricas cruzam o grão com o log
de eventos do mesmo cliente, filtram por `event_time < received_time` *antes* de
qualquer agregação, agregam, e re-anexam ao grão por left-join — assim uma linha
sem histórico sobrevive com contadores zerados, e nenhum evento igual ou
posterior ao recebimento alcança um acumulador.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PipelineConfig

_KEY = ["account_id", "offer_id", "received_time"]


def _offer_type_map(offers: DataFrame) -> DataFrame:
    return offers.select(F.col("id").alias("offer_id"), F.col("offer_type"))


def _transaction_features(base: DataFrame, events: DataFrame) -> DataFrame:
    """Features transacionais históricas: só transações com `time < received_time`."""
    txns = events.filter(F.col("event") == "transaction").select(
        F.col("account_id"),
        F.col("time").alias("txn_time"),
        F.col("amount").alias("txn_amount"),
    )

    prior = base.join(txns, on="account_id", how="inner").filter(
        F.col("txn_time") < F.col("received_time")
    )

    half = F.col("received_time") / 2
    first_half = F.when(F.col("txn_time") < half, F.col("txn_amount"))
    second_half = F.when(F.col("txn_time") >= half, F.col("txn_amount"))

    agg = prior.groupBy(*_KEY).agg(
        F.sum("txn_amount").alias("hist_spend_total"),
        F.count("txn_time").cast("int").alias("hist_txn_count"),
        F.avg("txn_amount").alias("hist_avg_ticket"),
        F.stddev("txn_amount").alias("hist_spend_std"),
        F.max("txn_time").alias("_last_txn_time"),
        F.avg(first_half).alias("_first_half_avg"),
        F.avg(second_half).alias("_second_half_avg"),
    )

    out = (
        base.select(*_KEY)
        .join(agg, on=_KEY, how="left")
        .withColumn("hist_spend_total", F.coalesce(F.col("hist_spend_total"), F.lit(0.0)))
        .withColumn("hist_txn_count", F.coalesce(F.col("hist_txn_count"), F.lit(0)))
        .withColumn("hist_avg_ticket", F.coalesce(F.col("hist_avg_ticket"), F.lit(0.0)))
        .withColumn("hist_spend_std", F.coalesce(F.col("hist_spend_std"), F.lit(0.0)))
        .withColumn(
            "hist_recency_days",
            F.when(
                F.col("_last_txn_time").isNotNull(),
                F.col("received_time") - F.col("_last_txn_time"),
            ).otherwise(F.lit(None)),
        )
        .withColumn(
            "hist_frequency",
            F.when(
                F.col("received_time") > 0, F.col("hist_txn_count") / F.col("received_time")
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "hist_spend_trend",
            F.coalesce(F.col("_second_half_avg"), F.lit(0.0))
            - F.coalesce(F.col("_first_half_avg"), F.lit(0.0)),
        )
    )
    return out.drop("_last_txn_time", "_first_half_avg", "_second_half_avg")


def _offer_response_features(base: DataFrame, events: DataFrame, offers: DataFrame) -> DataFrame:
    """Features de resposta a ofertas passadas: só eventos com `time < received_time`.

    Uma conversão histórica é medida sobre os eventos observáveis do cliente
    (received/viewed/completed), sem recomputar a atribuição.
    """
    type_map = _offer_type_map(offers).withColumnRenamed("offer_id", "hist_offer_id")

    offer_events = (
        events.filter(F.col("event").isin("offer received", "offer viewed", "offer completed"))
        .select(
            F.col("account_id"),
            F.col("event"),
            F.col("time").alias("event_time"),
            F.col("offer_ref").alias("hist_offer_id"),
        )
        .join(type_map, on="hist_offer_id", how="left")
    )

    prior = base.join(offer_events, on="account_id", how="inner").filter(
        F.col("event_time") < F.col("received_time")
    )

    is_received = F.col("event") == "offer received"
    is_viewed = F.col("event") == "offer viewed"
    is_completed = F.col("event") == "offer completed"

    def received_of(t: str):
        return (is_received & (F.col("offer_type") == t)).cast("int")

    def completed_of(t: str):
        return (is_completed & (F.col("offer_type") == t)).cast("int")

    agg = prior.groupBy(*_KEY).agg(
        F.sum(is_received.cast("int")).cast("int").alias("hist_offers_received"),
        F.sum(is_viewed.cast("int")).cast("int").alias("hist_offers_viewed"),
        F.sum(is_completed.cast("int")).cast("int").alias("hist_offers_completed"),
        F.sum(received_of("bogo")).cast("int").alias("hist_offers_received_bogo"),
        F.sum(received_of("discount")).cast("int").alias("hist_offers_received_discount"),
        F.sum(received_of("informational")).cast("int").alias("hist_offers_received_info"),
        F.sum(completed_of("bogo")).alias("_completed_bogo"),
        F.sum(completed_of("discount")).alias("_completed_discount"),
    )

    def rate(num, den):
        return F.when(den > 0, num / den).otherwise(F.lit(0.0))

    count_cols = [
        "hist_offers_received",
        "hist_offers_viewed",
        "hist_offers_completed",
        "hist_offers_received_bogo",
        "hist_offers_received_discount",
        "hist_offers_received_info",
    ]
    out = base.select(*_KEY).join(agg, on=_KEY, how="left")
    for col in count_cols:
        out = out.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    return (
        out.withColumn("hist_view_rate", rate(F.col("hist_offers_viewed"), F.col("hist_offers_received")))
        .withColumn("hist_conv_rate_bogo", rate(F.coalesce(F.col("_completed_bogo"), F.lit(0)), F.col("hist_offers_received_bogo")))
        .withColumn(
            "hist_conv_rate_discount",
            rate(F.coalesce(F.col("_completed_discount"), F.lit(0)), F.col("hist_offers_received_discount")),
        )
        .drop("_completed_bogo", "_completed_discount")
    )


def _completed_unseen_feature(base: DataFrame, events: DataFrame) -> DataFrame:
    """`hist_completed_unseen_flag`: já completou alguma oferta sem tê-la visto antes.

    Assinatura de "sure thing". Medido sobre pares (cliente, oferta) com
    `offer completed` cujo view precedente (se houver) foi posterior ao
    completed, tudo em `time < received_time`.
    """
    completed = events.filter(F.col("event") == "offer completed").select(
        F.col("account_id"),
        F.col("offer_ref").alias("hist_offer_id"),
        F.col("time").alias("completed_time"),
    )
    viewed = events.filter(F.col("event") == "offer viewed").select(
        F.col("account_id"),
        F.col("offer_ref").alias("hist_offer_id"),
        F.col("time").alias("view_time"),
    )

    completed_with_view = (
        completed.join(viewed, on=["account_id", "hist_offer_id"], how="left")
        .groupBy("account_id", "hist_offer_id", "completed_time")
        .agg(
            F.max(
                (F.col("view_time").isNotNull() & (F.col("view_time") <= F.col("completed_time"))).cast("int")
            ).alias("had_prior_view")
        )
        .withColumn("is_unseen", (F.col("had_prior_view") == 0).cast("int"))
    )

    prior = base.join(completed_with_view, on="account_id", how="inner").filter(
        F.col("completed_time") < F.col("received_time")
    )
    agg = prior.groupBy(*_KEY).agg(F.max("is_unseen").alias("hist_completed_unseen_flag"))

    return (
        base.select(*_KEY)
        .join(agg, on=_KEY, how="left")
        .withColumn(
            "hist_completed_unseen_flag",
            F.coalesce(F.col("hist_completed_unseen_flag"), F.lit(0)).cast("int"),
        )
    )


def _time_view_to_conv_feature(base: DataFrame, events: DataFrame) -> DataFrame:
    """`hist_time_view_to_conv`: tempo médio view→completed em ofertas passadas."""
    completed = events.filter(F.col("event") == "offer completed").select(
        F.col("account_id"),
        F.col("offer_ref").alias("hist_offer_id"),
        F.col("time").alias("completed_time"),
    )
    viewed = events.filter(F.col("event") == "offer viewed").select(
        F.col("account_id"),
        F.col("offer_ref").alias("hist_offer_id"),
        F.col("time").alias("view_time"),
    )

    view_to_conv = (
        completed.join(viewed, on=["account_id", "hist_offer_id"], how="inner")
        .filter(F.col("view_time") <= F.col("completed_time"))
        .groupBy("account_id", "hist_offer_id", "completed_time")
        .agg(F.max("view_time").alias("last_view_time"))
        .withColumn("view_to_conv", F.col("completed_time") - F.col("last_view_time"))
    )

    prior = base.join(view_to_conv, on="account_id", how="inner").filter(
        F.col("completed_time") < F.col("received_time")
    )
    agg = prior.groupBy(*_KEY).agg(F.avg("view_to_conv").alias("hist_time_view_to_conv"))

    return base.select(*_KEY).join(agg, on=_KEY, how="left")


def _offer_context_features(base: DataFrame, offers: DataFrame) -> DataFrame:
    """Features do catálogo da oferta e do contexto de recebimento.

    `n_concurrent_offers`: quantas outras ofertas o cliente tinha ativas (janela
    de validade contendo `received_time`) no momento deste recebimento.
    """
    catalog = offers.select(
        F.col("id").alias("offer_id"),
        F.col("discount_value"),
        F.col("min_value"),
        F.col("duration"),
        F.col("channels"),
    )

    with_catalog = (
        base.join(catalog, on="offer_id", how="left")
        .withColumn("n_channels", F.size("channels").cast("int"))
        .withColumn("channel_web", F.array_contains("channels", "web").cast("int"))
        .withColumn("channel_email", F.array_contains("channels", "email").cast("int"))
        .withColumn("channel_mobile", F.array_contains("channels", "mobile").cast("int"))
        .withColumn("channel_social", F.array_contains("channels", "social").cast("int"))
        .withColumn(
            "discount_to_minvalue_ratio",
            F.when(F.col("min_value") > 0, F.col("discount_value") / F.col("min_value")).otherwise(F.lit(0.0)),
        )
        .drop("channels")
    )

    all_received = base.select(
        F.col("account_id"),
        F.col("offer_id").alias("other_offer_id"),
        F.col("received_time").alias("other_received_time"),
    ).join(
        offers.select(F.col("id").alias("other_offer_id"), F.col("duration").alias("other_duration")),
        on="other_offer_id",
        how="left",
    )

    concurrency = (
        base.select(*_KEY)
        .join(all_received, on="account_id", how="left")
        .filter(
            (F.col("other_received_time") <= F.col("received_time"))
            & (F.col("received_time") <= F.col("other_received_time") + F.col("other_duration"))
            & ~(
                (F.col("other_offer_id") == F.col("offer_id"))
                & (F.col("other_received_time") == F.col("received_time"))
            )
        )
        .groupBy(*_KEY)
        .agg(F.count("*").cast("int").alias("n_concurrent_offers"))
    )

    return with_catalog.join(concurrency, on=_KEY, how="left").withColumn(
        "n_concurrent_offers", F.coalesce(F.col("n_concurrent_offers"), F.lit(0))
    )


def build(events: DataFrame, base: DataFrame, offers: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Anexa todas as features ao grão `base` (saída de `attribution.build_label`).

    `base` já traz `(account_id, offer_id, received_time)` e o label; esta função
    adiciona as features transacionais, de resposta a ofertas, e de oferta/contexto,
    todas sem leakage temporal (REQ-105, G2).
    """
    txn = _transaction_features(base, events)
    resp = _offer_response_features(base, events, offers)
    unseen = _completed_unseen_feature(base, events)
    v2c = _time_view_to_conv_feature(base, events)
    ctx = _offer_context_features(base, offers)

    return (
        ctx.join(txn, on=_KEY, how="left")
        .join(resp, on=_KEY, how="left")
        .join(unseen, on=_KEY, how="left")
        .join(v2c, on=_KEY, how="left")
    )
