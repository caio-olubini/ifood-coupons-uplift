"""Atribuição temporal oferta→transação e label de conversão (REQ-103, REQ-104).

Grão de saída: uma linha por `offer received` (account_id, offer_id, received_time).
Sob a Premissa 1 (uma oferta ativa por vez, por cliente), no máximo uma oferta
compete pela mesma transação; quando isso falha, a `AttributionPriority` da
config decide e a ocorrência é logada — nunca um `if` silencioso.

**O view não é condição da conversão.** `treatment` (viu / não viu) é a exposição;
`converted` (comprou na validade, atingindo o `min_value`) é o resultado. São eixos
independentes: um cliente pode não ver a oferta e ainda comprar na janela. Amarrar a
label ao view zeraria o resultado em todo o grupo de controle e destruiria o
contrafactual que o modelo de uplift precisa estimar (μ₀).

Uma transação só é elegível à atribuição se atinge o gasto mínimo da oferta
(`txn_amount >= min_value`, G10): abaixo desse valor o desconto nunca teria sido
concedido, e contar a compra como conversão faria o pipeline debitar um
`reward_cost` que a empresa não pagaria. `informational` tem `min_value = 0` e
portanto não é filtrada — não há gatilho de valor a atingir.
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


def _priority_order(cfg: PipelineConfig):
    """Ordenação de desempate entre recebimentos concorrentes (Premissa 1).

    `offer_id` entra como critério secundário estável: duas ofertas de uma mesma
    onda chegam no mesmo `received_time`, e sem esse desempate o `row_number`
    escolheria arbitrariamente, quebrando a reprodutibilidade entre execuções.
    """
    received_order = (
        F.col("received_time").asc()
        if cfg.attribution_priority == AttributionPriority.EARLIEST_RECEIVED
        else F.col("received_time").desc()
    )
    return [received_order, F.col("offer_id").asc()]


def _owned_views(base: DataFrame, viewed: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Atribui cada view física a um único recebimento e devolve a `view_time` por grão.

    Um evento de view só serve a um recebimento cuja janela `[received_time,
    valid_until]` o contenha — o que importa quando a MESMA oferta é reenviada
    em ondas cujas janelas se sobrepõem: uma única view cairia em ambas e, se
    compartilhada, marcaria `treatment=1` em dois recebimentos a partir de uma
    só exposição. Espelhando a exclusividade das transações, a view pertence a
    um recebimento só (desempate por `AttributionPriority`); cada recebimento
    fica com a primeira view que passou a lhe pertencer.
    """
    view_owner_window = Window.partitionBy("account_id", "offer_id", "view_time").orderBy(
        *_priority_order(cfg)
    )
    first_view_window = Window.partitionBy("account_id", "offer_id", "received_time").orderBy(
        "view_time"
    )
    return (
        base.join(viewed, on=["account_id", "offer_id"], how="inner")
        .filter(
            (F.col("view_time") >= F.col("received_time"))
            & (F.col("view_time") <= F.col("valid_until"))
        )
        .withColumn("view_owner_rank", F.row_number().over(view_owner_window))
        .filter(F.col("view_owner_rank") == 1)
        .withColumn("first_view_rank", F.row_number().over(first_view_window))
        .filter(F.col("first_view_rank") == 1)
        .select("account_id", "offer_id", "received_time", "view_time")
    )


def attribute(events: DataFrame, offers: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Constrói o grão (account_id, offer_id, received_time) com view e transações atribuídas.

    Uma linha `offer received` recebe, no máximo, uma `view_time` (a primeira
    view do cliente para aquela oferta após o recebimento) e no máximo UMA
    transação — a primeira elegível em `[received_time, received_time + duration]`
    — reportada em `assigned_txn_count` (0 ou 1), `assigned_txn_amount_sum` e
    `first_assigned_txn_time`. Conversão é um evento binário por recebimento;
    compras adicionais na mesma janela não são somadas nem contadas de novo.

    Só entram as transações que atingem o gasto mínimo da oferta (`min_value`):
    uma compra abaixo dele não dispara a recompensa, logo não é conversão.

    Tanto view quanto transação são **exclusivas**: um único evento físico é
    atribuído a no máximo um recebimento. Quando o mesmo evento é disputado por
    mais de uma oferta ativa (violação da Premissa 1), a `AttributionPriority`
    decide o dono e — no caso das transações — o conflito é logado.
    """
    received = _received(events)
    viewed = _viewed(events)
    # `txn_id` é gerado uma vez e cacheado: `monotonically_increasing_id` é
    # recalculado a cada action e há duas aqui (o log de sobreposição e a
    # materialização final). Sem cache, os ids poderiam divergir entre elas.
    txns = _transactions(events).withColumn("txn_id", F.monotonically_increasing_id()).cache()

    offers_meta = offers.select(
        F.col("id").alias("offer_id"),
        F.col("duration"),
        F.col("min_value").cast("double").alias("min_value"),
    )

    base = received.join(offers_meta, on="offer_id", how="left").withColumn(
        "valid_until", F.col("received_time") + F.col("duration")
    )

    base_with_view = base.join(
        _owned_views(base, viewed, cfg),
        on=["account_id", "offer_id", "received_time"],
        how="left",
    )

    # A janela de atribuição é a validade da oferta, `[received_time, valid_until]`,
    # e **não depende do view**. O view é o *tratamento* (exposição), não uma
    # condição da conversão: exigi-lo aqui zeraria a label em todo o grupo de
    # controle (`treatment=0 ⇒ converted=0` por construção), colapsando μ₀ ≡ 0 e
    # tornando o uplift τ = μ₁ − μ₀ ≡ μ₁ — isto é, não-causal. Quem não viu e
    # comprou dentro da validade converteu; é exatamente esse o contrafactual que
    # o X-learner precisa observar.
    #
    # O gasto mínimo (G10) é filtrado AQUI, antes da disputa de posse: uma compra
    # abaixo do `min_value` não ativa a recompensa, então a oferta não a "gastou"
    # e não pode reivindicá-la. Filtrar depois do `row_number` deixaria a oferta
    # inelegível vencer a disputa e descartar a transação de uma oferta elegível
    # de `min_value` menor, que legitimamente converteria com ela.
    in_window_candidates = base_with_view.join(txns, on="account_id", how="inner").filter(
        (F.col("txn_time") >= F.col("received_time"))
        & (F.col("txn_time") <= F.col("valid_until"))
        & (F.col("txn_amount") >= F.col("min_value"))
    )

    txn_owner_window = Window.partitionBy("account_id", "txn_id").orderBy(*_priority_order(cfg))

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

    # Uma oferta é atribuída à sua PRIMEIRA transação elegível na janela, não à soma
    # de todas: a conversão é o evento "o cliente comprou usando a oferta", que
    # acontece uma vez. Contar/somar todas as compras elegíveis do período inflaria
    # `assigned_txn_amount_sum` (e portanto `reward_cost` e `conversion_value`) com
    # compras que a mesma oferta não poderia ter causado mais de uma vez.
    first_txn_window = Window.partitionBy("account_id", "offer_id", "received_time").orderBy(
        "txn_time"
    )
    assigned = (
        owned_txns.withColumn("first_txn_rank", F.row_number().over(first_txn_window))
        .filter(F.col("first_txn_rank") == 1)
        .groupBy("account_id", "offer_id", "received_time")
        .agg(
            F.count("txn_id").alias("assigned_txn_count"),
            F.sum("txn_amount").alias("assigned_txn_amount_sum"),
            F.min("txn_time").alias("first_assigned_txn_time"),
        )
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
    """Deriva `converted` e `conversion_value` (REQ-104).

    `converted=1` sse há ao menos uma transação atribuída — e a atribuição em
    `attribute` exige que a transação ocorra dentro da validade
    `[received_time, valid_until]` e com valor ≥ `min_value` da oferta. **Não**
    exige view: ver a oferta é o tratamento, não o rótulo. Um recebimento não
    visto com compra na janela tem `converted=1`, e é justamente essa massa que
    dá μ₀ > 0 no grupo de controle.

    Nunca deriva de `offer completed` (cobre informational, G5).
    `conversion_value` soma as transações atribuídas.
    """
    converted = (F.col("assigned_txn_count") > 0).cast("int")

    return df.withColumn("converted", converted).withColumn(
        "conversion_value",
        F.when(converted == 1, F.col("assigned_txn_amount_sum")).otherwise(F.lit(0.0)),
    )


def add_recurrence_flag(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """`is_recurrent`: recebimento convertido cujo cliente converteu de novo
    (qualquer oferta) em até `cfg.recurrence_window_days` dias após esta compra.

    Derivada do target (`converted`), não uma feature preditiva — vazaria o
    próprio rótulo se entrasse em X. Mede recorrência no nível de campanha
    (todo o dataset), não por cliente isolado: cada linha convertida olha as
    demais compras atribuídas do mesmo `account_id`, em qualquer oferta, sem
    reduzir por cliente. Recebimentos com `converted=0` são sempre `is_recurrent=0`
    — não há segunda compra para ancorar a janela.
    """
    conversions = df.filter(F.col("converted") == 1).select(
        F.col("account_id").alias("_acc"),
        F.col("offer_id").alias("_other_offer_id"),
        F.col("received_time").alias("_other_received_time"),
        F.col("first_assigned_txn_time").alias("_other_txn_time"),
    )

    pairs = df.filter(F.col("converted") == 1).join(
        conversions, on=(F.col("account_id") == F.col("_acc")), how="left"
    ).filter(
        ~(
            (F.col("_other_offer_id") == F.col("offer_id"))
            & (F.col("_other_received_time") == F.col("received_time"))
        )
        & (F.col("_other_txn_time") > F.col("first_assigned_txn_time"))
        & (
            F.col("_other_txn_time")
            <= F.col("first_assigned_txn_time") + F.lit(cfg.recurrence_window_days)
        )
    )

    recurrent_keys = pairs.select("account_id", "offer_id", "received_time").distinct().withColumn(
        "is_recurrent", F.lit(1)
    )

    return (
        df.join(recurrent_keys, on=["account_id", "offer_id", "received_time"], how="left")
        .withColumn("is_recurrent", F.coalesce(F.col("is_recurrent"), F.lit(0)))
    )
