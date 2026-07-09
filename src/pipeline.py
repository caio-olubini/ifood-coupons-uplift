"""Orquestração bruto→processado e escrita conforme contrato (REQ-107, T-108).

`assemble_processed` encadeia os estágios puros de `src/` no grão do contrato,
deriva `treatment` (exposição) e `campaign_wave`, e projeta no `StructType` do
contrato. `run` valida o resultado (schema, nulos, amostra Pydantic) e escreve
`data/processed/`. O entrypoint CLI faz bruto→processado ponta a ponta.
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src import contract
from src.attribution import attribute, build_label
from src.clean import normalize_profile
from src.config import PipelineConfig, load
from src.cost import add_reward_cost
from src.features import build
from src.io import parse_events, read_offers, read_profile

logger = logging.getLogger(__name__)


def build_spark(cfg: PipelineConfig, app_name: str = "ifood-uplift") -> SparkSession:
    """SparkSession local configurada pela `cfg` — o heap default da JVM não roda o dado real."""
    return (
        SparkSession.builder.master(cfg.spark_master)
        .appName(app_name)
        .config("spark.driver.memory", cfg.spark_driver_memory)
        .config("spark.sql.shuffle.partitions", cfg.spark_shuffle_partitions)
        .getOrCreate()
    )


def _treatment(df: DataFrame) -> DataFrame:
    """`treatment=1` sse a oferta foi vista — exposição real, não recebimento."""
    return df.withColumn("treatment", F.col("view_time").isNotNull().cast("int"))


def _campaign_wave(df: DataFrame) -> DataFrame:
    """Onda de campanha: índice 0-based do `received_time` distinto (rank denso).

    As ofertas saem em disparos discretos (no dataset real, seis: t=0, 7, 14, 17,
    21 e 24), não em janelas de largura fixa — nenhum bucket de N dias separa 21 de
    24 e ao mesmo tempo produz exatamente seis ondas. A onda é, portanto, a posição
    do disparo na sequência de recebimentos distintos.

    O `Window` sem `partitionBy` é intencional: o rank é global (a onda é da
    campanha, não do cliente) e incide sobre o punhado de `received_time` distintos.
    """
    waves = (
        df.select("received_time")
        .distinct()
        .withColumn(
            "campaign_wave",
            (F.dense_rank().over(Window.orderBy("received_time")) - 1).cast("int"),
        )
    )
    return df.join(F.broadcast(waves), on="received_time", how="left")


def assemble_processed(
    events: DataFrame, offers: DataFrame, profile: DataFrame, cfg: PipelineConfig
) -> DataFrame:
    """Monta o dataset processado no contrato, a partir dos eventos e do perfil já lidos.

    `events` é a saída de `io.parse_events`; `profile` é a saída de
    `clean.normalize_profile`. Encadeia atribuição → label → features → custo,
    junta o perfil, deriva `treatment`/`campaign_wave` e projeta no contrato.
    """
    attributed = attribute(events, offers, cfg)
    labeled = build_label(attributed, cfg)
    featured = build(events, labeled, offers, cfg)
    priced = add_reward_cost(featured, offers, cfg)

    enriched = priced.join(profile, on="account_id", how="left")
    with_derived = _campaign_wave(_treatment(enriched))
    return contract.enforce_schema(with_derived)


def validate(df: DataFrame, cfg: PipelineConfig) -> None:
    """Impõe o contrato antes da escrita: schema, nulos (G8) e amostra Pydantic."""
    contract.assert_schema(df)
    contract.assert_no_unexpected_nulls(df)
    n = contract.validate_sample(df, cfg)
    logger.info("Contrato validado: schema ok, sem nulos indevidos, %d linhas na amostra Pydantic.", n)


def run(cfg: PipelineConfig, spark: SparkSession) -> DataFrame:
    """Executa o pipeline bruto→processado, valida e escreve `data/processed/`.

    Retorna o DataFrame processado (já materializado pela escrita).
    """
    events = parse_events(spark, cfg)
    offers = read_offers(spark, cfg)
    profile = normalize_profile(read_profile(spark, cfg), cfg)

    processed = assemble_processed(events, offers, profile, cfg)
    validate(processed, cfg)

    out = str(cfg.processed_dir)
    processed.write.mode("overwrite").parquet(out)
    logger.info("Dataset processado escrito em %s (%d linhas).", out, processed.count())
    return processed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pipeline iFood uplift: bruto → processado.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml (default: config.yaml).")
    args = parser.parse_args()

    cfg = load(config_path=args.config)
    spark = build_spark(cfg, app_name="ifood-uplift-pipeline")
    try:
        run(cfg, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
