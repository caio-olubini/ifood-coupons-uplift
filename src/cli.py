"""CLI de produto: `train` (ajusta e serializa) e `predict` (recomenda ações).

Fecha a fronteira treinar→prever que os wrappers de `src.models` habilitam:

    uv run coupons-uplift train
    uv run coupons-uplift predict --budget 5000 --out recomendacoes.csv

`train` ajusta o `BlendedUpliftModel` padrão da config no lado de treino do split
temporal (sem `informational`) e o escreve em `cfg.models_dir`. `predict` **não**
pontua a base histórica — carrega o modelo, monta a matriz de scoring (clientes ×
ofertas ativas as-of o instante de decisão, `src.serve`), pontua e devolve as
top-N ações com uma oferta por cliente. Toda a lógica vive em `src.models`,
`src.serve` e `src.split`; aqui só há orquestração e o parsing de argumentos.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from pyspark.sql import functions as F

from src import serve, split
from src.config import PipelineConfig, load
from src.models import BlendedUpliftModel
from src.pipeline import build_spark

logger = logging.getLogger(__name__)


def train(cfg: PipelineConfig) -> BlendedUpliftModel:
    """Ajusta o `BlendedUpliftModel` da config no treino e o serializa.

    Split temporal por `campaign_wave` e remoção de `informational` são os mesmos
    do notebook de modelagem — o modelo produtivo treina exatamente sobre o que a
    avaliação validou. O objeto ajustado inteiro (uplift + conversão + parâmetros
    do blend) vai para `cfg.models_dir`.
    """
    spark = build_spark(cfg, app_name="ifood-uplift-train")
    try:
        processed = spark.read.parquet(str(cfg.processed_dir))
        train_sdf, _ = split.temporal_split(processed, cfg)
        train_df = split.exclude_informational(train_sdf).toPandas()
    finally:
        spark.stop()

    logger.info("Treinando BlendedUpliftModel (mode=%s) em %d linhas.", cfg.blend_mode, len(train_df))
    model = BlendedUpliftModel.from_config(cfg).fit(train_df)
    path = model.save(cfg)
    logger.info("Modelo escrito em %s.", path)
    return model


def _default_decision_time(spark, cfg: PipelineConfig) -> float:
    """Instante de decisão default = fim do histórico observado (`max(time)`).

    Decidir "agora" significa pontuar com todo o histórico disponível; qualquer
    `event_time < decision_time` das features é então todo o log conhecido.
    """
    from src.io import parse_events

    return float(parse_events(spark, cfg).agg(F.max("time")).first()[0])


def predict(cfg: PipelineConfig, budget: int, decision_time: float | None, out: str | None):
    """Recomenda as `budget` ações de maior score, uma oferta por cliente.

    Monta a matriz de scoring (clientes × ofertas ativas de bogo/discount) as-of
    `decision_time`, pontua com o `BlendedUpliftModel` salvo e seleciona
    (`serve.recommend`, amostragem softmax por `cfg.blend_temperature`,
    reprodutível pela seed). Sem `--out`, imprime as recomendações; com `--out`,
    escreve o CSV.
    """
    model = BlendedUpliftModel.load(cfg)

    spark = build_spark(cfg, app_name="ifood-uplift-predict")
    try:
        if decision_time is None:
            decision_time = _default_decision_time(spark, cfg)
        # O modelo só conhece bogo/discount (informational fora da modelagem);
        # ofertas ativas candidatas são as com desconto no catálogo.
        offers = serve.read_offers(spark, cfg)
        active_ids = [
            r["id"]
            for r in offers.filter(F.col("offer_type").isin(list(split.MODELED_OFFER_TYPES)))
            .select("id")
            .collect()
        ]
        scoring_sdf = serve.build_scoring_frame(spark, cfg, decision_time, active_ids)
        scored = scoring_sdf.toPandas()
    finally:
        spark.stop()

    scored["score"] = model.score(scored).to_numpy()
    # Rank padrão dos modelos: amostragem softmax por `cfg.blend_temperature`,
    # reprodutível pela seed da config (ver `serve.recommend`).
    recs = serve.recommend(
        scored, budget,
        temperature=cfg.blend_temperature,
        rng=np.random.default_rng(cfg.seed),
    )

    logger.info(
        "Recomendadas %d ações (budget=%d) as-of t=%.2f, de %d pares candidatos.",
        len(recs), budget, decision_time, len(scored),
    )
    if out:
        recs.to_csv(out, index=False)
        logger.info("Recomendações escritas em %s.", out)
    else:
        print(recs.to_string(index=False))
    return recs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="iFood uplift — treino e recomendação (serve).")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml (default: config.yaml).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Ajusta o BlendedUpliftModel e o serializa em models_dir.")

    p_predict = sub.add_parser("predict", help="Recomenda as top-N ações (uma oferta por cliente).")
    p_predict.add_argument("--budget", type=int, default=None, help="Nº de ações a recomendar (default: cfg.predict_budget).")
    p_predict.add_argument("--decision-time", type=float, default=None, help="Instante de decisão as-of (default: fim do histórico).")
    p_predict.add_argument("--out", default=None, help="CSV de saída (default: imprime na tela).")

    args = parser.parse_args()
    cfg = load(config_path=args.config)

    if args.command == "train":
        train(cfg)
    elif args.command == "predict":
        predict(cfg, args.budget or cfg.predict_budget, args.decision_time, args.out)


if __name__ == "__main__":
    main()
