"""Unified CLI: `uv run coupons-uplift <command>`.

Dispatches to the existing pipeline, product CLI, and simulator export — no logic
duplicated here.
"""

from __future__ import annotations

import argparse
import logging

from simulator.export import run as export_simulator

from src.cli import predict, train
from src.config import load
from src.pipeline import build_spark, run as run_pipeline


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="coupons-uplift",
        description="iFood coupon uplift — pipeline, train, predict, and simulator export.",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config.yaml).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pipeline", help="Raw JSONs → data/processed/ (validate contract before writing).")
    sub.add_parser("train", help="Fit BlendedUpliftModel and serialize to models_dir.")
    sub.add_parser("export", help="Export static simulator artifacts to simulator/data/.")

    p_predict = sub.add_parser("predict", help="Recommend top-N actions (one offer per customer).")
    p_predict.add_argument(
        "--budget", type=int, default=None, help="Number of actions (default: cfg.predict_budget)."
    )
    p_predict.add_argument(
        "--decision-time",
        type=float,
        default=None,
        help="Decision instant as-of (default: end of observed history).",
    )
    p_predict.add_argument("--out", default=None, help="Output CSV path (default: print to stdout).")

    args = parser.parse_args(argv)
    cfg = load(config_path=args.config)

    if args.command == "pipeline":
        spark = build_spark(cfg, app_name="ifood-uplift-pipeline")
        try:
            run_pipeline(cfg, spark)
        finally:
            spark.stop()
    elif args.command == "train":
        train(cfg)
    elif args.command == "predict":
        predict(cfg, args.budget or cfg.predict_budget, args.decision_time, args.out)
    elif args.command == "export":
        export_simulator(cfg)


if __name__ == "__main__":
    main()
