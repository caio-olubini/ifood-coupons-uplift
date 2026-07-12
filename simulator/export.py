"""Export offline dos artefatos estáticos do simulador (spec 03, REQ-301/305).

`uv run coupons-uplift export [--config ...]`

Congela a matriz de scoring (clientes × ofertas ativas as-of um instante de
decisão) e o holdout rotulado em JSON estático, para a UI (JS no browser)
refazer a seleção do `serve.recommend` e o analytics de
`gaincurve.incremental_gain_curve` sem backend. Reusa as **mesmas** funções
puras do serving e da avaliação — nenhuma feature/score é reimplementado aqui.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from src import gaincurve, serve, split
from src.config import PipelineConfig, load
from src.gaincurve import NET_PROFIT_COLUMN, TREATMENT_COLUMN, _profit_per_treated_conversion
from src.models import BlendedUpliftModel
from src.pipeline import build_spark
from src.quadrant import QUADRANT_ORDER, classify_quadrant

logger = logging.getLogger(__name__)

#: Colunas da matriz de serving. O grão vem do serving; os scores/quadrante são
#: anexados pelo modelo (null nas linhas informational, que o modelo não conhece).
_GRAIN_COLUMNS = ["account_id", "offer_id", "offer_type"]
_SCORE_COLUMNS = ["uplift", "p_convert", "uncertainty", "score_dynamic", "quadrant"]
_MATRIX_COLUMNS = _GRAIN_COLUMNS + _SCORE_COLUMNS

#: Holdout rotulado para o card de analytics (REQ-305).
_HOLDOUT_COLUMNS = [
    "account_id",
    "offer_type",
    "treatment",
    "converted",
    "net_profit",
    "score_random",
    "p_convert",
    "score_dynamic",
    "quadrant",
]


def _decision_time(spark, cfg: PipelineConfig) -> float:
    """Instante de decisão = fim do histórico observado (`max(time)`)."""
    from src.io import parse_events

    return float(parse_events(spark, cfg).agg(F.max("time")).first()[0])


def _active_offers(spark, cfg: PipelineConfig) -> list[dict]:
    """Catálogo de ofertas para o `offers.json`."""
    rows = serve.read_offers(spark, cfg).select(
        F.col("id").alias("offer_id"),
        "offer_type",
        "discount_value",
        "min_value",
        "duration",
    ).collect()
    return [r.asDict() for r in rows]


def _json_safe(value):
    """NaN/NaT viram `None` (→ `null`): `JSON.parse` do browser rejeita `NaN` cru."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _columnar(pdf, columns: list[str]) -> dict[str, list]:
    """Converte um pandas DataFrame em dict colunar, saneando NaN → None."""
    return {col: [_json_safe(v) for v in pdf[col].tolist()] for col in columns}


def _hash_score(account_id: str, offer_id: str, seed: int) -> float:
    """Score aleatório seedado — espelho do `hashScore` em `simulator/index.html`."""
    h = 2166136261 ^ seed
    key = f"{account_id}|{offer_id}"
    for ch in key:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h & 0xFFFFFFFF) / 4294967296


def _attach_modeled_scores(
    frame: pd.DataFrame, model: BlendedUpliftModel, cfg: PipelineConfig
) -> pd.DataFrame:
    """Anexa uplift, p_convert, uncertainty, score_dynamic e quadrante."""
    result = frame.copy()
    for col in _SCORE_COLUMNS:
        result[col] = None

    modeled = frame[frame["offer_type"].isin(split.MODELED_OFFER_TYPES)]
    if modeled.empty:
        return result

    uplift_pred = model.uplift_model.predict(modeled)["uplift"]
    p_convert = model.conversion_model.predict_proba(modeled)
    uncertainty = model.uplift_model.predict_uncertainty(modeled)["uncertainty"]
    score_dynamic = gaincurve.dynamic_hybrid_score(
        uncertainty, uplift_pred, p_convert, cfg.simulator_score_gamma
    )
    stages = model.uplift_model.predict_stages(modeled)
    quadrant = classify_quadrant(stages, p_convert, cfg)

    result.loc[modeled.index, "uplift"] = uplift_pred
    result.loc[modeled.index, "p_convert"] = p_convert
    result.loc[modeled.index, "uncertainty"] = uncertainty
    result.loc[modeled.index, "score_dynamic"] = score_dynamic
    result.loc[modeled.index, "quadrant"] = quadrant
    return result


def _global_profit_per_conversion(holdout_df: pd.DataFrame) -> float:
    """Lucro médio por conversão tratada no holdout inteiro (insumo do lucro projetado)."""
    profit = holdout_df[NET_PROFIT_COLUMN].to_numpy()
    treated = (holdout_df[TREATMENT_COLUMN].to_numpy() == 1).astype(float)
    converted = holdout_df["converted"].to_numpy(dtype=float)
    ppc = _profit_per_treated_conversion(profit, treated, converted)
    return float(ppc[len(holdout_df)])


def _global_revenue_per_conversion(holdout_df: pd.DataFrame) -> float:
    """Receita bruta média por conversão tratada — insumo da receita projetada."""
    mask = (holdout_df[TREATMENT_COLUMN] == 1) & (holdout_df["converted"] == 1)
    subset = holdout_df.loc[mask, "conversion_value"]
    if subset.empty:
        return 0.0
    return float(subset.mean())


def _build_holdout(spark, cfg: PipelineConfig, model: BlendedUpliftModel) -> tuple[dict[str, list], pd.DataFrame]:
    """Holdout rotulado com scores e quadrante para métricas globais no metadata."""
    processed = spark.read.parquet(str(cfg.processed_dir))
    _, holdout_sdf = split.temporal_split(processed, cfg)
    holdout = split.exclude_informational(holdout_sdf).toPandas()
    holdout = gaincurve.add_net_profit(holdout)
    scored = _attach_modeled_scores(holdout, model, cfg)

    scored["net_profit"] = scored[NET_PROFIT_COLUMN]
    scored["score_random"] = [
        _hash_score(acc, oid, cfg.seed)
        for acc, oid in zip(scored["account_id"], scored["offer_id"], strict=True)
    ]
    return _columnar(scored, _HOLDOUT_COLUMNS), scored


def _metadata_labels() -> dict[str, object]:
    """Rótulos de UI para público não-técnico (REQ-313)."""
    return {
        "strategies": {
            "aleatorio": {
                "label": "Distribuir ao acaso",
                "help": "Manda cupom para quem calhar — o ponto de partida para comparar.",
            },
            "conversao": {
                "label": "Priorizar conversão",
                "help": "Prioriza quem tem mais chance de comprar — mas parte talvez comprasse sem o cupom.",
            },
            "uplift": {
                "label": "Priorizar uplift",
                "help": "Prioriza quem compra por causa do cupom — o efeito real da oferta.",
            },
        },
        "quadrants": {
            "persuadable": {
                "label": "Persuadables",
                "help": "Quem o cupom convence a comprar.",
            },
            "sure_thing": {
                "label": "Sure things",
                "help": "Quem já compra de qualquer jeito.",
            },
            "lost_cause": {
                "label": "Lost causes",
                "help": "Quem dificilmente converte.",
            },
            "sleeping_dog": {
                "label": "Sleeping dogs",
                "help": "Quem o cupom atrapalha — melhor não enviar.",
            },
        },
        "projection_disclaimer": (
            "Projeção (efeito esperado × médias do histórico) — lucro líquido e receita bruta por conversão."
        ),
        "analytics_disclaimer": (
            "Ganho medido no histórico de validação — o que já aconteceu nos dados, "
            "não uma previsão para frente."
        ),
        "glossary": [
            {
                "term": "Estratégia",
                "definition": (
                    "A regra que define em que ordem os clientes são priorizados "
                    "para receber cupom."
                ),
            },
            {
                "term": "Enviados",
                "definition": (
                    "Quantos clientes recebem cupom no orçamento escolhido — "
                    "um cupom por cliente."
                ),
            },
            {
                "term": "Conversões esperadas",
                "definition": (
                    "Quantas compras o modelo prevê no total, somando a chance de "
                    "conversão de cada cliente escolhido. Nem toda compra prevista "
                    "acontece por causa do cupom."
                ),
            },
            {
                "term": "Receita bruta esperada",
                "definition": (
                    "Valor total de compra previsto antes do desconto, calculado "
                    "a partir das conversões esperadas e do ticket médio histórico."
                ),
            },
            {
                "term": "Conversões incrementais",
                "definition": (
                    "Compras que só acontecem por causa do cupom — o efeito causal "
                    "da oferta, não quem já compraria de qualquer jeito."
                ),
            },
            {
                "term": "Lucro projetado",
                "definition": (
                    "Ganho líquido incremental estimado: conversões incrementais "
                    "multiplicadas pelo lucro médio por conversão no histórico."
                ),
            },
            {
                "term": "Desconto esperado",
                "definition": (
                    "Custo previsto em cupons: para cada cliente, chance de conversão "
                    "vezes o valor do desconto da oferta escolhida."
                ),
            },
            {
                "term": "Quadrantes",
                "definition": (
                    "Tipos de cliente segundo a resposta ao cupom: persuadables "
                    "(convencem a comprar), sure things (já compram), lost causes "
                    "(dificilmente convertem) e sleeping dogs (o cupom atrapalha)."
                ),
            },
            {
                "term": "Orçamento",
                "definition": "Quantos cupons você pretende enviar nesta campanha.",
            },
            {
                "term": "Exploração",
                "definition": (
                    "Mistura um pouco de acaso na ordem de prioridade — disponível "
                    "apenas em Priorizar uplift, para não ficar preso nos mesmos clientes."
                ),
            },
            {
                "term": "Projeção vs. comparativo",
                "definition": (
                    "A projeção estima o retorno para frente da campanha escolhida; "
                    "o comparativo roda as três estratégias com os mesmos filtros "
                    "para ver qual rende mais naquela configuração."
                ),
            },
        ],
    }


def build_artifacts(spark, cfg: PipelineConfig) -> dict[str, object]:
    """Monta os artefatos do simulador (dicts prontos para serializar em JSON)."""
    decision_time = _decision_time(spark, cfg)
    offers = _active_offers(spark, cfg)
    model = BlendedUpliftModel.load(cfg)

    scoring_sdf = serve.build_scoring_frame(spark, cfg, decision_time)
    serving = scoring_sdf.toPandas()
    scored = _attach_modeled_scores(serving, model, cfg)

    matrix = _columnar(scored, _MATRIX_COLUMNS)
    holdout, holdout_df = _build_holdout(spark, cfg, model)

    lucro_medio = _global_profit_per_conversion(holdout_df)
    receita_media = _global_revenue_per_conversion(holdout_df)

    n_clients = scored["account_id"].nunique()
    labels = _metadata_labels()

    metadata = {
        "decision_time": decision_time,
        "n_clients": int(n_clients),
        "n_rows": int(len(scored)),
        "n_holdout_rows": int(len(holdout_df)),
        "seed": cfg.seed,
        "default_budget": cfg.simulator_default_budget,
        "temperature_default": cfg.simulator_temperature_default,
        "temperature_max": cfg.simulator_temperature_max,
        "modeled_offer_types": list(split.MODELED_OFFER_TYPES),
        "lucro_medio_por_conversao_tratada": lucro_medio,
        "receita_media_por_conversao_tratada": receita_media,
        "analytics_budgets": list(cfg.gain_curve_budgets),
        "quadrant_order": list(QUADRANT_ORDER),
        **labels,
    }
    return {"matrix": matrix, "holdout": holdout, "offers": offers, "metadata": metadata}


def run(cfg: PipelineConfig) -> Path:
    """Executa o export ponta a ponta e escreve os JSONs em `cfg.simulator_output_dir`."""
    spark = build_spark(cfg, app_name="ifood-uplift-simulator-export")
    try:
        artifacts = build_artifacts(spark, cfg)
    finally:
        spark.stop()

    out_dir = cfg.simulator_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        logger.info("Escrito %s (%d bytes).", path, path.stat().st_size)

    logger.info(
        "Export do simulador concluído em %s (%d clientes, %d linhas de serving, %d holdout).",
        out_dir,
        artifacts["metadata"]["n_clients"],
        artifacts["metadata"]["n_rows"],
        artifacts["metadata"]["n_holdout_rows"],
    )
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export dos artefatos estáticos do simulador de cupons.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml (default: config.yaml).")
    args = parser.parse_args()
    run(load(config_path=args.config))


if __name__ == "__main__":
    main()
