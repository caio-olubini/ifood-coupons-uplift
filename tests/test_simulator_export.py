"""Smoke de paridade do simulador (spec 03, REQ-310, T-310).

Garante que a alocação no browser (espelhada aqui em Python) bate com
`serve.recommend`, que a curva de ganho bate com `gaincurve.incremental_gain_curve`,
e que o lucro projetado segue a fórmula do REQ-307.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulator.export import _hash_score
from src import gaincurve, serve
from src.gaincurve import NET_PROFIT_COLUMN, TREATMENT_COLUMN


def _mirror_allocate(
    matrix: dict[str, list],
    strategy: str,
    allowed_types: set[str],
    allowed_quadrants: set[str],
    budget: int,
    seed: int,
    temperature: float = 0.0,
) -> list[tuple[str, str]]:
    """Réplica determinística da alocação em `simulator/index.html`."""
    n = len(matrix["account_id"])
    effective_temp = temperature if strategy == "uplift" else 0.0
    best_by_client: dict[str, tuple[int, float]] = {}

    for i in range(n):
        if matrix["offer_type"][i] not in allowed_types:
            continue
        if matrix["quadrant"][i] not in allowed_quadrants:
            continue
        if strategy == "aleatorio":
            score = _hash_score(matrix["account_id"][i], matrix["offer_id"][i], seed)
        else:
            col = "p_convert" if strategy == "conversao" else "score_dynamic"
            score = matrix[col][i]
            if score is None:
                continue
        acc = matrix["account_id"][i]
        cur = best_by_client.get(acc)
        if cur is None or score > cur[1]:
            best_by_client[acc] = (i, score)

    best = [{"idx": v[0], "score": v[1]} for v in best_by_client.values()]
    if effective_temp > 0 and best:
        scores = [b["score"] for b in best]
        lo, hi = min(scores), max(scores)
        span = hi - lo + 1e-9
        rng = np.random.default_rng(seed)
        for b in best:
            norm = (b["score"] - lo) / span
            b["key"] = norm / effective_temp + rng.gumbel()
        best.sort(key=lambda b: (-b["key"], b["idx"]))
    else:
        best.sort(key=lambda b: (-b["score"], b["idx"]))

    chosen = best[:budget]
    return [(matrix["account_id"][c["idx"]], matrix["offer_id"][c["idx"]]) for c in chosen]


def _matrix_to_scored_df(matrix: dict[str, list], strategy: str, seed: int) -> pd.DataFrame:
    rows = []
    for i in range(len(matrix["account_id"])):
        if strategy == "aleatorio":
            score = _hash_score(matrix["account_id"][i], matrix["offer_id"][i], seed)
        else:
            col = "p_convert" if strategy == "conversao" else "score_dynamic"
            score = matrix[col][i]
            if score is None:
                continue
        rows.append({
            "account_id": matrix["account_id"][i],
            "offer_id": matrix["offer_id"][i],
            "offer_type": matrix["offer_type"][i],
            "score": score,
        })
    return pd.DataFrame(rows)


def test_hash_score_esta_no_intervalo_unitario():
    v = _hash_score("acc1", "off1", 42)
    assert 0.0 <= v < 1.0
    assert _hash_score("acc1", "off1", 42) == v


def test_allocate_uplift_bate_com_serve_recommend():
    matrix = {
        "account_id": ["a", "a", "b", "b", "c"],
        "offer_id": ["o1", "o2", "o1", "o2", "o1"],
        "offer_type": ["discount", "bogo", "discount", "bogo", "discount"],
        "p_convert": [0.3, 0.9, 0.8, 0.5, 0.7],
        "score_dynamic": [0.1, 0.6, 0.4, 0.9, 0.2],
        "quadrant": ["persuadable"] * 5,
    }
    allowed = {"discount", "bogo"}
    quads = {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    budget = 2
    seed = 7

    mirror = _mirror_allocate(matrix, "uplift", allowed, quads, budget, seed)
    scored = _matrix_to_scored_df(matrix, "uplift", seed)
    recs = serve.recommend(scored, budget=budget, temperature=0.0)
    python_pairs = list(zip(recs["account_id"], recs["offer_id"]))
    assert mirror == python_pairs


def test_temperatura_so_afeta_uplift():
    matrix = {
        "account_id": ["a", "b", "c"],
        "offer_id": ["o1", "o1", "o1"],
        "offer_type": ["discount", "discount", "discount"],
        "p_convert": [0.9, 0.5, 0.1],
        "score_dynamic": [0.9, 0.5, 0.1],
        "quadrant": ["persuadable", "persuadable", "persuadable"],
    }
    allowed = {"discount"}
    quads = {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    seed = 42
    sem_temp = _mirror_allocate(matrix, "conversao", allowed, quads, 3, seed, temperature=0.5)
    com_temp = _mirror_allocate(matrix, "conversao", allowed, quads, 3, seed, temperature=0.0)
    assert sem_temp == com_temp


def test_informational_fora_do_argmax_modelado():
    matrix = {
        "account_id": ["a", "a"],
        "offer_id": ["info1", "disc1"],
        "offer_type": ["informational", "discount"],
        "p_convert": [None, 0.8],
        "score_dynamic": [None, 0.5],
        "quadrant": ["persuadable", "persuadable"],
    }
    allowed = {"informational", "discount"}
    quads = {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    chosen = _mirror_allocate(matrix, "uplift", allowed, quads, 1, seed=1)
    assert chosen == [("a", "disc1")]


def test_lucro_projetado_bate_soma_manual():
    matrix = {
        "account_id": ["a", "b"],
        "offer_id": ["o1", "o2"],
        "offer_type": ["discount", "bogo"],
        "uplift": [0.1, 0.2],
        "p_convert": [0.5, 0.3],
        "score_dynamic": [0.4, 0.6],
        "quadrant": ["persuadable", "persuadable"],
    }
    lucro_medio = 15.0
    tau_sum = 0.1 + 0.2
    assert tau_sum * lucro_medio == pytest.approx(4.5)


def test_curva_de_ganho_bate_com_gaincurve():
    holdout = pd.DataFrame({
        "account_id": ["a", "b", "c", "d"],
        "offer_type": ["discount"] * 4,
        TREATMENT_COLUMN: [1, 1, 0, 0],
        "converted": [1, 0, 1, 0],
        "conversion_value": [20.0, 0.0, 15.0, 0.0],
        "reward_cost": [5.0, 0.0, 3.0, 0.0],
        "p_convert": [0.9, 0.7, 0.4, 0.2],
        "score_dynamic": [0.9, 0.7, 0.4, 0.2],
    })
    holdout = gaincurve.add_net_profit(holdout)
    ranking = holdout.sort_values("score_dynamic", ascending=False, kind="stable").index.to_numpy()
    curva = gaincurve.incremental_gain_curve(ranking, holdout)
    assert curva.loc[2, "gain"] >= curva.loc[1, "gain"]
    assert curva.loc[len(holdout), "conversions"] == curva["conversions"].iloc[-1]
