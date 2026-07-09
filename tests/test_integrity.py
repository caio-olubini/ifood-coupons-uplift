"""Testes de integridade end-to-end das garantias do contrato (G1, G7, G8).

Ao contrário dos testes por módulo, estes exercem invariantes sobre o dataset
que atravessa o pipeline inteiro — o lugar onde uma mina reaparece em silêncio.
"""

import json

from src import contract
from src.attribution import attribute, build_label
from src.clean import normalize_profile
from src.config import load
from src.cost import add_reward_cost
from src.features import build
from src.io import parse_events


def _setup(spark, tmp_path, events, offers, profiles):
    (tmp_path / "transactions.json").write_text(json.dumps(events))
    (tmp_path / "offers.json").write_text(json.dumps(offers))
    (tmp_path / "profile.json").write_text(json.dumps(profiles))
    cfg = load(raw_dir=tmp_path)
    parsed = parse_events(spark, cfg)
    offers_df = spark.read.option("multiLine", True).json(str(cfg.offers_path))
    profile_df = spark.read.option("multiLine", True).json(str(cfg.profile_path))
    return cfg, parsed, offers_df, profile_df


def _offer(offer_id, duration=7.0, offer_type="bogo", channels=None, discount_value=10, min_value=10):
    return {
        "channels": channels or ["web", "email"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": discount_value,
    }


def _received(account_id, offer_id, t):
    return {
        "event": "offer received", "account_id": account_id,
        "value": {"amount": None, "offer id": offer_id, "offer_id": None, "reward": None},
        "time_since_test_start": t,
    }


def _full_pipeline(spark, cfg, parsed, offers_df):
    attributed = attribute(parsed, offers_df, cfg)
    labeled = build_label(attributed, cfg)
    featured = build(parsed, labeled, offers_df, cfg)
    return add_reward_cost(featured, offers_df, cfg)


# --- G1: grão único -----------------------------------------------------------

def test_g1_unique_grain_no_duplicates(spark, tmp_path):
    # Dois recebimentos legítimos da mesma oferta (ondas distintas) NÃO colapsam:
    # geram duas linhas, ambas presentes, sem duplicar a chave.
    events = [
        _received("acc1", "off1", 0.0),
        _received("acc1", "off1", 10.0),
        _received("acc1", "off2", 3.0),
        _received("acc2", "off1", 0.0),
    ]
    offers = [_offer("off1"), _offer("off2")]
    profiles = [
        {"age": 40, "registered_on": "20180101", "gender": "M", "id": "acc1", "credit_card_limit": 1000},
        {"age": 30, "registered_on": "20180101", "gender": "F", "id": "acc2", "credit_card_limit": 2000},
    ]
    cfg, parsed, offers_df, _ = _setup(spark, tmp_path, events, offers, profiles)
    result = _full_pipeline(spark, cfg, parsed, offers_df)

    total = result.count()
    distinct = result.select("account_id", "offer_id", "received_time").distinct().count()
    assert total == 4
    assert distinct == total  # zero duplicatas na chave do grão


def test_g1_same_offer_two_waves_both_survive(spark, tmp_path):
    events = [_received("acc1", "off1", 0.0), _received("acc1", "off1", 15.0)]
    cfg, parsed, offers_df, _ = _setup(
        spark, tmp_path, events, [_offer("off1")],
        [{"age": 40, "registered_on": "20180101", "gender": "M", "id": "acc1", "credit_card_limit": 1000}],
    )
    received_times = sorted(r["received_time"] for r in _full_pipeline(spark, cfg, parsed, offers_df).collect())
    assert received_times == [0.0, 15.0]


# --- G7: sentinela de identidade (iff com os três campos ausentes) ------------

def test_g7_identity_missing_iff_three_fields_absent(spark, tmp_path):
    profiles = [
        # Sentinela canônica: age=118, gender e ccl nulos.
        {"age": 118, "registered_on": "20180101", "gender": None, "id": "sentinel", "credit_card_limit": None},
        # Cliente normal completo.
        {"age": 55, "registered_on": "20180101", "gender": "F", "id": "normal", "credit_card_limit": 5000},
    ]
    (tmp_path / "profile.json").write_text(json.dumps(profiles))
    cfg = load(raw_dir=tmp_path)
    df = spark.read.option("multiLine", True).json(str(cfg.profile_path))
    rows = {r["account_id"]: r for r in normalize_profile(df, cfg).collect()}

    # (⇒) identity_missing=1 implica os três campos ausentes.
    s = rows["sentinel"]
    assert s["identity_missing"] == 1
    assert s["age"] is None
    assert s["gender"] == "unknown"        # gender ausente normalizado
    assert s["credit_card_limit"] is None

    # (⇐) cliente com os três campos presentes NÃO recebe a flag.
    n = rows["normal"]
    assert n["identity_missing"] == 0
    assert n["age"] == 55
    assert n["credit_card_limit"] is not None

    # age nunca vale o sentinela após normalização (G7).
    ages = [r["age"] for r in rows.values() if r["age"] is not None]
    assert cfg.age_sentinel not in ages


# --- G8: nulos apenas onde o contrato permite ---------------------------------

# Este teste corre sobre a saída *intermediária* do pipeline (antes de
# `contract.enforce_schema`), por isso ainda enxerga colunas que o contrato
# descarta. As nullable do contrato vêm de `src/contract.py` — fonte única, para
# que este teste não possa discordar dele. G8 sobre o dataset final (já no
# contrato) é coberto em `tests/test_contract.py`.
_INTERMEDIATE_ONLY = {
    "view_time",                # intermediária: oferta recebida sem view
    "valid_until",              # intermediária de atribuição
    "first_assigned_txn_time",  # intermediária: sem transação atribuída
}
_ALLOWED_NULL = set(contract.NULLABLE_COLUMNS) | _INTERMEDIATE_ONLY


def test_g8_no_nulls_in_non_nullable_columns(spark, tmp_path):
    # Um dataset com histórico misto: cliente com e sem eventos, converte e não.
    events = [
        _received("acc1", "off1", 5.0),
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 6.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},  # pré-recebimento → alimenta hist_*
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 30.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 7.0},  # na janela → conversão
        _received("acc2", "off2", 0.0),  # cliente sem histórico algum
    ]
    offers = [_offer("off1"), _offer("off2", offer_type="informational", discount_value=0, min_value=0)]
    profiles = [
        {"age": 40, "registered_on": "20180101", "gender": "M", "id": "acc1", "credit_card_limit": 1000},
        {"age": 118, "registered_on": "20180101", "gender": None, "id": "acc2", "credit_card_limit": None},
    ]
    cfg, parsed, offers_df, profile_df = _setup(spark, tmp_path, events, offers, profiles)
    profile = normalize_profile(profile_df, cfg)
    featured = _full_pipeline(spark, cfg, parsed, offers_df)
    final = featured.join(profile, on="account_id", how="left")

    for row in final.collect():
        for col, value in row.asDict().items():
            if col not in _ALLOWED_NULL:
                assert value is not None, f"nulo inesperado em coluna não-nullable '{col}'"
