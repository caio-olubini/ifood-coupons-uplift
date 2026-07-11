import json

from src.attribution import attribute
from src.config import load
from src.io import parse_events


def _setup(spark, tmp_path, events, offers):
    tx_path = tmp_path / "transactions.json"
    tx_path.write_text(json.dumps(events))
    off_path = tmp_path / "offers.json"
    off_path.write_text(json.dumps(offers))
    cfg = load(raw_dir=tmp_path)
    parsed = parse_events(spark, cfg)
    offers_df = spark.read.option("multiLine", True).json(str(cfg.offers_path))
    return cfg, parsed, offers_df


def _offer(offer_id, duration=7.0, offer_type="bogo", min_value=10):
    return {
        "channels": ["email"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": 10,
    }


def test_overlapping_offers_apply_configured_priority(spark, tmp_path):
    # Ambas as ofertas vistas antes da transação, para que a regra estrita
    # (txn após view) não elimine a atribuição; o que se testa é a prioridade
    # em sobreposição, não o influence-aware.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    offers = [_offer("off1", duration=7.0), _offer("off2", duration=7.0)]

    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    cfg_earliest = load(raw_dir=cfg.raw_dir, attribution_priority="earliest_received")
    rows = {r["offer_id"]: r for r in attribute(parsed, offers_df, cfg_earliest).collect()}
    assert rows["off1"]["assigned_txn_count"] == 1
    assert rows["off2"]["assigned_txn_count"] == 0

    cfg_latest = load(raw_dir=cfg.raw_dir, attribution_priority="latest_received")
    rows = {r["offer_id"]: r for r in attribute(parsed, offers_df, cfg_latest).collect()}
    assert rows["off2"]["assigned_txn_count"] == 1
    assert rows["off1"]["assigned_txn_count"] == 0


def test_ineligible_offer_does_not_steal_transaction_from_an_eligible_one(spark, tmp_path):
    # Duas ofertas vistas disputam a mesma compra de R$ 15. A prioritária (off1,
    # recebida antes) exige R$ 50 e é INELEGÍVEL para essa compra; off2 exige R$ 10
    # e converteria com ela. O gasto mínimo é filtrado ANTES da disputa de posse,
    # então off1 não vence o desempate para depois descartar a transação — a
    # conversão fica com off2, que de fato a induziu.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 15.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    offers = [_offer("off1", min_value=50), _offer("off2", min_value=10)]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    rows = {r["offer_id"]: r for r in attribute(parsed, offers_df, cfg).collect()}

    assert rows["off1"]["assigned_txn_count"] == 0  # inelegível: 15 < 50
    assert rows["off2"]["assigned_txn_count"] == 1
    assert rows["off2"]["assigned_txn_amount_sum"] == 15.0


