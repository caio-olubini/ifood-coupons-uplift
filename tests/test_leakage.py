import json

from src.attribution import attribute, build_label
from src.config import load
from src.features import build
from src.io import parse_events


def _setup(spark, tmp_path, events, offers):
    (tmp_path / "transactions.json").write_text(json.dumps(events))
    (tmp_path / "offers.json").write_text(json.dumps(offers))
    cfg = load(raw_dir=tmp_path)
    parsed = parse_events(spark, cfg)
    offers_df = spark.read.option("multiLine", True).json(str(cfg.offers_path))
    return cfg, parsed, offers_df


def _offer(offer_id, duration=7.0, offer_type="bogo", channels=None, discount_value=10, min_value=10):
    return {
        "channels": channels or ["web", "email"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": discount_value,
    }


def _feature_row(spark, cfg, parsed, offers_df):
    attributed = attribute(parsed, offers_df, cfg)
    labeled = build_label(attributed, cfg)
    return build(parsed, labeled, offers_df, cfg).collect()


def test_post_receipt_transaction_does_not_leak_into_hist_features(spark, tmp_path):
    # G2: a transação em time=5 é DEPOIS do received_time=2 e não pode entrar
    # em nenhuma feature hist_*. Só a transação em time=1 (pré-recebimento) vale.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 10.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 999.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 5.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1")])
    row = _feature_row(spark, cfg, parsed, offers_df)[0]

    # Apenas a transação pré-recebimento (10.0) conta; a de 999.0 é ignorada.
    assert row["hist_spend_total"] == 10.0
    assert row["hist_txn_count"] == 1
    assert row["hist_avg_ticket"] == 10.0


def test_no_history_yields_zeroed_counts(spark, tmp_path):
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1")])
    row = _feature_row(spark, cfg, parsed, offers_df)[0]

    assert row["hist_spend_total"] == 0.0
    assert row["hist_txn_count"] == 0
    assert row["hist_offers_received"] == 0
    assert row["hist_completed_unseen_flag"] == 0


