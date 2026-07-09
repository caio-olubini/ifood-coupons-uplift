import json

from src.config import load
from src.io import parse_events


def _write_transactions(tmp_path, records):
    path = tmp_path / "transactions.json"
    path.write_text(json.dumps(records))
    return path


def test_received_and_viewed_read_offer_id_with_space(spark, tmp_path):
    records = [
        {
            "event": "offer received",
            "account_id": "acc1",
            "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
            "time_since_test_start": 0.0,
        },
        {
            "event": "offer viewed",
            "account_id": "acc1",
            "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
            "time_since_test_start": 1.0,
        },
    ]
    _write_transactions(tmp_path, records)
    cfg = load(raw_dir=tmp_path)
    df = parse_events(spark, cfg).collect()

    for row in df:
        assert row["offer_ref"] == "off1"


def test_completed_reads_offer_id_with_underscore(spark, tmp_path):
    records = [
        {
            "event": "offer completed",
            "account_id": "acc1",
            "value": {"amount": None, "offer id": None, "offer_id": "off1", "reward": 2.0},
            "time_since_test_start": 2.0,
        },
    ]
    _write_transactions(tmp_path, records)
    cfg = load(raw_dir=tmp_path)
    row = parse_events(spark, cfg).collect()[0]

    assert row["offer_ref"] == "off1"
    assert row["reward"] == 2.0


def test_transaction_has_amount_and_no_offer_ref(spark, tmp_path):
    records = [
        {
            "event": "transaction",
            "account_id": "acc1",
            "value": {"amount": 5.5, "offer id": None, "offer_id": None, "reward": None},
            "time_since_test_start": 3.0,
        },
    ]
    _write_transactions(tmp_path, records)
    cfg = load(raw_dir=tmp_path)
    row = parse_events(spark, cfg).collect()[0]

    assert row["amount"] == 5.5
    assert row["offer_ref"] is None
