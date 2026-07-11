import json

from src.attribution import attribute, build_label
from src.config import load
from src.cost import add_reward_cost
from src.io import parse_events


def _setup(spark, tmp_path, events, offers):
    (tmp_path / "transactions.json").write_text(json.dumps(events))
    (tmp_path / "offers.json").write_text(json.dumps(offers))
    cfg = load(raw_dir=tmp_path)
    parsed = parse_events(spark, cfg)
    offers_df = spark.read.option("multiLine", True).json(str(cfg.offers_path))
    return cfg, parsed, offers_df


def _offer(offer_id, duration=7.0, offer_type="bogo", discount_value=10, min_value=10):
    return {
        "channels": ["web"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": discount_value,
    }


def _labeled_with_cost(spark, cfg, parsed, offers_df):
    attributed = attribute(parsed, offers_df, cfg)
    labeled = build_label(attributed, cfg)
    return add_reward_cost(labeled, offers_df, cfg)


def test_unviewed_conversion_still_costs(spark, tmp_path):
    # O desconto é concedido a quem atinge o mínimo na validade, tenha visto a
    # oferta ou não — o custo segue a conversão, não a exposição.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 30.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", discount_value=10)])
    row = _labeled_with_cost(spark, cfg, parsed, offers_df).collect()[0]

    assert row["view_time"] is None
    assert row["converted"] == 1
    assert row["reward_cost"] == 10.0


def test_g6_invariant_holds_across_rows(spark, tmp_path):
    # G6: reward_cost > 0 ⇒ converted=1 e offer_type != informational.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 30.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "offer received", "account_id": "acc2",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc2",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc2",
         "value": {"amount": 30.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    offers = [
        _offer("off1", discount_value=10),
        _offer("off2", duration=4.0, offer_type="informational", discount_value=0, min_value=0),
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    rows = {r["account_id"]: r for r in _labeled_with_cost(spark, cfg, parsed, offers_df).collect()}

    assert rows["acc1"]["reward_cost"] == 10.0  # bogo convertido paga
    assert rows["acc2"]["reward_cost"] == 0.0   # informational convertido não paga (G6)
    for r in rows.values():
        if r["reward_cost"] > 0:
            assert r["converted"] == 1
            assert r["offer_type"] != "informational"
