import json

from src.attribution import attribute, build_label
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


def _offer(offer_id, duration=7.0, offer_type="bogo"):
    return {
        "channels": ["email"],
        "min_value": 10,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": 10,
    }


def test_completed_without_prior_view_is_not_converted(spark, tmp_path):
    # G3: sem view alguma, mesmo com transação na janela, não converte.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1")])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 0


def test_transaction_after_validity_does_not_convert(spark, tmp_path):
    # G4: transação um dia após o fim da validade não converte.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 8.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 0


def test_informational_converts_from_post_view_window_not_completed_event(spark, tmp_path):
    # G5: informational vista + transação na janela pós-view converte, sem depender de "offer completed".
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 15.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    cfg, parsed, offers_df = _setup(
        spark, tmp_path, events, [_offer("off1", duration=4.0, offer_type="informational")]
    )
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 1
    assert row["conversion_value"] == 15.0


def test_view_attributed_to_containing_receipt_window_across_waves(spark, tmp_path):
    # Armadilha: a MESMA oferta recebida em duas ondas. Um view em t=11 deve
    # pertencer à onda cuja janela o contém (t=10, válida até 17), não à onda
    # anterior (t=0, válida até 7). Sem isso, a onda 1 rouba o view e a onda 2
    # perde a conversão que lhe é devida.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 10.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 11.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 12.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    attributed = attribute(parsed, offers_df, cfg)
    rows = {r["received_time"]: r for r in build_label(attributed, cfg).collect()}

    # Onda 1 (t=0, até 7): o view em t=11 está fora da janela → não é atribuído.
    assert rows[0.0]["view_time"] is None
    assert rows[0.0]["converted"] == 0
    # Onda 2 (t=10, até 17): view em t=11 dentro da janela + txn em t=12 → converte.
    assert rows[10.0]["view_time"] == 11.0
    assert rows[10.0]["converted"] == 1


def test_transaction_before_view_does_not_convert(spark, tmp_path):
    # Regra influence-aware estrita: comprou ANTES de ver o anúncio, ainda que
    # ambos na janela de validade. A compra não pode ter sido induzida pela
    # visualização, então não conta como conversão.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},  # compra em t=2
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 4.0},  # ... só vê em t=4, depois da compra
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["view_time"] == 4.0        # o view existe e está na janela
    assert row["assigned_txn_count"] == 0  # mas a compra pré-view não é atribuída
    assert row["converted"] == 0


def test_transaction_after_view_converts(spark, tmp_path):
    # Mesmo cenário, mas a compra vem DEPOIS do view → conta.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 4.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 1
    assert row["conversion_value"] == 20.0


def test_viewed_and_in_window_transaction_converts(spark, tmp_path):
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 1
    assert row["conversion_value"] == 20.0
