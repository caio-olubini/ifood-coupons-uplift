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


def _offer(offer_id, duration=7.0, offer_type="bogo"):
    return {
        "channels": ["email"],
        "min_value": 10,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": 10,
    }


def test_transaction_outside_validity_window_not_assigned(spark, tmp_path):
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 10.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=7.0)])
    row = attribute(parsed, offers_df, cfg).collect()[0]

    assert row["assigned_txn_count"] == 0
    assert row["first_assigned_txn_time"] is None


def test_transaction_inside_validity_window_assigned(spark, tmp_path):
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
    row = attribute(parsed, offers_df, cfg).collect()[0]

    assert row["assigned_txn_count"] == 1
    assert row["first_assigned_txn_time"] == 3.0
    assert row["assigned_txn_amount_sum"] == 20.0


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


def test_single_view_serves_only_one_of_two_overlapping_receipts(spark, tmp_path):
    # A MESMA oferta reenviada em ondas cujas janelas se sobrepõem: recebida em
    # t=14 (válida até 24) e t=17 (válida até 27). Uma única view em t=18 cai nas
    # duas janelas. Como há um só evento de exposição, ela pode pertencer a um só
    # recebimento — senão uma exposição vira treatment=1 em duas linhas.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 14.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 17.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 18.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", duration=10.0)])
    rows = {r["received_time"]: r for r in attribute(parsed, offers_df, cfg).collect()}

    viewed_receipts = [t for t, r in rows.items() if r["view_time"] is not None]
    assert viewed_receipts == [14.0]  # earliest_received (default) fica com a view
    assert rows[17.0]["view_time"] is None


def test_unseen_offer_does_not_steal_transaction_from_a_seen_one(spark, tmp_path):
    # Oferta A recebida e NÃO vista; oferta B recebida e vista. A transação cai
    # na janela das duas, mas só a vista pôde tê-la induzido. A não-vista não
    # pode roubar a conversão da vista.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "offA", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "offB", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "offB", "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    offers = [_offer("offA", duration=7.0), _offer("offB", duration=7.0)]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    rows = {r["offer_id"]: r for r in attribute(parsed, offers_df, cfg).collect()}

    assert rows["offB"]["assigned_txn_count"] == 1
    assert rows["offA"]["assigned_txn_count"] == 0  # invisível não atribui


def test_identical_received_time_resolves_deterministically_by_offer_id(spark, tmp_path):
    # Duas ofertas da mesma onda chegam no mesmo received_time e ambas vistas
    # disputam a mesma transação. Com received_time empatado, o desempate estável
    # por offer_id garante o mesmo dono a cada execução (aqui, "off1").
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
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
    rows = {r["offer_id"]: r for r in attribute(parsed, offers_df, cfg).collect()}

    assert rows["off1"]["assigned_txn_count"] == 1
    assert rows["off2"]["assigned_txn_count"] == 0
