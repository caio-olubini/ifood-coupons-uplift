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


def _offer(offer_id, duration=7.0, offer_type="bogo", min_value=10):
    return {
        "channels": ["email"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": 10,
    }


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


def test_min_value_is_per_transaction_not_accumulated(spark, tmp_path):
    # G10: compra pós-view abaixo do mínimo não converte — a fronteira é fechada
    # (`txn_amount >= min_value`). E o limiar é POR transação, não sobre o gasto
    # acumulado: acc2 soma duas compras de R$ 6 (12 > min_value 10), mas nenhuma
    # delas sozinha ativou o desconto — a leitura acumulada foi rejeitada de propósito.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 4.65, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
        {"event": "offer received", "account_id": "acc2",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc2",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc2",
         "value": {"amount": 6.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc2",
         "value": {"amount": 6.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", min_value=10)])
    attributed = attribute(parsed, offers_df, cfg)
    rows = {r["account_id"]: r for r in build_label(attributed, cfg).collect()}

    assert rows["acc1"]["assigned_txn_count"] == 0   # compra única abaixo do mínimo
    assert rows["acc1"]["converted"] == 0
    assert rows["acc2"]["assigned_txn_count"] == 0   # soma > mínimo, mas nenhuma isolada
    assert rows["acc2"]["converted"] == 0
