import json

from src.attribution import add_recurrence_flag, attribute, build_label
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


def test_recurrence_flag_marks_second_conversion_inside_window(spark, tmp_path):
    # acc1 converte em off1 (txn t=3) e de novo em off2 (txn t=8): 8 - 3 = 5 dias,
    # dentro da janela padrão de 7. off1 fica is_recurrent=1 (tem outra conversão
    # depois, dentro da janela) — a recorrência é medida no nível de campanha,
    # olhando qualquer oferta do mesmo cliente, não só a própria oferta. off2 não
    # tem conversão POSTERIOR dentro da janela (a janela conta para a frente a
    # partir de cada conversão), então fica is_recurrent=0 — assim como acc2, que
    # converte uma única vez e não tem segunda compra para ancorar a janela.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 6.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 8.0},
        {"event": "offer received", "account_id": "acc2",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc2",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    offers = [_offer("off1", duration=4.0), _offer("off2", duration=4.0)]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    labeled = build_label(attribute(parsed, offers_df, cfg), cfg)
    rows = {(r["account_id"], r["offer_id"]): r for r in add_recurrence_flag(labeled, cfg).collect()}

    assert rows[("acc1", "off1")]["is_recurrent"] == 1
    assert rows[("acc1", "off2")]["is_recurrent"] == 0
    assert rows[("acc2", "off1")]["is_recurrent"] == 0


def test_recurrence_flag_respects_configurable_window(spark, tmp_path):
    # Mesmo cenário de duas conversões separadas por 5 dias, mas com N=3: a
    # segunda compra cai fora da janela e nenhum dos dois recebimentos é
    # recorrente. Prova que a janela é o parâmetro configurável, não um
    # valor mágico embutido em attribution.py.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off2", "offer_id": None, "reward": None},
         "time_since_test_start": 6.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 20.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 8.0},
    ]
    offers = [_offer("off1", duration=4.0), _offer("off2", duration=4.0)]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, offers)
    cfg_narrow = load(raw_dir=cfg.raw_dir, recurrence_window_days=3)
    labeled = build_label(attribute(parsed, offers_df, cfg_narrow), cfg_narrow)
    rows = {r["offer_id"]: r for r in add_recurrence_flag(labeled, cfg_narrow).collect()}

    assert rows["off1"]["is_recurrent"] == 0
    assert rows["off2"]["is_recurrent"] == 0


