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


def test_unviewed_offer_with_in_window_transaction_converts(spark, tmp_path):
    # O view é o TRATAMENTO, não condição do rótulo. Um recebimento não visto com
    # compra dentro da validade converte — é exatamente essa massa que dá μ₀ > 0
    # no grupo de controle e torna o uplift estimável.
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

    assert row["view_time"] is None   # controle: não viu
    assert row["converted"] == 1      # ... mas comprou na janela
    assert row["conversion_value"] == 20.0


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
    # `min_value=0` (como no catálogo real): informational não tem gatilho de valor,
    # então uma compra de R$ 3 converte — o filtro de gasto mínimo (G10) não a alcança.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 3.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
    ]
    cfg, parsed, offers_df = _setup(
        spark, tmp_path, events,
        [_offer("off1", duration=4.0, offer_type="informational", min_value=0)],
    )
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["converted"] == 1
    assert row["conversion_value"] == 3.0


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


def test_transaction_before_view_still_converts(spark, tmp_path):
    # A ordem view↔compra não altera o rótulo: a janela de atribuição é a
    # validade da oferta. Quem comprou antes de ver converteu; o que a ordem
    # afeta é a leitura causal, e essa é responsabilidade do modelo de uplift
    # (via `treatment`), não do label.
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

    assert row["view_time"] == 4.0         # tratado (viu, dentro da validade)
    assert row["assigned_txn_count"] == 1  # a compra na janela é atribuída
    assert row["converted"] == 1


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


def test_transaction_below_min_value_does_not_convert(spark, tmp_path):
    # G10: compra pós-view, na validade, mas abaixo do gasto mínimo da oferta.
    # O desconto nunca teria sido concedido — não é conversão, e não pode custar.
    # A fronteira é fechada (`txn_amount >= min_value`): a segunda compra, de
    # exatamente R$10, já dispara a recompensa e passa a converter sozinha.
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
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", min_value=10)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["view_time"] == 1.0          # foi vista
    assert row["assigned_txn_count"] == 0   # mas a compra não atingiu o mínimo
    assert row["converted"] == 0
    assert row["conversion_value"] == 0.0


def test_small_transactions_do_not_sum_past_min_value(spark, tmp_path):
    # O limiar é por transação, não sobre o gasto acumulado na janela: duas compras
    # de R$ 6 (soma 12 > min_value 10) não convertem, porque nenhuma delas sozinha
    # ativou o desconto. Fixa a regra escolhida contra a leitura acumulada.
    events = [
        {"event": "offer received", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 0.0},
        {"event": "offer viewed", "account_id": "acc1",
         "value": {"amount": None, "offer id": "off1", "offer_id": None, "reward": None},
         "time_since_test_start": 1.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 6.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 2.0},
        {"event": "transaction", "account_id": "acc1",
         "value": {"amount": 6.0, "offer id": None, "offer_id": None, "reward": None},
         "time_since_test_start": 3.0},
    ]
    cfg, parsed, offers_df = _setup(spark, tmp_path, events, [_offer("off1", min_value=10)])
    attributed = attribute(parsed, offers_df, cfg)
    row = build_label(attributed, cfg).collect()[0]

    assert row["assigned_txn_count"] == 0
    assert row["converted"] == 0


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
