"""Contrato do dataset processado e escrita (REQ-107, T-108).

Cobre a encarnação executável do contrato: o `StructType` e o Pydantic concordam
(defeito de contrato se divergirem), o dataset montado projeta exatamente o
contrato, `treatment`/`campaign_wave` derivam da config, G8 é imposto, e o
parquet escrito pela orquestração readmite o contrato ao ser relido.
"""

import json

import pytest

from src import contract
from src.clean import normalize_profile
from src.config import load
from src.io import parse_events, read_offers, read_profile
from src.pipeline import assemble_processed, run


def _setup(spark, tmp_path, events, offers, profiles, **overrides):
    (tmp_path / "transactions.json").write_text(json.dumps(events))
    (tmp_path / "offers.json").write_text(json.dumps(offers))
    (tmp_path / "profile.json").write_text(json.dumps(profiles))
    cfg = load(raw_dir=tmp_path, **overrides)
    events_df = parse_events(spark, cfg)
    offers_df = read_offers(spark, cfg)
    profile_df = normalize_profile(read_profile(spark, cfg), cfg)
    return cfg, events_df, offers_df, profile_df


def _offer(offer_id, duration=7.0, offer_type="bogo", discount_value=10, min_value=10):
    return {
        "channels": ["web", "email"],
        "min_value": min_value,
        "duration": duration,
        "id": offer_id,
        "offer_type": offer_type,
        "discount_value": discount_value,
    }


def _received(account_id, offer_id, t):
    return {"event": "offer received", "account_id": account_id,
            "value": {"amount": None, "offer id": offer_id, "offer_id": None, "reward": None},
            "time_since_test_start": t}


def _viewed(account_id, offer_id, t):
    return {"event": "offer viewed", "account_id": account_id,
            "value": {"amount": None, "offer id": offer_id, "offer_id": None, "reward": None},
            "time_since_test_start": t}


def _txn(account_id, amount, t):
    return {"event": "transaction", "account_id": account_id,
            "value": {"amount": amount, "offer id": None, "offer_id": None, "reward": None},
            "time_since_test_start": t}


def _profile(account_id, age=40, gender="M", ccl=1000, registered_on="20180101"):
    return {"age": age, "registered_on": registered_on, "gender": gender,
            "id": account_id, "credit_card_limit": ccl}


# Um cenário misto reutilizável: um cliente que vê e converte, e um sem histórico.
def _mixed_scenario(spark, tmp_path, **overrides):
    events = [
        _received("acc1", "off1", 5.0),
        _viewed("acc1", "off1", 6.0),
        _txn("acc1", 20.0, 2.0),   # pré-recebimento → alimenta hist_*
        _txn("acc1", 30.0, 7.0),   # pós-view, na janela → conversão
        _received("acc2", "off2", 0.0),  # sem histórico, sem view
    ]
    offers = [_offer("off1"), _offer("off2", offer_type="informational", discount_value=0, min_value=0)]
    profiles = [_profile("acc1"), _profile("acc2", age=118, gender=None, ccl=None)]
    return _setup(spark, tmp_path, events, offers, profiles, **overrides)


def test_g8_rejects_a_null_injected_into_a_non_nullable_column(spark, tmp_path):
    # As duas formas do contrato (StructType e Pydantic) saem da mesma lista;
    # divergir seria defeito.
    assert [f.name for f in contract.PROCESSED_SCHEMA.fields] == contract.CONTRACT_COLUMNS
    assert list(contract.ProcessedRow.model_fields) == contract.CONTRACT_COLUMNS

    cfg, events, offers, profile = _mixed_scenario(spark, tmp_path)
    processed = assemble_processed(events, offers, profile, cfg)

    # Nomes, ordem e tipos idênticos ao contrato — não levanta.
    contract.assert_schema(processed)
    assert processed.columns == contract.CONTRACT_COLUMNS
    contract.assert_no_unexpected_nulls(processed)  # o dataset limpo passa

    from pyspark.sql import functions as F
    corrupted = processed.withColumn("gender", F.lit(None).cast("string"))
    with pytest.raises(ValueError, match="G8"):
        contract.assert_no_unexpected_nulls(corrupted)


def test_run_writes_parquet_that_reconforms_to_the_contract(spark, tmp_path):
    # T-108 accept: o dataset escrito em data/processed/ relido bate com o contrato.
    processed_dir = tmp_path / "processed"
    cfg, _, _, _ = _mixed_scenario(spark, tmp_path, processed_dir=processed_dir)
    run(cfg, spark)

    reloaded = spark.read.parquet(str(processed_dir))
    assert reloaded.columns == contract.CONTRACT_COLUMNS
    contract.assert_schema(reloaded)
    contract.assert_no_unexpected_nulls(reloaded)
