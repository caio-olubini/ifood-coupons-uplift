"""T-split-temporal (T-202): split por onda nunca inverte a ordem temporal."""

import pytest

from src.config import load
from src.split import assert_temporal_order, temporal_split


def _rows(spark, rows):
    return spark.createDataFrame(rows, schema="account_id string, received_time double, campaign_wave int")


def test_split_respects_wave_cutoff(spark):
    cfg = load(validation_wave_cutoff=2, n_campaign_waves=4)
    df = _rows(spark, [
        ("acc1", 0.0, 0), ("acc2", 7.0, 1), ("acc1", 14.0, 2), ("acc3", 21.0, 3),
    ])
    train, holdout = temporal_split(df, cfg)

    assert sorted(r["campaign_wave"] for r in train.collect()) == [0, 1]
    assert sorted(r["campaign_wave"] for r in holdout.collect()) == [2, 3]


def test_same_client_across_waves_is_allowed(spark):
    # acc1 recebe em wave 0 (treino) e wave 3 (holdout) — não é vazamento,
    # é o mesmo cliente exposto a campanhas em momentos diferentes.
    cfg = load(validation_wave_cutoff=2, n_campaign_waves=4)
    df = _rows(spark, [("acc1", 0.0, 0), ("acc1", 21.0, 3)])
    train, holdout = temporal_split(df, cfg)

    assert_temporal_order(train, holdout)  # não levanta


def test_inverted_order_is_rejected(spark):
    # Onda "2" com received_time menor que a onda "1" seria um split quebrado
    # (dado corrompido ou config incoerente) — assert_temporal_order deve pegar.
    cfg = load(validation_wave_cutoff=1, n_campaign_waves=2)
    df = _rows(spark, [("acc1", 10.0, 0), ("acc2", 5.0, 1)])
    train, holdout = temporal_split(df, cfg)

    with pytest.raises(ValueError):
        assert_temporal_order(train, holdout)


def test_empty_side_is_rejected(spark):
    cfg = load(validation_wave_cutoff=1, n_campaign_waves=2)
    df = _rows(spark, [("acc1", 0.0, 0)])
    train, holdout = temporal_split(df, cfg)

    with pytest.raises(ValueError):
        assert_temporal_order(train, holdout)
