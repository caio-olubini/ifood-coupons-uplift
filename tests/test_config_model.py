"""Config de modelagem (spec 02, REQ-210, T-201).

Custo/hiperparâmetro/corte inválido deve falhar na carga, antes de qualquer
treino — mesma disciplina de fronteira da config do pipeline (spec 01).
"""

import pytest
from pydantic import ValidationError

from src.config import load


def test_default_model_config_loads():
    cfg = load()
    assert cfg.validation_wave_cutoff == 4
    assert cfg.lgbm_n_estimators == 200
    assert cfg.xlearner_n_estimators == 200


def test_wave_cutoff_at_or_above_n_waves_fails_at_load():
    with pytest.raises(ValidationError):
        load(validation_wave_cutoff=6, n_campaign_waves=6)


def test_zero_wave_cutoff_fails_at_load():
    with pytest.raises(ValidationError):
        load(validation_wave_cutoff=0)


def test_non_positive_lgbm_n_estimators_fails_at_load():
    with pytest.raises(ValidationError):
        load(lgbm_n_estimators=0)


def test_non_positive_lgbm_learning_rate_fails_at_load():
    with pytest.raises(ValidationError):
        load(lgbm_learning_rate=0)


def test_non_positive_xlearner_n_estimators_fails_at_load():
    with pytest.raises(ValidationError):
        load(xlearner_n_estimators=0)


def test_ipw_min_propensity_out_of_range_fails_at_load():
    with pytest.raises(ValidationError):
        load(ipw_min_propensity=0)
    with pytest.raises(ValidationError):
        load(ipw_min_propensity=1)


def test_ab_test_power_and_alpha_out_of_range_fail_at_load():
    with pytest.raises(ValidationError):
        load(ab_test_power=1.5)
    with pytest.raises(ValidationError):
        load(ab_test_alpha=0)
