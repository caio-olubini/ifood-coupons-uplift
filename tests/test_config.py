import pytest
from pydantic import ValidationError

from src.config import load


def test_default_config_loads():
    cfg = load()
    assert cfg.smd_threshold == 0.1
    assert cfg.age_sentinel == 118


def test_negative_smd_threshold_fails_at_load():
    with pytest.raises(ValidationError):
        load(smd_threshold=-0.1)


def test_zero_campaign_wave_days_fails_at_load():
    with pytest.raises(ValidationError):
        load(campaign_wave_days=0)
