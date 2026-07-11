import pytest
from pydantic import ValidationError

from src.config import load


def test_default_config_loads():
    cfg = load()
    assert cfg.smd_threshold == 0.1
    assert cfg.age_sentinel == 118


@pytest.mark.parametrize("overrides", [{"smd_threshold": -0.1}, {"n_campaign_waves": 0}])
def test_out_of_range_value_fails_at_load(overrides):
    with pytest.raises(ValidationError):
        load(**overrides)
