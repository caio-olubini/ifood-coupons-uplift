"""Config do simulador (spec 03, REQ-309, T-301).

Mesma disciplina de fronteira das outras configs: um parâmetro inválido do
simulador falha na carga, antes de qualquer export. Os defaults carregam sem
erro e são o que o `simulator.export`/JS assumem quando o YAML não sobrescreve.
"""

import pytest
from pydantic import ValidationError

from src.config import load


def test_simulator_defaults_load():
    cfg = load()
    assert cfg.simulator_output_dir.as_posix().endswith("simulator/data")
    assert cfg.simulator_default_budget > 0
    assert cfg.simulator_score_gamma > 0
    assert cfg.simulator_temperature_default <= cfg.simulator_temperature_max


@pytest.mark.parametrize("overrides", [
    {"simulator_default_budget": 0},
    {"simulator_score_gamma": 0},
    {"simulator_temperature_default": -0.1},
    {"simulator_temperature_max": 0},
])
def test_out_of_range_simulator_config_fails_at_load(overrides):
    with pytest.raises(ValidationError):
        load(**overrides)


def test_temperature_default_above_max_fails_at_load():
    """Validação cruzada entre dois campos: o slider não pode começar acima do
    próprio teto, senão a UI abriria num estado inválido."""
    with pytest.raises(ValidationError):
        load(simulator_temperature_default=1.5, simulator_temperature_max=1.0)
