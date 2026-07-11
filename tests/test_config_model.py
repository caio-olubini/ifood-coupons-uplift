"""Config de modelagem (spec 02, REQ-210, T-201).

Custo/hiperparâmetro/corte inválido deve falhar na carga, antes de qualquer
treino — mesma disciplina de fronteira da config do pipeline (spec 01).
"""

import pytest
from pydantic import ValidationError

from src.config import load


def test_wave_cutoff_at_or_above_n_waves_fails_at_load():
    """Validação cruzada entre dois campos (não um `Field(gt=0)` isolado): o
    corte de validação precisa ser estritamente menor que o número de ondas,
    senão o split de T-202 produziria um lado vazio. O default carrega sem
    erro (`cfg.validation_wave_cutoff=4 < cfg.n_campaign_waves=6`).
    """
    cfg = load()
    assert cfg.validation_wave_cutoff < cfg.n_campaign_waves

    with pytest.raises(ValidationError):
        load(validation_wave_cutoff=6, n_campaign_waves=6)


@pytest.mark.parametrize("overrides", [
    {"lgbm_n_estimators": 0},
    {"xlearner_n_estimators": 0},
    {"calibration_n_bins": 1},
    {"placebo_confidence_level": 1.5},
])
def test_out_of_range_hyperparameter_fails_at_load(overrides):
    with pytest.raises(ValidationError):
        load(**overrides)
