"""Wrappers de modelo (`src.models`): a superfície de `model train`/`model predict`.

Testes estruturais, não de resultado numérico: guardam que `from_config` lê os
campos certos da config, que o blend não altera a fórmula que `gaincurve` já
testa, e que `save`/`load` round-trip prevê idêntico — um pickle quebrado ou um
`from_config` que puxa o hiperparâmetro errado quebra o CLI em silêncio.
"""

import numpy as np
import pandas as pd
import pytest

from src import gaincurve
from src.config import load
from src.models import (
    BLENDED_MODEL_FILENAME,
    UPLIFT_MODEL_FILENAME,
    BlendedUpliftModel,
    ConversionModel,
    UpliftModel,
)
from src.split import MODELED_OFFER_TYPES
from tests.modeling_fixtures import synthetic_processed


def _modeled(df: pd.DataFrame) -> pd.DataFrame:
    """Só bogo/discount — os tipos que os modelos veem (informational fora)."""
    return df[df["offer_type"].isin(MODELED_OFFER_TYPES)].reset_index(drop=True)


def test_uplift_from_config_reads_xlearner_hyperparams_not_baseline():
    """`from_config` deve puxar os hiperparâmetros do X-learner, não os do LGBM
    baseline — os dois têm `n_estimators`/`max_depth`/`learning_rate` distintos
    na config, e trocar um pelo outro treinaria o modelo errado sem erro.
    """
    cfg = load(
        xlearner_n_estimators=77, xlearner_max_depth=5, xlearner_learning_rate=0.11,
        lgbm_n_estimators=200, seed=7,
    )
    model = UpliftModel.from_config(cfg)
    assert model.n_estimators == 77
    assert model.max_depth == 5
    assert model.learning_rate == 0.11
    assert model.seed == 7


def test_uplift_predict_before_fit_raises_not_returns_empty():
    """Prever sem ajustar deve falhar alto, não devolver um resultado vazio ou
    silencioso — o erro clássico de `model predict` apontando para um modelo não
    treinado precisa ser visível.
    """
    model = UpliftModel.from_config(load())
    with pytest.raises(RuntimeError):
        _ = model.models


def test_uplift_save_load_predicts_identically(tmp_path):
    """`save`/`load` round-trip deve prever exatamente igual — a garantia de que
    `model train` (escreve) e `model predict` (lê) veem o mesmo modelo.
    """
    df = _modeled(synthetic_processed(n=600, seed=3))
    cfg = load(xlearner_n_estimators=40, models_dir=tmp_path)

    model = UpliftModel.from_config(cfg).fit(df)
    path = model.save(cfg)
    assert path == tmp_path / UPLIFT_MODEL_FILENAME
    assert path.exists()

    reloaded = UpliftModel.load(cfg)
    np.testing.assert_array_equal(
        model.predict(df)["uplift"].to_numpy(),
        reloaded.predict(df)["uplift"].to_numpy(),
    )


def test_blended_fixed_score_is_exactly_the_hybrid_formula():
    """O wrapper não pode alterar a fórmula do blend: no modo fixo, `score` tem
    de bater `gaincurve.hybrid_score(uplift, p_convert, λ)` exatamente — o
    wrapper só liga as saídas dos dois componentes, `gaincurve` é quem testa a
    fórmula.
    """
    df = _modeled(synthetic_processed(n=600, seed=5))
    cfg = load(xlearner_n_estimators=40)

    uplift_model = UpliftModel.from_config(cfg).fit(df)
    conversion_model = ConversionModel.from_config(cfg).fit(df)
    blend = BlendedUpliftModel(uplift_model, conversion_model, mode="fixed", lambda_=0.3)

    uplift_pred = uplift_model.predict(df)["uplift"]
    p_convert = conversion_model.predict_proba(df)
    esperado = gaincurve.hybrid_score(uplift_pred, p_convert, 0.3)

    pd.testing.assert_series_equal(blend.score(df), esperado)


def test_uplift_causal_importance_is_normalized_over_xlearner_features():
    """A importância causal do X-learner soma 1 e é indexada pelas features do
    X-learner (sem `offer_type`, o eixo de estratificação) — um report que não
    normaliza ou vaza `offer_type` como feature interna está incoerente.
    """
    from src.uplift import _XLEARNER_FEATURES

    df = _modeled(synthetic_processed(n=800, seed=5))
    cfg = load(xlearner_n_estimators=40)
    model = UpliftModel.from_config(cfg).fit(df)

    imp = model.feature_importance(df)
    assert set(imp.index) == set(_XLEARNER_FEATURES)
    assert "offer_type" not in imp.index
    assert imp.min() >= 0.0
    np.testing.assert_allclose(imp.sum(), 1.0, rtol=1e-6)


def test_blended_combined_importance_mirrors_the_fixed_score_algebra():
    """A coluna `combined` do blend fixo tem de ser exatamente `imp_uplift +
    λ·imp_conversion` renormalizado — a mesma álgebra do score que ranqueia. Se
    a combinação divergir da fórmula do score, a importância do blend deixa de
    significar o que o blend faz.
    """
    df = _modeled(synthetic_processed(n=800, seed=6))
    cfg = load(xlearner_n_estimators=40)
    blend = BlendedUpliftModel.from_config(cfg.model_copy(update={"blend_mode": "fixed", "blend_lambda": 0.3})).fit(df)

    imp = blend.feature_importance(df)
    esperado = imp["uplift"] + 0.3 * imp["conversion"]
    esperado = esperado / esperado.sum()
    pd.testing.assert_series_equal(imp["combined"], esperado, check_names=False)
    np.testing.assert_allclose(imp[["uplift", "conversion", "combined"]].sum(), [1.0, 1.0, 1.0], rtol=1e-6)


def test_blended_dynamic_effective_lambda_is_mean_local_lambda():
    """No modo dinâmico o λ efetivo da combinação de importâncias é a média do
    `lambda_local` do score — o λ constante que reproduz, em média, a mesma
    mistura. Fixar outro λ tornaria a importância incoerente com o ranking.
    """
    df = _modeled(synthetic_processed(n=800, seed=7))
    cfg = load(xlearner_n_estimators=40)
    blend = BlendedUpliftModel.from_config(cfg.model_copy(update={"blend_mode": "dynamic", "blend_gamma": 1.0})).fit(df)

    uncertainty = blend.uplift_model.predict_uncertainty(df)["uncertainty"]
    lambda_local = (uncertainty / (uncertainty.max() + 1e-9)) ** 1.0
    np.testing.assert_allclose(blend._effective_lambda(df), lambda_local.mean(), rtol=1e-9)


def test_blended_from_config_reads_blend_defaults():
    """`from_config` deve materializar o blend padrão da config
    (`blend_mode`/`blend_lambda`/`blend_gamma`) — o default que `model predict`
    usa sem argumentos.
    """
    cfg = load(blend_mode="dynamic", blend_lambda=0.25, blend_gamma=1.5)
    blend = BlendedUpliftModel.from_config(cfg)
    assert blend.mode == "dynamic"
    assert blend.lambda_ == 0.25
    assert blend.gamma == 1.5


def test_blended_rejects_unknown_mode():
    """Modo fora de {fixed, dynamic} deve falhar na construção, não escolher um
    ramo silenciosamente — o construtor é a fronteira que valida o parâmetro.
    """
    with pytest.raises(ValueError):
        BlendedUpliftModel(
            UpliftModel.from_config(load()), ConversionModel.from_config(load()),
            mode="quadratic",
        )


def test_blended_save_load_ranks_identically(tmp_path):
    """Round-trip do modelo composto deve ranquear idêntico — o objeto de
    produção inteiro (uplift + conversão + parâmetros do blend) persiste junto.
    """
    df = _modeled(synthetic_processed(n=600, seed=8))
    cfg = load(xlearner_n_estimators=40, models_dir=tmp_path)

    blend = BlendedUpliftModel.from_config(cfg).fit(df)
    path = blend.save(cfg)
    assert path == tmp_path / BLENDED_MODEL_FILENAME

    reloaded = BlendedUpliftModel.load(cfg)
    np.testing.assert_array_equal(blend.rank(df), reloaded.rank(df))
