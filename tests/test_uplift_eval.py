"""T-205 — Qini/AUUC (REQ-203). T-212 — placebo por permutação (REQ-212)."""

import numpy as np
import pandas as pd

from src.config import load
from src.uplift_eval import (
    _permute_treatment_within_offer_type,
    auuc,
    placebo_qini_distribution,
    placebo_test,
    qini,
    qini_by_strategy,
    qini_curves_by_strategy,
)
from tests.modeling_fixtures import synthetic_processed


def test_qini_real_supera_o_limiar_da_nula_quando_ha_sinal_plantado():
    """Aceite de REQ-212: com efeito heterogêneo real, o Qini real deve furar
    o percentil da distribuição nula. A permutação em si preserva a proporção
    tratado/controle por `offer_type` — embaralhar globalmente mudaria essa
    proporção (as taxas de view divergem por tipo no dado real) e derrubaria o
    Qini nulo por composição de grupo, não por ausência de efeito.
    """
    df = synthetic_processed(n=2000, seed=3, n_waves=6)

    rng_perm = np.random.default_rng(0)
    permutado = _permute_treatment_within_offer_type(df, rng_perm)
    original_prop = df.groupby("offer_type")["treatment"].mean()
    nova_prop = df.assign(treatment=permutado).groupby("offer_type")["treatment"].mean()
    pd.testing.assert_series_equal(original_prop, nova_prop, check_names=False)
    assert not permutado.equals(df["treatment"])

    responde = df["hist_spend_total"] > df["hist_spend_total"].median()
    p_convert = np.where(responde, np.where(df["treatment"] == 1, 0.85, 0.15), 0.20)
    rng = np.random.default_rng(3)
    df["converted"] = rng.binomial(1, p_convert).astype("int32")

    train_df = df[df["campaign_wave"] < 4].reset_index(drop=True)
    holdout_df = df[df["campaign_wave"] >= 4].reset_index(drop=True)
    cfg = load(xlearner_n_estimators=50, placebo_n_permutations=8)

    from src.uplift import fit_xlearner, predict

    modelos = fit_xlearner(train_df, cfg)
    pred = predict(modelos, holdout_df)
    score_real = qini(holdout_df["converted"], pred["uplift"], holdout_df["treatment"])

    nula = placebo_qini_distribution(train_df, holdout_df, cfg)
    resultado = placebo_test(score_real, nula, cfg)
    assert resultado["passou"]


def test_qini_by_strategy_ranks_true_signal_above_random_score():
    """REQ-203 estendido: comparar Qini/AUUC de estratégias que não vêm do
    X-learner. Uma estratégia com o efeito verdadeiro embutido no score deve
    superar uma estratégia com score aleatório sem sinal, nas duas métricas.
    """
    rng = np.random.default_rng(5)
    n = 3000
    tau = rng.uniform(0.0, 1.0, size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    p = np.clip(0.2 + treatment * tau, 0.0, 1.0)
    y = pd.Series(rng.binomial(1, p))
    treatment = pd.Series(treatment)

    scores = {
        "sinal_real": pd.Series(tau),
        "aleatorio": pd.Series(rng.random(n)),
    }
    resumo = qini_by_strategy(y, treatment, scores)

    assert set(resumo["strategy"]) == {"sinal_real", "aleatorio"}
    sinal = resumo.set_index("strategy").loc["sinal_real"]
    aleatorio = resumo.set_index("strategy").loc["aleatorio"]
    assert sinal["qini"] > aleatorio["qini"]
    assert sinal["auuc"] > aleatorio["auuc"]

    # auuc/qini isolados devem bater com o que qini_by_strategy reporta.
    assert auuc(y, scores["sinal_real"], treatment) == sinal["auuc"]


def test_qini_curves_by_strategy_traz_uma_curva_por_estrategia():
    y = pd.Series([1, 0, 1, 0, 1, 0])
    treatment = pd.Series([1, 1, 0, 0, 1, 0])
    scores = {
        "a": pd.Series([0.9, 0.1, 0.5, 0.2, 0.8, 0.05]),
        "b": pd.Series([0.1, 0.9, 0.2, 0.5, 0.05, 0.8]),
    }
    curvas = qini_curves_by_strategy(y, treatment, scores)

    assert set(curvas.columns) == {"strategy", "n_treated", "gain"}
    assert set(curvas["strategy"]) == {"a", "b"}
    for _, grupo in curvas.groupby("strategy"):
        assert grupo["n_treated"].iloc[0] == 0
