"""T-208 — Curva de ganho incremental por budget top-N (REQ-206).

Contrafactual observado (estilo Qini), lucro líquido incremental, três
estratégias. Fixtures sintéticas minúsculas montadas para exercer a fórmula e
seus casos de fronteira — não amostras do dado real.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load
from src.gaincurve import (
    NET_PROFIT_COLUMN,
    add_net_profit,
    completion_ranking,
    dynamic_hybrid_ranking,
    dynamic_hybrid_score,
    gain_at_budget,
    gain_curves,
    gain_curves_with_ci,
    hybrid_ranking,
    hybrid_score,
    incremental_gain_curve,
    random_ranking,
    uplift_ranking,
)


def _holdout(rows):
    """Grão mínimo que a curva consome: um recebimento observado por linha.

    `converted` é derivado de `conversion_value > 0` quando a linha não passa
    o valor explicitamente — cobre as fixtures antigas que só tinham lucro.
    """
    columns = ["account_id", "offer_id", "treatment", "conversion_value", "reward_cost"]
    if rows and len(rows[0]) == 6:
        columns.append("converted")
        return pd.DataFrame(rows, columns=columns)
    df = pd.DataFrame(rows, columns=columns)
    return df.assign(converted=(df["conversion_value"] > 0).astype(int))


def test_net_profit_e_receita_menos_custo():
    holdout = _holdout([("a", "o", 1, 50.0, 5.0)])
    assert add_net_profit(holdout)[NET_PROFIT_COLUMN].iloc[0] == pytest.approx(45.0)


def test_net_profit_desconta_reward_cost_tambem_no_controle():
    """O desconto segue a conversão, não a exposição (`cost.add_reward_cost`,
    `test_unviewed_conversion_still_costs`): um controle que converteu paga
    reward_cost de verdade, e o lucro líquido precisa refletir isso — não há
    isenção de custo só por não ter visto a oferta.
    """
    holdout = _holdout([("a", "o", 0, 50.0, 10.0)])
    assert add_net_profit(holdout)[NET_PROFIT_COLUMN].iloc[0] == pytest.approx(40.0)


def test_net_profit_e_zero_quando_nao_converteu_sem_reward_cost_solto():
    """G6 garante reward_cost=0 fora de conversão; o lucro de uma linha não
    convertida não pode ficar negativo por um reward_cost que não deveria
    existir.
    """
    holdout = _holdout([("a", "o", 1, 0.0, 0.0), ("b", "o", 0, 0.0, 0.0)])
    lucro = add_net_profit(holdout)[NET_PROFIT_COLUMN]
    assert (lucro == 0.0).all()


def test_ganho_no_budget_total_e_o_contrafactual_qini_escalado():
    """No prefixo completo, o ganho é L_tratado − L_controle · (N_t/N_c) — o
    contrafactual dos tratados estimado a partir dos controles observados.
    Dois tratados (lucro 40 e 20) e um controle (lucro 10): ganho = 60 − 10·(2/1) = 40.
    """
    holdout = add_net_profit(_holdout([
        ("a", "o", 1, 45.0, 5.0),   # lucro 40, tratado
        ("b", "o", 1, 25.0, 5.0),   # lucro 20, tratado
        ("c", "o", 0, 10.0, 0.0),   # lucro 10, controle
    ]))
    curva = incremental_gain_curve(holdout.index.to_numpy(), holdout)

    assert curva["n"].tolist() == [0, 1, 2, 3]
    assert curva["gain"].iloc[0] == 0.0
    assert curva["gain"].iloc[-1] == pytest.approx(40.0)


def test_prefixo_sem_controle_nao_desconta_contrafactual():
    """Enquanto o prefixo só tem tratados, não há contrafactual a subtrair: o
    ganho é o lucro acumulado dos tratados, sem NaN nem salto. O controle só
    entra na última posição, então N=1 e N=2 são puro lucro tratado.
    """
    holdout = add_net_profit(_holdout([
        ("a", "o", 1, 45.0, 5.0),   # lucro 40, tratado
        ("b", "o", 1, 25.0, 5.0),   # lucro 20, tratado
        ("c", "o", 0, 10.0, 0.0),   # controle, por último
    ]))
    curva = incremental_gain_curve(holdout.index.to_numpy(), holdout)

    assert curva["gain"].iloc[1] == pytest.approx(40.0)   # só o 1º tratado
    assert curva["gain"].iloc[2] == pytest.approx(60.0)   # os dois tratados, sem contrafactual
    assert not curva["gain"].isna().any()


def test_ranking_de_uplift_ordena_por_tau_decrescente():
    holdout = _holdout([("a", "o", 1, 0.0, 0.0), ("b", "o", 1, 0.0, 0.0), ("c", "o", 1, 0.0, 0.0)])
    pred = pd.DataFrame({"uplift": [0.1, 0.9, 0.5]}, index=holdout.index)
    assert uplift_ranking(pred).tolist() == [1, 2, 0]


def test_ranking_de_completion_ordena_por_p_convert_decrescente():
    holdout = _holdout([("a", "o", 1, 0.0, 0.0), ("b", "o", 1, 0.0, 0.0), ("c", "o", 1, 0.0, 0.0)])
    p = pd.Series([0.2, 0.8, 0.5], index=holdout.index)
    assert completion_ranking(p).tolist() == [1, 2, 0]


def test_ranking_aleatorio_e_deterministico_pela_seed():
    holdout = _holdout([("a", "o", 1, 0.0, 0.0)] * 5)
    cfg = load()
    assert random_ranking(holdout, cfg).tolist() == random_ranking(holdout, cfg).tolist()


def test_hybrid_score_e_soma_direta_sem_normalizar():
    uplift_pred = pd.Series([0.1, 0.5, -0.2])
    p_convert = pd.Series([0.8, 0.2, 0.9])
    score = hybrid_score(uplift_pred, p_convert, lambda_=0.5)
    pd.testing.assert_series_equal(score, uplift_pred + 0.5 * p_convert)


def test_hybrid_com_lambda_zero_degenera_no_uplift_puro():
    holdout = _holdout([("a", "o", 1, 0.0, 0.0), ("b", "o", 1, 0.0, 0.0), ("c", "o", 1, 0.0, 0.0)])
    uplift_pred = pd.Series([0.1, 0.9, 0.5], index=holdout.index)
    p_convert = pd.Series([0.99, 0.01, 0.5], index=holdout.index)  # sinal oposto ao uplift

    ranking_hibrido_lambda0 = hybrid_ranking(uplift_pred, p_convert, lambda_=0.0)
    ranking_uplift = uplift_ranking(pd.DataFrame({"uplift": uplift_pred}))
    assert ranking_hibrido_lambda0.tolist() == ranking_uplift.tolist()


def test_hybrid_lambda_maior_puxa_ranking_em_direcao_a_conversao_crua():
    """Com sinais opostos entre uplift e p_convert, aumentar λ desloca o topo
    do ranking híbrido para quem a conversão crua favorece — a mistura
    literal `uplift + λ·p_convert` deve responder a λ, não ser um NaN
    disfarçado ou um ranking congelado.
    """
    uplift_pred = pd.Series([0.5, 0.4, 0.3])  # "a" no topo puro
    p_convert = pd.Series([0.0, 0.0, 1.0])    # "c" no topo de conversão crua

    top_lambda0 = hybrid_ranking(uplift_pred, p_convert, lambda_=0.0)[0]
    top_lambda_alto = hybrid_ranking(uplift_pred, p_convert, lambda_=0.5)[0]
    assert top_lambda0 == 0
    assert top_lambda_alto == 2


def test_dynamic_hybrid_score_pondera_por_incerteza_local():
    """Cliente com mu1==mu0 (incerteza 0) fica só com o uplift normalizado;
    cliente com maior discrepância |mu1-mu0| do grupo empresta mais peso do
    prior de conversão. gamma=1 é a resposta linear de referência.
    """
    mu0 = pd.Series([0.5, 0.2, 0.1])
    mu1 = pd.Series([0.5, 0.2, 0.9])   # "a" incerteza 0, "b" incerteza 0, "c" incerteza máxima
    uplift_pred = pd.Series([0.0, 0.0, 0.0])   # uplift_norm empatado em tudo
    p_convert = pd.Series([0.1, 0.5, 0.9])

    score = dynamic_hybrid_score(mu0, mu1, uplift_pred, p_convert, gamma=1.0)

    # "a" e "b" têm incerteza zero -> lambda_local=0 -> score = uplift_norm (empatado, 0
    # porque uplift_pred é constante e uplift_norm degenera em 0 pelo epsilon do denominador).
    assert score.iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert score.iloc[1] == pytest.approx(0.0, abs=1e-6)
    # "c" tem incerteza máxima (lambda_local=1) -> score = p_convert_norm = 1.0 (máximo do grupo).
    assert score.iloc[2] == pytest.approx(1.0, abs=1e-6)


def test_dynamic_hybrid_gamma_alto_e_mais_conservador():
    """gamma>1 concentra lambda_local nos extremos de incerteza: na linha de
    incerteza normalizada intermediária (0.5), gamma=2 dá um peso ao prior de
    conversão MENOR que gamma=0.5 (x**2 < x**0.5 para x em (0,1)) — a curva de
    resposta fica mais conservadora, não mais agressiva, com gamma alto. A
    linha 0 tem uplift alto/p_convert baixo, então "mais peso ao uplift" ali
    significa score MAIOR.
    """
    mu0 = pd.Series([0.0, 0.0])
    mu1 = pd.Series([0.5, 1.0])          # incerteza normalizada: 0.5 e 1.0
    uplift_pred = pd.Series([1.0, 0.0])  # linha 0 favorecida pelo uplift
    p_convert = pd.Series([0.0, 1.0])    # linha 0 desfavorecida pela conversão crua

    score_gamma_baixo = dynamic_hybrid_score(mu0, mu1, uplift_pred, p_convert, gamma=0.5)
    score_gamma_alto = dynamic_hybrid_score(mu0, mu1, uplift_pred, p_convert, gamma=2.0)

    # Na linha de incerteza intermediária (índice 0), gamma alto empresta MENOS
    # peso ao prior de conversão (lambda_local menor) -> score mais próximo do
    # uplift puro (mais alto, já que a linha 0 é favorecida pelo uplift).
    assert score_gamma_alto.iloc[0] > score_gamma_baixo.iloc[0]


def test_dynamic_hybrid_ranking_e_deterministico_e_ordena_decrescente():
    mu0 = pd.Series([0.1, 0.1, 0.1])
    mu1 = pd.Series([0.2, 0.9, 0.15])
    uplift_pred = pd.Series([0.1, 0.9, 0.5])
    p_convert = pd.Series([0.9, 0.1, 0.5])

    ranking = dynamic_hybrid_ranking(mu0, mu1, uplift_pred, p_convert, gamma=1.0)
    score = dynamic_hybrid_score(mu0, mu1, uplift_pred, p_convert, gamma=1.0)
    assert ranking.tolist() == score.sort_values(ascending=False, kind="stable").index.tolist()


def test_estrategia_com_sinal_supera_aleatorio_no_ganho_final():
    """Sanidade da comparação (REQ-206): quando o τ verdadeiro concentra o lucro
    incremental nos primeiros, ordenar por τ entrega mais cedo do que a ordem
    aleatória — a curva do uplift domina em budgets intermediários.
    """
    rng = np.random.default_rng(0)
    n = 200
    tau = np.linspace(1.0, 0.0, n)                 # sinal decrescente conhecido
    treatment = rng.integers(0, 2, size=n)
    # Lucro dos tratados cresce com tau; controle é ruído centrado em zero.
    profit = np.where(treatment == 1, tau * 100, rng.normal(0, 1, n))
    converted = rng.binomial(1, np.where(treatment == 1, tau, 0.1))
    holdout = pd.DataFrame({
        "account_id": [f"a{i}" for i in range(n)],
        "offer_id": "o",
        "treatment": treatment,
        "conversion_value": profit,
        "reward_cost": 0.0,
        "converted": converted,
    })

    pred = pd.DataFrame({"uplift": tau}, index=holdout.index)
    rankings = {
        "uplift": uplift_ranking(pred),
        "aleatorio": random_ranking(holdout, load()),
    }
    curvas = gain_curves(rankings, holdout)

    budget = n // 4
    ganho = gain_at_budget(curvas, budget).set_index("strategy")["gain"]
    assert ganho["uplift"] > ganho["aleatorio"]


def test_gain_at_budget_devolve_o_maior_n_dentro_do_orcamento():
    holdout = _holdout([("a", "o", 1, 10.0, 0.0), ("b", "o", 0, 5.0, 0.0), ("c", "o", 1, 8.0, 0.0)])
    curvas = gain_curves({"s": holdout.index.to_numpy()}, holdout)
    linha = gain_at_budget(curvas, budget=2)
    assert linha["n"].iloc[0] == 2


def test_conversao_incremental_usa_o_mesmo_contrafactual_escalado():
    """Mesma fórmula do lucro, mas sobre `converted`: dois tratados convertidos
    (converted=1) e um controle convertido (converted=1) — conversão
    incremental = 2 − 1·(2/1) = 0.
    """
    holdout = _holdout([
        ("a", "o", 1, 45.0, 5.0, 1),
        ("b", "o", 1, 25.0, 5.0, 1),
        ("c", "o", 0, 10.0, 0.0, 1),
    ])
    holdout = add_net_profit(holdout)
    curva = incremental_gain_curve(holdout.index.to_numpy(), holdout)
    assert curva["conversions"].iloc[-1] == pytest.approx(0.0)


def test_gain_curves_with_ci_traz_bandas_que_contem_o_ponto():
    """O IC bootstrap precisa conter a estimativa pontual e ser bem formado:
    lo <= gain <= hi em todo N, para lucro e conversão.
    """
    rng = np.random.default_rng(1)
    n = 60
    treatment = rng.integers(0, 2, size=n)
    profit = np.where(treatment == 1, rng.normal(20, 5, n), rng.normal(5, 5, n))
    converted = rng.binomial(1, np.where(treatment == 1, 0.5, 0.2))
    holdout = pd.DataFrame({
        "account_id": [f"a{i}" for i in range(n)],
        "offer_id": "o",
        "treatment": treatment,
        "conversion_value": profit,
        "reward_cost": 0.0,
        "converted": converted,
    })
    cfg = load(gain_curve_n_bootstrap=30)
    rankings = {"aleatorio": random_ranking(holdout, cfg)}
    curvas = gain_curves_with_ci(rankings, holdout, cfg)

    assert {"gain_lo", "gain_hi", "conversions_lo", "conversions_hi"} <= set(curvas.columns)
    assert (curvas["gain_lo"] <= curvas["gain"] + 1e-9).all()
    assert (curvas["gain"] <= curvas["gain_hi"] + 1e-9).all()
    assert (curvas["conversions_lo"] <= curvas["conversions"] + 1e-9).all()
    assert (curvas["conversions"] <= curvas["conversions_hi"] + 1e-9).all()
