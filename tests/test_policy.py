"""T-206/T-207 — política sensível a custo e baselines: T-policy-noturno."""

import numpy as np
import pandas as pd
import pytest

from src.config import load
from src.policy import (
    NO_SEND,
    allocate,
    expected_net_profit,
    offer_economics,
    policy_random,
    policy_send_all,
    policy_top_completion,
    validate_recommendations,
)


def _reference(rows):
    """Conjunto de referência mínimo no formato que `offer_economics` consome."""
    return pd.DataFrame(rows, columns=["account_id", "offer_id", "offer_type", "converted", "conversion_value", "reward_cost"])


def _uplift(rows):
    return pd.DataFrame(rows, columns=["account_id", "offer_id", "offer_type", "uplift"])


def test_policy_noturno_custo_maior_que_ganho_nao_envia():
    """T-policy-noturno (guarda REQ-204): oferta com uplift alto mas desconto
    caro tem lucro esperado negativo — a ação nula vence.

    Uplift 0,20 × receita 10 = ganho 2,00.
    P(converte|tratado) 0,90 × desconto 10 = custo 9,00.
    Ganho − custo = −7,00 < 0 ⇒ `nao_enviar`.
    """
    reference = _reference([
        ("acc1", "off_caro", "bogo", 1, 10.0, 10.0),
        ("acc1", "off_caro", "bogo", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)

    scored = expected_net_profit(
        _uplift([("acc1", "off_caro", "bogo", 0.20)]), economics, p_convert_treated=[0.90]
    )
    assert scored["net_profit"].iloc[0] == pytest.approx(-7.0)

    decisao = allocate(scored)
    assert decisao["chosen_action"].iloc[0] == NO_SEND
    assert decisao["expected_net_profit"].iloc[0] == 0.0


def test_oferta_lucrativa_e_escolhida_sobre_a_acao_nula():
    """O espelho do teste acima: ganho > custo ⇒ envia, e o lucro é o esperado."""
    reference = _reference([
        ("acc1", "off_bom", "discount", 1, 100.0, 2.0),
        ("acc1", "off_bom", "discount", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)

    # 0,30 × 100 = 30,0 de ganho; 0,50 × 2 = 1,0 de custo ⇒ 29,0.
    scored = expected_net_profit(
        _uplift([("acc1", "off_bom", "discount", 0.30)]), economics, p_convert_treated=[0.50]
    )
    decisao = allocate(scored)

    assert decisao["chosen_action"].iloc[0] == "off_bom"
    assert decisao["expected_net_profit"].iloc[0] == pytest.approx(29.0)


def test_informational_sem_desconto_tem_custo_esperado_zero():
    """`informational` não paga desconto: qualquer uplift positivo é lucrativo,
    sem que a política precise de regra especial para o tipo.
    """
    reference = _reference([
        ("acc1", "off_info", "informational", 1, 40.0, 0.0),
        ("acc1", "off_info", "informational", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)
    assert economics["discount_value"].iloc[0] == 0.0

    scored = expected_net_profit(
        _uplift([("acc1", "off_info", "informational", 0.01)]), economics, p_convert_treated=[0.99]
    )
    assert scored["expected_cost"].iloc[0] == 0.0
    assert allocate(scored)["chosen_action"].iloc[0] == "off_info"


def test_desconto_e_debitado_em_toda_conversao_nao_so_na_incremental():
    """A assimetria de REQ-204: receita é incremental (uplift), custo é total
    (P(converte|tratado)). Se o custo fosse cobrado só sobre o uplift, a oferta
    cara abaixo passaria a ser lucrativa e a política erraria.
    """
    reference = _reference([
        ("acc1", "off", "bogo", 1, 50.0, 10.0),
        ("acc1", "off", "bogo", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)
    scored = expected_net_profit(_uplift([("acc1", "off", "bogo", 0.10)]), economics, p_convert_treated=[0.80])

    assert scored["expected_revenue"].iloc[0] == pytest.approx(0.10 * 50.0)   # incremental
    assert scored["expected_cost"].iloc[0] == pytest.approx(0.80 * 10.0)      # total, não 0.10 * 10
    assert scored["net_profit"].iloc[0] < 0


def test_allocate_escolhe_a_melhor_entre_varias_ofertas_do_cliente():
    reference = _reference([
        ("acc1", "off_a", "bogo", 1, 20.0, 5.0),
        ("acc1", "off_b", "discount", 1, 80.0, 2.0),
        ("acc1", "off_a", "bogo", 0, 0.0, 0.0),
        ("acc1", "off_b", "discount", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)
    scored = expected_net_profit(
        _uplift([("acc1", "off_a", "bogo", 0.5), ("acc1", "off_b", "discount", 0.5)]),
        economics,
        p_convert_treated=[0.1, 0.1],
    )
    decisao = allocate(scored)

    assert len(decisao) == 1                          # uma linha por cliente
    assert decisao["chosen_action"].iloc[0] == "off_b"  # 40,0 − 0,2 vence 10,0 − 0,5


def test_allocate_e_deterministico_no_empate():
    """Empate de lucro resolve por `offer_id` — a política não pode variar entre
    execuções sobre a mesma entrada.
    """
    reference = _reference([
        ("acc1", "off_z", "bogo", 1, 10.0, 0.0),
        ("acc1", "off_a", "bogo", 1, 10.0, 0.0),
        ("acc1", "off_z", "bogo", 0, 0.0, 0.0),
        ("acc1", "off_a", "bogo", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)
    uplift = _uplift([("acc1", "off_z", "bogo", 0.3), ("acc1", "off_a", "bogo", 0.3)])

    escolhas = {
        allocate(expected_net_profit(uplift, economics, [0.0, 0.0]))["chosen_action"].iloc[0]
        for _ in range(5)
    }
    assert escolhas == {"off_a"}


def test_saida_da_politica_valida_no_contrato_tipado():
    reference = _reference([("acc1", "off", "bogo", 1, 50.0, 2.0), ("acc1", "off", "bogo", 0, 0.0, 0.0)])
    scored = expected_net_profit(_uplift([("acc1", "off", "bogo", 0.4)]), offer_economics(reference), [0.2])

    recomendacoes = validate_recommendations(allocate(scored))
    assert len(recomendacoes) == 1
    assert recomendacoes[0].account_id == "acc1"
    assert recomendacoes[0].chosen_action == "off"


def test_oferta_sem_conversao_observada_nao_e_escolhida():
    """Sem conversão no conjunto de referência não há receita observável: a
    oferta fica com receita 0 e perde para a ação nula, em vez de parecer
    lucrativa por ausência de evidência.
    """
    reference = _reference([("acc1", "off_novo", "bogo", 0, 0.0, 0.0)])
    economics = offer_economics(reference)
    assert economics["revenue_per_conversion"].iloc[0] == 0.0

    scored = expected_net_profit(_uplift([("acc1", "off_novo", "bogo", 0.9)]), economics, [0.5])
    assert allocate(scored)["chosen_action"].iloc[0] == NO_SEND


def test_oferta_ausente_do_conjunto_de_referencia_nao_produz_lucro_nan():
    """Uma oferta sem economia observável zera receita e custo, em vez de deixar um
    NaN correr até `allocate` — onde a decisão sairia por acidente, já que NaN perde
    toda comparação sem que ninguém tenha decidido isso.
    """
    reference = _reference([("acc1", "off_conhecida", "bogo", 1, 50.0, 5.0), ("acc1", "off_conhecida", "bogo", 0, 0.0, 0.0)])
    economics = offer_economics(reference)

    scored = expected_net_profit(
        _uplift([("acc1", "off_conhecida", "bogo", 0.3), ("acc1", "off_fantasma", "bogo", 0.9)]),
        economics,
        p_convert_treated=[0.1, 0.1],
    )
    assert scored["net_profit"].notna().all()
    assert scored.loc[scored["offer_id"] == "off_fantasma", "net_profit"].iloc[0] == 0.0
    assert allocate(scored)["chosen_action"].iloc[0] == "off_conhecida"


# --- Baselines (REQ-205) -------------------------------------------------------


def _reference_dois_clientes():
    return _reference([
        ("acc1", "off_a", "bogo", 1, 20.0, 5.0),
        ("acc1", "off_b", "discount", 1, 80.0, 2.0),
        ("acc2", "off_a", "bogo", 1, 20.0, 5.0),
        ("acc2", "off_b", "discount", 0, 0.0, 0.0),
    ])


def test_cada_baseline_produz_uma_recomendacao_por_cliente():
    """Aceite de T-207: os três baselines cobrem todo cliente do conjunto."""
    reference = _reference_dois_clientes()
    economics = offer_economics(reference)
    uplift = _uplift([
        ("acc1", "off_a", "bogo", 0.1), ("acc1", "off_b", "discount", 0.2),
        ("acc2", "off_a", "bogo", 0.1), ("acc2", "off_b", "discount", 0.2),
    ])
    scored = expected_net_profit(uplift, economics, p_convert_treated=[0.3, 0.3, 0.3, 0.3])
    cfg = load()

    clientes = set(reference["account_id"])
    for policy in (
        policy_random(reference, cfg),
        policy_send_all(scored),
        policy_top_completion(reference, p_convert=[0.1, 0.9, 0.9, 0.1]),
        allocate(scored),
    ):
        assert set(policy["account_id"]) == clientes
        assert len(policy) == len(clientes)


def test_send_all_nunca_escolhe_a_acao_nula_e_carrega_o_prejuizo():
    """O status quo envia sempre. Quando toda oferta dá prejuízo, `send_all`
    mostra o lucro negativo — é o custo que a política de uplift evita ao poder
    não enviar. Clipar em zero apagaria a comparação de REQ-206.
    """
    reference = _reference([("acc1", "off_caro", "bogo", 1, 10.0, 10.0), ("acc1", "off_caro", "bogo", 0, 0.0, 0.0)])
    scored = expected_net_profit(_uplift([("acc1", "off_caro", "bogo", 0.2)]), offer_economics(reference), [0.9])

    send_all = policy_send_all(scored)
    assert send_all["chosen_action"].iloc[0] == "off_caro"
    assert send_all["expected_net_profit"].iloc[0] == pytest.approx(-7.0)

    # Mesma entrada: a política sensível a custo recusa e fica em zero.
    assert allocate(scored)["chosen_action"].iloc[0] == NO_SEND


def test_top_completion_ordena_por_propensao_nao_por_uplift():
    """O baseline a bater (REQ-205): escolhe a oferta de maior P(converter),
    ainda que outra tenha uplift maior. Aqui `off_a` tem propensão alta e
    `off_b` uplift alto — top-completion pega `off_a`, a política pega `off_b`.
    """
    reference = _reference_dois_clientes()
    top = policy_top_completion(
        reference[reference["account_id"] == "acc1"], p_convert=[0.95, 0.05]
    )
    assert top["chosen_action"].iloc[0] == "off_a"

    economics = offer_economics(reference)
    scored = expected_net_profit(
        _uplift([("acc1", "off_a", "bogo", 0.01), ("acc1", "off_b", "discount", 0.50)]),
        economics,
        p_convert_treated=[0.95, 0.05],
    )
    assert allocate(scored)["chosen_action"].iloc[0] == "off_b"


def test_policy_random_e_reprodutivel_dada_a_seed():
    reference = _reference_dois_clientes()
    cfg = load()
    primeira = policy_random(reference, cfg)
    segunda = policy_random(reference, cfg)
    pd.testing.assert_frame_equal(primeira, segunda)

    # Só escolhe entre ofertas que o cliente de fato recebeu.
    recebidas = reference.groupby("account_id")["offer_id"].apply(set).to_dict()
    for _, row in primeira.iterrows():
        assert row["chosen_action"] in recebidas[row["account_id"]]
