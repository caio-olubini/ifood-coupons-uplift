"""T-206/T-207 — política sensível a custo e baselines: T-policy-noturno."""

import pandas as pd
import pytest

from src.config import load
from src.policy import (
    ELIGIBLE_OFFER_TYPES,
    NO_SEND,
    allocate,
    expected_net_profit,
    offer_economics,
    policy_random,
    policy_send_all,
    policy_top_completion,
)


def _reference(rows):
    """Conjunto de referência mínimo no formato que `offer_economics` consome."""
    return pd.DataFrame(rows, columns=["account_id", "offer_id", "offer_type", "converted", "conversion_value", "reward_cost"])


def _uplift(rows):
    return pd.DataFrame(rows, columns=["account_id", "offer_id", "offer_type", "uplift"])


def test_policy_noturno_custo_maior_que_ganho_nao_envia():
    """T-policy-noturno (guarda REQ-204): oferta com uplift alto mas desconto
    caro tem lucro esperado negativo — a ação nula vence. A assimetria de
    custo (receita incremental, custo total) é o que produz esse resultado:
    se o custo fosse cobrado só sobre o uplift, a oferta pareceria lucrativa.
    """
    reference = _reference([
        ("acc1", "off_caro", "bogo", 1, 10.0, 10.0),
        ("acc1", "off_caro", "bogo", 0, 0.0, 0.0),
    ])
    economics = offer_economics(reference)
    scored = expected_net_profit(
        _uplift([("acc1", "off_caro", "bogo", 0.20)]), economics, p_convert_treated=[0.90]
    )

    assert scored["expected_revenue"].iloc[0] == pytest.approx(0.20 * 10.0)   # incremental
    assert scored["expected_cost"].iloc[0] == pytest.approx(0.90 * 10.0)      # total, não 0.20 * 10

    decisao = allocate(scored)
    assert decisao["chosen_action"].iloc[0] == NO_SEND
    assert decisao["expected_net_profit"].iloc[0] == 0.0

    # `informational` não é cupom nem promoção — fica fora do universo que a
    # política aloca (escopo de REQ-204). Mesmo com uplift alto e custo zero,
    # nunca é escolhida entre as opções de um cliente.
    assert "informational" not in ELIGIBLE_OFFER_TYPES
    reference_mista = _reference([
        ("acc1", "off_info", "informational", 1, 1000.0, 0.0),
        ("acc1", "off_info", "informational", 0, 0.0, 0.0),
        ("acc1", "off_b", "bogo", 1, 20.0, 5.0),
        ("acc1", "off_b", "bogo", 0, 0.0, 0.0),
    ])
    scored_mista = expected_net_profit(
        _uplift([("acc1", "off_info", "informational", 0.9), ("acc1", "off_b", "bogo", 0.5)]),
        offer_economics(reference_mista),
        p_convert_treated=[0.01, 0.1],
    )
    assert allocate(scored_mista)["chosen_action"].iloc[0] == "off_b"


def test_baselines_cobrem_todo_cliente_e_send_all_carrega_o_prejuizo():
    """Aceite de T-207: os três baselines cobrem todo cliente do conjunto.
    `send_all` nunca escolhe a ação nula — quando toda oferta dá prejuízo, ele
    carrega o lucro negativo, que é o custo que a política de uplift evita ao
    poder recusar (clipar em zero apagaria a comparação de REQ-206).
    """
    reference = _reference([
        ("acc1", "off_a", "bogo", 1, 20.0, 5.0),
        ("acc1", "off_b", "discount", 1, 80.0, 2.0),
        ("acc2", "off_a", "bogo", 1, 20.0, 5.0),
        ("acc2", "off_b", "discount", 0, 0.0, 0.0),
    ])
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

    prejuizo = _reference([("acc1", "off_caro", "bogo", 1, 10.0, 10.0), ("acc1", "off_caro", "bogo", 0, 0.0, 0.0)])
    scored_prejuizo = expected_net_profit(_uplift([("acc1", "off_caro", "bogo", 0.2)]), offer_economics(prejuizo), [0.9])
    send_all = policy_send_all(scored_prejuizo)
    assert send_all["expected_net_profit"].iloc[0] == pytest.approx(-7.0)
    assert allocate(scored_prejuizo)["chosen_action"].iloc[0] == NO_SEND
