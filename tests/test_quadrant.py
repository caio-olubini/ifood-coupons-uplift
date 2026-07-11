"""Classificação de quadrante causal e seu cruzamento com o ranking de uma
estratégia (composição, decomposição de lucro, deixado na mesa). Fixtures
sintéticas minúsculas montadas para exercer a fórmula e seus casos de
fronteira — não amostras do dado real.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load
from src.quadrant import (
    LOST_CAUSE,
    PERSUADABLE,
    SLEEPING_DOG,
    SURE_THING,
    classify_quadrant,
    composition_at_budget,
    gain_by_quadrant_at_budget,
    left_on_table,
)

cfg = load()
EPS = cfg.quadrant_tau_epsilon


def _stages(tau):
    """`stages` mínimo: índice 0..n-1, um τ previsto por cliente."""
    return pd.DataFrame({"tau": tau})


def _p_convert(values):
    return pd.Series(values)


def test_classify_quadrant_cobre_os_quatro_tipos():
    """Fora da banda [-EPS, EPS], o sinal de tau decide sozinho; dentro dela,
    p_convert decide sure_thing (propensão alta) vs lost_cause (propensão baixa).
    """
    stages = _stages([2 * EPS, -2 * EPS, 0.0, 0.0])
    p_convert = _p_convert([0.5, 0.5, 0.9, 0.1])

    quadrante = classify_quadrant(stages, p_convert, cfg)

    assert quadrante.tolist() == [PERSUADABLE, SLEEPING_DOG, SURE_THING, LOST_CAUSE]


def test_classify_quadrant_tau_domina_p_convert_fora_da_banda():
    """Um tau claramente positivo é persuadable mesmo com p_convert baixo —
    o corte primário é sempre tau; p_convert só desempata dentro da banda.
    """
    stages = _stages([2 * EPS])
    p_convert = _p_convert([0.01])

    quadrante = classify_quadrant(stages, p_convert, cfg)

    assert quadrante.iloc[0] == PERSUADABLE


def test_composition_at_budget_conta_so_o_top_n_nao_o_holdout_inteiro():
    """Um cliente fora do top-N (budget menor que o ranking inteiro) não deve
    aparecer na composição — a pergunta é sobre quem a estratégia *escolhe*
    com aquele orçamento, não sobre o holdout inteiro.
    """
    stages = _stages([2 * EPS, 2 * EPS, -2 * EPS])  # 2 persuadable, 1 sleeping_dog
    p_convert = _p_convert([0.5, 0.5, 0.5])
    ranking = np.array([0, 1, 2])

    composicao = composition_at_budget(ranking, stages, p_convert, cfg, budget=2)

    assert set(composicao["quadrante"]) == {PERSUADABLE}
    linha = composicao.set_index("quadrante").loc[PERSUADABLE]
    assert linha["n"] == 2
    assert linha["pct"] == pytest.approx(1.0)


def test_composition_at_budget_pct_soma_um_entre_quadrantes_presentes():
    stages = _stages([2 * EPS, -2 * EPS, 0.0, 0.0])
    p_convert = _p_convert([0.5, 0.5, 0.9, 0.1])
    ranking = np.array([0, 1, 2, 3])

    composicao = composition_at_budget(ranking, stages, p_convert, cfg, budget=4)

    assert composicao["pct"].sum() == pytest.approx(1.0)
    assert len(composicao) == 4


def test_composition_at_budget_traz_tau_medio_por_quadrante():
    """Dois `sure_thing` (dentro da banda, p_convert alta) com τ plantado
    diferente (0.02·EPS e 0.9·EPS, ambos dentro da banda) — a média precisa
    refletir esses dois valores, não zerar por definição do quadrante.
    """
    stages = _stages([0.02 * EPS, 0.9 * EPS])
    p_convert = _p_convert([0.9, 0.9])
    ranking = np.array([0, 1])

    composicao = composition_at_budget(ranking, stages, p_convert, cfg, budget=2)

    tau_esperado = (0.02 * EPS + 0.9 * EPS) / 2
    assert composicao.set_index("quadrante").loc[SURE_THING, "tau_medio"] == pytest.approx(tau_esperado)


def _holdout(rows):
    columns = ["account_id", "offer_id", "treatment", "conversion_value", "reward_cost"]
    return pd.DataFrame(rows, columns=columns)


def test_gain_by_quadrant_marca_inavaliavel_sem_controle_no_quadrante():
    """Um quadrante cujo top-N não tem nenhum controle não tem contrafactual
    estimável (razão tratado/controle indefinida) — sai `avaliavel=False` e
    `gain` é NaN, nunca um número inventado (mesma disciplina de positividade
    de `uplift_eval.calibration_by_bin`).
    """
    holdout = _holdout([
        ("a", "o", 1, 50.0, 0.0),  # persuadable, tratado
        ("b", "o", 1, 60.0, 0.0),  # persuadable, tratado (sem controle no quadrante)
        ("c", "o", 1, 40.0, 0.0),  # sure_thing, tratado
        ("d", "o", 0, 10.0, 0.0),  # sure_thing, controle
    ])
    stages = _stages([2 * EPS, 2 * EPS, 0.0, 0.0])
    p_convert = _p_convert([0.5, 0.5, 0.9, 0.9])
    ranking = np.array([0, 1, 2, 3])

    resultado = gain_by_quadrant_at_budget(ranking, holdout, stages, p_convert, cfg, budget=4)
    por_quadrante = resultado.set_index("quadrante")

    assert not por_quadrante.loc[PERSUADABLE, "avaliavel"]
    assert np.isnan(por_quadrante.loc[PERSUADABLE, "gain"])
    assert por_quadrante.loc[SURE_THING, "avaliavel"]
    assert por_quadrante.loc[SURE_THING, "gain"] == pytest.approx(40.0 - 10.0)


def test_gain_by_quadrant_traz_tau_medio_mesmo_quando_inavaliavel():
    """`tau_medio` não depende de haver contrafactual estimável (`avaliavel`):
    o quadrante sem controle ainda tem τ previsto pelo X-learner para as linhas
    que tem, e essa média deve aparecer mesmo com `gain=NaN` — é o número que
    explica *por que* aquele gain não pôde ser calculado, não outro artefato
    do mesmo problema.
    """
    holdout = _holdout([
        ("a", "o", 1, 50.0, 0.0),
        ("b", "o", 1, 60.0, 0.0),
    ])
    stages = _stages([2 * EPS, 4 * EPS])
    p_convert = _p_convert([0.5, 0.5])
    ranking = np.array([0, 1])

    resultado = gain_by_quadrant_at_budget(ranking, holdout, stages, p_convert, cfg, budget=2)
    linha = resultado.set_index("quadrante").loc[PERSUADABLE]

    assert not linha["avaliavel"]
    assert np.isnan(linha["gain"])
    assert linha["tau_medio"] == pytest.approx(3 * EPS)


def test_gain_by_quadrant_soma_bate_com_gain_agregado_quando_um_so_quadrante():
    """Sanidade: com um único quadrante no top-N, o gain por quadrante deve
    coincidir com o contrafactual escalado do prefixo inteiro (mesma fórmula,
    só reaplicada num subconjunto que é o conjunto inteiro).
    """
    from src.gaincurve import incremental_gain_curve

    holdout = _holdout([
        ("a", "o", 1, 100.0, 0.0),
        ("b", "o", 0, 20.0, 0.0),
        ("c", "o", 1, 80.0, 0.0),
        ("d", "o", 0, 10.0, 0.0),
    ]).assign(converted=1)
    stages = _stages([2 * EPS] * 4)  # todo mundo persuadable
    p_convert = _p_convert([0.5] * 4)
    ranking = np.array([0, 1, 2, 3])

    gain_por_quadrante = gain_by_quadrant_at_budget(ranking, holdout, stages, p_convert, cfg, budget=4)
    gain_agregado = incremental_gain_curve(ranking, holdout.assign(
        net_profit_realized=holdout["conversion_value"] - holdout["reward_cost"]
    ))

    assert gain_por_quadrante.set_index("quadrante").loc[PERSUADABLE, "gain"] == pytest.approx(
        gain_agregado["gain"].iloc[-1]
    )


def test_left_on_table_conta_persuadable_que_a_escolhida_ignora():
    """3 clientes persuadable (índices 0,1,2); a referência os colocaria todos
    no budget=3. A estratégia escolhida prioriza o sure_thing (índice 3) e só
    um persuadable (índice 0) entra no seu top-2 — os outros dois (1 e 2)
    ficam de fora do budget=3.
    """
    stages = _stages([2 * EPS, 2 * EPS, 2 * EPS, 0.0])  # 3 persuadable, 1 sure_thing
    p_convert = _p_convert([0.5, 0.5, 0.5, 0.9])
    referencia = np.array([0, 1, 2, 3])
    escolhida = np.array([3, 0])  # sure_thing e um único persuadable

    resumo = left_on_table(referencia, escolhida, stages, p_convert, cfg, budget=3)

    assert resumo["persuadables_do_reference"].iloc[0] == 3
    assert resumo["deixados_de_fora"].iloc[0] == 2
    assert resumo["pct"].iloc[0] == pytest.approx(2 / 3)


def test_left_on_table_pct_nan_quando_referencia_nao_tem_persuadable_no_budget():
    """Sem persuadable no budget da referência, a fração não tem denominador —
    fica NaN, não zero (zero sugeriria "nada ficou de fora", que é uma resposta
    diferente de "não havia o que comparar").
    """
    stages = _stages([0.0, -2 * EPS])  # sure_thing, sleeping_dog
    p_convert = _p_convert([0.9, 0.5])
    referencia = np.array([0, 1])
    escolhida = np.array([1, 0])

    resumo = left_on_table(referencia, escolhida, stages, p_convert, cfg, budget=2)

    assert resumo["persuadables_do_reference"].iloc[0] == 0
    assert np.isnan(resumo["pct"].iloc[0])
