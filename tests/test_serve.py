"""Serving do `model predict`: seleção de recomendações e propensity de serve.

Testes estruturais dos invariantes que quebram o `predict` em silêncio: a
restrição "uma oferta por cliente", o corte por budget top-N, a escolha da melhor
oferta por cliente, e o guard de propensity que impede o X-learner de rejeitar o
grão de serve (todo `treatment=1`). A montagem da matriz de scoring
(`build_scoring_frame`) é Spark/IO e é verificada rodando o CLI, não aqui.
"""

import numpy as np
import pandas as pd

from src import serve
from src.uplift import fixed_propensity


def _scored(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """DataFrame de scoring mínimo: (account_id, offer_id, offer_type, score)."""
    return pd.DataFrame(rows, columns=["account_id", "offer_id", "offer_type", "score"])


def test_recommend_da_uma_oferta_por_cliente():
    """A restrição atual: cada cliente recebe no máximo uma oferta, mesmo tendo
    várias linhas candidatas no grão de scoring.
    """
    scored = _scored([
        ("a", "o1", "discount", 0.9),
        ("a", "o2", "bogo", 0.5),  # mesma conta, oferta pior — não deve entrar
        ("b", "o1", "discount", 0.8),
    ])
    recs = serve.recommend(scored, budget=10)
    assert recs["account_id"].nunique() == len(recs)
    assert recs.groupby("account_id").size().max() == 1


def test_recommend_escolhe_a_melhor_oferta_de_cada_cliente():
    """Entre as ofertas de um cliente, fica a de maior score — a oferta que mais
    move aquele cliente, não uma qualquer.
    """
    scored = _scored([
        ("a", "o1", "discount", 0.3),
        ("a", "o2", "bogo", 0.9),  # esta é a melhor de 'a'
    ])
    recs = serve.recommend(scored, budget=10)
    assert recs.loc[recs["account_id"] == "a", "offer_id"].iloc[0] == "o2"


def test_recommend_respeita_o_budget_top_n():
    """O budget corta o nº de ações (= nº de clientes, uma oferta cada), pegando
    os de maior score.
    """
    scored = _scored([
        ("a", "o1", "discount", 0.9),
        ("b", "o1", "discount", 0.8),
        ("c", "o1", "discount", 0.7),
    ])
    recs = serve.recommend(scored, budget=2)
    assert len(recs) == 2
    assert list(recs["account_id"]) == ["a", "b"]  # os dois maiores scores
    assert list(recs["rank"]) == [1, 2]  # rank 1-based, ordenado por score


def test_fixed_propensity_de_serve_nao_degenera_em_um():
    """Grão de serve (todo `treatment=1`) daria `mean()=1,0`, que o CausalML
    rejeita (p ∈ (0,1) aberto); o guard cai para 0,5 nesse caso.
    """
    serve_treatment = np.ones(50, dtype=int)
    p = fixed_propensity(serve_treatment)
    assert np.all((p > 0.0) & (p < 1.0))
    assert np.allclose(p, 0.5)


def test_fixed_propensity_com_os_dois_bracos_e_a_media_intacta():
    """Nos casos reais (ajuste/holdout, dois braços presentes), o guard não toca
    o número — `mean()` fica byte a byte igual ao de antes.
    """
    treatment = np.array([1, 1, 1, 0], dtype=int)
    p = fixed_propensity(treatment)
    assert np.allclose(p, 0.75)
