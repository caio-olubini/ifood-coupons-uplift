"""Visões da EDA e balanço de covariáveis (REQ-108, REQ-109).

As figuras não são testadas pixel a pixel; a **lógica** que as alimenta é — é ela
que pode mentir em silêncio (uma taxa dividida pelo denominador errado, um SMD
que vira NaN e some da tabela de diagnóstico).
"""

import math

import numpy as np
import pytest

from src.attribution import attribute
from src.config import load
from src.eda import (
    assign_segments,
    assignment_balance,
    campaign_waves,
    choose_k,
    client_features,
    cluster_matrix,
    cluster_scan,
    completed_unseen_by_type,
    conversion_by_type_and_segment,
    correlation_matrix,
    covariate_balance,
    fit_clusters,
    identity_null_overlap,
    naive_spend_lift,
    numeric_histogram,
    numeric_profile,
    positivity_by_offer_type,
    redundant_pairs,
    response_funnel,
    sanity_checks,
    segment_response,
    unattributable_transaction_share,
    window_spend,
)


def _processed(spark, linhas):
    return spark.createDataFrame(linhas, schema=(
        "account_id string, treatment int, age int, credit_card_limit double, "
        "tenure_days int, identity_missing int, gender string"))


def _evento(event, account_id, offer_ref, time):
    return (event, account_id, offer_ref, float(time))


def _events(spark, linhas):
    return spark.createDataFrame(linhas, schema="event string, account_id string, offer_ref string, time double")


def _offers(spark, pares):
    return spark.createDataFrame(pares, schema="id string, offer_type string")


# --- REQ-109: SMD --------------------------------------------------------------

def test_smd_matches_the_cohen_formula_on_known_values(spark):
    # tratado age=[1,3] (média 2, var 2); controle age=[3,5] (média 4, var 2).
    # pooled = sqrt((2+2)/2) = sqrt(2) → SMD = (2-4)/sqrt(2) = -1.41421…
    linhas = [
        ("a", 1, 1, 10.0, 100, 0, "M"), ("b", 1, 3, 10.0, 100, 0, "M"),
        ("c", 0, 3, 10.0, 100, 0, "M"), ("d", 0, 5, 10.0, 100, 0, "M"),
    ]
    cfg = load()
    frame = covariate_balance(_processed(spark, linhas), cfg).set_index("covariavel")

    assert frame.loc["age", "smd"] == pytest.approx(-2 / math.sqrt(2), rel=1e-6)
    assert frame.loc["age", "acima_do_limiar"]  # |−1.41| > 0.1


def test_zero_variance_covariate_yields_smd_zero_not_nan(spark):
    # tenure_days é constante nos dois grupos: pooled=0. Um NaN aqui sumiria da
    # tabela de diagnóstico em silêncio; o contrato é devolver 0.0.
    linhas = [
        ("a", 1, 20, 10.0, 100, 0, "M"), ("b", 1, 30, 10.0, 100, 0, "M"),
        ("c", 0, 20, 10.0, 100, 0, "M"), ("d", 0, 30, 10.0, 100, 0, "M"),
    ]
    frame = covariate_balance(_processed(spark, linhas), load()).set_index("covariavel")

    assert frame.loc["tenure_days", "smd"] == 0.0
    assert not np.isnan(frame.loc["tenure_days", "smd"])
    assert not frame.loc["tenure_days", "acima_do_limiar"]


def test_gender_enters_the_balance_as_one_indicator_per_level(spark):
    linhas = [
        ("a", 1, 20, 10.0, 100, 0, "F"), ("b", 1, 30, 10.0, 100, 0, "F"),
        ("c", 0, 20, 10.0, 100, 0, "M"), ("d", 0, 30, 10.0, 100, 0, "M"),
    ]
    frame = covariate_balance(_processed(spark, linhas), load())
    covariaveis = set(frame["covariavel"])

    assert {"gender=F", "gender=M"} <= covariaveis
    # Tratado é 100% F e controle 100% M: a covariável separa os grupos por completo.
    # Variância nula com médias distintas ⇒ |SMD| infinito, jamais 0.0 (que diria
    # "balanceado" no pior desbalanço possível).
    indicador = frame.set_index("covariavel").loc["gender=F"]
    assert np.isinf(indicador["smd"])
    assert indicador["acima_do_limiar"]


def test_assignment_balance_reports_worst_pair_over_received_offers(spark):
    # Verifica a Premissa 4 (envio aleatório), não viu/não-viu. offB recebe só
    # clientes idosos, offA só jovens: o par está maximamente desbalanceado em age.
    linhas = [
        ("a", 1, 20, 10.0, 100, 0, "M", "offA"), ("b", 1, 22, 10.0, 100, 0, "M", "offA"),
        ("c", 0, 70, 10.0, 100, 0, "M", "offB"), ("d", 0, 72, 10.0, 100, 0, "M", "offB"),
    ]
    df = spark.createDataFrame(linhas, schema=(
        "account_id string, treatment int, age int, credit_card_limit double, "
        "tenure_days int, identity_missing int, gender string, offer_id string"))
    frame = assignment_balance(df, load()).set_index("covariavel")
    assert frame.loc["age", "acima_do_limiar"]
    assert frame.loc["age", "pior_abs_smd"] > 1.0


def test_null_age_is_ignored_in_the_mean_not_treated_as_zero(spark):
    # Se o nulo virasse 0, a média do tratado despencaria e o SMD mentiria.
    linhas = [
        ("a", 1, None, 10.0, 100, 1, "unknown"), ("b", 1, 40, 10.0, 100, 0, "M"),
        ("c", 0, 40, 10.0, 100, 0, "M"), ("d", 0, 40, 10.0, 100, 0, "M"),
    ]
    frame = covariate_balance(_processed(spark, linhas), load()).set_index("covariavel")
    assert frame.loc["age", "media_tratado"] == 40.0  # só a linha não-nula


# --- REQ-108: completou sem ver -----------------------------------------------

def test_completed_unseen_rate_per_offer_type(spark):
    events = _events(spark, [
        _evento("offer completed", "acc1", "b", 5),          # bogo, sem view → sem_view
        _evento("offer viewed", "acc1", "d", 1),
        _evento("offer completed", "acc1", "d", 3),          # discount, com view → visto
    ])
    offers = _offers(spark, [("b", "bogo"), ("d", "discount"), ("i", "informational")])
    frame = completed_unseen_by_type(events, offers).set_index("offer_type")

    assert frame.loc["bogo", "taxa_sem_view"] == 1.0
    assert frame.loc["discount", "taxa_sem_view"] == 0.0


def test_informational_appears_with_zero_completed_and_null_rate(spark):
    # G5 em forma de visão: informational não emite `offer completed`. A linha
    # precisa existir (senão o tipo some do relatório) com taxa nula, não 0.0 —
    # zero diria "completou sempre vendo", o que é falso: não completou nunca.
    events = _events(spark, [_evento("offer completed", "acc1", "b", 5)])
    offers = _offers(spark, [("b", "bogo"), ("i", "informational")])
    frame = completed_unseen_by_type(events, offers).set_index("offer_type")

    assert frame.loc["informational", "completados"] == 0
    assert np.isnan(frame.loc["informational", "taxa_sem_view"])


def test_view_after_completion_does_not_count_as_seen(spark):
    # Viu DEPOIS de completar: a compra não foi induzida pela visualização.
    events = _events(spark, [
        _evento("offer completed", "acc1", "b", 3),
        _evento("offer viewed", "acc1", "b", 7),
    ])
    offers = _offers(spark, [("b", "bogo")])
    frame = completed_unseen_by_type(events, offers).set_index("offer_type")
    assert frame.loc["bogo", "taxa_sem_view"] == 1.0


# --- REQ-108: sentinela e histograma -------------------------------------------

def test_identity_null_overlap_counts_the_intersection(spark):
    cfg = load()
    perfil = spark.createDataFrame(
        [(118, None, None), (118, None, None), (40, "M", 1000.0)],
        schema="age int, gender string, credit_card_limit double")
    frame = identity_null_overlap(perfil, cfg).set_index("conjunto")

    assert frame.loc["os três, juntos", "clientes"] == 2
    assert frame.loc["ao menos um", "clientes"] == 2  # sobreposição perfeita
    assert frame.loc["os três, juntos", "fracao"] == pytest.approx(2 / 3)


def test_numeric_histogram_excludes_nulls_and_conserves_the_count(spark):
    df = spark.createDataFrame([(1.0,), (2.0,), (3.0,), (None,)], schema="x double")
    h = numeric_histogram(df, "x", bins=3)
    assert h["contagem"].sum() == 3  # o nulo não entra nem vira bucket


def test_numeric_histogram_handles_a_constant_column(spark):
    # min == max: largura de bucket seria 0 → divisão por zero se não tratado.
    df = spark.createDataFrame([(5.0,), (5.0,)], schema="x double")
    h = numeric_histogram(df, "x", bins=10)
    assert len(h) == 1
    assert h["contagem"].iloc[0] == 2


def test_campaign_waves_reports_view_rate_per_wave(spark):
    processed = spark.createDataFrame(
        [(0, 0.0, 1, 1), (0, 0.0, 0, 0), (1, 7.0, 1, 0)],
        schema="campaign_wave int, received_time double, treatment int, converted int")
    frame = campaign_waves(processed).set_index("campaign_wave")

    assert frame.loc[0, "recebimentos"] == 2
    assert frame.loc[0, "taxa_view"] == pytest.approx(0.5)
    assert frame.loc[1, "conversoes"] == 0


# --- Ato 3: compra fora de qualquer janela de oferta ---------------------------

def test_unattributable_share_counts_transactions_outside_every_window(spark):
    # off1 dura 7 dias a partir de t=0 (janela [0,7]); a compra em t=2 cai dentro,
    # a de t=20 não cai em NENHUMA janela — é espontânea por definição mais ampla
    # que a do label (não depende de view).
    events = spark.createDataFrame(
        [
            ("offer received", "acc1", "off1", 0.0, None),
            ("transaction", "acc1", None, 2.0, 20.0),
            ("transaction", "acc1", None, 20.0, 30.0),
        ],
        schema="event string, account_id string, offer_ref string, time double, amount double",
    )
    offers = spark.createDataFrame(
        [("off1", "bogo", 7.0, 10.0)],
        schema="id string, offer_type string, duration double, min_value double")
    attributed = attribute(events, offers, load())
    frame = unattributable_transaction_share(events, attributed).set_index("grupo")

    assert frame.loc["fora de qualquer janela de oferta", "transacoes"] == 1
    assert frame.loc["dentro de alguma janela", "transacoes"] == 1
    assert frame.loc["fora de qualquer janela de oferta", "fracao"] == pytest.approx(0.5)


# --- Ato 4: positividade --------------------------------------------------------

def test_positivity_counts_clients_who_never_received_a_type(spark):
    processed = spark.createDataFrame(
        [("a", "bogo"), ("b", "bogo"), ("a", "discount")],
        schema="account_id string, offer_type string")
    profile = spark.createDataFrame([("a",), ("b",), ("c",)], schema="account_id string")
    frame = positivity_by_offer_type(processed, profile).set_index("offer_type")

    assert frame.loc["bogo", "nunca_receberam"] == 1     # só "c" nunca recebeu bogo
    assert frame.loc["discount", "nunca_receberam"] == 2  # "b" e "c" nunca receberam discount
    assert frame.loc["discount", "clientes_total"] == 3


# --- Ato 5: heterogeneidade -----------------------------------------------------

def test_conversion_by_segment_splits_by_tenure_quartile(spark):
    # 8 clientes com tenure uniformemente espaçado formam quartis limpos e
    # previsíveis; os dois primeiros (menor tenure) nunca convertem, os dois
    # últimos (maior tenure) sempre convertem — o contraste que a visão existe
    # para revelar.
    contas = [f"c{i}" for i in range(8)]
    processed = spark.createDataFrame(
        [(c, "bogo", 1, 1 if i >= 6 else 0) for i, c in enumerate(contas)],
        schema="account_id string, offer_type string, treatment int, converted int")
    profile = spark.createDataFrame(
        [(c, (i + 1) * 100) for i, c in enumerate(contas)],
        schema="account_id string, tenure_days int")
    frame = conversion_by_type_and_segment(processed, profile, load())

    q1 = frame[frame["tenure_q"] == "Q1 (mais novo)"]
    q4 = frame[frame["tenure_q"] == "Q4 (mais antigo)"]
    assert q1["taxa_conversao"].iloc[0] == 0.0
    assert q4["taxa_conversao"].iloc[0] == 1.0


# --- REQ-108: perfil univariado, outliers, correlação, sanidade ----------------

def test_numeric_profile_separates_null_from_zero(spark):
    # Um nulo e um zero são coisas distintas: `frac_nulos` conta sobre o total de
    # linhas, `frac_zeros` sobre as linhas com valor. Trocar os denominadores é o
    # jeito mais fácil de uma tabela descritiva mentir.
    df = spark.createDataFrame([(0.0,), (0.0,), (10.0,), (None,)], schema="x double")
    linha = numeric_profile(df, ["x"], load()).set_index("coluna").loc["x"]

    assert linha["n"] == 3 and linha["nulos"] == 1
    assert linha["frac_nulos"] == pytest.approx(1 / 4)
    assert linha["zeros"] == 2
    assert linha["frac_zeros"] == pytest.approx(2 / 3)   # sobre os 3 não-nulos
    assert linha["media"] == pytest.approx(10 / 3)       # nulo fora da média


def test_numeric_profile_flags_the_tukey_fence_outlier(spark):
    # 1..9 mais um 100. Q1=3, Q3=8 (approxQuantile), IQR=5 → cerca superior 15.5;
    # só o 100 cai fora. O `min`/`max` continuam sendo os valores reais, não os cortes.
    valores = [(float(v),) for v in list(range(1, 10)) + [100]]
    linha = numeric_profile(spark.createDataFrame(valores, schema="x double"), ["x"], load()) \
        .set_index("coluna").loc["x"]

    assert linha["outliers"] == 1
    assert linha["max"] == 100.0
    assert linha["p50"] == pytest.approx(5.0)


def test_numeric_profile_survives_an_all_null_column(spark):
    df = spark.createDataFrame([(None,), (None,)], schema="x double")
    linha = numeric_profile(df, ["x"], load()).set_index("coluna").loc["x"]
    assert linha["n"] == 0 and linha["nulos"] == 2
    assert np.isnan(linha["p50"])      # sem quantil a reportar, e sem quebrar
    assert linha["outliers"] == 0      # sem cerca, ninguém cai fora


def test_correlation_uses_pairwise_deletion_not_zero_fill(spark):
    # y = 2x nas linhas onde y existe. Preencher o nulo de y com 0 arrastaria r
    # para longe de 1; a exclusão par a par preserva a relação perfeita.
    df = spark.createDataFrame(
        [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (100.0, None)], schema="x double, y double")
    corr = correlation_matrix(df, ["x", "y"])
    assert corr.loc["x", "y"] == pytest.approx(1.0)


def test_redundant_pairs_reports_each_pair_once_above_threshold(spark):
    df = spark.createDataFrame(
        [(1.0, 2.0, 5.0), (2.0, 4.0, 1.0), (3.0, 6.0, 9.0), (4.0, 8.0, 2.0)],
        schema="x double, y double, z double")
    pares = redundant_pairs(correlation_matrix(df, ["x", "y", "z"]), load())

    assert len(pares) == 1                       # (x,y) uma vez, não (y,x) também
    assert {pares.loc[0, "feature_a"], pares.loc[0, "feature_b"]} == {"x", "y"}
    assert pares.loc[0, "abs_r"] == pytest.approx(1.0)


def test_sanity_checks_count_the_impossible_rows(spark):
    # Uma linha sã e uma que viola G6 (custo sem conversão) — a tabela precisa
    # achar a segunda e deixar as outras verificações em zero.
    linhas = [
        ("a", "bogo", 1, 1, 30.0, 5.0, 10.0, 2, 50.0, 0.5, 40),
        ("b", "bogo", 1, 0, 0.0, 5.0, 0.0, 0, 0.0, 0.5, 40),
    ]
    processed = spark.createDataFrame(linhas, schema=(
        "account_id string, offer_type string, treatment int, converted int, "
        "conversion_value double, reward_cost double, hist_avg_ticket double, "
        "hist_txn_count int, hist_spend_total double, hist_view_rate double, age int"))
    frame = sanity_checks(processed).set_index("verificação")

    assert frame.loc["reward_cost > 0 sem conversão (viola G6)", "linhas"] == 1
    assert frame.loc["ticket médio histórico sem transação histórica", "linhas"] == 0
    # `converted=1` sem `treatment` não é mais anomalia — não deve nem ser checado.
    assert not any("G3" in v for v in frame.index)


def test_response_funnel_keeps_the_two_denominators_apart(spark):
    # 4 recebidos, 2 vistos, 1 convertido: 25% sobre recebidos, 50% sobre vistos.
    linhas = [("bogo", 1, 1, 20.0, 5.0), ("bogo", 1, 0, 0.0, 0.0),
              ("bogo", 0, 0, 0.0, 0.0), ("bogo", 0, 0, 0.0, 0.0)]
    processed = spark.createDataFrame(linhas, schema=(
        "offer_type string, treatment int, converted int, "
        "conversion_value double, reward_cost double"))
    linha = response_funnel(processed).set_index("offer_type").loc["bogo"]

    assert linha["taxa_conversao"] == pytest.approx(0.25)
    assert linha["taxa_conversao_vistos"] == pytest.approx(0.5)
    assert linha["margem_por_envio"] == pytest.approx((20.0 - 5.0) / 4)


def test_response_funnel_rate_over_viewed_is_null_when_nobody_viewed(spark):
    # Denominador zero: `nan` (não há resposta a medir), jamais 0.0 — que leria
    # "foi exposto e não converteu".
    processed = spark.createDataFrame(
        [("informational", 0, 0, 0.0, 0.0)], schema=(
            "offer_type string, treatment int, converted int, "
            "conversion_value double, reward_cost double"))
    linha = response_funnel(processed).set_index("offer_type").loc["informational"]
    assert np.isnan(linha["taxa_conversao_vistos"])


# --- REQ-111: segmentação por K-Means ------------------------------------------

def _clients(**colunas) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame(colunas)


def _base_clients(n: int, **sobrescreve):
    dados = {
        "account_id": [f"c{i}" for i in range(n)],
        "age": [30.0 + i for i in range(n)],
        "credit_card_limit": [1000.0 * (i + 1) for i in range(n)],
        "tenure_days": [100.0 + i for i in range(n)],
        "spend_total": [10.0 * (i + 1) for i in range(n)],
        "txn_count": [float(i + 1) for i in range(n)],
        "avg_ticket": [10.0 + 0.5 * i for i in range(n)],
        "identity_missing": [0] * n,
    }
    dados.update(sobrescreve)
    return _clients(**dados)


def test_client_features_gives_zero_ticket_to_a_client_who_never_bought(spark):
    # Ausência de compra é o valor 0, não um nulo: o cliente existe e comprou nada.
    # (Ao contrário de `age`, onde nulo é o segmento sentinela.)
    processed = spark.createDataFrame(
        [("a", 40, 1000.0, 200, 0, "M", 1, 0, 0.0, 0.0)], schema=(
            "account_id string, age int, credit_card_limit double, tenure_days int, "
            "identity_missing int, gender string, treatment int, converted int, "
            "conversion_value double, reward_cost double"))
    events = spark.createDataFrame(
        [("transaction", "outro", 1.0, 10.0)], schema="event string, account_id string, time double, amount double")
    linha = client_features(processed, events).set_index("account_id").loc["a"]

    assert linha["spend_total"] == 0.0
    assert linha["txn_count"] == 0
    assert linha["avg_ticket"] == 0.0        # e não NaN nem divisão por zero
    assert linha["view_rate"] == 1.0


def test_cluster_matrix_leaves_the_sentinel_segment_out_of_the_fit(spark):
    # O segmento de identidade ausente não é imputado nem descartado da análise:
    # sai do ajuste (não tem age/limit) e volta como segmento nomeado.
    clientes = _base_clients(4)
    clientes.loc[3, ["age", "credit_card_limit"]] = [np.nan, np.nan]
    clientes.loc[3, "identity_missing"] = 1

    matriz, ids, colunas = cluster_matrix(clientes)
    assert len(matriz) == 3
    assert "c3" not in set(ids)
    assert "identity_missing" not in colunas   # a flag não vira eixo do espaço


def test_cluster_matrix_standardizes_every_column(spark):
    # K-Means minimiza distância euclidiana: sem z-score, `credit_card_limit`
    # (milhares) esmagaria `age` (dezenas) e seria o único eixo real.
    matriz, _, _ = cluster_matrix(_base_clients(8))
    assert matriz.mean(axis=0) == pytest.approx(np.zeros(matriz.shape[1]), abs=1e-9)
    assert matriz.std(axis=0) == pytest.approx(np.ones(matriz.shape[1]), abs=1e-9)


def test_cluster_matrix_log_transforms_the_heavy_tails(spark):
    # `spend_total` entra como log1p antes do z-score; `tenure_days`, cru. Um cliente
    # 100× mais gastador não pode ficar 100× mais longe no espaço.
    from src.eda import CLUSTER_FEATURES
    clientes = _base_clients(5, spend_total=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    matriz, _, colunas = cluster_matrix(clientes)

    coluna = matriz[:, colunas.index("spend_total")]
    esperado = np.log1p(clientes["spend_total"].to_numpy())
    esperado = (esperado - esperado.mean()) / esperado.std(ddof=0)
    assert coluna == pytest.approx(esperado)
    assert "spend_total" in CLUSTER_FEATURES


def test_cluster_matrix_refuses_a_null_in_a_complete_profile_client(spark):
    # G7 promete que só o sentinela tem perfil ausente. Se aparecer outro nulo, o
    # KMeans do sklearn levantaria um erro genérico páginas depois — falha aqui, alto.
    clientes = _base_clients(3)
    clientes.loc[1, "age"] = np.nan          # identity_missing continua 0
    with pytest.raises(ValueError, match="perfil completo"):
        cluster_matrix(clientes)


def test_cluster_matrix_refuses_a_constant_feature(spark):
    clientes = _base_clients(4, tenure_days=[100.0] * 4)
    with pytest.raises(ValueError, match="constante"):
        cluster_matrix(clientes)


def test_fit_clusters_separates_two_obvious_blobs_and_is_deterministic(spark):
    # Dois grupos bem afastados em todas as dimensões: qualquer K-Means correto
    # com k=2 os separa, e a mesma seed devolve o mesmo particionamento.
    n = 6
    clientes = _base_clients(n,
        age=[25.0, 26.0, 27.0, 70.0, 71.0, 72.0],
        credit_card_limit=[500.0, 520.0, 540.0, 9000.0, 9200.0, 9400.0],
        tenure_days=[50.0, 55.0, 60.0, 900.0, 950.0, 1000.0],
        spend_total=[5.0, 6.0, 7.0, 5000.0, 5200.0, 5400.0],
        txn_count=[1.0, 1.0, 2.0, 50.0, 52.0, 54.0],
        avg_ticket=[5.0, 6.0, 3.5, 100.0, 100.0, 100.0])
    matriz, ids, _ = cluster_matrix(clientes)
    cfg = load()

    rotulos = fit_clusters(matriz, ids, 2, cfg).set_index("account_id")["cluster"]
    assert rotulos["c0"] == rotulos["c1"] == rotulos["c2"]
    assert rotulos["c3"] == rotulos["c4"] == rotulos["c5"]
    assert rotulos["c0"] != rotulos["c3"]

    de_novo = fit_clusters(matriz, ids, 2, cfg).set_index("account_id")["cluster"]
    assert (rotulos == de_novo).all()


def test_cluster_scan_covers_the_configured_range_and_picks_the_best_silhouette(spark):
    clientes = _base_clients(30,
        age=list(np.linspace(20, 70, 30)),
        credit_card_limit=list(np.linspace(500, 9000, 30)),
        tenure_days=list(np.linspace(30, 900, 30)),
        spend_total=list(np.linspace(1, 5000, 30)),
        txn_count=list(np.linspace(1, 40, 30)),
        avg_ticket=list(np.linspace(3, 120, 30)))
    cfg = load(cluster_k_min=2, cluster_k_max=4)
    matriz, _, _ = cluster_matrix(clientes)
    scan = cluster_scan(matriz, cfg)

    assert list(scan["k"]) == [2, 3, 4]
    assert scan["inercia"].is_monotonic_decreasing          # mais centróides, menos inércia
    assert scan["silhouette"].between(-1, 1).all()
    assert choose_k(scan) == int(scan.loc[scan["silhouette"].idxmax(), "k"])


def test_assign_segments_names_the_sentinel_instead_of_leaving_it_null(spark):
    from src.eda import MISSING_IDENTITY_SEGMENT
    clientes = _base_clients(3)
    clientes.loc[2, ["age", "credit_card_limit"]] = [np.nan, np.nan]
    clientes.loc[2, "identity_missing"] = 1
    matriz, ids, _ = cluster_matrix(clientes)
    rotulos = fit_clusters(matriz, ids, 2, load())

    segmentos = assign_segments(clientes, rotulos).set_index("account_id")["segmento"]
    assert segmentos["c2"] == MISSING_IDENTITY_SEGMENT
    assert segmentos["c0"].startswith("cluster ")


def test_segment_response_keeps_denominators_apart(spark):
    processed = spark.createDataFrame(
        [("a", "bogo", 1, 1, 30.0, 5.0), ("b", "bogo", 0, 0, 0.0, 0.0)],
        schema=("account_id string, offer_type string, treatment int, converted int, "
                "conversion_value double, reward_cost double"))
    segmentos = spark.createDataFrame(
        [("a", "cluster 0"), ("b", "cluster 0")], schema="account_id string, segmento string")
    linha = segment_response(processed, segmentos).set_index(["segmento", "offer_type"]).loc[("cluster 0", "bogo")]

    assert linha["envios"] == 2
    assert linha["taxa_conversao"] == pytest.approx(0.5)          # sobre envios
    assert linha["taxa_conversao_vistos"] == pytest.approx(1.0)   # sobre vistos
    assert linha["margem_por_envio"] == pytest.approx((30.0 - 5.0) / 2)


def test_window_spend_ignores_the_view_unlike_the_label(spark):
    # Compra em t=2, dentro da janela [0,7], sem nenhum `offer viewed`. O label a
    # descarta (G4); esta régua a conta — é justamente essa a diferença que ela mede.
    events = spark.createDataFrame(
        [("offer received", "acc1", "off1", 0.0, None),
         ("transaction", "acc1", None, 2.0, 20.0),
         ("transaction", "acc1", None, 20.0, 30.0)],
        schema="event string, account_id string, offer_ref string, time double, amount double")
    offers = spark.createDataFrame(
        [("off1", "bogo", 7.0, 10.0)],
        schema="id string, offer_type string, duration double, min_value double")
    attributed = attribute(events, offers, load())

    janela = window_spend(attributed, events).first()
    assert janela["window_spend"] == 20.0   # só a de t=2; a de t=20 caiu fora da janela
    assert janela["window_txns"] == 1


def test_naive_spend_lift_is_the_raw_difference_viewed_minus_unviewed(spark):
    processed = spark.createDataFrame(
        [("a", "o1", 0.0, 1), ("b", "o1", 0.0, 0)],
        schema="account_id string, offer_id string, received_time double, treatment int")
    window = spark.createDataFrame(
        [("a", "o1", 0.0, 30.0, 1), ("b", "o1", 0.0, 10.0, 1)],
        schema="account_id string, offer_id string, received_time double, window_spend double, window_txns int")
    segmentos = spark.createDataFrame(
        [("a", "cluster 0"), ("b", "cluster 0")], schema="account_id string, segmento string")

    linha = naive_spend_lift(processed, window, segmentos).set_index("segmento").loc["cluster 0"]
    assert linha["diferenca_bruta"] == pytest.approx(20.0)
    assert linha["envios_vistos"] == 1 and linha["envios_nao_vistos"] == 1
