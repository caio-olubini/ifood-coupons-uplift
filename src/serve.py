"""Caminho de serving do `model predict`: pontuar clientes × ofertas ativas.

`model predict` **não** pontua a base histórica — isso não é produto. Ele recebe
uma base de clientes, um catálogo de ofertas ativas e o histórico de eventos, e
responde "a quem mandar qual oferta, dado um budget de N ações". Este módulo
constrói a matriz de scoring desse cenário reusando as **mesmas** funções puras
do pipeline (`clean.normalize_profile`, `features.build`) — nenhuma feature é
reimplementada aqui.

A ideia central: as features `hist_*` de uma linha filtram `event_time <
received_time` *antes* de agregar (anti-leakage G2). Se eu montar o grão de
scoring como o produto (clientes × ofertas ativas) com `received_time =
t_decision`, o mesmo filtro passa a significar "todo o histórico até o instante
da decisão" — exatamente as features as-of a decisão, sem fabricar nada e sem
leakage. É o pipeline de treino avaliado num instante futuro, não um caminho
paralelo.

O grão de scoring **não** tem label (`converted`/`conversion_value`/`reward_cost`/
`is_recurrent`) nem `campaign_wave` — no momento da decisão nada disso ocorreu.
Essas colunas entram como sentinela só para o contrato fechar (`enforce_schema`);
a matriz de design do modelo as exclui por construção (`NON_FEATURE_COLUMNS`),
então nunca alimentam o score. `treatment=1` é a suposição de serving: estamos
decidindo **expor** a oferta.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import contract
from src.clean import normalize_profile
from src.cost import add_reward_cost
from src.features import build
from src.io import parse_events, read_offers, read_profile
from src.config import PipelineConfig

#: Colunas do contrato que não existem no instante da decisão (label + onda), com
#: o valor sentinela de cada uma. Entram só para `enforce_schema` fechar; a matriz
#: de design do modelo as ignora (todas estão em `model_baseline.NON_FEATURE_COLUMNS`),
#: então nunca alimentam o score. Os literais são construídos na função (não aqui),
#: porque `F.lit` exige um SparkContext ativo — inexistente no import do módulo.
_SENTINEL_LABEL_VALUES = {
    "converted": 0,
    "conversion_value": 0.0,
    "is_recurrent": 0,
    "campaign_wave": 0,
}


def _scoring_grain(
    profile: DataFrame, active_offers: DataFrame, decision_time: float
) -> DataFrame:
    """Grão de scoring = produto (clientes × ofertas ativas) em `decision_time`.

    Uma linha por par (cliente, oferta ativa), todas com o mesmo
    `received_time = decision_time` — é o cenário "se eu decidir agora, para cada
    cliente e cada oferta candidata". `crossJoin` é intencional: a decisão é sobre
    toda combinação, e o catálogo ativo é pequeno (dezenas de ofertas).
    """
    clients = profile.select("account_id")
    offer_ids = active_offers.select(F.col("id").alias("offer_id"))
    return clients.crossJoin(offer_ids).withColumn(
        "received_time", F.lit(float(decision_time))
    )


def build_scoring_frame(
    spark: SparkSession,
    cfg: PipelineConfig,
    decision_time: float,
    active_offer_ids: list[str] | None = None,
) -> DataFrame:
    """Monta a matriz de scoring (clientes × ofertas ativas) no contrato.

    Lê os brutos (`cfg.raw_dir`), normaliza o perfil, monta o grão de decisão em
    `decision_time`, computa as features com as mesmas funções do pipeline
    (as-of a decisão, sem leakage) e projeta no contrato. `active_offer_ids`
    restringe o catálogo às ofertas de fato ativas; `None` usa o catálogo inteiro.

    O DataFrame de saída obedece `PROCESSED_SCHEMA`, então o `.toPandas()` dele
    entra direto na matriz de design do modelo — as colunas de label são
    sentinela (ver docstring do módulo) e o modelo não as vê.
    """
    events = parse_events(spark, cfg)
    offers = read_offers(spark, cfg)
    profile = normalize_profile(read_profile(spark, cfg), cfg)

    active_offers = offers
    if active_offer_ids is not None:
        active_offers = offers.filter(F.col("id").isin(list(active_offer_ids)))

    base = _scoring_grain(profile, active_offers, decision_time)

    featured = build(events, base, offers, cfg)
    enriched = featured.join(profile, on="account_id", how="left")

    with_sentinels = enriched.withColumn("treatment", F.lit(1))
    for col, value in _SENTINEL_LABEL_VALUES.items():
        with_sentinels = with_sentinels.withColumn(col, F.lit(value))

    # `reward_cost` sai de `add_reward_cost` (0 aqui, pois `converted=0` sentinela),
    # mantendo a coluna coerente com o contrato sem reimplementar a regra.
    priced = add_reward_cost(with_sentinels, offers, cfg)

    return contract.enforce_schema(priced)


#: Colunas da recomendação final, na ordem de saída. `rank` é a prioridade de
#: envio (1 = primeiro do budget); `score` é o valor que ordenou.
RECOMMENDATION_COLUMNS = ["rank", "account_id", "offer_id", "offer_type", "score"]


def recommend(scored: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Seleciona as `budget` ações de maior score, uma oferta por cliente.

    `scored` é a matriz de scoring (`build_scoring_frame().toPandas()`) com uma
    coluna `score` já anexada — uma linha por par (cliente, oferta). A seleção é
    em dois passos, refletindo a restrição "uma oferta por cliente":

    1. **Melhor oferta por cliente**: entre as linhas de cada `account_id`, fica
       a de maior `score` (a oferta que mais move aquele cliente). Desempate
       estável pela ordem de entrada, para ser determinístico.
    2. **Top-N por budget**: esses melhores são ordenados por `score` decrescente
       e cortados nos primeiros `budget` — N ações, N clientes distintos.

    Retorna `RECOMMENDATION_COLUMNS` já ordenado por `rank` (1-based).
    """
    best_per_client = (
        scored.sort_values("score", ascending=False, kind="stable")
        .groupby("account_id", sort=False)
        .head(1)
    )
    top = best_per_client.sort_values("score", ascending=False, kind="stable").head(budget)
    return (
        top.assign(rank=range(1, len(top) + 1))
        [RECOMMENDATION_COLUMNS]
        .reset_index(drop=True)
    )
