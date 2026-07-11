"""Política de alocação sensível a custo (REQ-204) e baselines (REQ-205).

Escolhe, por cliente, a oferta que maximiza `uplift_receita − custo_esperado`
entre todas as ofertas **de cupom/promoção recebidas** (`bogo`, `discount`) e a
**ação nula** ("não enviar"), cujo lucro é exatamente zero. Assim "não enviar"
vence sempre que nenhuma oferta tiver ganho esperado positivo — o critério de
REQ-204.

`informational` fica fora do escopo desta política: não é cupom nem promoção,
é uma ação de comunicação sem desconto associado, e decidir alocação desse tipo
de ação é um estudo à parte. `offer_economics` filtra o tipo na entrada, então
`allocate` e os três baselines nunca o veem como candidato.

Três decisões de modelagem de lucro, cada uma imposta pelo dado, não escolhida:

1. **Custo é por `offer_id`, não por `offer_type`.** `cost.add_reward_cost`
   debita o `discount_value` do catálogo daquela oferta; duas ofertas do mesmo
   tipo têm descontos diferentes (2, 3, 5 e 10 no dado real). O `plan.md` fala
   em "custos por tipo na config", mas `config.yaml` já registra por que não há
   tal parâmetro: o custo é o `discount_value` do próprio catálogo.

2. **O desconto só é pago se o cliente converter.** O custo esperado de enviar
   é `p_convert_tratado × discount_value`, não o `discount_value` cheio. Cobrar
   o custo cheio faria "não enviar" ganhar por um artefato de contabilidade.

3. **`uplift` é Δ P(converted)**, um número adimensional. Vira receita
   multiplicando pela receita média por conversão da oferta
   (`revenue_per_conversion`), medida no conjunto de referência.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import PipelineConfig

NO_SEND = "nao_enviar"

#: Lucro da ação nula. Não enviar não gera receita nem custa desconto.
NO_SEND_NET_PROFIT = 0.0

#: Tipos de oferta que a política pode alocar. `informational` é ação de
#: comunicação, não cupom/promoção — fora do escopo desta política; se algum
#: dia entrar, é um estudo à parte, não uma extensão silenciosa desta lista.
ELIGIBLE_OFFER_TYPES = ("bogo", "discount")


def _only_eligible(reference: pd.DataFrame) -> pd.DataFrame:
    """Restringe o conjunto de referência aos tipos de oferta elegíveis (REQ-204)."""
    return reference[reference["offer_type"].isin(ELIGIBLE_OFFER_TYPES)]


class PolicyRecommendation(BaseModel):
    """Recomendação por cliente, validada antes de virar tabela/artefato MLflow.

    Pydantic nas bordas (Premissa 7): valida a saída da política, nunca linha a
    linha dentro do caminho quente.
    """

    account_id: str
    chosen_action: str = Field(description=f"`offer_id` escolhido, ou {NO_SEND!r}")
    # Sem piso em zero: `allocate` nunca produz negativo (a ação nula domina),
    # mas `policy_send_all` produz — enviar a todos é obrigado a mandar oferta
    # com prejuízo, e esconder isso apagaria justamente o que o baseline mostra.
    expected_net_profit: float = Field(description=f"Zero na ação nula ({NO_SEND!r}); negativo em baseline que envia à força")


def offer_economics(reference: pd.DataFrame) -> pd.DataFrame:
    """Receita por conversão e custo do desconto, por `offer_id`, medidos no dado.

    `revenue_per_conversion` é a média de `conversion_value` **entre as
    conversões** daquela oferta — dividir pelo total de recebimentos misturaria
    o preço do ticket com a taxa de conversão, que o uplift já modela.

    `discount_value` sai de `reward_cost`: `cost.add_reward_cost` grava lá o
    valor do catálogo nas conversões pagas. Uma oferta sem nenhuma conversão no
    conjunto de referência não tem receita observável; fica com receita 0 e
    nunca é escolhida — inavaliável, não lucrativa por omissão.

    Filtra a `ELIGIBLE_OFFER_TYPES` antes de tudo: `informational` não entra na
    economia da política, então nem aparece como candidato mais adiante.
    """
    reference = _only_eligible(reference)
    convertidas = reference[reference["converted"] == 1]

    receita = (
        convertidas.groupby("offer_id")["conversion_value"].mean().rename("revenue_per_conversion")
    )
    # O desconto é constante por oferta; `max` sobre as conversões pagas o recupera
    # (nas não-conversões `reward_cost` é 0 por construção, e diluiria a média).
    desconto = convertidas.groupby("offer_id")["reward_cost"].max().rename("discount_value")

    ofertas = reference[["offer_id", "offer_type"]].drop_duplicates().set_index("offer_id")
    return (
        ofertas.join(receita).join(desconto).fillna({"revenue_per_conversion": 0.0, "discount_value": 0.0}).reset_index()
    )


def expected_net_profit(
    uplift: pd.DataFrame, economics: pd.DataFrame, p_convert_treated: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Lucro líquido esperado de enviar cada oferta a cada cliente.

        lucro = uplift · receita_por_conversao − P(converte | tratado) · desconto

    O primeiro termo é **incremental** (só a conversão que a oferta causou); o
    segundo é **total**, porque o desconto é debitado em toda conversão da
    oferta, tenha ela sido causada ou não. Essa assimetria é o coração de
    REQ-204: uma oferta cara com uplift pequeno paga desconto para clientes que
    já iriam converter, e perde para a ação nula.

    `p_convert_treated` está alinhado posicionalmente a `uplift`, então o
    filtro de `ELIGIBLE_OFFER_TYPES` (`informational` fora) precisa acontecer
    **antes** de zipar os dois — nunca depois, ou o alinhamento quebra.
    """
    df = uplift.copy()
    df["p_convert_treated"] = np.asarray(p_convert_treated, dtype=float)
    df = _only_eligible(df)

    df = df.merge(economics, on=["offer_id", "offer_type"], how="left", validate="many_to_one")

    # Oferta ausente do conjunto de referência não tem economia observável. Deixar o
    # NaN correr faria `allocate` decidir por acidente (NaN perde toda comparação);
    # zerar a receita é a leitura explícita: sem evidência, a oferta não é escolhida.
    faltando = df["revenue_per_conversion"].isna()
    if faltando.any():
        df.loc[faltando, ["revenue_per_conversion", "discount_value"]] = 0.0

    df["expected_revenue"] = df["uplift"] * df["revenue_per_conversion"]
    df["expected_cost"] = df["p_convert_treated"] * df["discount_value"]
    df["net_profit"] = df["expected_revenue"] - df["expected_cost"]
    return df


def allocate(scored: pd.DataFrame) -> pd.DataFrame:
    """`argmax(net_profit)` por cliente, com a ação nula concorrendo em pé de igualdade.

    Espera a saída de `expected_net_profit`. Desempate estável por `offer_id`
    para que a política seja determinística dada a mesma entrada (o `max` do
    pandas não promete ordem entre empates).

    Retorna `[account_id, chosen_action, expected_net_profit]`, uma linha por
    cliente. Cliente cujo melhor lucro é ≤ 0 recebe `nao_enviar` e lucro 0.
    """
    ordenado = scored.sort_values(["account_id", "net_profit", "offer_id"], ascending=[True, False, True])
    melhor = ordenado.groupby("account_id", as_index=False).first()

    envia = melhor["net_profit"] > NO_SEND_NET_PROFIT
    return pd.DataFrame({
        "account_id": melhor["account_id"],
        "chosen_action": np.where(envia, melhor["offer_id"], NO_SEND),
        "expected_net_profit": np.where(envia, melhor["net_profit"], NO_SEND_NET_PROFIT),
    })


def validate_recommendations(policy: pd.DataFrame) -> list[PolicyRecommendation]:
    """Valida a saída da política contra o contrato tipado (REQ-204, Premissa 7)."""
    return [PolicyRecommendation(**row) for row in policy.to_dict(orient="records")]


# --- Baselines de política (REQ-205) -------------------------------------------
#
# Três políticas de comparação, todas no mesmo contrato de saída de `allocate`,
# para que a avaliação offline (T-208) não precise saber qual gerou qual tabela.


def policy_random(reference: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Aleatória: uma oferta uniforme entre as que o cliente recebeu. Seed da config.

    Só entre `ELIGIBLE_OFFER_TYPES` — o mesmo universo que `allocate` disputa,
    para que a comparação de REQ-206 seja justa.
    """
    reference = _only_eligible(reference)
    rng = np.random.default_rng(cfg.seed)
    escolhas = (
        reference[["account_id", "offer_id"]]
        .groupby("account_id")["offer_id"]
        .apply(lambda s: s.iloc[rng.integers(len(s))])
        .rename("chosen_action")
        .reset_index()
    )
    escolhas["expected_net_profit"] = NO_SEND_NET_PROFIT
    return escolhas


def policy_send_all(scored: pd.DataFrame) -> pd.DataFrame:
    """Enviar-a-todos: o status quo que gerou os dados. Nunca escolhe a ação nula.

    Entre as ofertas do cliente, manda a de maior lucro esperado — o status quo
    envia *alguma* oferta a todos, e a versão mais forte dele não é escolher ao
    acaso. Diferente de `allocate` só em não admitir `nao_enviar`: quando toda
    oferta do cliente dá prejuízo, este baseline manda a menos ruim e **carrega
    o lucro negativo**, que é exatamente o custo do status quo que a política de
    uplift pretende evitar.
    """
    ordenado = scored.sort_values(["account_id", "net_profit", "offer_id"], ascending=[True, False, True])
    melhor = ordenado.groupby("account_id", as_index=False).first()
    return pd.DataFrame({
        "account_id": melhor["account_id"],
        "chosen_action": melhor["offer_id"],
        "expected_net_profit": melhor["net_profit"],
    })


def policy_top_completion(reference: pd.DataFrame, p_convert: pd.Series | np.ndarray) -> pd.DataFrame:
    """Top-completion: aloca pela probabilidade prevista de completar (REQ-205).

    É o baseline que a política de uplift precisa **bater**: ordenar por μ₁
    (propensão a converter) ignora se a oferta causou a conversão. `p_convert`
    vem do baseline preditivo (`model_baseline`), não do X-learner.

    `p_convert` está alinhado posicionalmente a `reference`, então o filtro de
    `ELIGIBLE_OFFER_TYPES` acontece **depois** de zipar os dois — nunca antes,
    ou o alinhamento quebra.
    """
    df = reference[["account_id", "offer_id", "offer_type"]].copy()
    df["p_convert"] = np.asarray(p_convert, dtype=float)
    df = _only_eligible(df)

    ordenado = df.sort_values(["account_id", "p_convert", "offer_id"], ascending=[True, False, True])
    melhor = ordenado.groupby("account_id", as_index=False).first()
    return pd.DataFrame({
        "account_id": melhor["account_id"],
        "chosen_action": melhor["offer_id"],
        "expected_net_profit": NO_SEND_NET_PROFIT,
    })
