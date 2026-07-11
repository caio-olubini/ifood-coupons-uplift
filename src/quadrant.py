"""Classificação de quadrante de uplift e composição por política.

Diagnóstico exploratório, não um requisito formal da spec: quando as
estratégias de alocação divergem no ganho incremental por budget (`gaincurve`),
a pergunta seguinte é "divergem sobre *quem*?". Esta função abre essa
composição: de que tipo de cliente cada política está compondo sua escolha,
usando os quatro quadrantes clássicos de uplift sobre μ₀/μ₁ previstos
(`uplift.predict_stages`):

- **persuadable** (μ₀ baixo, μ₁ alto): converte só se tratado — o alvo ideal,
  τ real positivo.
- **sure thing** (μ₀ alto, μ₁ alto): converte de qualquer forma — enviar
  oferta paga desconto sem causar nada.
- **lost cause** (μ₀ baixo, μ₁ baixo): não converte de qualquer forma —
  enviar não muda o resultado, só teria custo se pagasse desconto.
- **sleeping dog** (μ₀ alto, μ₁ baixo): tratamento **atrapalha** — τ real
  negativo, "não enviar" é estritamente melhor.

O limiar (`cfg.quadrant_probability_threshold`, default 0,5) separa "alto" de
"baixo" em cada estágio — natural para μ₀/μ₁ serem probabilidades previstas de
um outcome binário.
"""

from __future__ import annotations

import pandas as pd

from src.config import PipelineConfig
from src.policy import NO_SEND

PERSUADABLE = "persuadable"
SURE_THING = "sure_thing"
LOST_CAUSE = "lost_cause"
SLEEPING_DOG = "sleeping_dog"


def classify_quadrant(stages: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    """Quadrante de cada linha de `stages` (saída de `uplift.predict_stages`),
    a partir de `mu0`/`mu1` previstos e `cfg.quadrant_probability_threshold`.
    """
    threshold = cfg.quadrant_probability_threshold
    mu0_alto = stages["mu0"] >= threshold
    mu1_alto = stages["mu1"] >= threshold

    quadrante = pd.Series(index=stages.index, dtype=object)
    quadrante[~mu0_alto & mu1_alto] = PERSUADABLE
    quadrante[mu0_alto & mu1_alto] = SURE_THING
    quadrante[~mu0_alto & ~mu1_alto] = LOST_CAUSE
    quadrante[mu0_alto & ~mu1_alto] = SLEEPING_DOG
    return quadrante


def quadrant_distribution(stages: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Distribuição de quadrantes por `offer_type` — visão geral antes de olhar
    por política: o quanto de cada tipo de cliente existe no holdout, período.
    """
    quadrante = classify_quadrant(stages, cfg)
    contagem = (
        stages.assign(quadrante=quadrante)
        .groupby(["offer_type", "quadrante"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_por_tipo = contagem.groupby("offer_type")["n"].transform("sum")
    contagem["pct"] = contagem["n"] / total_por_tipo
    return contagem


def _one_row_per_account_offer(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa `(account_id, offer_id)` a uma linha, de forma determinística.

    `policy.allocate`/`expected_net_profit` já operam no grão reduzido
    `(account_id, offer_id)` — sem `received_time` — porque a política não
    distingue ondas; aqui não está garantido que `received_time` sobreviva até
    `scored`/`stages`, então o desempate é só por ordem de índice, estável mas
    arbitrário. Existe para impedir merge em duplicata silencioso, não para
    escolher "o" recebimento certo — se isso importar, resolva a montante.
    """
    return df.drop_duplicates(subset=["account_id", "offer_id"], keep="first")


def policy_composition(
    policy: pd.DataFrame,
    scored: pd.DataFrame,
    stages: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Composição de uma política: por quadrante e por sinal do lucro esperado.

    `policy` é a saída de `policy.allocate`/baselines (`[account_id,
    chosen_action, expected_net_profit]`); `scored` é a saída de
    `policy.expected_net_profit` (`[account_id, offer_id, offer_type,
    net_profit, ...]`), usada para achar o `net_profit` **da ação
    recomendada**, não do melhor par teórico; `stages` é
    `uplift.predict_stages` (μ₀/μ₁ por `account_id`+`offer_id`).

    `nao_enviar` não casa com nenhuma linha de `scored`/`stages` (não é uma
    oferta) — fica com `quadrante`/`net_profit` nulos, contado à parte por
    `chosen_action == NO_SEND`, nunca herdando o quadrante de uma oferta que a
    política decidiu não mandar.

    Retorna uma linha por combinação `(quadrante, lucro_negativo)` presente,
    mais a linha de `nao_enviar`, com contagem e percentual sobre o total de
    clientes da política.
    """
    quadrante = classify_quadrant(stages, cfg)
    stages_com_quadrante = _one_row_per_account_offer(stages.assign(quadrante=quadrante))
    scored_unica = _one_row_per_account_offer(scored)

    enviados = policy[policy["chosen_action"] != NO_SEND]
    nao_enviados = policy[policy["chosen_action"] == NO_SEND]

    matched = enviados.merge(
        scored_unica[["account_id", "offer_id", "net_profit"]],
        left_on=["account_id", "chosen_action"],
        right_on=["account_id", "offer_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        stages_com_quadrante[["account_id", "offer_id", "quadrante"]],
        on=["account_id", "offer_id"],
        how="left",
        validate="one_to_one",
    )

    n_total = len(policy)
    linhas = []
    if not matched.empty:
        matched["lucro_negativo"] = matched["net_profit"] < 0
        resumo = (
            matched.groupby(["quadrante", "lucro_negativo"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        for _, row in resumo.iterrows():
            linhas.append({
                "quadrante": row["quadrante"],
                "lucro_negativo": bool(row["lucro_negativo"]),
                "n": int(row["n"]),
                "pct_do_total": row["n"] / n_total,
            })

    if len(nao_enviados):
        linhas.append({
            "quadrante": None,
            "lucro_negativo": None,
            "n": len(nao_enviados),
            "pct_do_total": len(nao_enviados) / n_total,
            "chosen_action": NO_SEND,
        })

    resultado = pd.DataFrame(linhas)
    if "chosen_action" not in resultado.columns:
        resultado["chosen_action"] = None
    return resultado


def negative_profit_share(policy: pd.DataFrame, scored: pd.DataFrame) -> float:
    """Fração de clientes cuja ação recomendada tem `net_profit < 0` (excluindo
    `nao_enviar`, que nunca é negativo no lucro esperado do modelo).

    Métrica direta pedida como resumo: quanto de cada política está, pelo
    próprio critério do modelo, enviando com prejuízo esperado.
    """
    enviados = policy[policy["chosen_action"] != NO_SEND]
    if enviados.empty:
        return 0.0

    scored_unica = _one_row_per_account_offer(scored)
    matched = enviados.merge(
        scored_unica[["account_id", "offer_id", "net_profit"]],
        left_on=["account_id", "chosen_action"],
        right_on=["account_id", "offer_id"],
        how="left",
        validate="one_to_one",
    )
    return float((matched["net_profit"] < 0).mean())
