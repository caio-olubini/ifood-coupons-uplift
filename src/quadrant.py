"""Classificação de quadrante de uplift.

Diagnóstico exploratório, não um requisito formal da spec: de que tipo de
cliente o holdout é composto, usando os quatro quadrantes clássicos de uplift:

- **persuadable** (τ > ε): converte só se tratado — o alvo ideal, efeito real
  e positivo.
- **sleeping dog** (τ < -ε): tratamento **atrapalha** — efeito real e
  negativo, "não enviar" é estritamente melhor.
- **sure thing** (|τ| ≤ ε, alta propensão a converter): dentro da banda onde
  τ é indistinguível de zero, mas já converteria de qualquer forma — enviar
  oferta paga desconto sem causar nada.
- **lost cause** (|τ| ≤ ε, baixa propensão a converter): dentro da mesma
  banda, mas não converte de qualquer forma.

O corte é em **τ**, não em μ₀/μ₁ separados: dois cortes independentes em
μ₀/μ₁ (a classificação anterior) não correspondem a uma curva de nível de τ
constante — um cliente com μ₀=0,49/μ₁=0,51 (τ≈0,02, quase nulo) e outro com
μ₀=0,05/μ₁=0,95 (τ=0,90, efeito grande) podiam cair em quadrantes diferentes
só pelo acaso de qual lado do limiar cada μ caía. Validado no dado real: variar
o limiar em μ₀/μ₁ nunca zerava o τ médio dentro de `sure_thing`/`lost_cause`
(ficava entre 0,04 e 0,07 em qualquer corte razoável); cortando direto em τ,
com `ε = cfg.quadrant_tau_epsilon`, o τ médio residual cai para ~0,0002 — a
banda de fato isola quem não tem efeito detectável.

Dentro da banda `|τ| ≤ ε`, `p_convert` (a propensão crua a converter, do
baseline preditivo) separa `sure_thing` de `lost_cause` — `τ` já não distingue
os dois ali, mas `p_convert` sim. `cfg.quadrant_p_convert_threshold` é o corte.

Três funções cruzam o quadrante com o ranking de uma estratégia (mesma
convenção de `gaincurve`: um array de índices, do mais prioritário ao menos),
para enriquecer a comparação de estratégias além de "qual dá mais Qini/R$":

- `composition_at_budget` — de quem o top-N de uma estratégia é feito, com o
  τ médio de cada quadrante como checagem de sanidade (deve ficar perto de
  zero dentro de `sure_thing`/`lost_cause`).
- `gain_by_quadrant_at_budget` — de qual quadrante vem o lucro incremental do
  top-N (o mesmo contrafactual escalado de `gaincurve`, aplicado por
  quadrante), também com τ médio ao lado.
- `left_on_table` — quantos `persuadable` uma estratégia de referência
  colocaria no budget que a estratégia escolhida deixa de fora.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from src import viz
from src.config import PipelineConfig
from src.gaincurve import (
    _profit_per_treated_conversion,
    _scaled_counterfactual_gain,
    add_net_profit,
)

PERSUADABLE = "persuadable"
SURE_THING = "sure_thing"
LOST_CAUSE = "lost_cause"
SLEEPING_DOG = "sleeping_dog"

#: Ordem de leitura dos quadrantes nas figuras — do alvo causal ideal
#: (persuadable) ao efeito negativo (sleeping_dog), passando pelos dois "sem
#: efeito detectável". Fixa a cor e a ordem de empilhamento entre as figuras.
QUADRANT_ORDER = (PERSUADABLE, SURE_THING, LOST_CAUSE, SLEEPING_DOG)


def classify_quadrant(stages: pd.DataFrame, p_convert: pd.Series, cfg: PipelineConfig) -> pd.Series:
    """Quadrante de cada linha de `stages` (saída de `uplift.predict_stages`),
    a partir de `tau` previsto e `p_convert` (propensão crua a converter,
    alinhada ao mesmo índice).

    Corte primário em `tau`: fora da banda `[-ε, ε]` (`cfg.quadrant_tau_epsilon`),
    o sinal de `tau` já decide `persuadable`/`sleeping_dog`. Dentro da banda,
    `tau` é indistinguível de zero e `p_convert` (limiar
    `cfg.quadrant_p_convert_threshold`) decide `sure_thing`/`lost_cause`.
    """
    epsilon = cfg.quadrant_tau_epsilon
    p_threshold = cfg.quadrant_p_convert_threshold

    tau = stages["tau"]
    persuadable = tau > epsilon
    sleeping_dog = tau < -epsilon
    sem_efeito_detectavel = ~persuadable & ~sleeping_dog

    quadrante = pd.Series(index=stages.index, dtype=object)
    quadrante[persuadable] = PERSUADABLE
    quadrante[sleeping_dog] = SLEEPING_DOG
    quadrante[sem_efeito_detectavel & (p_convert >= p_threshold)] = SURE_THING
    quadrante[sem_efeito_detectavel & (p_convert < p_threshold)] = LOST_CAUSE
    return quadrante


def quadrant_distribution(stages: pd.DataFrame, p_convert: pd.Series, cfg: PipelineConfig) -> pd.DataFrame:
    """Distribuição de quadrantes por `offer_type` — visão geral antes de olhar
    por política: o quanto de cada tipo de cliente existe no holdout, período.
    """
    quadrante = classify_quadrant(stages, p_convert, cfg)
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


def composition_at_budget(
    ranking: np.ndarray, stages: pd.DataFrame, p_convert: pd.Series, cfg: PipelineConfig, budget: int
) -> pd.DataFrame:
    """De quem é composto o top-N de uma estratégia, por quadrante causal.

    `ranking` é a ordem de prioridade da estratégia (mesma convenção de
    `gaincurve.uplift_ranking`/`hybrid_ranking`): os primeiros `budget` índices
    são o que ela escolheria com aquele orçamento. `stages` é
    `uplift.predict_stages`, indexado como o holdout — dá o quadrante de cada
    cliente escolhido.

    Responde "a estratégia X aposta em quem?": uma que persegue ticket alto
    deve concentrar `sure_thing`; uma que persegue causalidade deve concentrar
    `persuadable`. `tau_medio` acompanha a contagem como checagem de sanidade —
    dentro da banda `sure_thing`/`lost_cause` deve ficar perto de zero; se não
    ficar, `cfg.quadrant_tau_epsilon` está largo demais. Retorna `[quadrante, n,
    pct, tau_medio]`, um por quadrante presente no top-N (percentual sobre o
    próprio top-N, não sobre o holdout inteiro).
    """
    top_n = ranking[:budget]
    stages_top_n = stages.loc[top_n]
    quadrante_top_n = classify_quadrant(stages_top_n, p_convert.loc[top_n], cfg)

    contagem = quadrante_top_n.value_counts().rename("n").reset_index()
    contagem.columns = ["quadrante", "n"]
    contagem["pct"] = contagem["n"] / len(top_n)

    tau_por_quadrante = (
        stages_top_n["tau"].groupby(quadrante_top_n).mean().rename("tau_medio")
        .rename_axis("quadrante").reset_index()
    )
    contagem = contagem.merge(tau_por_quadrante, on="quadrante")
    return contagem.sort_values("quadrante").reset_index(drop=True)


def gain_by_quadrant_at_budget(
    ranking: np.ndarray,
    holdout_df: pd.DataFrame,
    stages: pd.DataFrame,
    p_convert: pd.Series,
    cfg: PipelineConfig,
    budget: int,
) -> pd.DataFrame:
    """Quanto do lucro incremental do top-N vem de cada quadrante causal.

    Mesma métrica de `gaincurve.incremental_gain_curve` (conversão incremental ×
    lucro médio por conversão tratada), mas aplicada **dentro de cada quadrante**
    em vez de no top-N inteiro — assim o "lucro incremental" aqui significa
    exatamente o de §5, só particionado. Um gain alto dentro de
    `sure_thing`/`lost_cause` (o quadrante "sem efeito detectável",
    `|τ| ≤ cfg.quadrant_tau_epsilon`) seria sinal de banda mal calibrada;
    `tau_medio` ao lado é a checagem direta disso.

    Um quadrante sem controle dentro do top-N não tem contrafactual estimável
    (a razão tratado/controle é indefinida) e sai marcado `avaliavel=False`,
    não com um gain inventado — o mesmo princípio de positividade que
    `_scaled_counterfactual_gain` exige.

    Retorna `[quadrante, n, gain, avaliavel, tau_medio]`, um por quadrante
    presente no top-N.
    """
    top_n = ranking[:budget]
    subset = add_net_profit(holdout_df.loc[top_n])
    quadrante = classify_quadrant(stages.loc[top_n], p_convert.loc[top_n], cfg)
    tau_top_n = stages.loc[top_n, "tau"]

    linhas = []
    for nome_quadrante, grupo in subset.assign(quadrante=quadrante).groupby("quadrante", observed=True):
        treated = (grupo["treatment"].to_numpy() == 1).astype(float)
        control = 1.0 - treated
        converted = grupo["converted"].to_numpy(dtype=float)
        profit = grupo["net_profit_realized"].to_numpy()
        avaliavel = control.sum() > 0

        incremental_conversions = _scaled_counterfactual_gain(converted, treated, control)
        profit_per_conv = _profit_per_treated_conversion(profit, treated, converted)
        gain = incremental_conversions * profit_per_conv
        linhas.append({
            "quadrante": nome_quadrante,
            "n": len(grupo),
            "gain": gain[-1] if avaliavel else float("nan"),
            "avaliavel": avaliavel,
            "tau_medio": tau_top_n.loc[grupo.index].mean(),
        })
    return pd.DataFrame(linhas).sort_values("quadrante").reset_index(drop=True)


def left_on_table(
    reference_ranking: np.ndarray,
    chosen_ranking: np.ndarray,
    stages: pd.DataFrame,
    p_convert: pd.Series,
    cfg: PipelineConfig,
    budget: int,
) -> pd.DataFrame:
    """Persuadables que `reference_ranking` escolheria e `chosen_ranking` não escolhe.

    Compara os top-`budget` de duas estratégias no mesmo orçamento: entre os
    `persuadable` (τ > ε, o alvo causal ideal) que `reference_ranking`
    colocaria dentro do budget, quantos `chosen_ranking` deixou de fora —
    clientes convertíveis pela oferta que a estratégia escolhida está
    ignorando, tipicamente porque persegue outro critério (ticket alto, por
    exemplo).

    Retorna um resumo `{persuadables_do_reference, deixados_de_fora, pct}`. Não
    afirma nada sobre o inverso (quem `chosen_ranking` pega e `reference_ranking`
    não) — é assimétrico por construção, a pergunta é sempre "o que a estratégia
    escolhida está perdendo perto de uma referência".
    """
    quadrante = classify_quadrant(stages, p_convert, cfg)
    persuadables = set(stages.index[quadrante == PERSUADABLE])

    persuadables_referencia = persuadables & set(reference_ranking[:budget])
    escolhidos = set(chosen_ranking[:budget])
    deixados_de_fora = persuadables_referencia - escolhidos

    n_referencia = len(persuadables_referencia)
    return pd.DataFrame([{
        "persuadables_do_reference": n_referencia,
        "deixados_de_fora": len(deixados_de_fora),
        "pct": len(deixados_de_fora) / n_referencia if n_referencia else float("nan"),
    }])


def _fig_stacked_by_quadrant(
    long: pd.DataFrame, value: str, title: str, subtitle: str, theme: str
) -> go.Figure:
    """Barras empilhadas: uma barra por estratégia, um segmento por quadrante.

    `long` é a tabela longa `[strategy, quadrante, <value>]` (saída pivotável de
    `composition_at_budget`/`gain_by_quadrant_at_budget`). Empilha os quadrantes
    na ordem fixa `QUADRANT_ORDER` — cor e ordem consistentes entre as duas
    figuras (share e receita), para o leitor comparar as duas lado a lado.
    """
    pivot = long.pivot(index="strategy", columns="quadrante", values=value)
    fig = viz.figure(title, subtitle, theme=theme, barmode="stack")
    cores = viz.palette(theme)
    for i, quad in enumerate(QUADRANT_ORDER):
        if quad not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[quad], name=quad,
            marker_color=cores[i % len(cores)], marker_line_width=0,
        ))
    return fig


def fig_composition_by_quadrant(composition: pd.DataFrame, budget: int, theme: str = "light") -> go.Figure:
    """Share de cada quadrante no top-N de cada estratégia, em barras empilhadas.

    `composition` é o empilhado de `composition_at_budget` por estratégia
    (`[strategy, quadrante, pct, ...]`). Mostra de quem cada estratégia é feita:
    quem persegue causalidade concentra `persuadable`, quem persegue ticket
    concentra `sure_thing`.
    """
    return _fig_stacked_by_quadrant(
        composition, "pct",
        "De quem cada estratégia é feita",
        f"Share de cada quadrante causal no top-{budget:,} de cada estratégia.",
        theme,
    )


def fig_gain_by_quadrant(gain: pd.DataFrame, budget: int, theme: str = "light") -> go.Figure:
    """Receita incremental por quadrante no top-N de cada estratégia, empilhada.

    `gain` é o empilhado de `gain_by_quadrant_at_budget` por estratégia
    (`[strategy, quadrante, gain, ...]`). Mostra de qual quadrante vem o lucro
    incremental de cada estratégia — quase tudo deve vir de `persuadable` se a
    ordenação captura efeito causal.
    """
    return _fig_stacked_by_quadrant(
        gain, "gain",
        "De qual quadrante vem o lucro incremental",
        f"Lucro líquido incremental (R$) por quadrante causal no top-{budget:,} de cada estratégia.",
        theme,
    )
