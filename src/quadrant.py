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
- `recurrence_gain_at_budget`/`recurrence_by_quadrant_at_budget` — taxa de
  recorrência **incremental** (`is_recurrent`, tratado − controle escalado,
  mesmo contrafactual estilo Qini de `gaincurve`) do top-N, consolidada e
  por quadrante, com N de suporte e IC bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    não com um gain inventado — mesmo princípio de positividade que
    `uplift_eval.calibration_by_bin` já aplicava por bin.

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


def _recurrence_gain(converted: pd.DataFrame) -> tuple[float, int]:
    """Taxa de recorrência incremental de um grupo de convertidos, estilo Qini.

    Mesma mecânica de `gaincurve._scaled_counterfactual_gain` — contrafactual do
    tratado estimado a partir do controle observado, escalado pela razão
    `n_tratado/n_controle` — mas aplicada só às linhas com `converted=1`
    (`is_recurrent` não é definida fora daí) e normalizada por `n_tratado` para
    virar uma **taxa** incremental, não uma soma: `[Σ(is_recurrent·tratado) −
    Σ(is_recurrent·controle)·n_tratado/n_controle] / n_tratado`. Sem controle
    convertido no grupo, o contrafactual não é estimável — devolve `nan`.

    Retorna `(recurrence_gain, n_convertidos_tratados)`; o segundo é o N de
    suporte que sustenta a taxa (o denominador da própria fórmula).
    """
    treated = (converted["treatment"].to_numpy() == 1).astype(float)
    control = 1.0 - treated
    n_treated = treated.sum()
    n_control = control.sum()
    if n_treated == 0 or n_control == 0:
        return float("nan"), int(n_treated)

    is_recurrent = converted["is_recurrent"].to_numpy(dtype=float)
    sum_treated = (is_recurrent * treated).sum()
    sum_control = (is_recurrent * control).sum()
    gain = (sum_treated - sum_control * (n_treated / n_control)) / n_treated
    return gain, int(n_treated)


def _recurrence_gain_ci(converted: pd.DataFrame, cfg: PipelineConfig) -> tuple[float, float]:
    """IC bootstrap não paramétrico de `_recurrence_gain` (mesmo padrão de
    `gaincurve.gain_curves_with_ci`): reamostra `converted` com reposição
    `cfg.gain_curve_n_bootstrap` vezes e recomputa a taxa por réplica; o
    intervalo é o percentil `cfg.gain_curve_confidence_level` das réplicas.
    `nan` quando não há linhas a reamostrar.
    """
    if len(converted) == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - cfg.gain_curve_confidence_level
    rng = np.random.default_rng(cfg.seed)
    replicas = np.empty(cfg.gain_curve_n_bootstrap)
    for i in range(cfg.gain_curve_n_bootstrap):
        sample_idx = rng.integers(0, len(converted), size=len(converted))
        replicas[i], _ = _recurrence_gain(converted.iloc[sample_idx])
    return tuple(np.nanquantile(replicas, [alpha / 2, 1 - alpha / 2]))


def recurrence_by_quadrant_at_budget(
    ranking: np.ndarray,
    holdout_df: pd.DataFrame,
    stages: pd.DataFrame,
    p_convert: pd.Series,
    cfg: PipelineConfig,
    budget: int,
) -> pd.DataFrame:
    """Taxa de recorrência **incremental** do top-N de uma estratégia, por quadrante causal.

    Mede uplift de recorrência — o objetivo aqui não é "quão recorrente é quem
    converteu" isoladamente, mas "quanto a oferta *causa* de recorrência a
    mais", do mesmo jeito que a curva de ganho (§5) mede lucro incremental: taxa
    do tratado menos a taxa do controle, escalada pelo contrafactual estilo Qini
    (`_recurrence_gain`). `is_recurrent` é derivada do target
    (`attribution.add_recurrence_flag`) — entra aqui só como outcome a decompor,
    nunca como insumo do ranking ou da classificação de quadrante. O grupo em
    cada braço é sempre quem converteu (`converted=1`): `is_recurrent` só é
    definida ali, e misturar não-convertidos (sempre 0) trocaria "a oferta faz
    quem converte recorrer mais?" por "a oferta faz mais gente converter e
    recorrer?" — duas perguntas diferentes.

    `n` é o N de suporte: quantos convertidos tratados sustentam a taxa em cada
    quadrante (o denominador da própria fórmula) — célula com `n` baixo é
    ruído, não achado. `recurrence_gain_lo`/`_hi` são o intervalo de confiança
    por bootstrap não paramétrico (`cfg.gain_curve_n_bootstrap` réplicas,
    `cfg.gain_curve_confidence_level`, mesmo padrão de `gaincurve.gain_curves_with_ci`):
    reamostra o top-N com reposição e recomputa `_recurrence_gain` por réplica.
    Quadrante sem convertido em algum braço fica `avaliavel=False`, sem taxa
    inventada — mesmo princípio de `gain_by_quadrant_at_budget`.

    Retorna `[quadrante, n, recurrence_gain, recurrence_gain_lo,
    recurrence_gain_hi, avaliavel, tau_medio]`, um por quadrante presente no
    top-N.
    """
    top_n = ranking[:budget]
    subset = holdout_df.loc[top_n]
    quadrante = classify_quadrant(stages.loc[top_n], p_convert.loc[top_n], cfg)
    tau_top_n = stages.loc[top_n, "tau"]

    linhas = []
    for nome_quadrante, grupo in subset.assign(quadrante=quadrante).groupby("quadrante", observed=True):
        convertidos = grupo[grupo["converted"] == 1]
        gain, n = _recurrence_gain(convertidos)
        avaliavel = not np.isnan(gain)
        gain_lo, gain_hi = _recurrence_gain_ci(convertidos, cfg) if avaliavel else (float("nan"), float("nan"))

        linhas.append({
            "quadrante": nome_quadrante,
            "n": n,
            "recurrence_gain": gain,
            "recurrence_gain_lo": gain_lo,
            "recurrence_gain_hi": gain_hi,
            "avaliavel": avaliavel,
            "tau_medio": tau_top_n.loc[grupo.index].mean(),
        })
    return pd.DataFrame(linhas).sort_values("quadrante").reset_index(drop=True)


def recurrence_gain_at_budget(
    ranking: np.ndarray,
    holdout_df: pd.DataFrame,
    cfg: PipelineConfig,
    budget: int,
) -> pd.Series:
    """Taxa de recorrência incremental consolidada do top-N de uma estratégia (todos os quadrantes juntos).

    Mesma fórmula de `recurrence_by_quadrant_at_budget`, mas sem quebrar por
    quadrante — o número único que resume "essa estratégia, com esse budget,
    causa quanta recorrência a mais". É a leitura de abertura antes de olhar
    onde esse efeito se concentra por quadrante.

    Retorna uma `Series` com `n`, `recurrence_gain`, `recurrence_gain_lo`,
    `recurrence_gain_hi`.
    """
    top_n = ranking[:budget]
    convertidos = holdout_df.loc[top_n].pipe(lambda df: df[df["converted"] == 1])
    gain, n = _recurrence_gain(convertidos)
    gain_lo, gain_hi = _recurrence_gain_ci(convertidos, cfg) if not np.isnan(gain) else (float("nan"), float("nan"))

    return pd.Series({
        "n": n,
        "recurrence_gain": gain,
        "recurrence_gain_lo": gain_lo,
        "recurrence_gain_hi": gain_hi,
    })


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
