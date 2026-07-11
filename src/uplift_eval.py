"""Avaliação de uplift: Qini/AUUC (REQ-203), placebo (REQ-212), calibração
(REQ-213) e correção isotônica pós-hoc (REQ-214).

Seleção e comparação de modelos de uplift nunca usa AUC/F1 (métrica de
classificação): um modelo pode classificar `converted` bem e ainda ordenar mal
o *efeito incremental* — Qini mede a segunda coisa. Reusa `sklift.metrics`
(implementação testada) em vez de reimplementar a curva.

Três olhares complementares sobre o mesmo modelo: Qini mede **ordenação**
(concentra o efeito nos top-ranqueados?), placebo mede **significância** (a
ordenação é real ou ruído?), calibração mede **magnitude** (o tamanho do uplift
previsto bate com o observado?). Qini alto não garante magnitude certa. Quando a
magnitude está errada, a correção isotônica (REQ-214) ajusta só isso — dentro de
cada fold do cross-fitting, é monotônica por construção (não pode desfazer a
ordenação que o Qini mediu); entre folds distintos, cada um usa uma isotônica
diferente e a ordem entre eles não é garantida.

`qini`/`auuc` não exigem que o score venha do X-learner — `qini_by_strategy`/
`qini_curves_by_strategy` reusam a mesma métrica para comparar o modelo de
uplift contra conversão crua (P(converte) previsto) e ranking aleatório, a
mesma pergunta que `gaincurve` responde em R$, aqui na métrica de ordenação
(REQ-203).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.isotonic import IsotonicRegression
from sklift.metrics import qini_auc_score, qini_curve, uplift_auc_score

from src import uplift, viz
from src.config import PipelineConfig


def qini(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> float:
    """Qini AUC: quanto a ordenação por uplift concentra o efeito incremental
    real nos clientes top-ranqueados, contra a curva de ganho aleatório.
    """
    return float(qini_auc_score(y_true, uplift, treatment))


def auuc(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> float:
    """AUUC (Area Under the Uplift Curve): mesma ideia do Qini, mas contra a
    curva de ganho aleatório em vez da curva Qini ótima — normalizações
    diferentes do mesmo ganho incremental acumulado (REQ-203).
    """
    return float(uplift_auc_score(y_true, uplift, treatment))


def qini_curve_points(y_true: pd.Series, uplift: pd.Series, treatment: pd.Series) -> pd.DataFrame:
    """Pontos `(n_treated, uplift_gain)` da curva Qini, para a figura."""
    n_treated, gain = qini_curve(y_true, uplift, treatment)
    return pd.DataFrame({"n_treated": n_treated, "gain": gain})


# --- Qini/AUUC por estratégia (REQ-203) -----------------------------------------
#
# `qini_auc_score`/`uplift_auc_score` só pedem um *score* para ordenar por —
# não precisa vir do X-learner. Ranquear por P(converte) ou por ordem aleatória
# e medir o mesmo Qini/AUUC responde "esse ranking também concentra efeito
# incremental, ou só o modelo de uplift faz isso?" (a mesma pergunta de
# `gaincurve`, mas na métrica de ordenação, não em R$).


def qini_by_strategy(
    y_true: pd.Series, treatment: pd.Series, scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """Qini AUC e AUUC de cada estratégia nomeada, no mesmo holdout.

    `scores` é `{nome_estrategia: score_por_linha}` — score maior é mais
    prioritário para tratar (mesma convenção de `gaincurve.uplift_ranking`/
    `completion_ranking`/`random_ranking`, mas aqui é o valor que ordena, não
    já um índice permutado). Ranking aleatório precisa de um score contínuo
    (ex.: `np.random.default_rng(cfg.seed).random(len(y_true))`), não da
    permutação de índice que `gaincurve.random_ranking` devolve — os dois
    servem propósitos diferentes (reordenar linhas vs. pontuar cada uma).

    Retorna `[strategy, qini, auuc]`, uma linha por estratégia.
    """
    linhas = []
    for nome, score in scores.items():
        linhas.append({
            "strategy": nome,
            "qini": qini(y_true, score, treatment),
            "auuc": auuc(y_true, score, treatment),
        })
    return pd.DataFrame(linhas)


def qini_curves_by_strategy(
    y_true: pd.Series, treatment: pd.Series, scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """`qini_curve_points` de cada estratégia nomeada, numa tabela longa.

    Mesmo `y_true`/`treatment` para todas — a comparação exige o mesmo
    holdout. Retorna `[strategy, n_treated, gain]`.
    """
    partes = []
    for nome, score in scores.items():
        curva = qini_curve_points(y_true, score, treatment)
        partes.append(curva.assign(strategy=nome))
    return pd.concat(partes, ignore_index=True)[["strategy", "n_treated", "gain"]]


# --- Teste de placebo por permutação (REQ-212) ---------------------------------
#
# A mesma distribuição nula serve dois propósitos: o percentil que o Qini real
# precisa superar (significância) e a dispersão da nula (intervalo de confiança
# do número reportado). Não são dois cálculos — é um só lido de duas formas.


def _permute_treatment_within_offer_type(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Embaralha `treatment` **dentro de cada `offer_type`**, preservando a
    proporção tratado/controle do grupo.

    Embaralhar globalmente mudaria essa proporção por tipo (os três `offer_type`
    têm taxas de view bem diferentes no dado real) e o Qini nulo cairia por
    composição de grupo, não pela ausência de efeito causal — exatamente o que
    este teste não quer medir. `X` e `y` (`converted`) ficam fixos; só o rótulo
    de tratamento embaralha.
    """
    permutado = df["treatment"].copy()
    for _, grupo in df.groupby("offer_type"):
        permutado.loc[grupo.index] = rng.permutation(grupo["treatment"].to_numpy())
    return permutado


def placebo_qini_distribution(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, cfg: PipelineConfig
) -> np.ndarray:
    """Distribuição nula do Qini: refita o X-learner `cfg.placebo_n_permutations`
    vezes com `treatment` embaralhado no treino, prevê no holdout real.

    Cada réplica usa uma seed derivada de `cfg.seed` — determinístico dado o
    config, mas distinto entre réplicas. Reusa `uplift.fit_xlearner`/`predict`
    (a mesma infraestrutura do modelo real), não uma reimplementação paralela.
    """
    scores = np.empty(cfg.placebo_n_permutations)
    for i in range(cfg.placebo_n_permutations):
        rng = np.random.default_rng(cfg.seed + i)
        placebo_train = train_df.copy()
        placebo_train["treatment"] = _permute_treatment_within_offer_type(placebo_train, rng)

        modelos = uplift.fit_xlearner(placebo_train, cfg)
        pred = uplift.predict(modelos, holdout_df)
        scores[i] = qini(holdout_df["converted"], pred["uplift"], holdout_df["treatment"])
    return scores


def placebo_test(
    qini_score: float, null_distribution: np.ndarray, cfg: PipelineConfig
) -> dict[str, float | bool]:
    """Compara o Qini real ao percentil `cfg.placebo_confidence_level` da nula.

    `p_value` é a fração de réplicas nulas que igualam ou superam o Qini real —
    o p-valor empírico da mesma distribuição, de graça (REQ-212).
    """
    limiar = float(np.quantile(null_distribution, cfg.placebo_confidence_level))
    p_value = float((null_distribution >= qini_score).mean())
    return {
        "qini_real": qini_score,
        "limiar_percentil": limiar,
        "passou": qini_score > limiar,
        "p_value": p_value,
        "null_mean": float(null_distribution.mean()),
        "null_std": float(null_distribution.std()),
    }


def fig_placebo_distribution(
    null_distribution: np.ndarray, qini_score: float, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Histograma da distribuição nula com o Qini real e o limiar marcados —
    a leitura visual do mesmo cálculo de `placebo_test`.
    """
    limiar = float(np.quantile(null_distribution, cfg.placebo_confidence_level))
    fig = viz.figure(
        f"Qini real ({qini_score:.3f}) supera o placebo — limiar p{int(100*cfg.placebo_confidence_level)} = {limiar:.3f}",
        f"Distribuição nula de {len(null_distribution)} permutações de `treatment` dentro de cada offer_type.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    ink_primary, secondary, _ = viz.ink(theme)
    fig.add_trace(go.Histogram(x=null_distribution, name="Qini sob placebo", marker_color=secondary))
    fig.add_vline(x=qini_score, line=dict(color=cor, width=2.5, dash="solid"))
    fig.add_vline(x=limiar, line=dict(color=ink_primary, width=1.5, dash="dot"))
    return fig


# --- Calibração da magnitude do uplift (REQ-213) -------------------------------
#
# Qini mede ordenação; calibração mede magnitude. Um modelo pode ordenar
# perfeitamente (Qini alto) e ainda errar o *tamanho* do efeito — prever 40 p.p.
# onde o real é 4. A política (REQ-204) multiplica o uplift por receita, então
# erro de magnitude vira erro de R$. Este bloco compara, por bin de τ previsto,
# o uplift previsto médio contra o observado (tratado − controle no bin).


def calibration_by_bin(
    uplift_pred: pd.Series, y_true: pd.Series, treatment: pd.Series, cfg: PipelineConfig
) -> pd.DataFrame:
    """Uplift previsto vs. observado, por bin de τ previsto (REQ-213).

    Bins de tamanho ~igual por quantil de `uplift_pred` (decis por default). Em
    cada bin, o uplift **observado** é `taxa_conversao(tratado) −
    taxa_conversao(controle)` — uma diferença de duas taxas, que exige os dois
    braços presentes no bin. Bin sem tratado ou sem controle fica com uplift
    observado `NaN` e `avaliavel=False` (positividade por bin, Premissa 8), nunca
    zero: ausência de contrafactual não é efeito nulo.

    Retorna uma linha por bin com `uplift_previsto`, `uplift_observado`, os
    tamanhos de cada braço e a flag `avaliavel`.
    """
    df = pd.DataFrame({
        "uplift_pred": np.asarray(uplift_pred, dtype=float),
        "y": np.asarray(y_true, dtype=float),
        "treatment": np.asarray(treatment, dtype=int),
    })
    # `duplicates="drop"` colapsa fronteiras repetidas quando τ tem muitos empates
    # (ex.: massa concentrada) — menos bins, mas nunca um corte inválido.
    df["bin"] = pd.qcut(df["uplift_pred"], q=cfg.calibration_n_bins, duplicates="drop")

    linhas = []
    for bin_id, (faixa, grupo) in enumerate(df.groupby("bin", observed=True)):
        tratado = grupo[grupo["treatment"] == 1]
        controle = grupo[grupo["treatment"] == 0]
        avaliavel = len(tratado) > 0 and len(controle) > 0
        observado = (tratado["y"].mean() - controle["y"].mean()) if avaliavel else float("nan")

        linhas.append({
            "bin": bin_id,
            "faixa_uplift_previsto": str(faixa),
            "n": len(grupo),
            "n_tratado": len(tratado),
            "n_controle": len(controle),
            "uplift_previsto": grupo["uplift_pred"].mean(),
            "uplift_observado": observado,
            "avaliavel": avaliavel,
        })
    return pd.DataFrame(linhas)


def calibration_error(calibration: pd.DataFrame) -> dict[str, float | int]:
    """Resumo do erro de calibração sobre os bins **avaliáveis** (REQ-213).

    `mae` é o erro absoluto médio entre uplift previsto e observado; `bias` é o
    erro com sinal (previsto − observado) médio — positivo indica que o modelo
    **superestima** a magnitude do efeito. `n_bins_inavaliaveis` diz quantos bins
    ficaram sem contrafactual e não entraram na conta.
    """
    avaliaveis = calibration[calibration["avaliavel"]]
    erro = avaliaveis["uplift_previsto"] - avaliaveis["uplift_observado"]
    return {
        "mae": float(erro.abs().mean()),
        "bias": float(erro.mean()),
        "n_bins_avaliados": int(len(avaliaveis)),
        "n_bins_inavaliaveis": int((~calibration["avaliavel"]).sum()),
    }


def fig_calibration(
    calibration: pd.DataFrame, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Uplift previsto (x) vs. observado (y) por bin, contra a diagonal perfeita.

    Pontos sobre a diagonal = magnitude calibrada. Acima = modelo subestima;
    abaixo = superestima. Só bins avaliáveis entram (os sem contrafactual não
    têm y observável). A diagonal cobre o intervalo dos pontos plotados.
    """
    avaliaveis = calibration[calibration["avaliavel"]]
    resumo = calibration_error(calibration)
    fig = viz.figure(
        f"Magnitude do uplift: previsto vs. observado — MAE = {resumo['mae']:.3f}",
        f"Um ponto por bin de τ previsto ({resumo['n_bins_avaliados']} avaliáveis). "
        "Sobre a diagonal = calibrado; abaixo = superestima.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    _, secondary, _ = viz.ink(theme)

    lo = float(min(avaliaveis["uplift_previsto"].min(), avaliaveis["uplift_observado"].min()))
    hi = float(max(avaliaveis["uplift_previsto"].max(), avaliaveis["uplift_observado"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], name="calibração perfeita",
        mode="lines", line=dict(color=secondary, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=avaliaveis["uplift_previsto"], y=avaliaveis["uplift_observado"],
        name="bins", mode="markers", marker=dict(color=cor, size=9),
    ))
    return fig


# --- Calibração isotônica pós-hoc (REQ-214) -------------------------------------
#
# A calibração por bin (REQ-213) diagnostica erro de magnitude; a isotônica
# corrige. `IsotonicRegression` ajusta um mapa τ_previsto → τ_calibrado
# monotônico crescente — por construção não pode inverter a ordem de dois τ,
# então não desfaz o que o Qini mediu. Só remapeia a escala.
#
# Cross-fitting evita que o "depois" aprenda no mesmo dado que avalia: cada
# fold do holdout é previsto por uma isotônica ajustada nos OUTROS folds.


def isotonic_calibrate_cross_fitted(
    uplift_pred: pd.Series, y_true: pd.Series, treatment: pd.Series, cfg: PipelineConfig
) -> np.ndarray:
    """τ calibrado por cross-fitting dentro do holdout (REQ-214).

    O alvo da isotônica não é `y_true` diretamente (é binário 0/1 por cliente,
    não o uplift em si): é o **uplift observado por bin** de τ previsto,
    calculado por `calibration_by_bin`. Para cada fold k, os bins (e portanto o
    alvo de regressão) são calculados só com as linhas fora do fold k; a
    isotônica ajustada nesses bins então prevê o τ calibrado das linhas
    **dentro** do fold k. Nenhuma linha informa a isotônica que a calibra.

    Monotonicidade é garantida **por fold**: dentro do mesmo fold, dois pontos
    com τ previsto distinto nunca saem com a ordem invertida (mesma isotônica).
    Entre folds diferentes, cada um usa sua própria isotônica (ajustada em bins
    diferentes) e a ordem relativa não é garantida — é o preço de nunca deixar
    um ponto se auto-calibrar.

    Retorna um array alinhado ao índice de `uplift_pred`, com o τ calibrado de
    cada linha.
    """
    n = len(uplift_pred)
    idx = np.asarray(uplift_pred.index)
    rng = np.random.default_rng(cfg.seed)
    fold = rng.integers(0, cfg.calibration_n_folds, size=n)

    calibrado = np.empty(n, dtype=float)
    uplift_arr = np.asarray(uplift_pred, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=int)

    for k in range(cfg.calibration_n_folds):
        fora = fold != k
        dentro = fold == k
        if not dentro.any():
            continue

        # Bins calculados só com os outros folds — o alvo da isotônica nunca
        # inclui o próprio fold que ela vai prever.
        calib_fora = calibration_by_bin(
            pd.Series(uplift_arr[fora]), pd.Series(y_arr[fora]), pd.Series(treatment_arr[fora]), cfg
        )
        avaliaveis = calib_fora[calib_fora["avaliavel"]]
        if len(avaliaveis) < 2:
            # Sem bins avaliáveis suficientes para ajustar uma isotônica não-trivial:
            # o fold sai sem correção (calibrado = previsto), não um valor inventado.
            calibrado[dentro] = uplift_arr[dentro]
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(avaliaveis["uplift_previsto"], avaliaveis["uplift_observado"])
        calibrado[dentro] = iso.predict(uplift_arr[dentro])

    return calibrado


def calibration_before_after(
    uplift_pred: pd.Series, y_true: pd.Series, treatment: pd.Series, cfg: PipelineConfig
) -> dict[str, dict[str, float | int]]:
    """Estatísticas de calibração antes e depois da correção isotônica (REQ-214),
    lado a lado no mesmo holdout — a comparação que REQ-214 pede.
    """
    calib_antes = calibration_by_bin(uplift_pred, y_true, treatment, cfg)
    uplift_calibrado = isotonic_calibrate_cross_fitted(uplift_pred, y_true, treatment, cfg)
    calib_depois = calibration_by_bin(pd.Series(uplift_calibrado, index=uplift_pred.index), y_true, treatment, cfg)

    return {
        "antes": calibration_error(calib_antes),
        "depois": calibration_error(calib_depois),
    }


def fig_calibration_before_after(
    calib_antes: pd.DataFrame, calib_depois: pd.DataFrame, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Previsto vs. observado, antes e depois da isotônica, na mesma figura (REQ-214).

    Os pontos "depois" devem se aproximar da diagonal em relação aos "antes" —
    a isotônica é monotônica, então não reordena os bins, só reescala o eixo x.
    """
    antes = calib_antes[calib_antes["avaliavel"]]
    depois = calib_depois[calib_depois["avaliavel"]]
    erro_antes, erro_depois = calibration_error(calib_antes), calibration_error(calib_depois)

    fig = viz.figure(
        f"Correção isotônica reduz o erro de magnitude — MAE {erro_antes['mae']:.3f} → {erro_depois['mae']:.3f}",
        "Previsto vs. observado por bin, antes (◇) e depois (●) da isotônica. Mais perto da diagonal = mais calibrado.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    _, secondary, _ = viz.ink(theme)

    lo = float(min(antes["uplift_previsto"].min(), antes["uplift_observado"].min(),
                   depois["uplift_previsto"].min(), depois["uplift_observado"].min()))
    hi = float(max(antes["uplift_previsto"].max(), antes["uplift_observado"].max(),
                   depois["uplift_previsto"].max(), depois["uplift_observado"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], name="calibração perfeita",
        mode="lines", line=dict(color=secondary, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=antes["uplift_previsto"], y=antes["uplift_observado"],
        name="antes", mode="markers", marker=dict(color=secondary, size=9, symbol="diamond-open"),
    ))
    fig.add_trace(go.Scatter(
        x=depois["uplift_previsto"], y=depois["uplift_observado"],
        name="depois", mode="markers", marker=dict(color=cor, size=9),
    ))
    return fig


def fig_qini_curve(
    curve: pd.DataFrame, qini_score: float, cfg: PipelineConfig, theme: str = "light"
) -> go.Figure:
    """Curva Qini no tema executivo: rótulo direto no fim da série (a paleta
    validada não deixa a cor sozinha carregar identidade — ver `src/viz.py`).
    """
    fig = viz.figure(
        f"Ordenar por uplift concentra o ganho incremental — Qini AUC = {qini_score:.3f}",
        "Ganho acumulado real vs. nº de clientes tratados, ordenados por uplift previsto.",
        theme=theme,
    )
    cor = viz.palette(theme)[0]
    fig.add_trace(go.Scatter(
        x=curve["n_treated"], y=curve["gain"], name="modelo de uplift",
        mode="lines", line=dict(color=cor, width=2.5),
    ))
    ink_primary, secondary, _ = viz.ink(theme)
    fig.add_trace(go.Scatter(
        x=[curve["n_treated"].iloc[0], curve["n_treated"].iloc[-1]],
        y=[curve["gain"].iloc[0], curve["gain"].iloc[-1]],
        name="aleatório", mode="lines", line=dict(color=secondary, width=1.5, dash="dot"),
    ))
    return viz.add_end_labels(fig, theme=theme)


def fig_qini_curves_by_strategy(curves: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Curva Qini de várias estratégias sobrepostas — modelo de uplift vs.
    conversão crua vs. aleatório, na mesma pergunta do Qini isolado
    (`fig_qini_curve`): ordenar por este score concentra o ganho incremental
    real, ou só parece concentrar? Rótulo direto no fim de cada série.
    """
    fig = viz.figure(
        "Qini por estratégia: quem concentra o ganho incremental real",
        "Ganho acumulado real vs. nº de clientes tratados, uma curva por estratégia de ranking.",
        theme=theme,
    )
    cores = viz.palette(theme)
    for i, (nome, grupo) in enumerate(curves.groupby("strategy")):
        cor = cores[i % len(cores)]
        fig.add_trace(go.Scatter(
            x=grupo["n_treated"], y=grupo["gain"], name=nome,
            mode="lines", line=dict(color=cor, width=2.5),
        ))
    return viz.add_end_labels(fig, theme=theme)
