"""Curva de ganho incremental por budget top-N (REQ-206).

Compara estratégias de alocação pela pergunta que o negócio de fato faz: **dado
um budget de N clientes, quanto lucro incremental cada estratégia entrega ao
escolher seus top-N?** Uma estratégia é um *ranking* de clientes; a curva varre
N de 0 ao total e, em cada N, mede o lucro líquido incremental **causal** dos
top-N daquela estratégia.

Três estratégias entram, todas rankeando o mesmo holdout:

- **modelo de uplift** — ordena por τ previsto (o X-learner);
- **conversão crua** — ordena por P(converte) previsto, o baseline que ignora
  incrementalidade e mira quem converte de qualquer jeito;
- **aleatório** — ordem uniforme (seed da config), a linha de base sem sinal.

Mais uma família, o **modelo híbrido** (`hybrid_score`/`hybrid_ranking`,
2026-07-10): `score = uplift_x_learner + λ · p_convert_cru`, um grid de λ
(`cfg.hybrid_lambda_grid`, default `[0, 0.1, 0.3, 0.5]`) gera uma estratégia por
valor. `λ=0` é o modelo de uplift puro — o ponto de controle do grid, não
tratado à parte. A pergunta que o híbrido responde: a curva de ganho em R$
favorece conversão crua (ticket alto) sobre o uplift causal puro
(`uplift_eval.qini_by_strategy` mostra o inverso em Qini/AUUC) — um híbrido
que empresta um pouco do sinal de conversão crua recupera parte do lucro em
R$ sem abrir mão de toda a ordenação causal?

E o **híbrido dinâmico** (`dynamic_hybrid_score`/`dynamic_hybrid_ranking`): em
vez de um λ fixo para o holdout inteiro, o peso varia por cliente, proporcional
à **incerteza da estimativa de τ** — a discordância interna do X-learner entre
seus dois estimadores de CATE (`|dhat_t − dhat_c|`, de
`uplift.predict_cate_uncertainty`). Onde a estimativa é menos confiável, o score
confia mais no prior de conversão; onde é confiável, confia no uplift quase puro.
`cfg.dynamic_hybrid_gamma_grid` (default `[0.5, 1.0, 2.0]`) varia a agressividade
da resposta à incerteza. `best_lambda_by_decile` é o diagnóstico exploratório que
motiva essa família (mostra se o λ ideal varia com o budget); `dynamic_lambda_by_budget`
mostra como o peso do híbrido dinâmico de fato se comporta ao longo do budget,
γ a γ.

**Por que não IPW nem Direct Method.** Ambos avaliam sobre *receita bruta
realizada* (`conversion_value − reward_cost` de fato ocorrido), que soma a
conversão que a oferta causou com a que aconteceria sem ela — não isolam o
incremental. A curva aqui mede só o **ganho causal**: a diferença tratado −
controle observada no RCT (Premissa 4), não a receita bruta de quem foi tratado.

**O contrafactual é observado, não previsto (estilo Qini).** O ranking vem do
modelo, mas o *ganho* de cada prefixo top-N sai do dado real. A **conversão
incremental** é o contrafactual escalado estilo Qini sobre `converted` (0/1):

    conversao_incremental(N) = C_tratado(N) − C_controle(N) · N_tratado(N)/N_controle(N)

Escalar o controle pela razão tratado/controle é a mesma correção da curva Qini
(`uplift_eval.qini_curve`): estima o contrafactual dos tratados a partir dos
controles observados no mesmo prefixo, sem assumir grupos de tamanho igual. Por
ser 0/1, sem variância de ticket, é a métrica **estável**.

**O lucro incremental é conversão incremental × lucro médio por conversão
tratada** (`_profit_per_treated_conversion`), não o contrafactual escalado
direto sobre o lucro por linha:

    lucro_incremental(N) = conversao_incremental(N) · lucro_medio_por_conversao_tratada(N)

Quantas conversões a oferta causa, vezes quanto vale uma conversão típica do
grupo escolhido. A versão anterior — `L_tratado(N) − L_controle(N)·razão` sobre
o lucro por linha — era **instável**: multiplicava o fator de escala volátil
pela soma acumulada de `conversion_value` (variância de ticket enorme, até
R$1015), fazendo o ganho *cair* com o budget. Fatorar em (contagem estável) ×
(lucro médio estável) remove a instabilidade na origem: o ticket alto de um
cliente entra diluído numa média, não multiplicado pela razão.

**Lucro líquido, por linha, sem assimetria por braço.** O lucro por linha é
`conversion_value − reward_cost`, igual para tratado e controle. `reward_cost`
já vem zerado em quem não converteu (`cost.add_reward_cost`, G6): o desconto é
concedido a quem atinge o `min_value` na validade, **view ou não**
(`test_unviewed_conversion_still_costs`). O lucro médio por conversão tratada é
sobre esse `net_profit` real — a conversão incremental é uma conversão causada,
que se materializa como conversão de tratado e carrega seu desconto pago.

**Conversão incremental é a mesma fórmula sobre `converted`.** `converted` é
0/1 por linha (sem a assimetria de custo do lucro — não há desconto a debitar
numa contagem de conversões), então
`conversao_incremental(N) = C_tratado(N) − C_controle(N) · N_tratado(N)/N_controle(N)`
lê "quantas conversões a mais os top-N tiveram por causa da oferta", no mesmo
contrafactual estilo Qini. É o numerador em unidades de clientes, não reais —
complementa o lucro (R$) quando a pergunta é "quantas conversões a política
realmente causa", não só "quanto lucro".

**Intervalo de confiança por bootstrap não paramétrico.** A curva é uma
estatística não trivial (razão de somas cumulativas) sem forma fechada de
variância; reamostra-se o holdout inteiro com reposição
(`cfg.gain_curve_n_bootstrap` réplicas, seed da config) e recomputa-se a curva
completa por réplica — mesmo padrão de reamostragem do teste de placebo
(`uplift_eval.placebo_qini_distribution`), mas aqui a variação é amostral, não
uma distribuição nula. O IC em cada N é o percentil
`cfg.gain_curve_confidence_level` das réplicas nesse N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig

TREATMENT_COLUMN = "treatment"

#: Lucro líquido realizado por linha do holdout: receita da conversão menos o
#: desconto pago. É o mesmo reward que a política otimiza, medido no dado.
NET_PROFIT_COLUMN = "net_profit_realized"


def add_net_profit(holdout_df: pd.DataFrame) -> pd.DataFrame:
    """Anexa o lucro líquido realizado por linha: `conversion_value − reward_cost`."""
    return holdout_df.assign(
        **{NET_PROFIT_COLUMN: holdout_df["conversion_value"] - holdout_df["reward_cost"]}
    )


def _scaled_counterfactual_gain(
    values: np.ndarray, treated: np.ndarray, control: np.ndarray
) -> np.ndarray:
    """Contrafactual escalado estilo Qini de uma métrica por linha, acumulado por N.

    `V_tratado(N) − V_controle(N) · N_tratado(N)/N_controle(N)`: o contrafactual
    dos tratados estimado a partir dos controles observados no mesmo prefixo.
    Onde ainda não há controle no prefixo, a razão é indefinida — mantém-se o
    ganho só do braço tratado, sem subtrair nada (nenhum salto, nenhum NaN).
    """
    cum_treated = np.concatenate([[0.0], np.cumsum(values * treated)])
    cum_control = np.concatenate([[0.0], np.cumsum(values * control)])
    cum_n_treated = np.concatenate([[0.0], np.cumsum(treated)])
    cum_n_control = np.concatenate([[0.0], np.cumsum(control)])

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(cum_n_control > 0, cum_n_treated / cum_n_control, 0.0)
    return cum_treated - cum_control * scale


def _monotone_envelope(cumulative: np.ndarray) -> np.ndarray:
    """Máximo acumulado (`cummax`) do ganho por prefixo — o envelope não-decrescente.

    O contrafactual escalado (`_scaled_counterfactual_gain`) **não é monótono**
    em N: `scale = n_tratado/n_controle` reescala *todo* o histórico de controle
    a cada passo, então um lote de controles convertendo com ticket alto entrando
    no prefixo faz o termo subtraído crescer mais rápido que o dos tratados, e o
    ganho cru cai — instabilidade conhecida do estimador de razão estilo Qini
    quando o balanço tratado/controle varia ao longo do ranking (aqui o controle
    é ~30% do holdout: ver é escolha do cliente, não braço aleatório). A queda é
    real no estimador, não nos dados (a curva de `conversions`, sem a variância
    de ticket, é quase lisa), mas não corresponde a nenhuma decisão: ninguém é
    obrigado a gastar o budget inteiro — pode parar no melhor prefixo ≤ N.

    O envelope responde essa pergunta de negócio — "com budget N, o melhor lucro
    incremental que consigo travar" — e é o que a curva reporta. Aplicado também
    a `conversions` e às bandas de IC, para envelope e banda ficarem coerentes.
    """
    return np.maximum.accumulate(cumulative)


def _profit_per_treated_conversion(profit: np.ndarray, treated: np.ndarray, converted: np.ndarray) -> np.ndarray:
    """Lucro médio por conversão tratada, acumulado por prefixo top-N.

    Uma conversão *incremental* é uma conversão que a oferta causou; ela se
    materializa como conversão de tratado e carrega um `net_profit` real
    (`conversion_value − reward_cost`). O valor de uma conversão incremental é,
    portanto, o lucro médio das conversões de tratado **dentro do prefixo** — a
    estatística coerente com o grupo que o ranking está de fato escolhendo.

    É uma **média** (soma do lucro tratado-convertido ÷ nº de tratados
    convertidos), não uma soma escalada: estabiliza conforme N cresce em vez de
    ser reescalada a cada passo pela razão tratado/controle. É isso que torna o
    ganho estável — o ticket alto de um cliente entra diluído numa média, não
    multiplicado pelo fator de escala volátil. Prefixo sem nenhum tratado
    convertido ainda → 0 (não há conversão a valorar; o nº de conversões
    incrementais ali também é ~0, então o ganho é ~0 de qualquer forma).
    """
    treated_conv = treated * converted
    cum_profit = np.concatenate([[0.0], np.cumsum(profit * treated_conv)])
    cum_count = np.concatenate([[0.0], np.cumsum(treated_conv)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_count > 0, cum_profit / cum_count, 0.0)


def incremental_gain_curve(ranking: np.ndarray, holdout_df: pd.DataFrame) -> pd.DataFrame:
    """Curva `(n, gain, conversions)` de uma estratégia sobre o holdout.

    `ranking` é a ordem de prioridade dos clientes (índices das linhas de
    `holdout_df`, do mais prioritário ao menos): a estratégia escolhe os top-N
    tomando os primeiros N desse vetor. `holdout_df` precisa de `NET_PROFIT_COLUMN`
    (via `add_net_profit`), `treatment` e `converted`.

    `conversions` (conversões incrementais, contagem) é o contrafactual escalado
    estilo Qini sobre `converted`: soma nos tratados menos a soma nos controles
    **escalada** pela razão tratado/controle do prefixo — o contrafactual dos
    tratados estimado a partir dos controles observados ali. É a métrica
    **estável**: 0/1 por linha, sem a variância de ticket.

    `gain` (lucro líquido incremental, R$) é `conversions × lucro médio por
    conversão tratada no prefixo` (`_profit_per_treated_conversion`) — quantas
    conversões a oferta causa, vezes quanto vale uma conversão típica desse
    grupo. **Substitui** o contrafactual escalado direto sobre o lucro por
    linha, que era instável: aquele multiplicava o fator de escala volátil pela
    soma acumulada de `conversion_value` (variância de ticket enorme), fazendo o
    ganho cair com o budget. Fatorar em (contagem estável) × (lucro médio
    estável) remove essa instabilidade na origem, não só na apresentação.

    Ambas passam pelo **envelope monótono** (`_monotone_envelope`) como salvaguarda
    — a nova fórmula já é quase monótona, mas o envelope garante a leitura de
    negócio "com budget N, o melhor que consigo travar" (parar no melhor prefixo
    ≤ N), não-decrescente.

    Retorna `[n, gain, conversions]` com uma linha por N de 0 ao total; N=0 é 0.
    """
    ordered = holdout_df.loc[ranking]
    treated = (ordered[TREATMENT_COLUMN].to_numpy() == 1).astype(float)
    control = 1.0 - treated
    profit = ordered[NET_PROFIT_COLUMN].to_numpy()
    converted = ordered["converted"].to_numpy(dtype=float)

    conversions_raw = _scaled_counterfactual_gain(converted, treated, control)
    profit_per_conv = _profit_per_treated_conversion(profit, treated, converted)

    gain = _monotone_envelope(conversions_raw * profit_per_conv)
    conversions = _monotone_envelope(conversions_raw)

    return pd.DataFrame(
        {"n": np.arange(len(gain)), "gain": gain, "conversions": conversions}
    )


def uplift_ranking(uplift_pred: pd.DataFrame) -> np.ndarray:
    """Ranking pela estratégia do modelo de uplift: τ previsto decrescente.

    `uplift_pred` é a saída de `uplift.predict` (coluna `uplift`), alinhada por
    índice ao holdout. Empate resolvido pela ordem estável do índice, para a
    curva ser determinística dada a mesma entrada.
    """
    return uplift_pred.sort_values("uplift", ascending=False, kind="stable").index.to_numpy()


def completion_ranking(p_convert: pd.Series) -> np.ndarray:
    """Ranking pela conversão crua: P(converte) previsto decrescente.

    Mira quem tem maior propensão a converter, ignorando se a oferta *causa* a
    conversão — a estratégia que o modelo de uplift precisa bater.
    `p_convert` vem do baseline preditivo, alinhado ao holdout.
    """
    return p_convert.sort_values(ascending=False, kind="stable").index.to_numpy()


def hybrid_score(uplift_pred: pd.Series, p_convert: pd.Series, lambda_: float) -> pd.Series:
    """Score híbrido X-learner + conversão crua: `uplift + λ · p_convert`.

    Soma direta, sem normalizar: os dois termos já vivem em escalas parecidas
    (τ do X-learner ∈ aprox. [-1, 1], `p_convert` ∈ [0, 1]) e `λ` funciona como
    o peso literal que a fórmula pede, não um peso relativo pós-padronização.
    `λ=0` degenera no modelo de uplift puro (`uplift_ranking`) — é o ponto de
    controle do grid `cfg.hybrid_lambda_grid`, não um caso especial tratado à
    parte. `uplift_pred`/`p_convert` precisam estar alinhados pelo mesmo índice
    do holdout.
    """
    return uplift_pred + lambda_ * p_convert


def hybrid_ranking(uplift_pred: pd.Series, p_convert: pd.Series, lambda_: float) -> np.ndarray:
    """Ranking pelo score híbrido (`hybrid_score`) decrescente.

    Mesma convenção de `uplift_ranking`/`completion_ranking`: desempate estável
    pela ordem do índice, para a curva ser determinística dada a mesma entrada.
    """
    return hybrid_score(uplift_pred, p_convert, lambda_).sort_values(
        ascending=False, kind="stable"
    ).index.to_numpy()


def _minmax(s: pd.Series) -> pd.Series:
    """Normaliza para [0, 1] no próprio holdout (epsilon evita divisão por zero)."""
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def dynamic_hybrid_score(
    uncertainty: pd.Series, uplift_pred: pd.Series, p_convert: pd.Series, gamma: float = 1.0
) -> pd.Series:
    """Blend uplift + conversão crua, com peso local pela **incerteza** do τ.

    Onde `hybrid_score` usa um λ constante para o holdout inteiro,
    `dynamic_hybrid_score` deixa o peso variar **por cliente**: onde a estimativa
    de τ é menos confiável, o score empresta mais peso do prior de conversão;
    onde é confiável, confia no uplift quase puro. `uncertainty` é essa medida
    de confiança — `uplift.predict_cate_uncertainty` fornece a discordância
    interna do X-learner (`|dhat_t − dhat_c|`), a incerteza da própria
    estimativa, não o tamanho do efeito.

    Os dois termos entram **normalizados min-max** antes do blend — diferente de
    `hybrid_score`, que soma os scores brutos. Aqui a normalização é necessária:
    `lambda_local` é uma fração em [0, 1] e a combinação convexa
    `(1-λ)·uplift + λ·p_convert` só faz sentido com os dois termos na mesma escala.

        incerteza_norm = uncertainty / max(uncertainty)
        lambda_local = incerteza_norm ** gamma
        score = (1 - lambda_local) · uplift_norm + lambda_local · p_convert_norm

    A incerteza é escalada pelo **máximo** (não min-max): incerteza 0 precisa
    mapear em `lambda_local = 0` (confiança total no uplift), o que subtrair o
    mínimo do grupo destruiria. Já `uplift_pred`/`p_convert` são min-max: ali só
    a posição relativa importa, não o zero.

    `gamma` controla a agressividade da resposta à incerteza: `gamma=1` responde
    linearmente; `gamma>1` só empresta peso relevante na incerteza muito alta
    (conservador); `gamma<1` empresta peso já em incerteza moderada (agressivo).
    Todas as séries precisam estar alinhadas pelo mesmo índice do holdout.
    """
    lambda_local = (uncertainty / (uncertainty.max() + 1e-9)) ** gamma
    return (1 - lambda_local) * _minmax(uplift_pred) + lambda_local * _minmax(p_convert)


def dynamic_hybrid_ranking(
    uncertainty: pd.Series, uplift_pred: pd.Series, p_convert: pd.Series, gamma: float = 1.0
) -> np.ndarray:
    """Ranking pelo score híbrido dinâmico (`dynamic_hybrid_score`) decrescente.

    Mesma convenção de desempate estável de `uplift_ranking`/`hybrid_ranking`.
    """
    return dynamic_hybrid_score(uncertainty, uplift_pred, p_convert, gamma).sort_values(
        ascending=False, kind="stable"
    ).index.to_numpy()


def _marginal_conversions_by_decile(curva: pd.DataFrame, n: int) -> np.ndarray:
    """Conversões incrementais marginais por decil, a partir da curva acumulada."""
    budgets = [0] + [round(n * d / 10) for d in range(1, 11)]
    cumulative = curva.set_index("n").loc[budgets, "conversions"].to_numpy()
    return np.diff(cumulative)


def incremental_conversions_by_decile(
    ranking: np.ndarray,
    holdout_df: pd.DataFrame,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Conversões incrementais **marginais** por decil de um ranking.

    Cada decil cobre 10% do holdout na ordem do ranking (D1 = topo, D10 =
    cauda). A métrica é a diferença do acumulado de `incremental_gain_curve`
    entre dois cortes de budget — quantas conversões causadas pela oferta cada
    faixa do ranking contribui, no mesmo contrafactual escalado estilo Qini de
    `conversions`.

    Com `cfg`, anexa IC bootstrap (`conversions_lo`/`conversions_hi`) por decil
    (mesmo padrão de `gain_curves_with_ci`). Retorna `[decil, conversions]`
    (+ IC se `cfg`).
    """
    holdout_with_profit = add_net_profit(holdout_df)
    n = len(holdout_df)
    ordered = holdout_with_profit.loc[ranking].reset_index(drop=True)

    point_curve = incremental_gain_curve(np.arange(n), ordered)
    marginals = _marginal_conversions_by_decile(point_curve, n)

    out = pd.DataFrame({"decil": np.arange(1, 11), "conversions": marginals})

    if cfg is None:
        return out

    alpha = 1.0 - cfg.gain_curve_confidence_level
    rng = np.random.default_rng(cfg.seed)
    replicas = np.empty((cfg.gain_curve_n_bootstrap, 10))

    for i in range(cfg.gain_curve_n_bootstrap):
        sample_idx = np.sort(rng.integers(0, n, size=n))
        resampled = ordered.iloc[sample_idx].reset_index(drop=True)
        curva = incremental_gain_curve(np.arange(len(resampled)), resampled)
        replicas[i] = _marginal_conversions_by_decile(curva, n)

    conv_lo, conv_hi = np.quantile(replicas, [alpha / 2, 1 - alpha / 2], axis=0)
    return out.assign(conversions_lo=conv_lo, conversions_hi=conv_hi)


def best_lambda_by_decile(
    uplift_pred: pd.Series,
    p_convert: pd.Series,
    holdout_df: pd.DataFrame,
    lambda_grid: list[float],
) -> pd.DataFrame:
    """Exploração suja: por decil de budget, qual λ fixo maximiza o lucro ali?

    Para cada λ do grid, mede o lucro incremental do híbrido (`hybrid_ranking`)
    nos cortes de 10%, 20%, … 100% do holdout; para cada decil, devolve o λ que
    rende mais. Se o λ ótimo variar de decil para decil, um λ **dinâmico** tem o
    que ganhar sobre um λ fixo; se for constante, não. Diagnóstico que motiva (ou
    descarta) o híbrido dinâmico — não entra na política.

    Retorna `[decil, melhor_lambda, gain]`, um por decil.
    """
    holdout_with_profit = add_net_profit(holdout_df)
    n = len(holdout_df)
    decile_budgets = [round(n * d / 10) for d in range(1, 11)]

    gain_por_lambda = {}
    for lam in lambda_grid:
        ranking = hybrid_ranking(uplift_pred, p_convert, lam)
        curva = incremental_gain_curve(np.arange(n), holdout_with_profit.loc[ranking].reset_index(drop=True))
        gain_por_lambda[lam] = curva.set_index("n").loc[decile_budgets, "gain"].to_numpy()

    tabela = pd.DataFrame(gain_por_lambda, index=range(1, 11))
    return pd.DataFrame({
        "decil": tabela.index,
        "melhor_lambda": tabela.idxmax(axis=1).to_numpy(),
        "gain": tabela.max(axis=1).to_numpy(),
    })


def dynamic_lambda_by_budget(
    uncertainty: pd.Series,
    uplift_pred: pd.Series,
    p_convert: pd.Series,
    gamma_grid: list[float],
) -> pd.DataFrame:
    """λ_local médio acumulado nos top-N do ranking, para cada γ do grid.

    `dynamic_hybrid_score` calcula um `lambda_local` por cliente; esta função
    ordena o holdout pelo score de cada γ (mesmo ranking usado para o ganho) e
    acumula a média de `lambda_local` dentro de cada prefixo top-N — como o
    peso efetivo do prior de conversão se comporta ao longo do budget, γ a γ.

    Retorna `[gamma, n, lambda_medio]`, uma linha por N (todo o holdout) e γ.
    """
    partes = []
    for gamma in gamma_grid:
        lambda_local = (uncertainty / (uncertainty.max() + 1e-9)) ** gamma
        ranking = dynamic_hybrid_ranking(uncertainty, uplift_pred, p_convert, gamma)
        lambda_ordenado = lambda_local.loc[ranking].to_numpy()
        media_acumulada = np.cumsum(lambda_ordenado) / np.arange(1, len(lambda_ordenado) + 1)
        partes.append(pd.DataFrame({
            "gamma": gamma,
            "n": np.arange(1, len(lambda_ordenado) + 1),
            "lambda_medio": media_acumulada,
        }))
    return pd.concat(partes, ignore_index=True)


def random_ranking(holdout_df: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    """Ranking aleatório: permutação uniforme dos clientes, seed da config.

    A linha de base sem sinal — se uma estratégia não bate esta, não há sinal
    de incrementalidade a explorar. Seed fixa para reprodutibilidade (REQ-110).
    """
    rng = np.random.default_rng(cfg.seed)
    return rng.permutation(holdout_df.index.to_numpy())


def softmax_ranking(
    score: pd.Series, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    """Ranking por **amostragem softmax (Boltzmann)**: chance ∝ `exp(score/τ)`.

    Em vez do corte duro por score decrescente (`uplift_ranking` e afins), cada
    cliente entra na ordem com probabilidade proporcional a `exp(score_norm/τ)` —
    quem está logo abaixo do topo ganha chance real de ser priorizado, quem está
    muito abaixo ganha chance pequena mas não-nula. É o mecanismo de rank padrão
    dos modelos (`BlendedUpliftModel.rank`, `serve.recommend`): dá exploração ao
    ranqueamento sem abandonar o sinal do score.

    `score` entra **normalizado min-max** para [0, 1] (`_minmax`) antes do
    softmax — sem isso, o mesmo `temperature` produziria suavidades diferentes
    conforme a escala do score (o blend fixo vive em ~[-1, 1.5], o dinâmico em
    [0, 1]); normalizado, `τ` é uma fração comparável da amplitude do score entre
    modos e datasets.

    A permutação é uma amostra exata de **Plackett-Luce** (`p_i ∝ exp(score_i/τ)`,
    sequencial sem reposição) via o **truque de Gumbel-max**: somar ruído
    `Gumbel(0,1)` a `score_norm/τ` e ordenar por essa chave decrescente produz a
    mesma distribuição em O(n log n), sem o laço de renormalização. `temperature`
    controla a suavidade:

    - `τ → 0`: o score domina o ruído → ordena por score (o determinístico atual,
      o caso especial). Tratado explicitamente (sem divisão por zero).
    - `τ → ∞`: o ruído domina → permutação uniforme, o limite de `random_ranking`.

    Determinístico dado `rng` (mesma seed → mesma permutação), como
    `random_ranking`. `score` precisa estar alinhado ao índice do holdout; a saída
    é a ordem de prioridade (índices, do mais ao menos prioritário).
    """
    if temperature <= 0:
        return score.sort_values(ascending=False, kind="stable").index.to_numpy()

    score_norm = _minmax(score).to_numpy()
    keys = score_norm / temperature + rng.gumbel(size=len(score_norm))
    order = np.argsort(-keys, kind="stable")
    return score.index.to_numpy()[order]


def gain_curves(
    rankings: dict[str, np.ndarray], holdout_df: pd.DataFrame
) -> pd.DataFrame:
    """`incremental_gain_curve` para cada estratégia nomeada, numa tabela longa.

    Mesmo holdout para todas — a comparação exige que nenhuma estratégia veja
    uma base diferente das outras. Retorna `[strategy, n, gain, conversions]`.
    """
    holdout_with_profit = add_net_profit(holdout_df)
    partes = []
    for nome, ranking in rankings.items():
        curva = incremental_gain_curve(ranking, holdout_with_profit)
        partes.append(curva.assign(strategy=nome))
    return pd.concat(partes, ignore_index=True)[["strategy", "n", "gain", "conversions"]]


def gain_curves_with_ci(
    rankings: dict[str, np.ndarray],
    holdout_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """`gain_curves` com intervalo de confiança por bootstrap não paramétrico.

    Reamostra o holdout inteiro com reposição `cfg.gain_curve_n_bootstrap`
    vezes (seed da config) e recomputa a curva completa por réplica, para cada
    estratégia. Cada ranking é reaplicado à reamostra pela **posição relativa**
    dos clientes originais (o ranking já ordena o holdout; reamostrar preserva
    essa ordem, só duplica/omite linhas), não recalculado — a estratégia em si
    não muda entre réplicas, só a base amostrada.

    O IC em cada N é o percentil `cfg.gain_curve_confidence_level` das réplicas
    nesse N. Retorna `[strategy, n, gain, gain_lo, gain_hi, conversions,
    conversions_lo, conversions_hi]`.
    """
    holdout_with_profit = add_net_profit(holdout_df)
    alpha = 1.0 - cfg.gain_curve_confidence_level
    rng = np.random.default_rng(cfg.seed)
    n_rows = len(holdout_with_profit)

    partes = []
    for nome, ranking in rankings.items():
        ordered = holdout_with_profit.loc[ranking].reset_index(drop=True)
        point = incremental_gain_curve(np.arange(n_rows), ordered)

        replicas_gain = np.empty((cfg.gain_curve_n_bootstrap, n_rows + 1))
        replicas_conv = np.empty((cfg.gain_curve_n_bootstrap, n_rows + 1))
        for i in range(cfg.gain_curve_n_bootstrap):
            sample_idx = np.sort(rng.integers(0, n_rows, size=n_rows))
            resampled = ordered.iloc[sample_idx].reset_index(drop=True)
            curva = incremental_gain_curve(np.arange(len(resampled)), resampled)
            replicas_gain[i] = curva["gain"].to_numpy()
            replicas_conv[i] = curva["conversions"].to_numpy()

        gain_lo, gain_hi = np.quantile(replicas_gain, [alpha / 2, 1 - alpha / 2], axis=0)
        conv_lo, conv_hi = np.quantile(replicas_conv, [alpha / 2, 1 - alpha / 2], axis=0)

        partes.append(point.assign(
            strategy=nome,
            gain_lo=gain_lo, gain_hi=gain_hi,
            conversions_lo=conv_lo, conversions_hi=conv_hi,
        ))

    return pd.concat(partes, ignore_index=True)[[
        "strategy", "n", "gain", "gain_lo", "gain_hi",
        "conversions", "conversions_lo", "conversions_hi",
    ]]


def gain_at_budget(curves: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Métricas de cada estratégia num budget fixo N (leitura da curva).

    `budget` é o N de clientes; devolve a linha da curva no maior N ≤ `budget`
    de cada estratégia (o valor efetivamente entregue com aquele orçamento) —
    todas as colunas de `curves` (`gain`/`conversions`, com ou sem IC). Serve o
    "se meu budget for N, quanto ganho?" direto, sem procurar na curva inteira.
    """
    cortadas = curves[curves["n"] <= budget]
    return (
        cortadas.sort_values("n").groupby("strategy", as_index=False).last()
    )
