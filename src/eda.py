"""Visões da EDA (REQ-108) e diagnóstico de balanço de covariáveis (REQ-109).

Cada visão é uma função **pura** que agrega no Spark e devolve um pandas
pequeno. O notebook importa esse pandas e o desenha com as primitivas genéricas
de `src/viz.py` (`line_series`, `vertical_bars`, `heatmap`, `faceted`, …) — a
figura é a *chamada* da primitiva, não um construtor por gráfico aqui (figura
`fig_*` por visão é o anti-padrão que essas primitivas substituem). Nenhuma
transformação vive no notebook (NFR "notebooks apenas para análise").

O estilo das figuras (paleta validada, rótulo direto) vem do mesmo `src/viz.py`;
nada de estilo ad hoc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PipelineConfig

# Covariáveis de cliente do contrato. `gender` entra como indicadores (é categórica).
CLIENT_COVARIATES = ("age", "credit_card_limit", "tenure_days", "identity_missing")

HIST_COMPRA_FEATURES = (
    "hist_spend_total", "hist_txn_count", "hist_avg_ticket", "hist_spend_std",
    "hist_recency_days", "hist_frequency", "hist_spend_trend",
)
HIST_OFERTA_FEATURES = (
    "hist_offers_received", "hist_offers_viewed", "hist_offers_completed",
    "hist_offers_received_bogo", "hist_offers_received_discount",
    "hist_view_rate", "hist_conv_rate_bogo", "hist_conv_rate_discount",
    "hist_completed_unseen_flag", "hist_time_view_to_conv",
)
HIST_BALANCE_FEATURES = HIST_COMPRA_FEATURES + HIST_OFERTA_FEATURES

EVENT_ORDER = ("offer received", "offer viewed", "offer completed", "transaction")


# --- Visão 1: distribuição dos quatro eventos no tempo -------------------------

def events_over_time(events: DataFrame) -> pd.DataFrame:
    """Contagem de cada tipo de evento por dia inteiro de `time`."""
    por_dia = (
        events.withColumn("dia", F.floor(F.col("time")).cast("int"))
        .groupBy("event", "dia")
        .agg(F.count("*").alias("eventos"))
        .toPandas()
    )
    return por_dia.sort_values(["event", "dia"]).reset_index(drop=True)


# --- Visão 2: as seis ondas de campanha ----------------------------------------

def campaign_waves(processed: DataFrame) -> pd.DataFrame:
    """Recebimentos, exposição e conversão por onda de campanha."""
    return (
        processed.groupBy("campaign_wave", "received_time")
        .agg(
            F.count("*").alias("recebimentos"),
            F.sum("treatment").alias("vistos"),
            F.sum("converted").alias("conversoes"),
        )
        .orderBy("campaign_wave")
        .toPandas()
        .assign(taxa_view=lambda d: d["vistos"] / d["recebimentos"])
    )


def campaign_validity_windows(processed: DataFrame, events: DataFrame) -> tuple[pd.DataFrame, float]:
    """Janela de validade por onda e quanto dela é observável antes do fim dos dados.

    `valid_until = received_time + duration`. Quando ultrapassa o último evento bruto,
    a diferença é censura à direita — conversão e taxas das ondas tardias ficam
    subestimadas por construção, não por fadiga do cliente.
    """
    fim = float(events.agg(F.max("time").alias("fim")).first()["fim"])
    ondas = (
        processed.groupBy("campaign_wave", "received_time")
        .agg(
            F.avg("duration").alias("duration"),
            F.count("*").alias("recebimentos"),
        )
        .orderBy("campaign_wave")
        .toPandas()
    )
    ondas["valid_until"] = ondas["received_time"] + ondas["duration"]
    ondas["janela_observavel"] = ondas["valid_until"].clip(upper=fim)
    ondas["censurada"] = ondas["valid_until"] > fim
    ondas["dias_censurados"] = (ondas["valid_until"] - fim).clip(lower=0)
    ondas["rotulo"] = ondas.apply(
        lambda r: f"onda {int(r.campaign_wave)} · t={r.received_time:g} · {r.duration:g}d",
        axis=1,
    )
    ondas["recebimentos_censurados"] = np.where(ondas["censurada"], ondas["recebimentos"], 0)
    return ondas, fim


# --- Visão 3: completou sem ver, por tipo de oferta -----------------------------

def completed_unseen_by_type(events: DataFrame, offers: DataFrame) -> pd.DataFrame:
    """Taxa de `offer completed` sem view precedente, por tipo de oferta.

    `informational` não emite `offer completed` — a linha existe com zero eventos
    e taxa nula, e é exatamente esse vazio que a garantia G5 protege.
    """
    completados = events.filter(F.col("event") == "offer completed").select(
        "account_id", F.col("offer_ref").alias("offer_id"), F.col("time").alias("completed_time"))
    vistos = events.filter(F.col("event") == "offer viewed").select(
        "account_id", F.col("offer_ref").alias("offer_id"), F.col("time").alias("view_time"))

    pares = (
        completados.join(vistos, on=["account_id", "offer_id"], how="left")
        .groupBy("account_id", "offer_id", "completed_time")
        .agg(F.max((F.col("view_time") <= F.col("completed_time")).cast("int")).alias("teve_view"))
    )
    tipado = pares.join(offers.select(F.col("id").alias("offer_id"), "offer_type"), on="offer_id", how="left")
    agregado = tipado.groupBy("offer_type").agg(
        F.count("*").alias("completados"),
        F.sum((F.coalesce(F.col("teve_view"), F.lit(0)) == 0).cast("int")).alias("sem_view"),
    )
    todos_tipos = offers.select("offer_type").distinct()
    frame = (
        todos_tipos.join(agregado, on="offer_type", how="left")
        .fillna({"completados": 0, "sem_view": 0})
        .orderBy("offer_type")
        .toPandas()
    )
    frame["taxa_sem_view"] = np.where(frame["completados"] > 0, frame["sem_view"] / frame["completados"], np.nan)
    return frame


# --- Visão 4: sobreposição dos nulos de perfil ---------------------------------

def identity_null_overlap(raw_profile: DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Contagens dos três campos ausentes e da interseção (Premissa 3)."""
    sentinela = F.col("age") == cfg.age_sentinel
    sem_genero = F.col("gender").isNull()
    sem_limite = F.col("credit_card_limit").isNull()

    linha = raw_profile.select(
        F.count("*").alias("clientes"),
        F.sum(sentinela.cast("int")).alias(f"age={cfg.age_sentinel}"),
        F.sum(sem_genero.cast("int")).alias("gender nulo"),
        F.sum(sem_limite.cast("int")).alias("credit_card_limit nulo"),
        F.sum((sentinela & sem_genero & sem_limite).cast("int")).alias("os três, juntos"),
        F.sum((sentinela | sem_genero | sem_limite).cast("int")).alias("ao menos um"),
    ).first().asDict()

    total = linha.pop("clientes")
    return pd.DataFrame(
        [{"conjunto": k, "clientes": v, "fracao": v / total} for k, v in linha.items()]
    ).assign(total_clientes=total)


# --- Ato 3: compra fora de qualquer janela de oferta ---------------------------

def unattributable_transaction_share(events: DataFrame, attributed: DataFrame) -> pd.DataFrame:
    """Fração das transações que não cai em NENHUMA janela de recebimento.

    Diferente da reconciliação do audit (que mede pós-view, para o label), aqui a
    pergunta é mais ampla: essa compra tinha ALGUMA oferta ativa por perto, vista
    ou não? Se não, é comportamento espontâneo — nem o label mais generoso a
    prenderia a um envio. Evidência de que "sure thing" é um fenômeno real e
    grande, não um artefato da regra de atribuição.
    """
    txns = (
        events.filter(F.col("event") == "transaction")
        .select("account_id", F.col("time").alias("txn_time"))
        .withColumn("row_id", F.monotonically_increasing_id())
    )
    total = txns.count()
    em_alguma_janela = (
        attributed.join(txns, on="account_id", how="inner")
        .filter((F.col("txn_time") >= F.col("received_time")) & (F.col("txn_time") <= F.col("valid_until")))
        .select("row_id").distinct().count()
    )
    fora = total - em_alguma_janela
    return pd.DataFrame([
        {"grupo": "fora de qualquer janela de oferta", "transacoes": fora, "fracao": fora / total},
        {"grupo": "dentro de alguma janela", "transacoes": em_alguma_janela, "fracao": em_alguma_janela / total},
    ])


# --- Ato 4: positividade — todo perfil recebeu todo tipo de oferta? ------------

def positivity_by_offer_type(processed: DataFrame, profile: DataFrame) -> pd.DataFrame:
    """Para cada tipo de oferta, quantos clientes da base NUNCA o receberam.

    IPW exige sobreposição: um cliente que nunca teve chance de receber um tipo
    não tem contrafactual observável para ele (Premissa 8). Isso não é sobre
    `identity_missing` — é sobre o desenho do envio em si.
    """
    total = profile.select("account_id").distinct().count()
    tipos = [r["offer_type"] for r in processed.select("offer_type").distinct().collect()]
    linhas = []
    for tipo in sorted(tipos):
        recebeu = processed.filter(F.col("offer_type") == tipo).select("account_id").distinct()
        n_recebeu = recebeu.count()
        linhas.append({
            "offer_type": tipo, "clientes_total": total, "receberam": n_recebeu,
            "nunca_receberam": total - n_recebeu, "cobertura": n_recebeu / total,
        })
    return pd.DataFrame(linhas)


# --- Ato 5: heterogeneidade — conversão por tipo × segmento --------------------

def conversion_by_type_and_segment(processed: DataFrame, profile: DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Taxa de conversão influence-aware por tipo de oferta × quartil de tenure.

    `tenure_days` é a mesma covariável já auditada em `covariate_balance` — aqui
    ela vira eixo de heterogeneidade, não de diagnóstico: onde a conversão varia
    forte por tipo × segmento é onde o uplift tem o que aprender.

    Três taxas, com denominadores explícitos porque é onde uma EDA mente sem
    querer: `taxa_conversao` divide por **recebimentos** (mistura quem nem viu),
    `taxa_view` isola a exposição, e `taxa_conversao_vistos` divide por **vistos**
    — a única que fala de resposta ao estímulo. Vale a identidade
    `taxa_conversao = taxa_view × taxa_conversao_vistos` (G3: converter exige ver).
    """
    q1, q2, q3 = profile.approxQuantile("tenure_days", [0.25, 0.5, 0.75], cfg.quantile_rel_error)
    faixa = (
        F.when(F.col("tenure_days") <= q1, "Q1 (mais novo)")
        .when(F.col("tenure_days") <= q2, "Q2")
        .when(F.col("tenure_days") <= q3, "Q3")
        .otherwise("Q4 (mais antigo)")
    )
    segmentado = profile.withColumn("tenure_q", faixa).select("account_id", "tenure_q")
    frame = (
        processed.join(segmentado, on="account_id", how="left")
        .groupBy("offer_type", "tenure_q")
        .agg(
            F.count("*").alias("n"),
            F.sum("treatment").alias("vistos"),
            F.sum("converted").alias("conversoes"),
            F.avg("converted").alias("taxa_conversao"),
        )
        .toPandas()
        .sort_values(["offer_type", "tenure_q"])
        .reset_index(drop=True)
    )
    frame["taxa_view"] = frame["vistos"] / frame["n"]
    frame["taxa_conversao_vistos"] = np.where(
        frame["vistos"] > 0, frame["conversoes"] / frame["vistos"], np.nan)
    return frame


# --- Visão 5: distribuições de features-chave ----------------------------------

def numeric_histogram(df: DataFrame, column: str, bins: int) -> pd.DataFrame:
    """Histograma calculado no Spark (nulos excluídos), devolvido como pandas."""
    limites = df.select(F.min(column).alias("lo"), F.max(column).alias("hi")).first()
    lo, hi = limites["lo"], limites["hi"]
    if lo is None:
        return pd.DataFrame({"centro": [], "contagem": []})
    if hi == lo:
        n = df.filter(F.col(column).isNotNull()).count()
        return pd.DataFrame({"centro": [float(lo)], "contagem": [n]})

    largura = (hi - lo) / bins
    bucket = F.least(F.floor((F.col(column) - F.lit(lo)) / F.lit(largura)), F.lit(bins - 1)).cast("int")
    contagens = (
        df.filter(F.col(column).isNotNull())
        .withColumn("bucket", bucket)
        .groupBy("bucket").agg(F.count("*").alias("contagem"))
        .orderBy("bucket").toPandas()
    )
    contagens["centro"] = lo + (contagens["bucket"] + 0.5) * largura
    return contagens[["centro", "contagem"]]


# --- REQ-109: balanço de covariáveis (SMD) -------------------------------------

def _with_gender_indicators(processed: DataFrame) -> tuple[DataFrame, list[str]]:
    niveis = sorted(r["gender"] for r in processed.select("gender").distinct().collect())
    colunas = []
    saida = processed
    for nivel in niveis:
        nome = f"gender={nivel}"
        saida = saida.withColumn(nome, (F.col("gender") == nivel).cast("double"))
        colunas.append(nome)
    return saida, colunas


def covariate_balance(processed: DataFrame, cfg: PipelineConfig, group_col: str = "treatment") -> pd.DataFrame:
    """SMD por covariável entre grupo 1 e grupo 0 de `group_col` — REQ-109.

    `SMD = (média_1 - média_0) / sqrt((var_1 + var_0) / 2)`, a definição usual de
    Cohen para covariáveis contínuas e binárias. Nulos de `age`/`credit_card_limit`
    são ignorados nas médias (é o segmento sentinela, cuja ausência é informativa
    e aparece por conta própria em `identity_missing`).

    Com `group_col="treatment"` (default) compara quem **viu** a oferta contra quem
    não viu — o pedido do REQ-109. Note que ver é comportamento **pós-tratamento**:
    balanço aqui é tranquilizador, mas não é o que verifica a Premissa 4 (a
    aleatorização do *envio*). Para isso, ver `assignment_balance`.

    Diagnóstico, não gate: acima do limiar a leitura causal fica qualificada, o
    estimador não muda (Premissas 4 e 5).
    """
    preparado, colunas_genero = _with_gender_indicators(processed)
    covariaveis = list(CLIENT_COVARIATES) + list(HIST_BALANCE_FEATURES) + colunas_genero

    agregacoes = []
    for c in covariaveis:
        agregacoes += [F.avg(F.col(c)).alias(f"media::{c}"), F.var_samp(F.col(c)).alias(f"var::{c}")]
    estatisticas = preparado.groupBy(group_col).agg(*agregacoes).toPandas().set_index(group_col)

    linhas = []
    for c in covariaveis:
        media_t, media_c = estatisticas.loc[1, f"media::{c}"], estatisticas.loc[0, f"media::{c}"]
        var_t, var_c = estatisticas.loc[1, f"var::{c}"], estatisticas.loc[0, f"var::{c}"]
        pooled = np.sqrt((np.nan_to_num(var_t) + np.nan_to_num(var_c)) / 2)
        diferenca = media_t - media_c
        if pooled > 0:
            smd = diferenca / pooled
        elif diferenca == 0:
            smd = 0.0            # sem variação e sem diferença: genuinamente balanceada
        else:
            # Variância nula nos dois grupos e médias distintas: a covariável separa
            # tratado de controle perfeitamente. A diferença padronizada é infinita —
            # devolver 0.0 aqui esconderia o desbalanço máximo em silêncio.
            smd = np.inf * np.sign(diferenca)
        linhas.append({"covariavel": c, "media_tratado": media_t, "media_controle": media_c, "smd": smd})

    frame = pd.DataFrame(linhas)
    frame["abs_smd"] = frame["smd"].abs()
    frame["acima_do_limiar"] = frame["abs_smd"] > cfg.smd_threshold
    return frame.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def assignment_balance(processed: DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """SMD entre quem **recebeu** cada oferta e quem recebeu outra — verifica a Premissa 4.

    A Premissa 4 diz que o *envio* é aleatorizado (RCT), logo as covariáveis de
    cliente devem estar balanceadas entre os braços de recebimento. Diferente de
    `covariate_balance`, que compara viu/não-viu (pós-tratamento), aqui o grupo é o
    `offer_id` recebido — o que a aleatorização de fato controla.

    Devolve o **pior** |SMD| de cada covariável entre todos os pares de ofertas
    (o teste mais exigente): se nem o par mais desbalanceado passa do limiar, o
    balanço do desenho se sustenta.
    """
    preparado, colunas_genero = _with_gender_indicators(processed)
    covariaveis = list(CLIENT_COVARIATES) + colunas_genero
    ofertas = [r["offer_id"] for r in processed.select("offer_id").distinct().collect()]

    agregacoes = []
    for c in covariaveis:
        agregacoes += [F.avg(F.col(c)).alias(f"media::{c}"), F.var_samp(F.col(c)).alias(f"var::{c}")]
    est = preparado.groupBy("offer_id").agg(*agregacoes).toPandas().set_index("offer_id")

    linhas = []
    for c in covariaveis:
        pior = 0.0
        for i in range(len(ofertas)):
            for j in range(i + 1, len(ofertas)):
                a, b = ofertas[i], ofertas[j]
                pooled = np.sqrt((np.nan_to_num(est.loc[a, f"var::{c}"]) + np.nan_to_num(est.loc[b, f"var::{c}"])) / 2)
                if pooled > 0:
                    pior = max(pior, abs(est.loc[a, f"media::{c}"] - est.loc[b, f"media::{c}"]) / pooled)
        linhas.append({"covariavel": c, "pior_abs_smd": pior, "acima_do_limiar": pior > cfg.smd_threshold})

    return pd.DataFrame(linhas).sort_values("pior_abs_smd", ascending=False).reset_index(drop=True)


def treatment_group_comparison(processed: DataFrame) -> pd.DataFrame:
    """Taxa de conversão e ticket médio entre viu (`treatment=1`) e não viu (`treatment=0`).

    Leitura bruta, não causal: `treatment` é pós-tratamento (o cliente escolheu ver),
    então a diferença aqui mistura efeito real com o mesmo confundimento que
    `covariate_balance` qualifica. Complementa o balanço de covariáveis com a métrica
    de resposta que ele não mostra — o "quanto" ao lado do "os grupos são comparáveis".
    """
    frame = (
        processed.groupBy("treatment")
        .agg(
            F.count("*").alias("recebidos"),
            F.avg("converted").alias("taxa_conversao"),
            F.avg(F.when(F.col("converted") == 1, F.col("conversion_value"))).alias("ticket_medio"),
            F.avg("conversion_value").alias("receita_media"),
        )
        .orderBy("treatment")
        .toPandas()
    )
    frame["treatment"] = frame["treatment"].map({0: "não viu", 1: "viu"})
    return frame


# --- Perfil univariado das features (REQ-108) ----------------------------------

def numeric_profile(df: DataFrame, columns: list[str], cfg: PipelineConfig) -> pd.DataFrame:
    """Estatística descritiva por coluna numérica, com nulos, zeros e outliers.

    Outlier é definido pela cerca de Tukey — fora de `[Q1 − k·IQR, Q3 + k·IQR]`,
    com `k = cfg.outlier_iqr_multiplier`. É um **rótulo de cauda**, não um veredito
    de erro: `hist_spend_total` tem cauda longa legítima (cliente que compra muito),
    enquanto uma coluna 0/1 vira "100% outlier" se a classe minoritária é rara.
    Ler a linha junto com `zeros` e os percentis, nunca a coluna `outliers` sozinha.

    Quantis vêm de `approxQuantile` com `cfg.quantile_rel_error` (nulos ignorados,
    como nas médias). Duas passadas: a primeira mede, a segunda conta quem caiu fora.
    """
    total = df.count()

    agregacoes = []
    for c in columns:
        agregacoes += [
            F.count(F.col(c)).alias(f"n::{c}"),
            F.avg(F.col(c)).alias(f"media::{c}"),
            F.stddev_samp(F.col(c)).alias(f"desvio::{c}"),
            F.min(F.col(c)).alias(f"min::{c}"),
            F.max(F.col(c)).alias(f"max::{c}"),
            F.sum((F.col(c) == 0).cast("long")).alias(f"zeros::{c}"),
        ]
    base = df.agg(*agregacoes).first().asDict()

    quantis = df.approxQuantile(columns, [0.01, 0.25, 0.5, 0.75, 0.99], cfg.quantile_rel_error)

    cercas: dict[str, tuple[float, float]] = {}
    for c, qs in zip(columns, quantis):
        if len(qs) < 5:               # coluna inteiramente nula: não há cerca a calcular
            continue
        _, q1, _, q3, _ = qs
        iqr = q3 - q1
        cercas[c] = (q1 - cfg.outlier_iqr_multiplier * iqr, q3 + cfg.outlier_iqr_multiplier * iqr)

    if cercas:
        fora = df.agg(*[
            F.sum(((F.col(c) < lo) | (F.col(c) > hi)).cast("long")).alias(f"fora::{c}")
            for c, (lo, hi) in cercas.items()
        ]).first().asDict()
    else:
        fora = {}

    linhas = []
    for c, qs in zip(columns, quantis):
        n = base[f"n::{c}"]
        nulos = total - n
        p1, p25, p50, p75, p99 = qs if len(qs) == 5 else (np.nan,) * 5
        n_fora = fora.get(f"fora::{c}") or 0
        linhas.append({
            "coluna": c, "n": n, "nulos": nulos, "frac_nulos": nulos / total,
            "zeros": base[f"zeros::{c}"] or 0,
            "frac_zeros": (base[f"zeros::{c}"] or 0) / n if n else np.nan,
            "media": base[f"media::{c}"], "desvio": base[f"desvio::{c}"],
            "min": base[f"min::{c}"], "p1": p1, "p25": p25, "p50": p50, "p75": p75, "p99": p99,
            "max": base[f"max::{c}"],
            "outliers": n_fora, "frac_outliers": n_fora / n if n else np.nan,
        })
    return pd.DataFrame(linhas)


def categorical_profile(df: DataFrame, columns: list[str]) -> pd.DataFrame:
    """Contagem e frequência de cada nível, por coluna categórica."""
    total = df.count()
    partes = []
    for c in columns:
        parte = (
            df.groupBy(F.col(c).cast("string").alias("nivel"))
            .agg(F.count("*").alias("linhas"))
            .toPandas()
            .assign(coluna=c)
        )
        partes.append(parte)
    frame = pd.concat(partes, ignore_index=True)
    frame["fracao"] = frame["linhas"] / total
    return frame.sort_values(["coluna", "linhas"], ascending=[True, False]).reset_index(drop=True)


def correlation_matrix(df: DataFrame, columns: list[str]) -> pd.DataFrame:
    """Correlação de Pearson entre as colunas numéricas, par a par.

    Colhe o grão inteiro em pandas — ~76 mil linhas × dezenas de colunas, alguns
    MB — porque `pandas.corr` faz **exclusão par a par** dos nulos: a correlação de
    `age` com `duration` usa todas as linhas com idade, mesmo que `hist_recency_days`
    seja nula ali. Preencher nulo com zero para caber num `VectorAssembler` inventaria
    correlação onde há apenas ausência.
    """
    return df.select(columns).toPandas().astype("float64").corr(method="pearson")


def redundant_pairs(corr: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Pares acima de `cfg.correlation_threshold` em módulo — candidatos a redundância."""
    triangulo = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    empilhado = triangulo.stack()
    fortes = empilhado[empilhado.abs() >= cfg.correlation_threshold]
    return (
        pd.DataFrame({"feature_a": [i[0] for i in fortes.index],
                      "feature_b": [i[1] for i in fortes.index],
                      "r": fortes.to_numpy()})
        .assign(abs_r=lambda d: d["r"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )


def sanity_checks(processed: DataFrame) -> pd.DataFrame:
    """Contagens de combinações que **não deveriam existir** no grão processado.

    Não substitui `0_pipeline_audit.ipynb` (que faz `assert`): aqui as mesmas
    condições viram números numa tabela, para que a EDA leia o dado com os olhos
    abertos em vez de assumir a auditoria de cor. Zero em todas as linhas é o
    resultado esperado; qualquer coisa diferente é achado de EDA, não de estilo.
    """
    condicoes = {
        # `converted=1` com `treatment=0` é ESPERADO: quem não viu a oferta e
        # comprou na validade converteu. É a massa que dá μ₀ > 0 e torna o
        # uplift estimável — checar o contrário zeraria o contrafactual.
        "converted=1 com conversion_value = 0":
            (F.col("converted") == 1) & (F.col("conversion_value") <= 0),
        "conversion_value > 0 sem converted":
            (F.col("conversion_value") > 0) & (F.col("converted") == 0),
        "reward_cost > 0 sem conversão (viola G6)":
            (F.col("reward_cost") > 0) & (F.col("converted") == 0),
        "reward_cost > 0 em informational (viola G6)":
            (F.col("reward_cost") > 0) & (F.col("offer_type") == "informational"),
        "reward_cost acima da receita da conversão":
            (F.col("converted") == 1) & (F.col("reward_cost") > F.col("conversion_value")),
        "ticket médio histórico sem transação histórica":
            (F.col("hist_avg_ticket") > 0) & (F.col("hist_txn_count") == 0),
        "gasto histórico negativo":
            F.col("hist_spend_total") < 0,
        "taxa de view histórica fora de [0,1]":
            (F.col("hist_view_rate") < 0) | (F.col("hist_view_rate") > 1),
        "idade igual à sentinela (viola G7)":
            F.col("age") == 118,
    }
    total = processed.count()
    linha = processed.agg(*[
        F.sum(c.cast("long")).alias(nome) for nome, c in condicoes.items()
    ]).first().asDict()
    return pd.DataFrame([
        {"verificação": nome, "linhas": int(v or 0), "fracao": (v or 0) / total}
        for nome, v in linha.items()
    ])


# --- Funil de resposta ----------------------------------------------------------

def response_funnel(processed: DataFrame) -> pd.DataFrame:
    """Recebido → visto → convertido → recorrente, por tipo de oferta.

    `taxa_conversao` (sobre recebidos) e `taxa_conversao_vistos` (sobre vistos) medem
    coisas diferentes: a primeira é a performance da campanha como enviada, a segunda
    é a resposta de quem foi de fato exposto. Confundir as duas infla a leitura de
    qualquer tipo com baixa taxa de view.

    Recorrência: `is_recurrent=1` só em `converted=1` — outra compra na janela configurada
    após a conversão. `taxa_recorrencia` divide por recebidos; `taxa_recorrencia_convertidos`
    divide por convertidos (denominador correto para recompra entre quem comprou).
    """
    frame = (
        processed.groupBy("offer_type")
        .agg(
            F.count("*").alias("recebidos"),
            F.sum("treatment").alias("vistos"),
            F.sum("converted").alias("convertidos"),
            F.sum("is_recurrent").alias("recorrentes"),
            F.sum("conversion_value").alias("receita"),
            F.sum("reward_cost").alias("custo"),
        )
        .orderBy("offer_type")
        .toPandas()
    )
    frame["taxa_view"] = frame["vistos"] / frame["recebidos"]
    frame["taxa_conversao"] = frame["convertidos"] / frame["recebidos"]
    frame["taxa_conversao_vistos"] = np.where(
        frame["vistos"] > 0, frame["convertidos"] / frame["vistos"], np.nan)
    frame["taxa_recorrencia"] = frame["recorrentes"] / frame["recebidos"]
    frame["taxa_recorrencia_convertidos"] = np.where(
        frame["convertidos"] > 0, frame["recorrentes"] / frame["convertidos"], np.nan)
    frame["margem_por_envio"] = (frame["receita"] - frame["custo"]) / frame["recebidos"]
    return frame


def recurrence_by_wave(processed: DataFrame) -> pd.DataFrame:
    """Recorrência de recompra por onda de campanha, com denominadores explícitos."""
    return (
        processed.groupBy("campaign_wave")
        .agg(
            F.count("*").alias("recebidos"),
            F.sum("converted").alias("convertidos"),
            F.sum("is_recurrent").alias("recorrentes"),
        )
        .orderBy("campaign_wave")
        .toPandas()
        .assign(
            taxa_recorrencia=lambda d: d["recorrentes"] / d["recebidos"],
            taxa_recorrencia_convertidos=lambda d: np.where(
                d["convertidos"] > 0, d["recorrentes"] / d["convertidos"], np.nan),
        )
    )


def recurrence_by_treatment(processed: DataFrame) -> pd.DataFrame:
    """Recorrência por tipo de oferta × braço (viu/não viu)."""
    return (
        processed.groupBy("offer_type", "treatment")
        .agg(
            F.count("*").alias("recebidos"),
            F.sum("converted").alias("convertidos"),
            F.sum("is_recurrent").alias("recorrentes"),
        )
        .toPandas()
        .assign(
            taxa_recorrencia=lambda d: d["recorrentes"] / d["recebidos"],
            taxa_recorrencia_convertidos=lambda d: np.where(
                d["convertidos"] > 0, d["recorrentes"] / d["convertidos"], np.nan),
        )
        .sort_values(["offer_type", "treatment"])
        .reset_index(drop=True)
    )


# --- Segmentação de clientes por K-Means (REQ-111) ------------------------------

# Quem o cliente é (perfil) e quanto ele compra (comportamento). Fora da matriz:
# `view_rate`, `converted` e derivados — são a **resposta** que os segmentos vão
# explicar; clusterizar por eles e depois comparar resposta entre clusters seria
# circular. Fora também `identity_missing`, pelo motivo em `cluster_matrix`.
CLUSTER_FEATURES = ("age", "credit_card_limit", "tenure_days",
                    "spend_total", "txn_count", "avg_ticket")
# Contagens e valores monetários são fortemente assimétricos à direita: sem log,
# a distância euclidiana do K-Means é ditada pela cauda e os centróides perseguem
# os maiores gastadores. `log1p` (e não `log`) porque zero é frequente e legítimo.
LOG_FEATURES = ("spend_total", "txn_count", "avg_ticket")

MISSING_IDENTITY_SEGMENT = "identidade ausente"


def client_features(processed: DataFrame, events: DataFrame) -> pd.DataFrame:
    """Uma linha por cliente: perfil, comportamento de compra e resposta observada.

    Descritivo, **não modelável**: `spend_total` e `view_rate` olham a janela inteira
    do teste, inclusive depois de cada `received_time`. Serve para nomear segmentos e
    ler resultados; como feature do X-learner seria leakage puro (G2).

    `F.max` no lugar de `F.first` para as colunas de perfil: elas são constantes por
    cliente, e `max` ignora nulo e independe da ordem das partições — `first` daria
    resultado dependente do shuffle.
    """
    perfil = processed.groupBy("account_id").agg(
        F.max("age").alias("age"),
        F.max("credit_card_limit").alias("credit_card_limit"),
        F.max("tenure_days").alias("tenure_days"),
        F.max("identity_missing").alias("identity_missing"),
        F.max("gender").alias("gender"),
        F.count("*").alias("offers_received"),
        F.sum("treatment").alias("offers_viewed"),
        F.sum("converted").alias("conversions"),
        F.sum("conversion_value").alias("conversion_value"),
        F.sum("reward_cost").alias("reward_cost"),
    )
    compras = (
        events.filter(F.col("event") == "transaction")
        .groupBy("account_id")
        .agg(F.sum("amount").alias("spend_total"), F.count("*").alias("txn_count"))
    )
    frame = (
        perfil.join(compras, on="account_id", how="left")
        .fillna({"spend_total": 0.0, "txn_count": 0})
        .toPandas()
    )
    # Cliente sem transação tem ticket médio 0, não NaN: "não comprou" é um valor,
    # não uma ausência. (Diferente de `age`, onde o nulo é o segmento sentinela.)
    frame["avg_ticket"] = np.where(frame["txn_count"] > 0, frame["spend_total"] / frame["txn_count"], 0.0)
    frame["view_rate"] = frame["offers_viewed"] / frame["offers_received"]
    frame["conv_rate"] = frame["conversions"] / frame["offers_received"]
    frame["margem"] = frame["conversion_value"] - frame["reward_cost"]
    return frame.sort_values("account_id").reset_index(drop=True)


def cluster_matrix(clients: pd.DataFrame) -> tuple[np.ndarray, pd.Index, list[str]]:
    """Matriz padronizada para o K-Means, só com clientes de perfil completo.

    Três decisões que fazem a distância euclidiana significar alguma coisa:

    1. **Sem imputação.** Os clientes com `identity_missing=1` não têm `age` nem
       `credit_card_limit` (Premissa 3). Imputar a mediana os empurraria para o
       centro do espaço — inventaria clientes medianos e ainda contaria a mesma
       ausência três vezes (age, limite e a flag). Eles já **são** um segmento;
       ficam fora do ajuste e entram na leitura como segmento nomeado.
    2. **`log1p` nas caudas** (`LOG_FEATURES`): K-Means minimiza soma de quadrados,
       logo um gastador extremo puxa o centróide sozinho.
    3. **z-score depois do log**: a distância euclidiana soma diferenças ao quadrado
       de features em unidades distintas (anos, reais, dias). Sem padronizar, quem
       tem a maior variância bruta — `credit_card_limit` — vira o único eixo real.

    Devolve `(X, account_ids, colunas)` com `X` na ordem de `account_ids`.
    """
    completos = clients.loc[clients["identity_missing"] == 0]
    bruto = completos[list(CLUSTER_FEATURES)].astype("float64")
    if bruto.isna().to_numpy().any():
        colunas_nulas = bruto.columns[bruto.isna().any()].tolist()
        raise ValueError(
            f"nulo em cliente de perfil completo: {colunas_nulas}. "
            "G7 diz que só o segmento sentinela tem perfil ausente — o contrato foi violado."
        )

    transformado = bruto.copy()
    for coluna in LOG_FEATURES:
        transformado[coluna] = np.log1p(transformado[coluna])

    desvio = transformado.std(ddof=0)
    if (desvio == 0).any():
        constantes = desvio.index[desvio == 0].tolist()
        raise ValueError(f"feature constante não pode ser padronizada: {constantes}")
    padronizado = (transformado - transformado.mean()) / desvio

    return padronizado.to_numpy(), completos["account_id"].reset_index(drop=True), list(CLUSTER_FEATURES)


def cluster_scan(matrix: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    """Varre `k` em `[cluster_k_min, cluster_k_max]`: inércia (cotovelo) e silhouette.

    A silhouette é medida na **mesma matriz padronizada** que alimenta o ajuste e com
    a mesma métrica (euclidiana) — avaliar no espaço bruto compararia um agrupamento
    com uma geometria que ele nunca viu. É O(n²), daí a amostra de
    `cfg.cluster_silhouette_sample` linhas, fixada por `cfg.seed`.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    linhas = []
    for k in range(cfg.cluster_k_min, cfg.cluster_k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(matrix)
        silhueta = silhouette_score(
            matrix, modelo.labels_, metric="euclidean",
            sample_size=min(cfg.cluster_silhouette_sample, len(matrix)), random_state=cfg.seed,
        )
        linhas.append({"k": k, "inercia": modelo.inertia_, "silhouette": silhueta})
    return pd.DataFrame(linhas)


def choose_k(scan: pd.DataFrame) -> int:
    """O `k` de maior silhouette. A varredura inteira vai na figura — o critério é visível."""
    return int(scan.loc[scan["silhouette"].idxmax(), "k"])


def fit_clusters(matrix: np.ndarray, account_ids: pd.Index, k: int, cfg: PipelineConfig) -> pd.DataFrame:
    """Rótulo de cluster por cliente. Determinístico dado `cfg.seed`."""
    from sklearn.cluster import KMeans

    modelo = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(matrix)
    return pd.DataFrame({"account_id": account_ids.to_numpy(), "cluster": modelo.labels_})


def assign_segments(clients: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Junta o rótulo do K-Means e nomeia o segmento sentinela, que não foi ajustado."""
    unido = clients.merge(clusters, on="account_id", how="left")
    unido["segmento"] = np.where(
        unido["identity_missing"] == 1,
        MISSING_IDENTITY_SEGMENT,
        "cluster " + unido["cluster"].astype("Int64").astype("string"),
    )
    return unido


def segment_profile(segments: pd.DataFrame) -> pd.DataFrame:
    """Média de cada feature por segmento, em unidades originais, com tamanho e resposta."""
    colunas = list(CLUSTER_FEATURES) + ["offers_received", "view_rate", "conv_rate", "margem"]
    perfil = segments.groupby("segmento")[colunas].mean()
    perfil.insert(0, "clientes", segments.groupby("segmento").size())
    perfil.insert(1, "fracao", perfil["clientes"] / len(segments))
    return perfil.reset_index()


def segment_response(processed: DataFrame, segments: DataFrame) -> pd.DataFrame:
    """Resposta observada por segmento × tipo de oferta.

    Denominadores explícitos (mesma disciplina de `response_funnel`) e economia por
    envio: `margem_por_envio = (receita − custo) / envios` é a grandeza que a política
    da spec 02 vai maximizar, aqui na sua versão **observacional** — média do que
    aconteceu, não o efeito causal de enviar.
    """
    frame = (
        processed.join(segments, on="account_id", how="inner")
        .groupBy("segmento", "offer_type")
        .agg(
            F.count("*").alias("envios"),
            F.sum("treatment").alias("vistos"),
            F.sum("converted").alias("conversoes"),
            F.sum("conversion_value").alias("receita"),
            F.sum("reward_cost").alias("custo"),
        )
        .toPandas()
        .sort_values(["segmento", "offer_type"])
        .reset_index(drop=True)
    )
    frame["taxa_view"] = frame["vistos"] / frame["envios"]
    frame["taxa_conversao"] = frame["conversoes"] / frame["envios"]
    frame["taxa_conversao_vistos"] = np.where(
        frame["vistos"] > 0, frame["conversoes"] / frame["vistos"], np.nan)
    frame["margem_por_envio"] = (frame["receita"] - frame["custo"]) / frame["envios"]
    return frame


def window_spend(attributed: DataFrame, events: DataFrame) -> DataFrame:
    """Gasto do cliente dentro de `[received_time, valid_until]`, **visto ou não**.

    Diferente de `conversion_value`, que só conta transação após o view (G4). Aqui a
    janela é a mesma, o filtro de view não existe, e a exclusividade também não: uma
    compra em janelas sobrepostas soma nas duas linhas. Justamente por isso este número
    **não pode ser label** — é uma régua de comparação entre expostos e não expostos,
    para a qual o denominador é o envio, não a transação.
    """
    txns = events.filter(F.col("event") == "transaction").select(
        "account_id", F.col("time").alias("txn_time"), F.col("amount").alias("txn_amount"))
    grao = ["account_id", "offer_id", "received_time"]
    na_janela = (
        attributed.select(*grao, "valid_until")
        .join(txns, on="account_id", how="inner")
        .filter((F.col("txn_time") >= F.col("received_time")) & (F.col("txn_time") <= F.col("valid_until")))
        .groupBy(*grao)
        .agg(F.sum("txn_amount").alias("window_spend"), F.count("*").alias("window_txns"))
    )
    return (
        attributed.select(*grao)
        .join(na_janela, on=grao, how="left")
        .fillna({"window_spend": 0.0, "window_txns": 0})
    )


def naive_spend_lift(processed: DataFrame, window: DataFrame, segments: DataFrame) -> pd.DataFrame:
    """Diferença bruta de gasto na janela entre quem viu e quem não viu, por segmento.

    **Não é uplift.** Ver é escolha do cliente (pós-tratamento): quem abre a oferta
    tende a ser quem já estava mais ativo, e essa seleção entra inteira na diferença.
    O número serve para duas coisas honestas: mostrar que a resposta ao estímulo varia
    por segmento (heterogeneidade que o X-learner vai modelar) e fixar a ordem de
    grandeza que uma estimativa causal terá de bater — se o efeito causal vier maior
    que esta diferença confundida, é sinal de erro, não de descoberta.
    """
    grao = ["account_id", "offer_id", "received_time"]
    unido = processed.select(*grao, "treatment").join(window, on=grao, how="inner").join(
        segments, on="account_id", how="inner")
    frame = (
        unido.groupBy("segmento", "treatment")
        .agg(F.avg("window_spend").alias("gasto_medio"), F.count("*").alias("envios"))
        .toPandas()
        .pivot(index="segmento", columns="treatment", values=["gasto_medio", "envios"])
    )
    saida = pd.DataFrame({
        "segmento": frame.index,
        "gasto_visto": frame[("gasto_medio", 1)].to_numpy(),
        "gasto_nao_visto": frame[("gasto_medio", 0)].to_numpy(),
        "envios_vistos": frame[("envios", 1)].to_numpy(),
        "envios_nao_vistos": frame[("envios", 0)].to_numpy(),
    })
    saida["diferenca_bruta"] = saida["gasto_visto"] - saida["gasto_nao_visto"]
    return saida.sort_values("diferenca_bruta", ascending=False).reset_index(drop=True)


def paid_below_minimum(processed: DataFrame) -> pd.DataFrame:
    """Verifica G10 sobre o dado real: nenhuma conversão paga fica abaixo do `min_value`.

    Enquanto a atribuição aceitava qualquer transação pós-view na janela, o label
    (G4) e o custo (REQ-106) discordavam: uma compra abaixo do gasto mínimo virava
    conversão e debitava um desconto que nunca teria sido concedido — inflando o
    custo do lado errado da função de lucro. G10 fechou essa fenda na atribuição.

    A função sobrevive como **auditoria**, não como achado: `abaixo_do_minimo` e
    `custo_sob_suspeita` devem ser zero em toda linha. Qualquer valor positivo é
    regressão de G10. `custo_acima_da_receita` continua podendo ser não-zero — um
    desconto de R$ 10 numa compra de R$ 10 é legítimo e não viola nada.
    """
    pagas = processed.filter((F.col("converted") == 1) & (F.col("offer_type") != "informational"))
    frame = (
        pagas.groupBy("offer_type")
        .agg(
            F.count("*").alias("conversoes_pagas"),
            F.sum((F.col("conversion_value") < F.col("min_value")).cast("long")).alias("abaixo_do_minimo"),
            F.sum((F.col("reward_cost") > F.col("conversion_value")).cast("long")).alias("custo_acima_da_receita"),
            F.sum(F.col("reward_cost")).alias("custo_total"),
            F.sum(F.when(F.col("conversion_value") < F.col("min_value"), F.col("reward_cost"))
                  .otherwise(F.lit(0.0))).alias("custo_sob_suspeita"),
        )
        .orderBy("offer_type")
        .toPandas()
    )
    frame["frac_abaixo_do_minimo"] = frame["abaixo_do_minimo"] / frame["conversoes_pagas"]
    frame["frac_custo_sob_suspeita"] = frame["custo_sob_suspeita"] / frame["custo_total"]
    return frame
