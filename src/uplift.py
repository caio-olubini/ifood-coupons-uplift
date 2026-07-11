"""X-learner de uplift: efeito de ver a oferta sobre `converted` (REQ-202).

Um X-learner por `offer_type` — cada tipo tem mecanismo de recompensa distinto
(bogo/discount pagam, informational não) e a saída precisa ser comparável por
tipo (REQ-202: "uplift por cliente × tipo de oferta"). `treatment` do contrato
(view=1/0) é o tratamento causal; `converted` é o outcome. Só as features
`hist_*`/cliente/oferta entram em X — nunca `converted`, `conversion_value`,
`reward_cost` (seriam leakage do próprio rótulo) nem `treatment` (é o
tratamento, não uma feature).

Escolhido por robustez a grupos de tamanhos desiguais e μ₀ mal-estimado
(Premissa 5) — a fraqueza exata do T-learner: um cliente "sure thing" (que já
converteria de qualquer forma, μ₀ alto) deve receber uplift ≈ 0, não um
artefato de dois modelos mal calibrados subtraídos.

Propensity é passada explicitamente (taxa de view observada, constante por
`offer_type`) em vez de deixar o CausalML estimar a sua internamente: o
projeto não estima propensity para de-confounding (RCT, Premissa 4/Não-objetivo
em `00-clarify.md`), e o estimador default do CausalML (`LogisticRegressionCV`)
não tolera os nulos legítimos de `age`/`credit_card_limit`/`hist_*` (G8) —
usar propensity fixa evita alimentar esse modelo auxiliar de qualquer forma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from lightgbm import LGBMRegressor

from src.config import PipelineConfig
from src.model_baseline import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMN = "converted"
OFFER_TYPE_COLUMN = "offer_type"

#: Grão do contrato (`specification/schema-processed.md`). `offer_id` sozinho não
#: identifica o recebimento: a mesma oferta chega ao mesmo cliente em ondas distintas.
GRAIN_COLUMNS = ["account_id", "offer_id", "received_time"]

# offer_type é o eixo de estratificação (um modelo por tipo); não entra como
# feature dentro de cada modelo — seria constante no grupo.
_XLEARNER_FEATURES = [c for c in FEATURE_COLUMNS if c != OFFER_TYPE_COLUMN]
_XLEARNER_CATEGORICAL = [c for c in CATEGORICAL_COLUMNS if c != OFFER_TYPE_COLUMN]


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[_XLEARNER_FEATURES].copy()
    for col in _XLEARNER_CATEGORICAL:
        X[col] = X[col].astype("category").cat.codes
    return X


def fixed_propensity(treatment: np.ndarray) -> np.ndarray:
    """Propensity constante = taxa de view observada no grupo (Premissa 4: RCT,
    propensity conhecida, não estimada). Mesmo valor repetido por linha —
    CausalML espera um vetor do tamanho de `treatment`, não um escalar.

    No serving (`src.serve`) todo o grupo entra com `treatment=1` (decidimos
    **expor**, o view ainda não ocorreu), e `mean()` degeneraria em 1,0 — que o
    CausalML rejeita (p ∈ (0,1) aberto). Quando o grupo é de um braço só, cai
    para 0,5 (peso neutro entre os dois estimadores de CATE do X-learner); nos
    casos reais (ajuste e holdout, com os dois braços presentes) `mean()` fica
    intacto — mesmo número de antes, byte a byte.
    """
    rate = treatment.mean()
    if rate <= 0.0 or rate >= 1.0:
        rate = 0.5
    return np.full(treatment.shape, fill_value=rate, dtype=float)


def _make_learner(cfg: PipelineConfig) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=cfg.xlearner_n_estimators,
        max_depth=cfg.xlearner_max_depth,
        learning_rate=cfg.xlearner_learning_rate,
        random_state=cfg.seed,
        verbose=-1,
    )


def fit_xlearner(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, BaseXRegressor]:
    """Ajusta um X-learner por `offer_type` presente em `df`.

    Retorna `{offer_type: modelo_ajustado}`. Cada modelo vê só as linhas do
    seu tipo — `informational` não tem `reward_cost`, misturar os tipos no
    mesmo modelo confundiria mecanismos de recompensa diferentes.
    """
    models: dict[str, BaseXRegressor] = {}
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        X = _design_matrix(group)
        treatment = group[TREATMENT_COLUMN].to_numpy()
        y = group[OUTCOME_COLUMN].to_numpy()
        p = fixed_propensity(treatment)

        model = BaseXRegressor(learner=_make_learner(cfg), control_name=0)
        model.fit(X=X, treatment=treatment, y=y, p=p)
        models[offer_type] = model
    return models


def predict(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """Uplift por linha, usando o modelo do `offer_type` daquela linha.

    Retorna `[account_id, offer_id, received_time, offer_type, uplift]` **na
    mesma ordem de linha de `df`**, uma estimativa por recebimento (REQ-202).

    `received_time` faz parte do grão do contrato e vai junto: um cliente pode
    receber a mesma oferta em duas ondas, então `(account_id, offer_id,
    offer_type)` **não** é chave única no dado real — juntar por ela produz um
    produto cartesiano silencioso (1.896 linhas do holdout têm grão duplicado).
    A ordem preservada permite atribuir a coluna direto, sem join nenhum.
    """
    uplift = pd.Series(index=df.index, dtype=float)
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        p = fixed_propensity(group[TREATMENT_COLUMN].to_numpy())
        te = model.predict(X=_design_matrix(group), p=p)
        uplift.loc[group.index] = te[:, 0] if te.ndim == 2 else te.ravel()

    return df[[*GRAIN_COLUMNS, OFFER_TYPE_COLUMN]].assign(uplift=uplift)


# --- Diagnóstico: de onde vem o número de uplift -------------------------------
#
# O X-learner é τ(x) = p·τ_c(x) + (1−p)·τ_t(x), construído sobre dois modelos de
# resultado: μ₀ (treinado só no controle) e μ₁ (só no tratado). Um uplift alto
# demais quase sempre é μ₀ degenerado, não efeito causal grande — e μ₀ degenera
# quando o label é impossível no controle. Estas funções abrem a caixa.


def predict_stages(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """μ₀, μ₁ e τ previstos **por linha** (não agregados), no grão do contrato.

    Mesmo cálculo de `stage_diagnostics`, mas devolvendo a estimativa individual
    em vez da média por `offer_type` — é o que a classificação de quadrante
    (`quadrant.classify_quadrant`, que corta em `tau` e `p_convert`) precisa:
    τ por cliente, não a média do grupo. `predict` já devolve τ por linha; esta
    função adiciona μ₀/μ₁ ao lado, reaproveitando o mesmo laço em vez de rodar
    o X-learner duas vezes.
    """
    mu0 = pd.Series(index=df.index, dtype=float)
    mu1 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(index=df.index, dtype=float)

    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        X = _design_matrix(group)
        arm = model.t_groups[0]
        p = fixed_propensity(group[TREATMENT_COLUMN].to_numpy())

        mu0.loc[group.index] = model.model_mu_c.predict(X)
        mu1.loc[group.index] = model.models_mu_t[arm].predict(X)
        te = model.predict(X=X, p=p)
        tau.loc[group.index] = te[:, 0] if te.ndim == 2 else te.ravel()

    return df[[*GRAIN_COLUMNS, OFFER_TYPE_COLUMN]].assign(mu0=mu0, mu1=mu1, tau=tau)


def predict_cate_uncertainty(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """Incerteza da estimativa de τ **por linha**: a discordância interna do X-learner.

    O X-learner combina dois estimadores de CATE — `dhat_c`, ajustado sobre os
    controles, e `dhat_t`, sobre os tratados — em `τ = p·dhat_c + (1−p)·dhat_t`.
    Onde os dois discordam, o efeito é menos identificado: `|dhat_t − dhat_c|`
    é uma medida honesta de **incerteza da própria estimativa**, não do tamanho
    do efeito (o que `|mu1 − mu0|` mediria). `predict(..., return_components=True)`
    devolve os dois componentes sem custo extra de ajuste.

    Retorna `[account_id, offer_id, received_time, offer_type, uncertainty]` na
    ordem de linha de `df` — a mesma convenção de `predict`.
    """
    uncertainty = pd.Series(index=df.index, dtype=float)
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        p = fixed_propensity(group[TREATMENT_COLUMN].to_numpy())
        _, dhat_c, dhat_t = model.predict(
            X=_design_matrix(group), p=p, return_components=True, verbose=False
        )
        arm = model.t_groups[0]
        uncertainty.loc[group.index] = np.abs(dhat_t[arm] - dhat_c[arm])

    return df[[*GRAIN_COLUMNS, OFFER_TYPE_COLUMN]].assign(uncertainty=uncertainty)


def causal_importance(
    models: dict[str, BaseXRegressor], df: pd.DataFrame, cfg: PipelineConfig
) -> pd.Series:
    """Importância **causal** das features: o que dirige o τ estimado, não o outcome.

    Usa a API avançada do CausalML (`BaseXRegressor.get_importance`): para cada
    `offer_type`, ajusta um meta-modelo `model_tau_feature` (LGBM, os
    hiperparâmetros de estágio da config) sobre `X → τ̂(X)` e mede a importância
    das features **nesse** meta-modelo. É a diferença que a instrução pede em
    relação a `feature_importances_` cru: aqui a variável importa se explica o
    *efeito da oferta*, não a propensão a converter.

    `method='permutation'` em vez de `'auto'` para um report **suave e
    estatisticamente coerente**: a importância é a queda média de acurácia ao
    permutar cada coluna (grandeza comparável entre features e entre os dois
    tipos de oferta), não o ganho de split cru (dependente da escala e da
    cardinalidade de cada feature). Os dois grupos de `offer_type` são
    reconciliados por **média ponderada pelo nº de linhas** — a importância
    agregada reflete a composição real do holdout, não a média simples de dois
    grupos de tamanhos desiguais. Normalizada para somar 1, para ser lida como
    participação relativa.

    Retorna uma `Series` indexada pelas features do X-learner (sem `offer_type`,
    que é o eixo de estratificação), em ordem decrescente de importância.
    """
    features = _XLEARNER_FEATURES
    total = pd.Series(0.0, index=features)
    n_total = 0

    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        X = _design_matrix(group)
        p = fixed_propensity(group[TREATMENT_COLUMN].to_numpy())
        tau = model.predict(X=X, p=p)

        imp = model.get_importance(
            X=X,
            tau=tau,
            model_tau_feature=_make_learner(cfg),
            features=np.array(features),
            method="permutation",
            random_state=cfg.seed,
        )
        # `get_importance` devolve {grupo_de_tratamento: Series}. Com um só braço
        # de tratamento (view=1), há uma entrada; somamos ponderando por linhas.
        (group_imp,) = imp.values()
        # permutação pode dar importância negativa (feature que atrapalha o
        # meta-modelo); piso em 0 antes de agregar — participação não é negativa.
        group_imp = group_imp.reindex(features).fillna(0.0).clip(lower=0.0)
        total = total + group_imp * len(group)
        n_total += len(group)

    mean_imp = total / n_total
    normalized = mean_imp / mean_imp.sum() if mean_imp.sum() > 0 else mean_imp
    return normalized.sort_values(ascending=False)


def label_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Estrutura do outcome dentro de cada braço, por `offer_type`.

    Se `taxa_outcome` no controle for exatamente 0, μ₀ ≡ 0 e o "uplift" degenera
    em μ₁ — o modelo devolve a taxa de conversão dos tratados, não um efeito.
    """
    linhas = []
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        for arm in (0, 1):
            braco = group[group[TREATMENT_COLUMN] == arm]
            linhas.append({
                "offer_type": offer_type,
                "braco": "controle (não viu)" if arm == 0 else "tratado (viu)",
                "n": len(braco),
                "outcome_positivo": int(braco[OUTCOME_COLUMN].sum()),
                "taxa_outcome": braco[OUTCOME_COLUMN].mean() if len(braco) else float("nan"),
            })
    return pd.DataFrame(linhas)


def stage_diagnostics(models: dict[str, BaseXRegressor], df: pd.DataFrame) -> pd.DataFrame:
    """μ₀, μ₁ e τ previstos em `df`, por `offer_type` — os estágios do X-learner.

    `mu0_*` vem de `model_mu_c` (ajustado só no controle) e `mu1_*` de
    `models_mu_t` (só no tratado). `tau_medio` é o uplift final. A identidade
    `tau ≈ mu1 − mu0` deve valer; quando `mu0` é constante zero, `tau ≈ mu1` e
    o número não é causal.
    """
    linhas = []
    for offer_type, group in df.groupby(OFFER_TYPE_COLUMN):
        model = models[offer_type]
        X = _design_matrix(group)
        arm = model.t_groups[0]

        mu0 = model.model_mu_c.predict(X)
        mu1 = model.models_mu_t[arm].predict(X)
        tau = model.predict(X=X, p=fixed_propensity(group[TREATMENT_COLUMN].to_numpy()))
        tau = tau[:, 0] if tau.ndim == 2 else tau.ravel()

        linhas.append({
            "offer_type": offer_type,
            "mu0_medio": mu0.mean(), "mu0_desvio": mu0.std(),
            "mu1_medio": mu1.mean(), "mu1_desvio": mu1.std(),
            "mu1_menos_mu0": mu1.mean() - mu0.mean(),
            "tau_medio": tau.mean(),
        })
    return pd.DataFrame(linhas)
