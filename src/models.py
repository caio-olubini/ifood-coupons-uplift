"""Wrappers de modelo prontos para produto: `train`/`predict` num objeto só.

As funções de `src.uplift`, `src.model_baseline` e `src.gaincurve` são puras e
testáveis, mas espalhadas — treinar e depois pontuar exige orquestrar
`fit_xlearner` + `predict` + o baseline de conversão + a fórmula do blend na mão
(é o que o notebook faz). Estas classes empacotam essa orquestração num objeto
que se instancia com parâmetros simples, se ajusta com `fit(train_df)` e pontua
com `predict`/`score` — a superfície que os comandos `model train` e `model
predict` do CLI vão chamar.

Cada wrapper tem `from_config(cfg)`: a config é a **fonte dos defaults**, não um
argumento do construtor. Instanciar direto (`UpliftModel(n_estimators=300)`)
serve o ajuste fino; `from_config` serve o caminho produtivo, onde todo default
vem de `config.yaml` (REQ-110). `save`/`load` persistem o objeto ajustado inteiro
em `models_dir`, fechando a fronteira entre treinar (escreve) e prever (lê).

Dois modelos:

- `UpliftModel` — o X-learner por `offer_type`. `predict` devolve τ por linha;
  o objeto ajustado carrega os `BaseXRegressor` internos.
- `BlendedUpliftModel` — o modelo de produção. Compõe um `UpliftModel` com um
  prior de conversão (`ConversionModel`, o LGBM de `converted`) e mistura os dois
  num score de ranqueamento, fixo (λ) ou dinâmico por incerteza (γ). É o que a
  avaliação mostrou dominar em Qini e recuperar lucro em R$ (ver CLAUDE.md).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from lightgbm import LGBMClassifier

from src import gaincurve, model_baseline, uplift
from src.config import PipelineConfig

#: Nomes de arquivo dos modelos serializados dentro de `cfg.models_dir`. Estáveis
#: para que `model predict` saiba onde `model train` escreveu, sem parâmetro.
UPLIFT_MODEL_FILENAME = "uplift_model.pkl"
BLENDED_MODEL_FILENAME = "blended_uplift_model.pkl"


def _save_pickle(obj: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return path


def _load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


class ConversionModel:
    """Prior de conversão: P(converte | x) do LGBM baseline (`model_baseline`).

    Não é o modelo de uplift — é a propensão crua a converter, μ₁, que o blend
    empresta onde o τ é incerto. Encapsula só a metade LGBM de
    `model_baseline.train` (a logística é âncora de diagnóstico, não entra no
    blend), para o `BlendedUpliftModel` compor sem reimplementar o treino.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._model: LGBMClassifier | None = None

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "ConversionModel":
        return cls(
            n_estimators=cfg.lgbm_n_estimators,
            max_depth=cfg.lgbm_max_depth,
            learning_rate=cfg.lgbm_learning_rate,
            seed=cfg.seed,
        )

    def fit(self, train_df: pd.DataFrame) -> "ConversionModel":
        model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbose=-1,
        )
        X = model_baseline._design_matrix(train_df)
        model.fit(X, train_df[model_baseline.TARGET_COLUMN],
                  categorical_feature=model_baseline.CATEGORICAL_COLUMNS)
        self._model = model
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """P(converte | x), alinhada ao índice de `df`. Exige `fit` antes."""
        if self._model is None:
            raise RuntimeError("ConversionModel não ajustado — chame fit() antes de predict_proba().")
        return model_baseline.predict_conversion_probability(self._model, df)


class UpliftModel:
    """X-learner de uplift por `offer_type`, empacotado (`src.uplift`).

    Instancia-se com os hiperparâmetros dos regressores de estágio; `fit` ajusta
    um `BaseXRegressor` por tipo de oferta e `predict` devolve τ por linha no
    grão do contrato. `predict_stages`/`predict_uncertainty` expõem os
    diagnósticos que o blend dinâmico e a classificação de quadrante consomem.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._models: dict[str, BaseXRegressor] | None = None

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "UpliftModel":
        return cls(
            n_estimators=cfg.xlearner_n_estimators,
            max_depth=cfg.xlearner_max_depth,
            learning_rate=cfg.xlearner_learning_rate,
            seed=cfg.seed,
        )

    def _as_config(self) -> PipelineConfig:
        """Config mínima com os hiperparâmetros do X-learner, para reusar as
        funções de `src.uplift` (que recebem `cfg`) sem duplicar a fórmula do
        LGBM de estágio aqui — os wrappers orquestram, não reimplementam.
        """
        return PipelineConfig(
            xlearner_n_estimators=self.n_estimators,
            xlearner_max_depth=self.max_depth,
            xlearner_learning_rate=self.learning_rate,
            seed=self.seed,
        )

    def fit(self, train_df: pd.DataFrame) -> "UpliftModel":
        self._models = uplift.fit_xlearner(train_df, self._as_config())
        return self

    @property
    def models(self) -> dict[str, BaseXRegressor]:
        if self._models is None:
            raise RuntimeError("UpliftModel não ajustado — chame fit() antes.")
        return self._models

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """τ por linha no grão do contrato (`[*grão, offer_type, uplift]`)."""
        return uplift.predict(self.models, df)

    def predict_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """μ₀, μ₁ e τ por linha — insumo da classificação de quadrante."""
        return uplift.predict_stages(self.models, df)

    def predict_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Incerteza da estimativa de τ por linha — peso do blend dinâmico."""
        return uplift.predict_cate_uncertainty(self.models, df)

    def save(self, cfg: PipelineConfig) -> Path:
        return _save_pickle(self, cfg.models_dir / UPLIFT_MODEL_FILENAME)

    @classmethod
    def load(cls, cfg: PipelineConfig) -> "UpliftModel":
        return _load_pickle(cfg.models_dir / UPLIFT_MODEL_FILENAME)  # type: ignore[return-value]


class BlendedUpliftModel:
    """Modelo de produção: X-learner + prior de conversão, num score de ranking.

    Compõe um `UpliftModel` (τ causal) com um `ConversionModel` (P(converte)) e
    mistura os dois no score que ordena os clientes — a estratégia que a
    avaliação mostrou dominar em Qini e recuperar lucro em R$ (ver CLAUDE.md).
    Dois modos, escolhidos por `mode`:

    - `"fixed"` — `score = τ + λ · p_convert` (`gaincurve.hybrid_score`), um peso
      global. `λ` (`self.lambda_`) é o único parâmetro.
    - `"dynamic"` — peso local pela incerteza do τ
      (`gaincurve.dynamic_hybrid_score`), agressividade controlada por `γ`
      (`self.gamma`). Exige a incerteza do X-learner, então só vale para o modo
      dinâmico calcular `predict_uncertainty` no `score`.

    `score(df)` devolve o score por linha; `rank(df)` a ordem de prioridade
    (índices, do mais ao menos prioritário) — a saída que `model predict`
    entrega e a curva de ganho consome.
    """

    def __init__(
        self,
        uplift_model: UpliftModel,
        conversion_model: ConversionModel,
        mode: str = "fixed",
        lambda_: float = 0.3,
        gamma: float = 1.0,
    ) -> None:
        if mode not in ("fixed", "dynamic"):
            raise ValueError(f"mode deve ser 'fixed' ou 'dynamic', não {mode!r}")
        self.uplift_model = uplift_model
        self.conversion_model = conversion_model
        self.mode = mode
        self.lambda_ = lambda_
        self.gamma = gamma

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "BlendedUpliftModel":
        return cls(
            uplift_model=UpliftModel.from_config(cfg),
            conversion_model=ConversionModel.from_config(cfg),
            mode=cfg.blend_mode,
            lambda_=cfg.blend_lambda,
            gamma=cfg.blend_gamma,
        )

    def fit(self, train_df: pd.DataFrame) -> "BlendedUpliftModel":
        """Ajusta os dois componentes no mesmo `train_df` (uplift e conversão)."""
        self.uplift_model.fit(train_df)
        self.conversion_model.fit(train_df)
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Score de ranqueamento por linha, alinhado ao índice de `df`.

        No modo fixo, `τ + λ · p_convert`; no dinâmico, a combinação convexa
        ponderada pela incerteza do τ. Ambos delegam a fórmula a `gaincurve`,
        que já a testa — o wrapper só liga as saídas dos dois componentes.
        """
        # `uplift.predict`/`predict_cate_uncertainty` preservam o índice de `df`
        # (constroem a saída via `df[...].assign(...)`), então a coluna já vem
        # alinhada — sem reindexar.
        uplift_pred = self.uplift_model.predict(df)["uplift"]
        p_convert = self.conversion_model.predict_proba(df)

        if self.mode == "fixed":
            return gaincurve.hybrid_score(uplift_pred, p_convert, self.lambda_)

        uncertainty = self.uplift_model.predict_uncertainty(df)["uncertainty"]
        return gaincurve.dynamic_hybrid_score(uncertainty, uplift_pred, p_convert, self.gamma)

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        """Ordem de prioridade dos clientes: índices de `df`, score decrescente.

        Desempate estável pela ordem do índice — mesma convenção dos rankings de
        `gaincurve`, para o resultado ser determinístico dada a mesma entrada.
        """
        return self.score(df).sort_values(ascending=False, kind="stable").index.to_numpy()

    def save(self, cfg: PipelineConfig) -> Path:
        return _save_pickle(self, cfg.models_dir / BLENDED_MODEL_FILENAME)

    @classmethod
    def load(cls, cfg: PipelineConfig) -> "BlendedUpliftModel":
        return _load_pickle(cfg.models_dir / BLENDED_MODEL_FILENAME)  # type: ignore[return-value]
