"""Config tipada e validada do pipeline (REQ-110).

Nenhum parâmetro de janela/limiar/caminho/seed pode aparecer literal fora
deste módulo — qualquer valor assim no corpo de uma função é um defeito.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")


class AttributionPriority(str, Enum):
    """Regra de desempate quando duas ofertas estão ativas no mesmo intervalo (Premissa 1)."""

    EARLIEST_RECEIVED = "earliest_received"
    LATEST_RECEIVED = "latest_received"


class PipelineConfig(BaseSettings):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    offers_filename: str = "offers.json"
    profile_filename: str = "profile.json"
    transactions_filename: str = "transactions.json"

    age_sentinel: int = 118
    test_start_date: str = "20180726"
    smd_threshold: float = Field(default=0.1, gt=0)
    attribution_priority: AttributionPriority = AttributionPriority.EARLIEST_RECEIVED

    # A onda é o rank do `received_time` distinto (disparos discretos), não um
    # bucket de largura fixa; `n_campaign_waves` é o número esperado de disparos,
    # verificado na auditoria — não um parâmetro de derivação.
    n_campaign_waves: int = Field(default=6, gt=0)

    # Janela de recorrência: um recebimento convertido é "recorrente" se o mesmo
    # cliente tem outra conversão (qualquer oferta) em até N dias após esta.
    # É derivada do target (`converted`), não uma feature — nunca entra em X.
    recurrence_window_days: int = Field(default=7, gt=0)

    contract_sample_size: int = Field(default=1000, gt=0)

    # EDA: parâmetros das visões descritivas (REQ-108, REQ-111).
    histogram_bins: int = Field(default=40, gt=1)
    quantile_rel_error: float = Field(default=0.001, ge=0, lt=1)
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0)
    correlation_threshold: float = Field(default=0.8, gt=0, le=1)

    # Segmentação K-Means (REQ-111). `k` é escolhido por silhouette dentro da
    # faixa; a varredura inteira é reportada, nunca só o vencedor.
    cluster_k_min: int = Field(default=2, gt=1)
    cluster_k_max: int = Field(default=8, gt=1)
    cluster_silhouette_sample: int = Field(default=5000, gt=1)

    # Infraestrutura do Spark (execução local). Não é semântica de pipeline, mas
    # também não é valor mágico: o default de heap da JVM não roda o dado real.
    spark_master: str = "local[*]"
    spark_driver_memory: str = "4g"
    spark_shuffle_partitions: int = Field(default=16, gt=0)

    seed: int = 42

    # Modelagem (spec 02). Split treino/validação é por `campaign_wave` (rank
    # discreto do received_time, não uma data): ondas < cutoff treinam, ondas
    # >= cutoff validam — nunca split aleatório (REQ-201 NFR, T-202).
    validation_wave_cutoff: int = Field(default=4, gt=0)

    # Hiperparâmetros do baseline preditivo (REQ-201). LGBM trata nulos
    # nativamente; a lista fica pequena porque o objetivo é âncora, não tuning.
    lgbm_n_estimators: int = Field(default=200, gt=0)
    lgbm_max_depth: int = Field(default=-1)
    lgbm_learning_rate: float = Field(default=0.05, gt=0)
    logit_max_iter: int = Field(default=1000, gt=0)

    # X-learner (REQ-202): hiperparâmetros dos regressores de estágio (CausalML
    # usa LGBM/sklearn por baixo; expostos aqui para não hardcode em src/).
    xlearner_n_estimators: int = Field(default=200, gt=0)
    xlearner_max_depth: int = Field(default=-1)
    xlearner_learning_rate: float = Field(default=0.05, gt=0)

    # Onde os modelos treinados (`src.models`) são serializados. É a fronteira
    # entre `model train` (escreve) e `model predict` (lê) no CLI — o par de
    # comandos produtivos que os wrappers habilitam.
    models_dir: Path = Path("models")

    # Serving (`model predict`): budget default (nº de ações a recomendar) quando
    # o CLI não recebe `--budget`. A restrição atual é uma oferta por cliente, então
    # o budget é também o nº de clientes distintos atendidos.
    predict_budget: int = Field(default=1000, gt=0)

    # Blend padrão do `BlendedUpliftModel` quando nenhum parâmetro é passado.
    # Os dois melhores blends medidos no holdout real (ver CLAUDE.md): λ=0,3
    # fixo e γ=1,0 dinâmico. `blend_mode` escolhe qual dos dois um blend sem
    # argumentos usa; o grid completo (`hybrid_lambda_grid`/
    # `dynamic_hybrid_gamma_grid`) continua servindo o estudo comparativo.
    blend_mode: str = Field(default="fixed", pattern="^(fixed|dynamic)$")
    blend_lambda: float = Field(default=0.3, ge=0)
    blend_gamma: float = Field(default=1.0, gt=0)

    # Avaliação offline: curva de ganho incremental por budget top-N (REQ-206).
    # Cada estratégia (uplift, conversão crua, aleatório) é um ranking; a curva
    # mede o lucro líquido incremental causal dos top-N. Estes budgets são os
    # pontos tabelados na leitura "se meu budget for N, quanto ganho?" — a curva
    # inteira é varrida, não só eles.
    gain_curve_budgets: list[int] = Field(default=[1000, 5000, 10000])

    # Intervalo de confiança da curva de ganho (lucro e conversão incremental),
    # por bootstrap não paramétrico (reamostragem do holdout com reposição,
    # recomputando a curva por réplica — mesmo padrão do placebo em
    # `placebo_n_permutations`, mas para incerteza amostral, não a nula causal).
    gain_curve_n_bootstrap: int = Field(default=200, gt=0)
    gain_curve_confidence_level: float = Field(default=0.95, gt=0, lt=1)

    # Estratégia híbrida (gaincurve.hybrid_score): score = uplift_x_learner +
    # λ · p_convert_cru, soma direta sem normalizar (os dois já vivem em escalas
    # parecidas: τ ∈ [-1, 1] aprox., p_convert ∈ [0, 1]). λ=0 degenera no modelo
    # de uplift puro — é o ponto de controle do grid, não um caso especial.
    hybrid_lambda_grid: list[float] = Field(default=[0.0, 0.1, 0.3, 0.5])

    # Estratégia híbrida dinâmica (gaincurve.dynamic_hybrid_score): peso λ_local
    # por cliente, proporcional à incerteza do τ (discordância interna do
    # X-learner, uplift.predict_cate_uncertainty) elevada a γ. γ=1 é resposta
    # linear; γ>1 concentra o blend nos extremos de incerteza (conservador);
    # γ<1 espalha o blend por mais do ranking (agressivo).
    dynamic_hybrid_gamma_grid: list[float] = Field(default=[0.5, 1.0, 2.0])

    # Grid fino de λ para a exploração suja gaincurve.best_lambda_by_decile —
    # qual λ fixo maximiza cada decil de budget. Diagnóstico exploratório (motiva
    # o híbrido dinâmico), não entra na política; por isso um grid mais denso que
    # hybrid_lambda_grid, sem custo de manter os dois alinhados.
    blend_lambda_scan: list[float] = Field(default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])

    # Classificação de quadrante de uplift (persuadable/sure thing/lost cause/
    # sleeping dog): corte direto em τ (o efeito em si) em vez de μ₀/μ₁
    # separados — dois cortes independentes em μ₀/μ₁ não correspondem a uma
    # curva de nível de τ constante, então um cliente com μ₀=0,49/μ₁=0,51
    # (τ≈0,02) e outro com μ₀=0,05/μ₁=0,95 (τ=0,90) podiam cair no mesmo
    # quadrante por acaso de qual lado do limiar cada μ caía. `quadrant_tau_epsilon`
    # é a banda |τ| < ε onde o efeito é indistinguível de zero: medido como o
    # desvio da distribuição nula do teste de placebo
    # (`uplift_eval.placebo_qini_distribution`) no dado real — o ruído do
    # próprio X-learner, não um número arbitrário. Dentro da banda,
    # `quadrant_p_convert_threshold` separa sure_thing (converte de qualquer
    # jeito) de lost_cause (não converte de qualquer jeito); 0,5 é o ponto
    # natural para uma probabilidade.
    quadrant_tau_epsilon: float = Field(default=0.0143, gt=0)
    quadrant_p_convert_threshold: float = Field(default=0.5, gt=0, lt=1)

    # Teste de placebo por permutação (REQ-212): réplicas do embaralhamento e o
    # percentil da nula que o Qini real precisa superar. `placebo_confidence_level`
    # é o mesmo cálculo do intervalo de confiança do Qini — reuso de infraestrutura,
    # não coincidência (ver `uplift_eval.placebo_test`).
    placebo_n_permutations: int = Field(default=20, gt=0)
    placebo_confidence_level: float = Field(default=0.95, gt=0, lt=1)

    # Tracking de experimentos (REQ-209): SQLite local, sem servidor — MLflow
    # 3.x descontinuou o backend de arquivo puro (./mlruns) em favor de banco.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "ifood-uplift"

    model_config = {"frozen": True, "yaml_file": DEFAULT_CONFIG_PATH}

    @model_validator(mode="after")
    def _check_positive_semantics(self) -> "PipelineConfig":
        if self.smd_threshold <= 0:
            raise ValueError("smd_threshold deve ser > 0")
        if self.n_campaign_waves <= 0:
            raise ValueError("n_campaign_waves deve ser > 0")
        if self.cluster_k_max <= self.cluster_k_min:
            raise ValueError("cluster_k_max deve ser > cluster_k_min")
        if self.validation_wave_cutoff >= self.n_campaign_waves:
            raise ValueError("validation_wave_cutoff deve ser < n_campaign_waves")
        return self

    @property
    def offers_path(self) -> Path:
        return self.raw_dir / self.offers_filename

    @property
    def profile_path(self) -> Path:
        return self.raw_dir / self.profile_filename

    @property
    def transactions_path(self) -> Path:
        return self.raw_dir / self.transactions_filename

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, YamlConfigSettingsSource(settings_cls))


def load(config_path: Path | str | None = None, **overrides) -> PipelineConfig:
    """Carrega a config, validando na fronteira. Falha antes de tocar em qualquer dado.

    Lê `config_path` (default `config.yaml`, se existir) e aplica `overrides`
    por cima — overrides explícitos sempre vencem o arquivo.
    """
    if config_path is None:
        return PipelineConfig(**overrides)

    class _PipelineConfigWithPath(PipelineConfig):
        model_config = {**PipelineConfig.model_config, "yaml_file": Path(config_path)}

    return _PipelineConfigWithPath(**overrides)
