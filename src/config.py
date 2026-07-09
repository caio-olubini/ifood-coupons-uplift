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

    contract_sample_size: int = Field(default=1000, gt=0)

    # Infraestrutura do Spark (execução local). Não é semântica de pipeline, mas
    # também não é valor mágico: o default de heap da JVM não roda o dado real.
    spark_master: str = "local[*]"
    spark_driver_memory: str = "4g"
    spark_shuffle_partitions: int = Field(default=16, gt=0)

    seed: int = 42

    model_config = {"frozen": True, "yaml_file": DEFAULT_CONFIG_PATH}

    @model_validator(mode="after")
    def _check_positive_semantics(self) -> "PipelineConfig":
        if self.smd_threshold <= 0:
            raise ValueError("smd_threshold deve ser > 0")
        if self.n_campaign_waves <= 0:
            raise ValueError("n_campaign_waves deve ser > 0")
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
