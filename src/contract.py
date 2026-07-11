"""Contrato do dataset processado: `StructType` + validação Pydantic (REQ-107).

Este módulo é a **encarnação executável** do contrato descrito em
`specification/schema-processed.md`. As duas formas exigidas — o `StructType`
imposto na escrita e o modelo Pydantic que valida uma amostra — são geradas de
uma **única** lista canônica de colunas (`_COLUMNS`), de modo que divergir entre
elas seja impossível por construção, não por disciplina.

Nulos: apenas as colunas em `NULLABLE_COLUMNS` admitem null. `age` e
`credit_card_limit` porque o dado de perfil pode faltar (Premissa 3); as duas
features de histórico porque `null` ali significa "não há histórico" — um sinal
que o modelo (LGBM/árvores) consome direto, e que fabricar um número destruiria.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, create_model
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DataType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

_INT = IntegerType()
_DBL = DoubleType()
_STR = StringType()

# Fonte única do contrato: (nome, tipo Spark, tipo Python, nullable).
# A ordem é a ordem das colunas no dataset processado e segue schema-processed.md.
_COLUMNS: list[tuple[str, DataType, type, bool]] = [
    # Identificação e tratamento
    ("account_id", _STR, str, False),
    ("offer_id", _STR, str, False),
    ("offer_type", _STR, str, False),
    ("received_time", _DBL, float, False),
    ("campaign_wave", _INT, int, False),
    ("treatment", _INT, int, False),
    # Label
    ("converted", _INT, int, False),
    ("conversion_value", _DBL, float, False),
    ("reward_cost", _DBL, float, False),
    ("is_recurrent", _INT, int, False),
    # Features de cliente
    ("age", _INT, int, True),
    ("gender", _STR, str, False),
    ("credit_card_limit", _DBL, float, True),
    ("identity_missing", _INT, int, False),
    ("tenure_days", _INT, int, False),
    # Features transacionais históricas
    ("hist_spend_total", _DBL, float, False),
    ("hist_txn_count", _INT, int, False),
    ("hist_avg_ticket", _DBL, float, False),
    ("hist_spend_std", _DBL, float, False),
    ("hist_recency_days", _DBL, float, True),
    ("hist_frequency", _DBL, float, False),
    ("hist_spend_trend", _DBL, float, False),
    # Features de histórico de resposta a ofertas
    ("hist_offers_received", _INT, int, False),
    ("hist_offers_received_bogo", _INT, int, False),
    ("hist_offers_received_discount", _INT, int, False),
    ("hist_offers_received_info", _INT, int, False),
    ("hist_offers_viewed", _INT, int, False),
    ("hist_offers_completed", _INT, int, False),
    ("hist_view_rate", _DBL, float, False),
    ("hist_conv_rate_bogo", _DBL, float, False),
    ("hist_conv_rate_discount", _DBL, float, False),
    ("hist_completed_unseen_flag", _INT, int, False),
    ("hist_time_view_to_conv", _DBL, float, True),
    # Features da oferta e contexto
    ("discount_value", _DBL, float, False),
    ("min_value", _DBL, float, False),
    ("duration", _DBL, float, False),
    ("n_channels", _INT, int, False),
    ("channel_web", _INT, int, False),
    ("channel_email", _INT, int, False),
    ("channel_mobile", _INT, int, False),
    ("channel_social", _INT, int, False),
    ("discount_to_minvalue_ratio", _DBL, float, False),
    ("n_concurrent_offers", _INT, int, False),
]

CONTRACT_COLUMNS: list[str] = [name for name, *_ in _COLUMNS]
NULLABLE_COLUMNS: frozenset[str] = frozenset(name for name, _, _, nullable in _COLUMNS if nullable)

PROCESSED_SCHEMA = StructType(
    [StructField(name, spark_type, nullable) for name, spark_type, _, nullable in _COLUMNS]
)

ProcessedRow: type[BaseModel] = create_model(
    "ProcessedRow",
    **{
        name: (Optional[py_type], None) if nullable else (py_type, ...)
        for name, _, py_type, nullable in _COLUMNS
    },
)


def enforce_schema(df: DataFrame) -> DataFrame:
    """Projeta `df` exatamente no contrato: colunas na ordem e nos tipos declarados.

    Falha se faltar qualquer coluna do contrato. Colunas intermediárias do
    pipeline (`view_time`, `valid_until`, `assigned_*`) são descartadas por não
    constarem de `CONTRACT_COLUMNS`.
    """
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas do contrato ausentes no dataset: {missing}")

    typed = {name: spark_type for name, spark_type, _, _ in _COLUMNS}
    return df.select(*[df[name].cast(typed[name]).alias(name) for name in CONTRACT_COLUMNS])


def assert_schema(df: DataFrame) -> None:
    """Garante que o schema de `df` é idêntico ao contrato (nomes, ordem, tipos).

    Divergência aqui é o defeito de contrato que schema-processed.md nomeia — a
    escrita não deve prosseguir sem essa igualdade.
    """
    actual = [(f.name, f.dataType) for f in df.schema.fields]
    expected = [(name, spark_type) for name, spark_type, _, _ in _COLUMNS]
    if actual != expected:
        raise ValueError(f"Schema fora do contrato.\nesperado: {expected}\nobtido:   {actual}")


def assert_no_unexpected_nulls(df: DataFrame) -> None:
    """G8: nenhuma coluna fora de `NULLABLE_COLUMNS` contém null."""
    from pyspark.sql import functions as F

    non_nullable = [c for c in CONTRACT_COLUMNS if c not in NULLABLE_COLUMNS]
    null_counts = df.select(
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in non_nullable]
    ).first()
    offenders = {c: null_counts[c] for c in non_nullable if null_counts[c]}
    if offenders:
        raise ValueError(f"G8 violado: nulos em colunas não-nullable {offenders}")


def validate_sample(df: DataFrame, cfg) -> int:
    """Valida uma amostra determinística do dataset contra `ProcessedRow` (Premissa 7).

    Pydantic vive na borda: valida `cfg.contract_sample_size` linhas, não o
    caminho quente do Spark. Levanta `pydantic.ValidationError` se qualquer linha
    fugir do contrato. Retorna quantas linhas foram validadas.
    """
    sample = df.limit(cfg.contract_sample_size).collect()
    for row in sample:
        ProcessedRow(**row.asDict())
    return len(sample)
