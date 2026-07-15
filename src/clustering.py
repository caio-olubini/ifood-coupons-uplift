"""Segmentação de clientes → personas de negócio (K-Means), nível cliente.

Este módulo é o **delta** sobre a segmentação que já vive em `src/eda.py`
(REQ-111): reutiliza `eda.client_features` (grão cliente), `eda.choose_k`,
`eda.fit_clusters`, `eda.assign_segments`, `eda.segment_profile` e
`eda.segment_response` como estão — não reimplementa nada disso. Só três coisas
são genuinamente novas aqui e por isso justificam o módulo:

1. `design_matrix` — a matriz padronizada parametrizada por `cfg.cluster_features`
   (o `eda.cluster_matrix` usa a constante fixa de 6 features; as personas usam 7,
   com `view_rate`) e que devolve `means`/`stds` para interpretar centróides de
   volta em unidade original.
2. `scan_k` — varre `k` com quatro índices internos (inércia, silhouette,
   Davies-Bouldin, Calinski-Harabasz), contra os dois de `eda.cluster_scan`, para
   a escolha de `k` ficar visível por mais de um critério.
3. `name_personas` — traduz os rótulos arbitrários do K-Means em rótulos de
   negócio derivados do próprio perfil.

O tratamento de variável (sem imputar o sentinela, `log1p` nas caudas, z-score) é
o mesmo que `eda.cluster_matrix` documenta; `LOG_FEATURES`/`MISSING_IDENTITY_SEGMENT`
são reusados de `eda` para não divergir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import eda
from src.config import PipelineConfig
from src.eda import LOG_FEATURES, MISSING_IDENTITY_SEGMENT

# Reexporta as funções reutilizadas de `eda`, para o notebook importar tudo de
# `clustering` e a fronteira ficar num lugar só. Não são wrappers — são as
# mesmas funções.
build_client_frame = eda.client_features
choose_k = eda.choose_k
fit = eda.fit_clusters
assign_segments = eda.assign_segments
profile_segments = eda.segment_profile
segment_response = eda.segment_response

__all__ = [
    "build_client_frame",
    "design_matrix",
    "DesignMatrix",
    "scan_k",
    "choose_k",
    "fit",
    "assign_segments",
    "profile_segments",
    "name_personas",
    "segment_response",
    "LOG_FEATURES",
    "MISSING_IDENTITY_SEGMENT",
]


@dataclass(frozen=True)
class DesignMatrix:
    """A matriz padronizada e o suficiente para interpretar centróides de volta.

    `X` está na ordem de `account_ids`. `means`/`stds` são medidos no espaço
    **log-transformado** (a mesma escala do z-score): desfazer o z de um centróide
    e depois `expm1` nas `log_columns` recupera o valor em unidade original.
    """

    X: np.ndarray
    account_ids: pd.Series
    columns: list[str]
    means: pd.Series
    stds: pd.Series
    log_columns: list[str]


def design_matrix(clients: pd.DataFrame, cfg: PipelineConfig) -> DesignMatrix:
    """Matriz de design padronizada, parametrizada por `cfg.cluster_features`.

    Mesmo tratamento de `eda.cluster_matrix` (sem imputar o sentinela, `log1p` nas
    caudas de `LOG_FEATURES`, z-score depois), mas com a lista de features vinda da
    config (REQ-110) e devolvendo `means`/`stds` para o back-transform dos
    centróides. Recusa nulo em cliente de perfil completo (G7) e feature constante.
    """
    columns = list(cfg.cluster_features)
    completos = clients.loc[clients["identity_missing"] == 0]
    bruto = completos[columns].astype("float64")
    if bruto.isna().to_numpy().any():
        colunas_nulas = bruto.columns[bruto.isna().any()].tolist()
        raise ValueError(
            f"nulo em cliente de perfil completo: {colunas_nulas}. "
            "G7 diz que só o segmento sentinela tem perfil ausente — o contrato foi violado."
        )

    log_columns = [c for c in LOG_FEATURES if c in columns]
    transformado = bruto.copy()
    for coluna in log_columns:
        transformado[coluna] = np.log1p(transformado[coluna])

    stds = transformado.std(ddof=0)
    if (stds == 0).any():
        constantes = stds.index[stds == 0].tolist()
        raise ValueError(f"feature constante não pode ser padronizada: {constantes}")
    means = transformado.mean()
    padronizado = (transformado - means) / stds

    return DesignMatrix(
        X=padronizado.to_numpy(),
        account_ids=completos["account_id"].reset_index(drop=True),
        columns=columns,
        means=means,
        stds=stds,
        log_columns=log_columns,
    )


def scan_k(matrix: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    """Varre `k` em `[cluster_k_min, cluster_k_max]` com quatro índices internos.

    Estende `eda.cluster_scan` (inércia + silhouette) com Davies-Bouldin (↓) e
    Calinski-Harabasz (↑), para a escolha de `k` não depender de um critério só.
    Todos na mesma matriz padronizada do ajuste, com a metric euclidiana e a
    `cfg.seed`; silhouette amostrada (O(n²)), DB/CH na matriz inteira (O(n)).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    linhas = []
    for k in range(cfg.cluster_k_min, cfg.cluster_k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(matrix)
        labels = modelo.labels_
        silhueta = silhouette_score(
            matrix, labels, metric="euclidean",
            sample_size=min(cfg.cluster_silhouette_sample, len(matrix)), random_state=cfg.seed,
        )
        linhas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhueta,
            "davies_bouldin": davies_bouldin_score(matrix, labels),
            "calinski_harabasz": calinski_harabasz_score(matrix, labels),
        })
    return pd.DataFrame(linhas)


def name_personas(profile: pd.DataFrame) -> pd.DataFrame:
    """Mapeia cada segmento a um rótulo de negócio, derivado do perfil.

    Os rótulos do K-Means (`cluster 0/1/…`) são arbitrários (dependem da
    inicialização). Regra: o segmento sentinela é sempre "Incomplete registration".
    Entre os clusters ajustados, ordena por `spend_total` (valor econômico
    realizado) e nomeia por posição — o de maior gasto vira "High value", o de
    menor "Low ticket"; intermediários levam "Mid value" numerado.
    Devolve `profile` com uma coluna `persona`. Reprodutível apesar da numeração.
    """
    ajustados = profile.loc[profile["segmento"] != MISSING_IDENTITY_SEGMENT]
    ordenados = ajustados.sort_values("spend_total", ascending=False).reset_index(drop=True)

    nomes: dict[str, str] = {}
    n = len(ordenados)
    for pos, seg in enumerate(ordenados["segmento"]):
        if n == 1:
            rotulo = "Base"
        elif pos == 0:
            rotulo = "High value"
        elif pos == n - 1:
            rotulo = "Low ticket"
        else:
            rotulo = f"Mid value {pos}"
        nomes[seg] = rotulo
    nomes[MISSING_IDENTITY_SEGMENT] = "Incomplete registration"

    saida = profile.copy()
    saida["persona"] = saida["segmento"].map(nomes)
    return saida
