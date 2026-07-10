"""Fixture sintética determinística para os testes de modelagem (T-203..T-205).

Gera um pandas no formato do contrato (`src.contract.CONTRACT_COLUMNS`), com
sinal aprendível plantado (não ruído puro) para que baseline e X-learner
tenham algo a aprender sem depender do dado real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contract import CONTRACT_COLUMNS


def synthetic_processed(n: int = 400, seed: int = 42, n_waves: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    offer_type = rng.choice(["bogo", "discount", "informational"], size=n)
    campaign_wave = rng.integers(0, n_waves, size=n)
    received_time = campaign_wave.astype(float) * 7.0
    treatment = rng.binomial(1, 0.7, size=n)  # viu a oferta

    hist_spend_total = rng.gamma(shape=2.0, scale=20.0, size=n)
    hist_txn_count = rng.poisson(3, size=n)
    hist_view_rate = rng.uniform(0, 1, size=n)

    # Sinal plantado: gasto histórico alto + visto ⇒ mais propenso a converter.
    logit = -1.5 + 0.02 * hist_spend_total + 1.2 * treatment + 0.5 * hist_view_rate
    p_convert = 1 / (1 + np.exp(-logit))
    converted = rng.binomial(1, p_convert)
    conversion_value = np.where(converted == 1, rng.uniform(10, 100, size=n), 0.0)
    reward_cost = np.where(
        (converted == 1) & (offer_type != "informational"), rng.uniform(2, 10, size=n), 0.0
    )

    df = pd.DataFrame({
        "account_id": [f"acc{i}" for i in range(n)],
        "offer_id": [f"off{i}" for i in range(n)],
        "offer_type": offer_type,
        "received_time": received_time,
        "campaign_wave": campaign_wave.astype("int32"),
        "treatment": treatment.astype("int32"),
        "converted": converted.astype("int32"),
        "conversion_value": conversion_value,
        "reward_cost": reward_cost,
        "age": rng.integers(18, 90, size=n),
        "gender": rng.choice(["M", "F", "O", "unknown"], size=n),
        "credit_card_limit": rng.uniform(500, 20000, size=n),
        "identity_missing": np.zeros(n, dtype="int32"),
        "tenure_days": rng.integers(0, 2000, size=n),
        "hist_spend_total": hist_spend_total,
        "hist_txn_count": hist_txn_count.astype("int32"),
        "hist_avg_ticket": hist_spend_total / np.maximum(hist_txn_count, 1),
        "hist_spend_std": rng.uniform(0, 20, size=n),
        "hist_recency_days": rng.uniform(0, 30, size=n),
        "hist_frequency": rng.uniform(0, 1, size=n),
        "hist_spend_trend": rng.normal(0, 1, size=n),
        "hist_offers_received": rng.integers(0, 5, size=n).astype("int32"),
        "hist_offers_received_bogo": rng.integers(0, 3, size=n).astype("int32"),
        "hist_offers_received_discount": rng.integers(0, 3, size=n).astype("int32"),
        "hist_offers_received_info": rng.integers(0, 3, size=n).astype("int32"),
        "hist_offers_viewed": rng.integers(0, 5, size=n).astype("int32"),
        "hist_offers_completed": rng.integers(0, 3, size=n).astype("int32"),
        "hist_view_rate": hist_view_rate,
        "hist_conv_rate_bogo": rng.uniform(0, 1, size=n),
        "hist_conv_rate_discount": rng.uniform(0, 1, size=n),
        "hist_completed_unseen_flag": rng.binomial(1, 0.05, size=n).astype("int32"),
        "hist_time_view_to_conv": rng.uniform(0, 5, size=n),
        "discount_value": rng.uniform(2, 10, size=n),
        "min_value": rng.uniform(5, 20, size=n),
        "duration": rng.choice([3.0, 5.0, 7.0, 10.0], size=n),
        "n_channels": rng.integers(1, 4, size=n).astype("int32"),
        "channel_web": rng.binomial(1, 0.8, size=n).astype("int32"),
        "channel_email": rng.binomial(1, 0.8, size=n).astype("int32"),
        "channel_mobile": rng.binomial(1, 0.5, size=n).astype("int32"),
        "channel_social": rng.binomial(1, 0.3, size=n).astype("int32"),
        "discount_to_minvalue_ratio": rng.uniform(0.1, 1.0, size=n),
        "n_concurrent_offers": rng.integers(0, 3, size=n).astype("int32"),
    })
    return df[CONTRACT_COLUMNS]
