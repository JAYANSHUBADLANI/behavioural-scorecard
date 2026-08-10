"""Revenue, exposure and expected loss arithmetic for the limit increase decision.

This dataset carries no interest margin, no fee schedule and no recovery data, so every
money figure here rests on stated assumptions in `config/config.yaml` rather than on
observed profitability. What the data does supply is balances, drawings and credit limits,
which are the quantities the assumptions get applied to. Treat the ratios and the ranking
as the result, and the absolute currency amounts as illustrative.

Exposure is the reason a limit strategy cannot be judged on revenue alone. Granting extra
limit creates undrawn exposure immediately, and a share of it converts to drawn balance
exactly when an account deteriorates, which is what the credit conversion factor prices.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EconomicsConfig:
    annual_interest_margin: float = 0.18
    interchange_rate: float = 0.015
    credit_conversion_factor: float = 0.65
    loss_given_default: float = 0.75


def revenue(
    mean_balance: np.ndarray,
    total_drawings: np.ndarray,
    config: EconomicsConfig,
    window_months: int,
) -> np.ndarray:
    """Revenue proxy: margin on the revolving balance plus interchange on spend."""
    years = window_months / 12.0
    carried = np.nan_to_num(mean_balance, nan=0.0) * config.annual_interest_margin * years
    spend = np.nan_to_num(total_drawings, nan=0.0) * config.interchange_rate
    return (carried + spend).astype("float64")


def exposure_at_default(
    balance: np.ndarray, limit: np.ndarray, config: EconomicsConfig
) -> np.ndarray:
    """Drawn balance plus the converted share of the undrawn line."""
    balance = np.nan_to_num(balance, nan=0.0)
    limit = np.nan_to_num(limit, nan=0.0)
    undrawn = np.clip(limit - balance, 0.0, None)
    return balance + config.credit_conversion_factor * undrawn


def expected_loss(
    exposure: np.ndarray, probability_of_default: np.ndarray, config: EconomicsConfig
) -> np.ndarray:
    return exposure * np.clip(probability_of_default, 0.0, 1.0) * config.loss_given_default


def incremental_exposure(
    limit_increase: np.ndarray, incremental_balance: np.ndarray, config: EconomicsConfig
) -> np.ndarray:
    """Extra exposure created by granting more limit.

    The undrawn part of the increase converts at the credit conversion factor. Any extra
    balance the account actually carries is already drawn, so it counts in full.
    """
    increase = np.clip(np.nan_to_num(limit_increase, nan=0.0), 0.0, None)
    extra_balance = np.clip(np.nan_to_num(incremental_balance, nan=0.0), 0.0, None)
    undrawn_increase = np.clip(increase - extra_balance, 0.0, None)
    return extra_balance + config.credit_conversion_factor * undrawn_increase


def incremental_expected_loss(
    baseline_exposure: np.ndarray,
    incremental_exposure_amount: np.ndarray,
    baseline_probability: np.ndarray,
    incremental_probability: np.ndarray,
    config: EconomicsConfig,
) -> np.ndarray:
    """Expected loss after the increase minus expected loss without it.

    Three things move at once: the existing exposure now carries a higher default
    probability, the new exposure carries the baseline probability, and the new exposure
    also carries the incremental probability. Netting the two expected losses captures
    all three rather than only the headline probability change.
    """
    base_pd = np.clip(np.nan_to_num(baseline_probability, nan=0.0), 0.0, 1.0)
    new_pd = np.clip(base_pd + np.nan_to_num(incremental_probability, nan=0.0), 0.0, 1.0)
    before = baseline_exposure * base_pd * config.loss_given_default
    after = (baseline_exposure + incremental_exposure_amount) * new_pd * config.loss_given_default
    return after - before


def net_benefit(
    incremental_revenue: np.ndarray, incremental_loss: np.ndarray
) -> np.ndarray:
    """What a risk committee actually decides on: revenue upside net of loss cost."""
    return np.nan_to_num(incremental_revenue, nan=0.0) - np.nan_to_num(
        incremental_loss, nan=0.0
    )
