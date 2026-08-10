"""Behavioural feature engineering from the pre-observation panel history.

Every feature is computed strictly from months at or before the observation point. The
outcome window is never touched here, which is the whole reason feature construction and
target construction are separate modules sharing one eligibility rule.

Features come in short and long lookback pairs where the trend matters. A single account
appears once per observation point, so the same account contributes several rows built
from overlapping but different history.
"""

import warnings
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import WindowConfig
from .windows import PanelMatrices, eligibility_mask

PANEL_VALUE_COLUMNS = (
    "AMT_PAYMENT_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_ATM_CURRENT",
)

DEFAULT_LOOKBACKS = (3, 6, 12)


def _nan_agg(values: np.ndarray, how: str) -> np.ndarray:
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(np, f"nan{how}")(values, axis=1).astype("float32")


def _slope(values: np.ndarray) -> np.ndarray:
    """Least squares slope per row against month index, ignoring missing months."""
    n_rows, n_cols = values.shape
    if n_cols < 2:
        return np.full(n_rows, np.nan, dtype="float32")
    x = np.arange(n_cols, dtype="float32")
    observed = ~np.isnan(values)
    counts = observed.sum(axis=1)
    filled = np.where(observed, values, 0.0)
    x_sum = (observed * x).sum(axis=1)
    y_sum = filled.sum(axis=1)
    xy_sum = (filled * x).sum(axis=1)
    xx_sum = (observed * x * x).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = counts * xx_sum - x_sum ** 2
        slope = (counts * xy_sum - x_sum * y_sum) / denominator
    return np.where(counts >= 2, slope, np.nan).astype("float32")


def _months_since_last(flags: np.ndarray) -> np.ndarray:
    """Months between the observation point and the most recent True, nan if never."""
    n_rows, n_cols = flags.shape
    if n_cols == 0:
        return np.full(n_rows, np.nan, dtype="float32")
    position = np.where(flags, np.arange(n_cols, dtype="float32"), -1.0)
    last = position.max(axis=1)
    return np.where(last >= 0, (n_cols - 1) - last, np.nan).astype("float32")


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan).astype("float32")


def build_features(
    panel: PanelMatrices,
    observation_month: int,
    window: WindowConfig,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    feature_offset: int = 0,
) -> pd.DataFrame:
    """Behavioural features for every account scoreable at one observation point.

    `feature_offset` moves the feature anchor earlier than the observation point while
    leaving eligibility defined at the observation point. The scorecard uses an offset of
    zero, since all history up to the decision point is fair game there. The uplift stage
    must use an offset of at least the treatment lookback, because a credit limit increase
    inside the feature window leaks the treatment into the covariates: utilisation is
    balance over limit, so raising the limit mechanically moves every utilisation feature.
    """
    anchor_month = observation_month - feature_offset
    if anchor_month < int(panel.months.min()):
        return pd.DataFrame()
    obs = panel.column_of(anchor_month)
    eligible, _ = eligibility_mask(panel, observation_month, window)

    present = panel.present[:, : obs + 1][eligible]
    utilisation = np.where(present, panel.utilisation[:, : obs + 1][eligible], np.nan)
    dpd = np.where(present, panel.dpd[:, : obs + 1][eligible], np.nan)
    limit = np.where(present, panel.limit[:, : obs + 1][eligible], np.nan)
    balance = np.where(present, panel.balance[:, : obs + 1][eligible], np.nan)

    def extra(name: str) -> np.ndarray:
        matrix = panel.extra.get(name)
        if matrix is None:
            return np.full(present.shape, np.nan, dtype="float32")
        return np.where(present, matrix[:, : obs + 1][eligible], np.nan)

    payment = extra("AMT_PAYMENT_CURRENT")
    minimum_due = extra("AMT_INST_MIN_REGULARITY")
    drawings = extra("AMT_DRAWINGS_CURRENT")
    drawings_atm = extra("AMT_DRAWINGS_ATM_CURRENT")

    payment_ratio = _safe_ratio(payment, minimum_due)
    features: Dict[str, np.ndarray] = {
        "SK_ID_PREV": panel.contract_ids[eligible],
        "observation_month": observation_month,
    }

    features["tenure_months"] = present.sum(axis=1).astype("float32")
    features["util_at_observation"] = utilisation[:, -1]
    features["limit_at_observation_log"] = np.log1p(
        np.clip(np.nan_to_num(limit[:, -1], nan=0.0), 0, None)
    ).astype("float32")

    for months in lookbacks:
        tail = slice(-months, None)
        util_tail = utilisation[:, tail]
        dpd_tail = dpd[:, tail]

        features[f"util_mean_{months}m"] = _nan_agg(util_tail, "mean")
        features[f"util_max_{months}m"] = _nan_agg(util_tail, "max")
        features[f"util_std_{months}m"] = _nan_agg(util_tail, "std")
        features[f"months_overlimit_{months}m"] = np.nansum(util_tail > 1.0, axis=1).astype("float32")
        features[f"dpd_max_{months}m"] = _nan_agg(dpd_tail, "max")
        features[f"months_in_arrears_{months}m"] = np.nansum(dpd_tail > 0, axis=1).astype("float32")
        features[f"payment_ratio_mean_{months}m"] = _nan_agg(payment_ratio[:, tail], "mean")
        features[f"payment_ratio_std_{months}m"] = _nan_agg(payment_ratio[:, tail], "std")
        features[f"months_zero_payment_{months}m"] = np.nansum(
            np.nan_to_num(payment[:, tail], nan=0.0) <= 0, axis=1
        ).astype("float32")
        features[f"atm_share_{months}m"] = _safe_ratio(
            np.nansum(drawings_atm[:, tail], axis=1), np.nansum(drawings[:, tail], axis=1)
        )
        features[f"drawings_to_limit_{months}m"] = _safe_ratio(
            np.nansum(drawings[:, tail], axis=1), np.nan_to_num(limit[:, -1], nan=0.0)
        )

    longest = max(lookbacks)
    features[f"util_slope_{longest}m"] = _slope(utilisation[:, -longest:])
    features[f"balance_slope_{longest}m"] = _slope(balance[:, -longest:])
    features["util_trend_ratio"] = _safe_ratio(
        features[f"util_mean_{min(lookbacks)}m"], features[f"util_mean_{longest}m"]
    )
    features["months_since_arrears"] = _months_since_last(np.nan_to_num(dpd, nan=0.0) > 0)
    features["months_since_overlimit"] = _months_since_last(np.nan_to_num(utilisation, nan=0.0) > 1.0)
    features["dpd_max_ever"] = _nan_agg(dpd, "max")
    features["months_in_arrears_ever"] = np.nansum(dpd > 0, axis=1).astype("float32")
    features["limit_changes_ever"] = (
        np.abs(np.diff(np.nan_to_num(limit, nan=0.0), axis=1)) > 0
    ).sum(axis=1).astype("float32")

    return pd.DataFrame(features)


def build_feature_table(
    panel: PanelMatrices,
    window: WindowConfig,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    feature_offset: int = 0,
) -> pd.DataFrame:
    """Stack features across every observation point."""
    frames: List[pd.DataFrame] = [
        build_features(panel, month, window, lookbacks, feature_offset)
        for month in window.observation_months()
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True)


KEY_COLUMNS = ("SK_ID_PREV", "observation_month")

FORBIDDEN_PREFIXES = ("outcome_", "target")
FORBIDDEN_NAMES = frozenset({
    "treated", "limit_increase_amount", "observed_revenue", "baseline_exposure",
    "pre_max_dpd", "pre_max_utilisation", "recent_max_dpd", "recent_max_utilisation",
    "limit_at_observation", "balance_at_observation",
})


def feature_columns(feature_frame: pd.DataFrame) -> List[str]:
    """Model inputs, taken from the feature frame itself rather than by exclusion.

    This deliberately reads the columns the feature builder produced instead of removing
    known-bad names from a merged frame. An exclusion list silently stops being correct
    the moment a new column is added upstream, which is exactly what happened here: the
    treatment amount and two outcome window aggregates were added to the labelled
    population and leaked straight into the covariate matrix, taking the propensity model
    to an AUC of 0.9953 and destroying every overlap diagnostic.
    """
    columns = [c for c in feature_frame.columns if c not in KEY_COLUMNS]
    assert_no_leakage(columns)
    return columns


def assert_no_leakage(columns: List[str]) -> None:
    """Fail loudly if a treatment or outcome column reaches the model inputs."""
    offenders = [
        c for c in columns
        if c in FORBIDDEN_NAMES or c.startswith(FORBIDDEN_PREFIXES)
    ]
    if offenders:
        raise ValueError(
            f"treatment or outcome columns reached the feature set: {sorted(offenders)}"
        )
