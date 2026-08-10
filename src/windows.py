"""Observation point and outcome window construction.

This module owns the entire definition of the behavioural target. Window length,
delinquency threshold, history requirements and observation spacing all arrive from
configuration, so a different performance window is a config change rather than a code
change. Nothing downstream is allowed to redefine what a bad account is.

The panel is held as contract by month matrices. A single account contributes one row
per observation point, so the resulting population is a stacked panel and any train and
test split must group on SK_ID_PREV to avoid leaking an account across folds.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import TreatmentConfig, WindowConfig

EXCLUSION_ORDER = (
    "not_observed_at_observation_point",
    "insufficient_pre_history",
    "insufficient_outcome_window",
    "no_active_credit_limit",
    "already_delinquent_at_observation",
)


@dataclass
class PanelMatrices:
    """Contract by month views of the card panel, aligned on a shared month index."""

    contract_ids: np.ndarray
    client_ids: np.ndarray
    months: np.ndarray
    dpd: np.ndarray
    limit: np.ndarray
    balance: np.ndarray
    present: np.ndarray
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def column_of(self, month: int) -> int:
        matches = np.flatnonzero(self.months == month)
        if matches.size == 0:
            raise KeyError(f"month {month} is outside the panel range")
        return int(matches[0])

    def matrix(self, name: str) -> np.ndarray:
        if name in self.extra:
            return self.extra[name]
        raise KeyError(f"{name} was not loaded into the panel")

    @property
    def utilisation(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.limit > 0, self.balance / self.limit, np.nan)


def build_panel(
    frame: pd.DataFrame, months: Sequence[int], extra_values: Sequence[str] = ()
) -> PanelMatrices:
    """Pivot the long card panel into aligned contract by month matrices."""
    months = np.asarray(list(months), dtype=int)
    contract_ids = np.sort(frame["SK_ID_PREV"].unique())

    def pivot(value: str, aggfunc: str = "max") -> pd.DataFrame:
        """Pivot onto the full contract by month grid.

        pivot_table silently drops any contract whose values are all null, which for the
        20 percent null payment columns would misalign the matrices against the panel.
        Reindexing on the full contract list keeps every matrix the same shape.
        """
        return frame.pivot_table(
            index="SK_ID_PREV", columns="MONTHS_BALANCE", values=value, aggfunc=aggfunc
        ).reindex(index=contract_ids, columns=months)

    dpd = pivot("SK_DPD")
    client_lookup = (
        frame.groupby("SK_ID_PREV", observed=True)["SK_ID_CURR"].first().reindex(contract_ids)
    )

    dpd_matrix = dpd.to_numpy(dtype="float32")
    extra = {
        name: pivot(name, "mean").to_numpy(dtype="float32")
        for name in extra_values
        if name in frame.columns
    }
    return PanelMatrices(
        contract_ids=contract_ids,
        client_ids=client_lookup.to_numpy(),
        months=months,
        dpd=dpd_matrix,
        limit=pivot("AMT_CREDIT_LIMIT_ACTUAL").to_numpy(dtype="float32"),
        balance=pivot("AMT_BALANCE").to_numpy(dtype="float32"),
        present=~np.isnan(dpd_matrix),
        extra=extra,
    )


def eligibility_mask(panel: PanelMatrices, observation_month: int, window: WindowConfig):
    """Boolean mask of accounts scoreable at an observation point, plus exclusion counts.

    Shared by target construction and feature engineering so the two cannot drift apart.
    """
    obs = panel.column_of(observation_month)
    outcome_slice = slice(obs + 1, obs + 1 + window.outcome_window_months)

    pre_mask = panel.present[:, : obs + 1]
    outcome_months = panel.present[:, outcome_slice].sum(axis=1)
    pre_dpd = np.nan_to_num(_row_max(panel.dpd[:, : obs + 1], pre_mask), nan=0.0)

    observed_at_obs = panel.present[:, obs]
    has_history = pre_mask.sum(axis=1) >= window.min_pre_history_months
    if window.require_full_outcome_window:
        has_outcome = outcome_months == window.outcome_window_months
    else:
        has_outcome = outcome_months >= 1
    already_bad = (
        pre_dpd >= window.dpd_threshold
        if window.exclude_already_delinquent
        else np.zeros(len(pre_dpd), dtype=bool)
    )
    no_limit = (
        ~(np.nan_to_num(panel.limit[:, obs], nan=0.0) > 0)
        if window.require_active_limit_at_observation
        else np.zeros(len(pre_dpd), dtype=bool)
    )

    failures = [~observed_at_obs, ~has_history, ~has_outcome, no_limit, already_bad]
    exclusions: Dict[str, int] = {}
    remaining = np.ones(len(observed_at_obs), dtype=bool)
    for reason, failed in zip(EXCLUSION_ORDER, failures):
        exclusions[reason] = int((remaining & failed).sum())
        remaining = remaining & ~failed
    return remaining, exclusions


def _row_max(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row wise maximum over masked cells, returning nan for rows with no data."""
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.where(mask, values, np.nan), axis=1)


def _row_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row wise mean over masked cells, returning nan for rows with no data."""
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(mask, values, np.nan), axis=1).astype("float32")


def _row_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row wise sum over masked cells, treating unobserved months as zero."""
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan, dtype="float32")
    return np.nansum(np.where(mask, values, np.nan), axis=1).astype("float32")


def limit_increase_matrix(panel: PanelMatrices, min_increase_ratio: float) -> np.ndarray:
    """Month on month credit limit increases exceeding the configured relative size."""
    previous = panel.limit[:, :-1]
    delta = np.diff(panel.limit, axis=1)
    observed = panel.present[:, :-1] & panel.present[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(previous > 0, delta / previous, np.inf)
    return (delta > 0) & (ratio > min_increase_ratio) & observed


def label_at(
    panel: PanelMatrices,
    observation_month: int,
    window: WindowConfig,
    treatment: TreatmentConfig,
    increases: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build the labelled cohort for a single observation point.

    Returns the eligible rows and a count of accounts removed by each exclusion rule,
    applied in a fixed order so the reasons partition the excluded population.
    """
    obs = panel.column_of(observation_month)
    outcome_slice = slice(obs + 1, obs + 1 + window.outcome_window_months)

    pre_mask = panel.present[:, : obs + 1]
    outcome_mask = panel.present[:, outcome_slice]
    pre_months = pre_mask.sum(axis=1)
    outcome_months = outcome_mask.sum(axis=1)

    pre_dpd = np.nan_to_num(_row_max(panel.dpd[:, : obs + 1], pre_mask), nan=0.0)
    utilisation = panel.utilisation
    pre_util = _row_max(utilisation[:, : obs + 1], pre_mask)

    eligible, exclusions = eligibility_mask(panel, observation_month, window)
    outcome_dpd = np.nan_to_num(_row_max(panel.dpd[:, outcome_slice], outcome_mask), nan=0.0)
    outcome_util = _row_max(utilisation[:, outcome_slice], outcome_mask)

    treat_start = max(obs - treatment.lookback_months + 1, 1)
    treated = increases[:, treat_start - 1 : obs].sum(axis=1) > 0
    limit_before = panel.limit[:, max(obs - treatment.lookback_months, 0)]
    limit_increase = np.nan_to_num(panel.limit[:, obs], nan=0.0) - np.nan_to_num(
        limit_before, nan=0.0
    )

    recent = slice(max(obs - window.recent_state_months + 1, 0), obs + 1)
    recent_mask = panel.present[:, recent]
    recent_dpd = np.nan_to_num(_row_max(panel.dpd[:, recent], recent_mask), nan=0.0)
    recent_util = _row_max(utilisation[:, recent], recent_mask)

    drawings = panel.extra.get("AMT_DRAWINGS_CURRENT")
    outcome_balance = _row_mean(panel.balance[:, outcome_slice], outcome_mask)
    outcome_drawings = (
        _row_sum(drawings[:, outcome_slice], outcome_mask)
        if drawings is not None
        else np.full(len(pre_dpd), np.nan, dtype="float32")
    )

    cohort = pd.DataFrame(
        {
            "SK_ID_PREV": panel.contract_ids[eligible],
            "SK_ID_CURR": panel.client_ids[eligible],
            "observation_month": observation_month,
            "pre_history_months": pre_months[eligible].astype("int16"),
            "outcome_months_observed": outcome_months[eligible].astype("int16"),
            "pre_max_dpd": pre_dpd[eligible].astype("float32"),
            "pre_max_utilisation": pre_util[eligible].astype("float32"),
            "recent_max_dpd": recent_dpd[eligible].astype("float32"),
            "recent_max_utilisation": recent_util[eligible].astype("float32"),
            "limit_at_observation": panel.limit[eligible, obs],
            "balance_at_observation": panel.balance[eligible, obs],
            "limit_increase_amount": limit_increase[eligible].astype("float32"),
            "treated": treated[eligible],
            "target": (outcome_dpd[eligible] >= window.dpd_threshold).astype("int8"),
            "outcome_max_dpd": outcome_dpd[eligible].astype("float32"),
            "outcome_max_utilisation": outcome_util[eligible].astype("float32"),
            "outcome_mean_balance": outcome_balance[eligible].astype("float32"),
            "outcome_total_drawings": outcome_drawings[eligible].astype("float32"),
        }
    )
    return cohort, exclusions


def build_labelled_population(
    panel: PanelMatrices,
    window: WindowConfig,
    treatment: TreatmentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stack every observation point into one labelled population.

    The second frame is the exclusion ledger: how many accounts each rule removed at
    each observation point, so the reported sample size is reconcilable to the panel.
    """
    increases = limit_increase_matrix(panel, treatment.min_increase_ratio)
    cohorts: List[pd.DataFrame] = []
    ledger: List[Dict] = []

    for month in window.observation_months():
        cohort, exclusions = label_at(panel, month, window, treatment, increases)
        cohorts.append(cohort)
        ledger.append(
            {
                "observation_month": month,
                "contracts_in_panel": len(panel.contract_ids),
                **exclusions,
                "eligible": len(cohort),
                "bads": int(cohort["target"].sum()),
                "treated": int(cohort["treated"].sum()),
            }
        )

    population = pd.concat(cohorts, ignore_index=True)
    return population, pd.DataFrame(ledger)


def add_secondary_targets(
    population: pd.DataFrame, overlimit_threshold: float, soft_dpd_threshold: int
) -> pd.DataFrame:
    """Attach the lower severity risk outcomes used by the uplift stage.

    The 90 plus delinquency target is too rare among treated accounts to support an
    incremental risk estimate, so the uplift stage models these instead. Each carries
    its own already in state exclusion flag rather than reusing the primary one.

    Overlimit uses the recent state rather than the whole pre window. Being over limit is
    transient, unlike 90 plus delinquency which persists, so one breach three years ago
    should not disqualify an account from the overlimit analysis.
    """
    result = population.copy()
    result["target_soft_dpd"] = (result["outcome_max_dpd"] >= soft_dpd_threshold).astype("int8")
    result["target_overlimit"] = (
        result["outcome_max_utilisation"] > overlimit_threshold
    ).astype("int8")
    result["already_soft_dpd_at_observation"] = (
        result["pre_max_dpd"] >= soft_dpd_threshold
    ).astype("int8")
    result["already_overlimit_at_observation"] = (
        result["recent_max_utilisation"] > overlimit_threshold
    ).astype("int8")
    return result
