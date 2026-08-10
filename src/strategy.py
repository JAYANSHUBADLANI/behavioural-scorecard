"""Decision bands and champion against challenger policy comparison.

A ranked model is not a strategy. What a credit committee signs off is a set of decision
bands and the rules that move an account between them, so the uplift output is translated
into four actions here: auto increase, manual review, hold, and decrease or monitor.

Policies are compared by off policy value rather than by their own predicted uplift.
Each account was either given a limit increase or not, so for any proposed policy only the
accounts whose actual treatment matches the policy's recommendation carry information, and
those get reweighted by the inverse probability of receiving the treatment they got. That
is a real comparison against what happened, not a comparison of one model's predictions
against another model's predictions.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

AUTO_INCREASE = "auto increase"
MANUAL_REVIEW = "manual review"
HOLD = "hold"
DECREASE_MONITOR = "decrease or monitor"

BAND_ORDER = (AUTO_INCREASE, MANUAL_REVIEW, HOLD, DECREASE_MONITOR)

SCORE_COLUMN = "score"


@dataclass
class StrategyConfig:
    auto_increase_score_percentile: float = 0.70
    review_score_percentile: float = 0.40
    net_benefit_percentile: float = 0.60
    deteriorating_dpd: int = 1
    monitor_score_percentile: float = 0.10
    challenger_utilisation_low: float = 0.30
    challenger_utilisation_high: float = 0.90
    ipw_clip: float = 0.05
    bootstrap_samples: int = 200


def assign_bands(
    frame: pd.DataFrame,
    config: StrategyConfig,
    score_column: str = "score",
    benefit_column: str = "conservative_net_benefit",
) -> pd.Series:
    """Two dimensional policy grid: risk on the score, opportunity on net benefit.

    Deterioration overrides everything. An account already in arrears does not get a limit
    increase however attractive its estimated net benefit looks, which is a rule a credit
    committee would impose regardless of what the model says.
    """
    score = frame[score_column]
    benefit = frame[benefit_column]
    auto_cut = score.quantile(config.auto_increase_score_percentile)
    review_cut = score.quantile(config.review_score_percentile)
    benefit_cut = benefit.quantile(config.net_benefit_percentile)

    bands = pd.Series(HOLD, index=frame.index, dtype=object)
    bands[(score >= auto_cut) & (benefit >= benefit_cut)] = AUTO_INCREASE
    bands[(score >= auto_cut) & (benefit < benefit_cut)] = MANUAL_REVIEW
    bands[(score >= review_cut) & (score < auto_cut) & (benefit >= benefit_cut)] = MANUAL_REVIEW
    bands[deteriorating(frame, config)] = DECREASE_MONITOR
    return bands


def deteriorating(frame: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Accounts to pull back on rather than grow.

    The modelling population already excludes anything that reached the 30+ DPD threshold
    before the observation point, so a 30 day trigger can never fire here and would leave
    this band permanently empty. Any arrears at all inside the recent window is the usable
    signal, and it carries a 30+ DPD rate roughly four times the sample baseline. The
    bottom score decile is added because an account can deteriorate without having missed
    a payment yet.
    """
    score_floor = frame[SCORE_COLUMN].quantile(config.monitor_score_percentile)
    return (
        (frame["recent_max_dpd"] >= config.deteriorating_dpd)
        | (frame[SCORE_COLUMN] <= score_floor)
    )


def champion_policy(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    """Offer a limit increase to the auto increase band only."""
    return (assign_bands(frame, config) == AUTO_INCREASE).to_numpy().astype(int)


def challenger_policy(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    """The rule a bank would run without any model at all.

    Offer more limit to accounts using a healthy share of the line, since they have shown
    demand, but not to accounts already close to the limit or in arrears, since those look
    like distress rather than opportunity.
    """
    utilisation = frame["recent_max_utilisation"]
    return (
        (utilisation >= config.challenger_utilisation_low)
        & (utilisation <= config.challenger_utilisation_high)
        & ~deteriorating(frame, config)
    ).to_numpy().astype(int)


def scorecard_policy(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    """Rank by the behavioural scorecard alone, ignoring uplift."""
    cut = frame[SCORE_COLUMN].quantile(config.auto_increase_score_percentile)
    return (
        (frame[SCORE_COLUMN] >= cut) & ~deteriorating(frame, config)
    ).to_numpy().astype(int)


def treat_all(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    return np.ones(len(frame), dtype=int)


def treat_none(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    return np.zeros(len(frame), dtype=int)


def random_policy(frame: pd.DataFrame, config: StrategyConfig, share: float = 0.30) -> np.ndarray:
    generator = np.random.default_rng(42)
    return (generator.uniform(size=len(frame)) < share).astype(int)


def realised_value(
    frame: pd.DataFrame, loss_given_default: float, risk_column: str = "target_soft_dpd"
) -> np.ndarray:
    """What each account actually delivered: revenue earned less loss actually incurred.

    This uses observed outcomes rather than modelled ones, which is the whole point. A
    policy that looks good only under its own predictions is not evidence of anything.
    """
    revenue = np.nan_to_num(frame["observed_revenue"].to_numpy(), nan=0.0)
    exposure = np.nan_to_num(frame["baseline_exposure"].to_numpy(), nan=0.0)
    event = frame[risk_column].to_numpy(dtype="float64")
    return revenue - loss_given_default * exposure * event


def policy_value(
    value: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    recommendation: np.ndarray,
    clip: float = 0.05,
) -> Dict[str, float]:
    """Inverse probability weighted value of a policy, per account.

    Only accounts whose observed treatment matches the recommendation contribute, weighted
    by the inverse probability of the treatment they actually received. The effective
    sample size is reported because extreme weights can leave a headline number resting on
    very few accounts.
    """
    propensity = np.clip(propensity, clip, 1 - clip)
    matched = treated == recommendation
    weights = np.where(recommendation == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
    weights = np.where(matched, weights, 0.0)

    total_weight = weights.sum()
    if total_weight <= 0:
        return {"value_per_account": float("nan"), "matched_accounts": 0,
                "effective_sample_size": 0.0, "treated_share_pct": 0.0}

    estimate = float((weights * value).sum() / total_weight)
    effective = float(total_weight ** 2 / (weights ** 2).sum())
    return {
        "value_per_account": round(estimate, 2),
        "matched_accounts": int(matched.sum()),
        "effective_sample_size": round(effective, 1),
        "treated_share_pct": round(float(recommendation.mean()) * 100, 2),
    }


def bootstrap_value(
    value: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    recommendation: np.ndarray,
    groups: np.ndarray,
    clip: float,
    samples: int,
    seed: int = 42,
) -> Dict[str, float]:
    """Confidence interval by resampling accounts, not rows.

    The panel repeats each account across observation points, so resampling rows would
    understate the interval badly.
    """
    generator = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    estimates = []
    for _ in range(samples):
        drawn = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([index_by_group[g] for g in drawn])
        result = policy_value(
            value[rows], treated[rows], propensity[rows], recommendation[rows], clip
        )
        if not np.isnan(result["value_per_account"]):
            estimates.append(result["value_per_account"])

    if not estimates:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "ci_low": round(float(np.quantile(estimates, 0.05)), 2),
        "ci_high": round(float(np.quantile(estimates, 0.95)), 2),
    }


def compare_policies(
    frame: pd.DataFrame,
    policies: Dict[str, Callable[[pd.DataFrame, StrategyConfig], np.ndarray]],
    config: StrategyConfig,
    loss_given_default: float,
    bootstrap: bool = True,
) -> pd.DataFrame:
    """Off policy value for every candidate policy, ranked best first."""
    value = realised_value(frame, loss_given_default)
    treated = frame["treated"].to_numpy().astype(int)
    propensity = frame["propensity"].to_numpy()
    groups = frame["SK_ID_PREV"].to_numpy()

    rows = []
    for name, policy in policies.items():
        recommendation = policy(frame, config)
        result = {"policy": name, **policy_value(
            value, treated, propensity, recommendation, config.ipw_clip
        )}
        if bootstrap and config.bootstrap_samples > 0:
            result.update(bootstrap_value(
                value, treated, propensity, recommendation, groups,
                config.ipw_clip, config.bootstrap_samples,
            ))
        rows.append(result)
    return pd.DataFrame(rows).sort_values("value_per_account", ascending=False).reset_index(
        drop=True
    )


def band_profile(frame: pd.DataFrame, bands: pd.Series, risk_column: str) -> pd.DataFrame:
    """What sits in each decision band, so the policy can be read as a table."""
    working = frame.assign(band=bands)
    rows = []
    for band in BAND_ORDER:
        subset = working[working["band"] == band]
        if subset.empty:
            continue
        rows.append({
            "band": band,
            "accounts": int(len(subset)),
            "share_pct": round(len(subset) / len(working) * 100, 2),
            "mean_score": round(float(subset["score"].mean()), 1),
            "observed_risk_rate_pct": round(float(subset[risk_column].mean()) * 100, 3),
            "mean_utilisation": round(float(subset["recent_max_utilisation"].mean()), 3),
            "mean_baseline_exposure": round(float(subset["baseline_exposure"].mean()), 0),
            "mean_net_benefit": round(float(subset["conservative_net_benefit"].mean()), 0),
        })
    return pd.DataFrame(rows)


def champion_ranking(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    ranking = frame["conservative_net_benefit"].rank(pct=True).to_numpy()
    return np.where(deteriorating(frame, config), -1.0, ranking)


def scorecard_ranking(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    ranking = frame[SCORE_COLUMN].rank(pct=True).to_numpy()
    return np.where(deteriorating(frame, config), -1.0, ranking)


def challenger_ranking(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    """Rule eligibility first, then demand, so the rule becomes a ranking."""
    eligible = challenger_policy(frame, config).astype(bool)
    within = frame["recent_max_utilisation"].rank(pct=True).to_numpy()
    return np.where(eligible, 1.0 + within, np.where(deteriorating(frame, config), -1.0, within))


def random_ranking(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    return np.random.default_rng(42).uniform(size=len(frame))


RANKINGS = {
    "champion, uplift net benefit": champion_ranking,
    "challenger, utilisation rule": challenger_ranking,
    "scorecard only": scorecard_ranking,
    "random": random_ranking,
}


def compare_at_depth(
    frame: pd.DataFrame,
    config: StrategyConfig,
    loss_given_default: float,
    depths: Sequence[float] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Score every ranking at identical targeting depths.

    The free depth comparison is not apples to apples, because a policy that recommends
    treating more accounts is scored on a larger slice of the positively selected treated
    population and looks better for that reason alone. Holding the targeted share fixed
    isolates ranking quality, which is the thing actually in dispute.
    """
    value = realised_value(frame, loss_given_default)
    treated = frame["treated"].to_numpy().astype(int)
    propensity = frame["propensity"].to_numpy()

    rows = []
    for name, ranking_function in RANKINGS.items():
        ranking = ranking_function(frame, config)
        order = np.argsort(-ranking)
        for depth in depths:
            cutoff = int(len(frame) * depth)
            recommendation = np.zeros(len(frame), dtype=int)
            recommendation[order[:cutoff]] = 1
            rows.append({
                "policy": name,
                "depth": depth,
                **policy_value(value, treated, propensity, recommendation, config.ipw_clip),
            })
    return pd.DataFrame(rows)
