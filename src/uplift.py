"""Two model uplift estimation on observational data, with Qini evaluation.

Nothing here recovers a causal effect on its own. Credit limit increases in this book were
not randomised, they were granted to accounts the lender already judged to be good, so the
raw treated minus control difference measures selection far more than treatment. What the
propensity model does is make that selection explicit, let the sample be trimmed to the
region where treated and control accounts actually look alike, and weight what remains.
The result is an observational approximation with a stated overlap diagnostic, not a
substitute for an A/B test on limit increases.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


@dataclass
class UpliftConfig:
    propensity_trim: float = 0.02
    n_estimators: int = 200
    max_leaf_nodes: int = 31
    learning_rate: float = 0.08
    random_state: int = 42
    qini_bins: int = 20


def _classifier(config: UpliftConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.n_estimators, max_leaf_nodes=config.max_leaf_nodes,
        learning_rate=config.learning_rate, random_state=config.random_state,
    )


def _regressor(config: UpliftConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.n_estimators, max_leaf_nodes=config.max_leaf_nodes,
        learning_rate=config.learning_rate, random_state=config.random_state,
    )


class PropensityModel:
    """Probability of having received a limit increase, given pre-treatment behaviour."""

    def __init__(self, config: UpliftConfig):
        self.config = config
        self.model = _classifier(config)

    def fit(self, features: np.ndarray, treated: np.ndarray) -> "PropensityModel":
        self.model.fit(features, treated)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def diagnostics(self, features: np.ndarray, treated: np.ndarray) -> Dict[str, float]:
        """A high AUC here is bad news for the uplift estimate, not good news.

        It means treatment is almost perfectly predictable from pre-treatment behaviour,
        so treated and control accounts occupy different regions of feature space and
        there is little genuine overlap to compare across.
        """
        propensity = self.predict(features)
        return {
            "propensity_auc": round(float(roc_auc_score(treated, propensity)), 4),
            "treated_mean_propensity": round(float(propensity[treated == 1].mean()), 4),
            "control_mean_propensity": round(float(propensity[treated == 0].mean()), 4),
        }


def cross_fitted_propensity(
    features: np.ndarray,
    treated: np.ndarray,
    groups: np.ndarray,
    config: UpliftConfig,
    n_splits: int = 5,
) -> np.ndarray:
    """Out of fold propensity scores.

    In sample propensity scores from a boosted model are overfitted: the model partly
    memorises who was treated, which inflates the reported AUC and pushes the common
    support trimming and the inverse probability weights around. Every row therefore gets
    a score from a model that never saw it. Folds are grouped by account, because the same
    account appears at several observation points.
    """
    folds = GroupKFold(n_splits=n_splits)
    out_of_fold = np.zeros(len(treated), dtype="float64")
    for train_index, test_index in folds.split(features, treated, groups=groups):
        model = _classifier(config)
        model.fit(features[train_index], treated[train_index])
        out_of_fold[test_index] = model.predict_proba(features[test_index])[:, 1]
    return out_of_fold


def cross_fitted_arms(
    features: np.ndarray,
    treated: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    config: UpliftConfig,
    binary: bool,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out of fold predictions from both arms, so Qini is not scored on fitted values.

    Returning the untreated arm separately matters for the exposure arithmetic: expected
    loss needs each account's own baseline default probability, not the portfolio average.
    """
    folds = GroupKFold(n_splits=n_splits)
    control = np.zeros(len(treated), dtype="float64")
    treated_arm = np.zeros(len(treated), dtype="float64")
    for train_index, test_index in folds.split(features, treated, groups=groups):
        train_treated = treated[train_index]
        if train_treated.sum() == 0 or (1 - train_treated).sum() == 0:
            continue
        learner = TLearner(config, binary=binary).fit(
            features[train_index], train_treated, outcome[train_index]
        )
        control[test_index] = learner._predict(learner.control_model, features[test_index])
        treated_arm[test_index] = learner._predict(learner.treated_model, features[test_index])
    return control, treated_arm


def cross_fitted_cate(
    features: np.ndarray,
    treated: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    config: UpliftConfig,
    binary: bool,
    n_splits: int = 5,
) -> np.ndarray:
    """Out of fold treatment effect estimates."""
    control, treated_arm = cross_fitted_arms(
        features, treated, outcome, groups, config, binary, n_splits
    )
    return treated_arm - control


def propensity_diagnostics(propensity: np.ndarray, treated: np.ndarray) -> Dict[str, float]:
    """A high AUC here is bad news for the uplift estimate, not good news.

    It means treatment is close to predictable from pre-treatment behaviour, so treated
    and control accounts occupy different regions of feature space and there is little
    genuine overlap left to compare across.
    """
    return {
        "propensity_auc": round(float(roc_auc_score(treated, propensity)), 4),
        "treated_mean_propensity": round(float(propensity[treated == 1].mean()), 4),
        "control_mean_propensity": round(float(propensity[treated == 0].mean()), 4),
    }


def common_support(propensity: np.ndarray, treated: np.ndarray, trim: float) -> np.ndarray:
    """Keep rows whose propensity falls inside the overlap of both arms."""
    treated_scores = propensity[treated == 1]
    control_scores = propensity[treated == 0]
    if treated_scores.size == 0 or control_scores.size == 0:
        return np.zeros(len(propensity), dtype=bool)
    lower = max(np.quantile(treated_scores, trim), np.quantile(control_scores, trim))
    upper = min(
        np.quantile(treated_scores, 1 - trim), np.quantile(control_scores, 1 - trim)
    )
    return (propensity >= lower) & (propensity <= upper)


class TLearner:
    """Separate outcome models for the treated and control arms."""

    def __init__(self, config: UpliftConfig, binary: bool):
        self.config = config
        self.binary = binary
        self.treated_model = _classifier(config) if binary else _regressor(config)
        self.control_model = _classifier(config) if binary else _regressor(config)

    def fit(self, features: np.ndarray, treated: np.ndarray, outcome: np.ndarray) -> "TLearner":
        self.treated_model.fit(features[treated == 1], outcome[treated == 1])
        self.control_model.fit(features[treated == 0], outcome[treated == 0])
        return self

    def _predict(self, model, features: np.ndarray) -> np.ndarray:
        if self.binary:
            return model.predict_proba(features)[:, 1]
        return model.predict(features)

    def predict_cate(self, features: np.ndarray) -> np.ndarray:
        """Estimated effect of a limit increase for each account."""
        return self._predict(self.treated_model, features) - self._predict(
            self.control_model, features
        )


def inverse_probability_ate(
    outcome: np.ndarray, treated: np.ndarray, propensity: np.ndarray, clip: float = 0.01
) -> float:
    """Average treatment effect reweighted by inverse propensity."""
    propensity = np.clip(propensity, clip, 1 - clip)
    treated_term = (treated * outcome / propensity).sum() / (treated / propensity).sum()
    control_term = ((1 - treated) * outcome / (1 - propensity)).sum() / (
        (1 - treated) / (1 - propensity)
    ).sum()
    return float(treated_term - control_term)


def naive_ate(outcome: np.ndarray, treated: np.ndarray) -> float:
    return float(outcome[treated == 1].mean() - outcome[treated == 0].mean())


def qini_curve(
    outcome: np.ndarray, treated: np.ndarray, ranking: np.ndarray, bins: int = 20
) -> pd.DataFrame:
    """Cumulative incremental outcome as targeting depth increases.

    At each depth the treated response is compared with the control response scaled to
    the same treated population size, which is the standard Qini adjustment for unequal
    arm sizes inside the targeted group.
    """
    order = np.argsort(-np.asarray(ranking, dtype="float64"))
    outcome = np.asarray(outcome, dtype="float64")[order]
    treated = np.asarray(treated, dtype="int8")[order]
    n = len(outcome)

    cumulative_treated = np.cumsum(treated)
    cumulative_control = np.cumsum(1 - treated)
    cumulative_treated_outcome = np.cumsum(outcome * treated)
    cumulative_control_outcome = np.cumsum(outcome * (1 - treated))

    depths = np.unique(np.linspace(0, n, bins + 1).astype(int))
    rows = []
    for depth in depths:
        if depth == 0:
            rows.append({"targeted": 0, "fraction_targeted": 0.0, "qini": 0.0,
                         "treated": 0, "control": 0})
            continue
        index = depth - 1
        n_treated = cumulative_treated[index]
        n_control = cumulative_control[index]
        ratio = n_treated / n_control if n_control > 0 else 0.0
        qini = cumulative_treated_outcome[index] - cumulative_control_outcome[index] * ratio
        rows.append({
            "targeted": int(depth),
            "fraction_targeted": depth / n,
            "qini": float(qini),
            "treated": int(n_treated),
            "control": int(n_control),
        })
    curve = pd.DataFrame(rows)
    final = curve["qini"].iloc[-1]
    curve["random"] = curve["fraction_targeted"] * final
    return curve


def qini_coefficient(curve: pd.DataFrame) -> Dict[str, float]:
    """Area between the Qini curve and the random targeting line."""
    x = curve["fraction_targeted"].to_numpy()
    actual = np.trapezoid(curve["qini"].to_numpy(), x)
    random = np.trapezoid(curve["random"].to_numpy(), x)
    return {
        "qini_auc": round(float(actual), 2),
        "random_auc": round(float(random), 2),
        "qini_coefficient": round(float(actual - random), 2),
        "lift_over_random": round(float(actual / random), 4) if random != 0 else float("nan"),
    }


def decile_effects(
    outcome: np.ndarray, treated: np.ndarray, ranking: np.ndarray, groups: int = 10
) -> pd.DataFrame:
    """Observed treated minus control outcome within each band of estimated effect."""
    frame = pd.DataFrame({"outcome": outcome, "treated": treated, "ranking": ranking})
    frame["band"] = pd.qcut(frame["ranking"].rank(method="first"), groups, labels=False)
    rows = []
    for band, subset in frame.groupby("band", observed=True):
        treated_arm = subset[subset["treated"] == 1]["outcome"]
        control_arm = subset[subset["treated"] == 0]["outcome"]
        rows.append({
            "band": int(groups - band),
            "accounts": int(len(subset)),
            "treated": int(len(treated_arm)),
            "control": int(len(control_arm)),
            "treated_outcome": float(treated_arm.mean()) if len(treated_arm) else np.nan,
            "control_outcome": float(control_arm.mean()) if len(control_arm) else np.nan,
            "observed_effect": (
                float(treated_arm.mean() - control_arm.mean())
                if len(treated_arm) and len(control_arm) else np.nan
            ),
            "estimated_effect": float(subset["ranking"].mean()),
        })
    return pd.DataFrame(rows).sort_values("band").reset_index(drop=True)
