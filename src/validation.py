"""Discrimination, separation, stability and calibration measures.

Gini and KS answer whether the score ranks risk. PSI answers whether the population the
score is applied to still looks like the one it was fitted on. Calibration answers
whether the predicted probability means what it claims. A behavioural scorecard needs all
four, and on this dataset the stability measures are the interesting ones, because the
bad rate moves sharply across observation points.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(target: np.ndarray, score: np.ndarray) -> float:
    """Gini coefficient, where the score is oriented so higher means safer."""
    target = np.asarray(target)
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(2 * roc_auc_score(target, -np.asarray(score)) - 1)


def ks_statistic(target: np.ndarray, score: np.ndarray) -> float:
    """Maximum separation between the cumulative bad and good score distributions."""
    target = np.asarray(target)
    score = np.asarray(score)
    if len(np.unique(target)) < 2:
        return float("nan")
    order = np.argsort(score)
    sorted_target = target[order]
    bads = np.cumsum(sorted_target) / max(sorted_target.sum(), 1)
    goods = np.cumsum(1 - sorted_target) / max((1 - sorted_target).sum(), 1)
    return float(np.max(np.abs(bads - goods)))


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, bins: int = 10
) -> float:
    """PSI of a comparison sample against reference deciles.

    Bin edges come from the reference distribution, and empty bins are floored so a
    single missing bucket cannot send the index to infinity.
    """
    reference = np.asarray(reference, dtype="float64")
    comparison = np.asarray(comparison, dtype="float64")
    reference = reference[~np.isnan(reference)]
    comparison = comparison[~np.isnan(comparison)]
    if reference.size == 0 or comparison.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / reference.size
    comparison_share = np.histogram(comparison, bins=edges)[0] / comparison.size
    floor = 1e-4
    reference_share = np.clip(reference_share, floor, None)
    comparison_share = np.clip(comparison_share, floor, None)
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def feature_psi(
    reference: pd.DataFrame, comparison: pd.DataFrame, columns: List[str], bins: int = 10
) -> pd.DataFrame:
    rows = [
        {
            "feature": column,
            "psi": population_stability_index(
                reference[column].to_numpy(), comparison[column].to_numpy(), bins
            ),
        }
        for column in columns
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def score_bands(
    target: np.ndarray, score: np.ndarray, predicted: Optional[np.ndarray] = None, bands: int = 10
) -> pd.DataFrame:
    """Actual and predicted bad rate by score band, worst band first."""
    frame = pd.DataFrame({"target": np.asarray(target), "score": np.asarray(score)})
    if predicted is not None:
        frame["predicted"] = np.asarray(predicted)
    frame["band"] = pd.qcut(frame["score"], bands, duplicates="drop")

    grouped = frame.groupby("band", observed=True)
    table = grouped.agg(
        accounts=("target", "size"),
        bads=("target", "sum"),
        actual_bad_rate=("target", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()
    if predicted is not None:
        table["predicted_bad_rate"] = grouped["predicted"].mean().to_numpy()
    table["band"] = table["band"].astype(str)
    return table.sort_values("min_score").reset_index(drop=True)


def calibration_error(target: np.ndarray, predicted: np.ndarray, bands: int = 10) -> float:
    """Mean absolute gap between predicted and actual bad rate across score bands."""
    frame = pd.DataFrame({"target": np.asarray(target), "predicted": np.asarray(predicted)})
    frame["band"] = pd.qcut(frame["predicted"], bands, duplicates="drop")
    grouped = frame.groupby("band", observed=True).agg(
        actual=("target", "mean"), predicted=("predicted", "mean")
    )
    return float((grouped["actual"] - grouped["predicted"]).abs().mean())


def summarise_performance(
    target: np.ndarray, score: np.ndarray, predicted: Optional[np.ndarray] = None
) -> Dict[str, float]:
    result = {
        "accounts": int(len(target)),
        "bads": int(np.sum(target)),
        "bad_rate_pct": round(float(np.mean(target)) * 100, 3),
        "gini": round(gini(target, score), 4),
        "ks": round(ks_statistic(target, score), 4),
    }
    if predicted is not None:
        result["calibration_error_pct"] = round(calibration_error(target, predicted) * 100, 4)
    return result
