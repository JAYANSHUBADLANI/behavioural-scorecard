"""Logistic scorecard on WOE transformed features, with points scaling.

Scaling follows the usual convention: a higher score means a better account, and the
score falls by exactly `pdo` points every time the odds of going bad double. Points are
allocated back to individual features so the card can be read as a table rather than as
a set of coefficients, which is what a credit committee actually reviews.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import WOETransformer


@dataclass
class ScalingConfig:
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score + self.factor * np.log(1.0 / self.base_odds)


def select_features(
    transformer: WOETransformer,
    woe_frame: pd.DataFrame,
    min_iv: float = 0.02,
    max_correlation: float = 0.75,
) -> List[str]:
    """Keep features above an IV floor, dropping the weaker of any correlated pair."""
    ranked = transformer.iv_table()
    candidates = [f for f in ranked.loc[ranked["iv"] >= min_iv, "feature"] if f in woe_frame]
    if not candidates:
        return []

    correlations = woe_frame[candidates].corr().abs()
    kept: List[str] = []
    for feature in candidates:
        if all(correlations.loc[feature, other] < max_correlation for other in kept):
            kept.append(feature)
    return kept


class Scorecard:
    """WOE logistic regression plus the points transformation."""

    def __init__(self, scaling: Optional[ScalingConfig] = None, C: float = 1.0):
        self.scaling = scaling or ScalingConfig()
        self.model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        self.features: List[str] = []

    def fit(self, woe_frame: pd.DataFrame, target: np.ndarray, features: List[str]) -> "Scorecard":
        self.features = features
        self.model.fit(woe_frame[features].to_numpy(), target)
        return self

    def log_odds(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(woe_frame[self.features].to_numpy())

    def predict_proba(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(woe_frame[self.features].to_numpy())[:, 1]

    def score(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.scaling.offset - self.scaling.factor * self.log_odds(woe_frame)

    def points(self, woe_frame: pd.DataFrame) -> pd.DataFrame:
        """Points contributed by each feature, summing to the total score."""
        n = len(self.features)
        intercept = float(self.model.intercept_[0])
        allocation: Dict[str, np.ndarray] = {}
        for coefficient, feature in zip(self.model.coef_[0], self.features):
            contribution = coefficient * woe_frame[feature].to_numpy()
            allocation[feature] = (
                -self.scaling.factor * (contribution + intercept / n) + self.scaling.offset / n
            )
        frame = pd.DataFrame(allocation, index=woe_frame.index)
        frame["total_score"] = frame.sum(axis=1)
        return frame

    def coefficient_table(self, transformer: WOETransformer) -> pd.DataFrame:
        rows = []
        for coefficient, feature in zip(self.model.coef_[0], self.features):
            rows.append({
                "feature": feature,
                "coefficient": float(coefficient),
                "iv": transformer.bins[feature].iv,
                "bins": len(transformer.bins[feature].woe),
            })
        table = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        table.attrs["intercept"] = float(self.model.intercept_[0])
        return table

    def scorecard_table(self, transformer: WOETransformer) -> pd.DataFrame:
        """The card itself: every bin of every retained feature with its points."""
        n = len(self.features)
        intercept = float(self.model.intercept_[0])
        coefficients = dict(zip(self.features, self.model.coef_[0]))
        rows = []
        for feature in self.features:
            binning = transformer.bins[feature]
            table = binning.table.copy()
            table["points"] = (
                -self.scaling.factor * (coefficients[feature] * table["woe"] + intercept / n)
                + self.scaling.offset / n
            ).round(1)
            rows.append(table)
        return pd.concat(rows, ignore_index=True)
