"""Monotonic weight of evidence binning and information value.

Bins start as quantile cuts and are then merged until two conditions hold: every bin
carries at least the configured share of the population, and the bad rate moves in one
direction across bins. Monotonicity is not cosmetic here. A scorecard that says rising
utilisation is worse up to 60 percent and then better again cannot be signed off by a
credit committee, and it is usually a sign the model is fitting sampling noise.

Missing values get their own bin rather than being imputed, because a null payment ratio
in this data means no payment record for that month, which is itself informative.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SMOOTHING = 0.5
MISSING_LABEL = "missing"


@dataclass
class BinningConfig:
    max_prebins: int = 20
    min_bin_fraction: float = 0.03
    min_bin_bads: int = 5
    enforce_monotonic: bool = True


@dataclass
class FeatureBinning:
    """Fitted bins for one feature, including its own missing value bin."""

    name: str
    edges: np.ndarray
    woe: np.ndarray
    missing_woe: float
    table: pd.DataFrame
    iv: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype="float64")
        out = np.full(values.shape, self.missing_woe, dtype="float32")
        observed = ~np.isnan(values)
        if self.edges.size and observed.any():
            index = np.searchsorted(self.edges, values[observed], side="right")
            index = np.clip(index, 0, len(self.woe) - 1)
            out[observed] = self.woe[index]
        elif observed.any():
            out[observed] = self.woe[0] if self.woe.size else 0.0
        return out


def _woe_and_iv(bads: np.ndarray, goods: np.ndarray):
    total_bad = bads.sum()
    total_good = goods.sum()
    if total_bad == 0 or total_good == 0:
        return np.zeros(len(bads), dtype="float64"), 0.0
    bad_rate = (bads + SMOOTHING) / (total_bad + SMOOTHING * len(bads))
    good_rate = (goods + SMOOTHING) / (total_good + SMOOTHING * len(goods))
    woe = np.log(bad_rate / good_rate)
    iv = float(((bad_rate - good_rate) * woe).sum())
    return woe, iv


def _merge(counts: List[int], bads: List[int], edges: List[float], position: int):
    counts[position] += counts[position + 1]
    bads[position] += bads[position + 1]
    del counts[position + 1]
    del bads[position + 1]
    del edges[position]


def _enforce_minimums(counts, bads, edges, min_count, min_bads):
    changed = True
    while changed and len(counts) > 1:
        changed = False
        for i in range(len(counts)):
            goods = counts[i] - bads[i]
            too_small = counts[i] < min_count or bads[i] < min_bads or goods < 1
            if not too_small:
                continue
            position = i - 1 if i == len(counts) - 1 else i
            _merge(counts, bads, edges, position)
            changed = True
            break
    return counts, bads, edges


def _enforce_monotonic(counts, bads, edges):
    while len(counts) > 2:
        rates = np.array(bads, dtype="float64") / np.array(counts, dtype="float64")
        direction = np.sign(np.corrcoef(np.arange(len(rates)), rates)[0, 1])
        if np.isnan(direction) or direction == 0:
            direction = 1.0
        diffs = np.diff(rates) * direction
        violations = np.flatnonzero(diffs < 0)
        if violations.size == 0:
            break
        worst = int(violations[np.argmin(diffs[violations])])
        _merge(counts, bads, edges, worst)
    return counts, bads, edges


def fit_feature(
    values: np.ndarray, target: np.ndarray, name: str, config: BinningConfig
) -> FeatureBinning:
    """Fit monotonic WOE bins for a single numeric feature."""
    values = np.asarray(values, dtype="float64")
    target = np.asarray(target, dtype="int8")

    missing = np.isnan(values)
    missing_bads = int(target[missing].sum())
    missing_count = int(missing.sum())

    observed_values = values[~missing]
    observed_target = target[~missing]

    if observed_values.size == 0 or np.unique(observed_values).size < 2:
        if observed_values.size == 0 or missing_count == 0:
            # Only one real group exists here, either every value is missing or none
            # is, so there is nothing to compare missingness against and woe/iv are
            # both genuinely zero rather than merely unset.
            woe_missing, iv = 0.0, 0.0
            rows = []
            if observed_values.size:
                observed_bads = int(observed_target.sum())
                rows.append({"feature": name, "bin": "(-inf, inf]",
                             "count": int(observed_values.size), "bads": observed_bads,
                             "bad_rate": observed_bads / observed_values.size, "woe": 0.0})
            if missing_count:
                rows.append({"feature": name, "bin": MISSING_LABEL, "count": missing_count,
                             "bads": missing_bads, "bad_rate": missing_bads / missing_count,
                             "woe": 0.0})
            table = pd.DataFrame(rows)
            table["iv"] = iv
            woe_array = np.array([0.0])
        else:
            # Two real groups: the single constant non-missing bin, and missing. A
            # single-bin call here would compare the missing group against itself and
            # always return zero regardless of how strongly missingness predicts bad
            # rate, so both groups go through the same paired woe/iv computation used
            # for ordinary bins below.
            observed_bads = int(observed_target.sum())
            observed_goods = int(observed_values.size - observed_bads)
            pair_woe, iv = _woe_and_iv(
                np.array([observed_bads, missing_bads]),
                np.array([observed_goods, missing_count - missing_bads]),
            )
            observed_woe, woe_missing = float(pair_woe[0]), float(pair_woe[1])
            table = pd.DataFrame([
                {"feature": name, "bin": "(-inf, inf]", "count": int(observed_values.size),
                 "bads": observed_bads, "bad_rate": observed_bads / observed_values.size,
                 "woe": observed_woe},
                {"feature": name, "bin": MISSING_LABEL, "count": missing_count,
                 "bads": missing_bads, "bad_rate": missing_bads / missing_count,
                 "woe": woe_missing},
            ])
            table["iv"] = iv
            woe_array = np.array([observed_woe])
        return FeatureBinning(name, np.array([]), woe_array, woe_missing, table, iv)

    quantiles = np.linspace(0, 1, config.max_prebins + 1)[1:-1]
    edges = list(np.unique(np.quantile(observed_values, quantiles)))
    index = np.searchsorted(np.array(edges), observed_values, side="right")
    n_bins = len(edges) + 1
    counts = list(np.bincount(index, minlength=n_bins).astype(int))
    bads = list(np.bincount(index, weights=observed_target, minlength=n_bins).astype(int))

    min_count = max(int(config.min_bin_fraction * len(values)), 1)
    counts, bads, edges = _enforce_minimums(
        counts, bads, edges, min_count, config.min_bin_bads
    )
    if config.enforce_monotonic:
        counts, bads, edges = _enforce_monotonic(counts, bads, edges)

    counts_array = np.array(counts, dtype="float64")
    bads_array = np.array(bads, dtype="float64")
    goods_array = counts_array - bads_array

    all_bads = np.append(bads_array, missing_bads)
    all_goods = np.append(goods_array, missing_count - missing_bads)
    woe_all, iv = _woe_and_iv(all_bads, all_goods)
    woe = woe_all[:-1]
    missing_woe = float(woe_all[-1]) if missing_count else 0.0

    labels = []
    boundaries = [-np.inf] + list(edges) + [np.inf]
    for i in range(len(counts)):
        labels.append(f"({boundaries[i]:.4g}, {boundaries[i + 1]:.4g}]")
    rows = [
        {"feature": name, "bin": labels[i], "count": int(counts[i]), "bads": int(bads[i]),
         "bad_rate": float(bads[i] / counts[i]) if counts[i] else 0.0, "woe": float(woe[i])}
        for i in range(len(counts))
    ]
    if missing_count:
        rows.append({"feature": name, "bin": MISSING_LABEL, "count": missing_count,
                     "bads": missing_bads,
                     "bad_rate": missing_bads / missing_count, "woe": missing_woe})
    table = pd.DataFrame(rows)
    table["iv"] = iv
    return FeatureBinning(name, np.array(edges), woe.astype("float64"), missing_woe, table, iv)


class WOETransformer:
    """Fits and applies WOE bins across a set of features."""

    def __init__(self, config: Optional[BinningConfig] = None):
        self.config = config or BinningConfig()
        self.bins: Dict[str, FeatureBinning] = {}

    def fit(self, frame: pd.DataFrame, target: np.ndarray, columns: List[str]) -> "WOETransformer":
        for column in columns:
            self.bins[column] = fit_feature(
                frame[column].to_numpy(), target, column, self.config
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: binning.transform(frame[name].to_numpy())
             for name, binning in self.bins.items()},
            index=frame.index,
        )

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": name, "iv": binning.iv, "bins": len(binning.woe)}
                for name, binning in self.bins.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def bin_table(self) -> pd.DataFrame:
        return pd.concat([b.table for b in self.bins.values()], ignore_index=True)
