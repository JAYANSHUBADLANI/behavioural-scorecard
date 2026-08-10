"""Tests for monotonic WOE binning and information value."""

import numpy as np
import pandas as pd
import pytest

from src.binning import BinningConfig, MISSING_LABEL, WOETransformer, fit_feature

RNG = np.random.default_rng(7)


def graded_feature(n=6000, strength=3.0):
    """A feature where higher values carry monotonically higher bad rates."""
    x = RNG.uniform(0, 1, n)
    probability = 1 / (1 + np.exp(-(strength * (x - 0.5))))
    y = RNG.binomial(1, probability * 0.3)
    return x, y


class TestWeightOfEvidence:
    def test_higher_bad_rate_gives_higher_woe(self):
        x = np.concatenate([np.zeros(1000), np.ones(1000)])
        y = np.concatenate([
            np.ones(50), np.zeros(950), np.ones(200), np.zeros(800)
        ]).astype(int)
        binning = fit_feature(x, y, "f", BinningConfig(min_bin_fraction=0.1, min_bin_bads=5))
        table = binning.table
        worst = table.loc[table["bad_rate"].idxmax()]
        best = table.loc[table["bad_rate"].idxmin()]
        assert worst["woe"] > best["woe"]
        assert len(binning.woe) == 2

    def test_bin_with_no_bads_is_merged_away(self):
        x = np.concatenate([np.zeros(1000), np.ones(1000)])
        y = np.concatenate([np.zeros(1000), np.ones(200), np.zeros(800)]).astype(int)
        strict = fit_feature(x, y, "f", BinningConfig(min_bin_fraction=0.1, min_bin_bads=5))
        permissive = fit_feature(x, y, "f", BinningConfig(min_bin_fraction=0.1, min_bin_bads=0))
        assert len(strict.woe) == 1
        assert len(permissive.woe) == 2

    def test_predictive_feature_has_higher_iv_than_noise(self):
        x, y = graded_feature()
        noise = RNG.uniform(0, 1, len(y))
        config = BinningConfig()
        signal_iv = fit_feature(x, y, "signal", config).iv
        noise_iv = fit_feature(noise, y, "noise", config).iv
        assert signal_iv > noise_iv
        assert signal_iv > 0.02

    def test_iv_is_non_negative(self):
        x, y = graded_feature()
        assert fit_feature(x, y, "f", BinningConfig()).iv >= 0


class TestMonotonicity:
    def test_bad_rate_is_monotonic_after_fitting(self):
        x = RNG.uniform(0, 1, 8000)
        wobble = np.where((x > 0.4) & (x < 0.6), 0.9, 0.1)
        y = RNG.binomial(1, wobble)
        binning = fit_feature(x, y, "f", BinningConfig(enforce_monotonic=True))
        rates = binning.table.loc[binning.table["bin"] != MISSING_LABEL, "bad_rate"].to_numpy()
        differences = np.diff(rates)
        assert np.all(differences >= 0) or np.all(differences <= 0)

    def test_disabling_monotonicity_keeps_more_bins(self):
        x = RNG.uniform(0, 1, 8000)
        wobble = np.where((x > 0.4) & (x < 0.6), 0.9, 0.1)
        y = RNG.binomial(1, wobble)
        strict = fit_feature(x, y, "f", BinningConfig(enforce_monotonic=True))
        loose = fit_feature(x, y, "f", BinningConfig(enforce_monotonic=False))
        assert len(loose.woe) > len(strict.woe)


class TestBinConstraints:
    def test_every_bin_meets_the_minimum_share(self):
        x, y = graded_feature(n=5000)
        config = BinningConfig(min_bin_fraction=0.05)
        binning = fit_feature(x, y, "f", config)
        counts = binning.table.loc[binning.table["bin"] != MISSING_LABEL, "count"]
        assert counts.min() >= 0.05 * len(x)

    def test_constant_feature_is_handled(self):
        x = np.full(500, 3.0)
        y = RNG.binomial(1, 0.1, 500)
        binning = fit_feature(x, y, "f", BinningConfig())
        assert binning.iv == 0.0
        assert np.all(binning.transform(x) == 0.0)

    def test_all_missing_feature_is_handled(self):
        x = np.full(500, np.nan)
        y = RNG.binomial(1, 0.1, 500)
        binning = fit_feature(x, y, "f", BinningConfig())
        assert binning.iv == 0.0
        assert not np.isnan(binning.transform(x)).any()


class TestMissingValues:
    def test_missing_values_get_their_own_bin(self):
        x, y = graded_feature(n=4000)
        x = x.copy()
        x[:400] = np.nan
        binning = fit_feature(x, y, "f", BinningConfig())
        assert MISSING_LABEL in binning.table["bin"].tolist()
        row = binning.table.loc[binning.table["bin"] == MISSING_LABEL].iloc[0]
        assert row["count"] == 400

    def test_missing_values_map_to_the_missing_woe(self):
        x, y = graded_feature(n=4000)
        x = x.copy()
        x[:400] = np.nan
        binning = fit_feature(x, y, "f", BinningConfig())
        transformed = binning.transform(x)
        assert np.allclose(transformed[:400], binning.missing_woe)
        assert not np.isnan(transformed).any()

    def test_missing_rate_influences_missing_woe(self):
        n = 4000
        x = RNG.uniform(0, 1, n)
        y = np.zeros(n, dtype=int)
        y[:200] = 1
        x[:200] = np.nan
        binning = fit_feature(x, y, "f", BinningConfig())
        assert binning.missing_woe > 0


class TestTransformer:
    def test_transform_produces_one_column_per_feature(self):
        frame = pd.DataFrame({"a": RNG.uniform(0, 1, 2000), "b": RNG.uniform(0, 1, 2000)})
        y = RNG.binomial(1, 0.1, 2000)
        transformer = WOETransformer().fit(frame, y, ["a", "b"])
        out = transformer.transform(frame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2000

    def test_iv_table_is_sorted_descending(self):
        x, y = graded_feature(n=4000)
        frame = pd.DataFrame({"signal": x, "noise": RNG.uniform(0, 1, len(y))})
        transformer = WOETransformer().fit(frame, y, ["signal", "noise"])
        table = transformer.iv_table()
        assert table["iv"].is_monotonic_decreasing
        assert table.iloc[0]["feature"] == "signal"

    def test_transform_on_unseen_data_stays_within_fitted_woe_values(self):
        x, y = graded_feature(n=4000)
        frame = pd.DataFrame({"a": x})
        transformer = WOETransformer().fit(frame, y, ["a"])
        unseen = pd.DataFrame({"a": np.array([-99.0, 0.5, 99.0])})
        values = transformer.transform(unseen)["a"].to_numpy()
        allowed = transformer.bins["a"].woe
        assert all(np.isclose(value, allowed, atol=1e-6).any() for value in values)
