"""Tests for discrimination, separation, stability and calibration measures."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    calibration_error,
    gini,
    ks_statistic,
    population_stability_index,
    score_bands,
    summarise_performance,
)

RNG = np.random.default_rng(3)


class TestGini:
    def test_perfect_ranking_gives_one(self):
        target = np.array([0] * 500 + [1] * 500)
        score = np.array([1.0] * 500 + [0.0] * 500)
        assert np.isclose(gini(target, score), 1.0)

    def test_random_ranking_is_near_zero(self):
        target = RNG.binomial(1, 0.1, 20000)
        score = RNG.uniform(0, 1, 20000)
        assert abs(gini(target, score)) < 0.05

    def test_reversed_score_flips_the_sign(self):
        target = RNG.binomial(1, 0.2, 5000)
        score = target + RNG.normal(0, 0.5, 5000)
        assert np.isclose(gini(target, score), -gini(target, -score))

    def test_single_class_returns_nan(self):
        assert np.isnan(gini(np.zeros(100), RNG.uniform(0, 1, 100)))


class TestKS:
    def test_perfect_separation_gives_one(self):
        target = np.array([0] * 500 + [1] * 500)
        score = np.array([1.0] * 500 + [0.0] * 500)
        assert np.isclose(ks_statistic(target, score), 1.0)

    def test_bounded_between_zero_and_one(self):
        target = RNG.binomial(1, 0.15, 5000)
        score = RNG.uniform(0, 1, 5000)
        value = ks_statistic(target, score)
        assert 0.0 <= value <= 1.0

    def test_single_class_returns_nan(self):
        assert np.isnan(ks_statistic(np.ones(100), RNG.uniform(0, 1, 100)))


class TestPSI:
    def test_identical_distributions_give_near_zero(self):
        sample = RNG.normal(0, 1, 20000)
        assert population_stability_index(sample, sample.copy()) < 1e-6

    def test_shifted_distribution_is_flagged(self):
        reference = RNG.normal(0, 1, 20000)
        shifted = RNG.normal(1.5, 1, 20000)
        assert population_stability_index(reference, shifted) > 0.25

    def test_larger_shift_gives_larger_psi(self):
        reference = RNG.normal(0, 1, 20000)
        small = population_stability_index(reference, RNG.normal(0.3, 1, 20000))
        large = population_stability_index(reference, RNG.normal(2.0, 1, 20000))
        assert large > small

    def test_empty_input_returns_nan(self):
        assert np.isnan(population_stability_index(np.array([]), np.array([1.0, 2.0])))

    def test_constant_reference_returns_zero(self):
        assert population_stability_index(np.ones(500), np.ones(500)) == 0.0


class TestBandsAndCalibration:
    def test_bands_are_ordered_by_score(self):
        target = RNG.binomial(1, 0.1, 5000)
        score = RNG.uniform(300, 800, 5000)
        table = score_bands(target, score, bands=10)
        assert table["min_score"].is_monotonic_increasing
        assert table["accounts"].sum() == 5000

    def test_bands_report_predicted_when_supplied(self):
        target = RNG.binomial(1, 0.1, 5000)
        predicted = RNG.uniform(0.01, 0.3, 5000)
        table = score_bands(target, -predicted, predicted, bands=5)
        assert "predicted_bad_rate" in table.columns

    def test_perfect_calibration_gives_near_zero_error(self):
        probability = RNG.uniform(0.01, 0.5, 40000)
        target = RNG.binomial(1, probability)
        assert calibration_error(target, probability) < 0.02

    def test_biased_prediction_raises_calibration_error(self):
        probability = RNG.uniform(0.01, 0.5, 40000)
        target = RNG.binomial(1, probability)
        assert calibration_error(target, probability * 0.5) > 0.05


class TestSummary:
    def test_summary_reports_every_headline_metric(self):
        target = RNG.binomial(1, 0.1, 5000)
        score = 600 - 100 * target + RNG.normal(0, 30, 5000)
        summary = summarise_performance(target, score, RNG.uniform(0.01, 0.3, 5000))
        assert set(summary) == {
            "accounts", "bads", "bad_rate_pct", "gini", "ks", "calibration_error_pct"
        }
        assert summary["accounts"] == 5000
        assert summary["gini"] > 0.5
