"""Tests for the revenue, exposure and expected loss arithmetic."""

import numpy as np
import pytest

from src.economics import (
    EconomicsConfig,
    exposure_at_default,
    expected_loss,
    incremental_exposure,
    incremental_expected_loss,
    net_benefit,
    revenue,
)

CONFIG = EconomicsConfig(
    annual_interest_margin=0.20,
    interchange_rate=0.02,
    credit_conversion_factor=0.50,
    loss_given_default=0.80,
)


class TestRevenue:
    def test_margin_applies_to_the_carried_balance(self):
        value = revenue(np.array([10_000.0]), np.array([0.0]), CONFIG, window_months=12)
        assert np.isclose(value[0], 2_000.0)

    def test_window_length_scales_the_margin(self):
        full = revenue(np.array([10_000.0]), np.array([0.0]), CONFIG, 12)
        half = revenue(np.array([10_000.0]), np.array([0.0]), CONFIG, 6)
        assert np.isclose(half[0], full[0] / 2)

    def test_interchange_applies_to_drawings(self):
        value = revenue(np.array([0.0]), np.array([50_000.0]), CONFIG, 12)
        assert np.isclose(value[0], 1_000.0)

    def test_nulls_are_treated_as_zero(self):
        value = revenue(np.array([np.nan]), np.array([np.nan]), CONFIG, 12)
        assert value[0] == 0.0


class TestExposure:
    def test_fully_drawn_account_has_exposure_equal_to_balance(self):
        value = exposure_at_default(np.array([50_000.0]), np.array([50_000.0]), CONFIG)
        assert np.isclose(value[0], 50_000.0)

    def test_undrawn_line_converts_at_the_stated_factor(self):
        value = exposure_at_default(np.array([20_000.0]), np.array([100_000.0]), CONFIG)
        assert np.isclose(value[0], 20_000.0 + 0.5 * 80_000.0)

    def test_overdrawn_account_does_not_get_negative_undrawn_credit(self):
        value = exposure_at_default(np.array([120_000.0]), np.array([100_000.0]), CONFIG)
        assert np.isclose(value[0], 120_000.0)


class TestIncrementalExposure:
    def test_unused_limit_increase_converts_at_the_factor(self):
        value = incremental_exposure(np.array([100_000.0]), np.array([0.0]), CONFIG)
        assert np.isclose(value[0], 50_000.0)

    def test_fully_drawn_increase_counts_in_full(self):
        value = incremental_exposure(np.array([100_000.0]), np.array([100_000.0]), CONFIG)
        assert np.isclose(value[0], 100_000.0)

    def test_partly_drawn_increase_splits_between_drawn_and_converted(self):
        value = incremental_exposure(np.array([100_000.0]), np.array([40_000.0]), CONFIG)
        assert np.isclose(value[0], 40_000.0 + 0.5 * 60_000.0)

    def test_no_increase_creates_no_exposure(self):
        assert incremental_exposure(np.array([0.0]), np.array([0.0]), CONFIG)[0] == 0.0

    def test_negative_increase_is_floored_at_zero(self):
        assert incremental_exposure(np.array([-50_000.0]), np.array([0.0]), CONFIG)[0] == 0.0


class TestExpectedLoss:
    def test_expected_loss_is_exposure_times_pd_times_lgd(self):
        value = expected_loss(np.array([100_000.0]), np.array([0.05]), CONFIG)
        assert np.isclose(value[0], 100_000.0 * 0.05 * 0.80)

    def test_incremental_loss_captures_both_exposure_and_probability(self):
        value = incremental_expected_loss(
            baseline_exposure=np.array([100_000.0]),
            incremental_exposure_amount=np.array([50_000.0]),
            baseline_probability=np.array([0.02]),
            incremental_probability=np.array([0.01]),
            config=CONFIG,
        )
        before = 100_000.0 * 0.02 * 0.80
        after = 150_000.0 * 0.03 * 0.80
        assert np.isclose(value[0], after - before)

    def test_extra_exposure_alone_still_costs_money(self):
        value = incremental_expected_loss(
            np.array([100_000.0]), np.array([50_000.0]),
            np.array([0.02]), np.array([0.0]), CONFIG,
        )
        assert value[0] > 0

    def test_no_change_gives_no_incremental_loss(self):
        value = incremental_expected_loss(
            np.array([100_000.0]), np.array([0.0]),
            np.array([0.02]), np.array([0.0]), CONFIG,
        )
        assert np.isclose(value[0], 0.0)

    def test_probability_cannot_exceed_one(self):
        value = incremental_expected_loss(
            np.array([100_000.0]), np.array([0.0]),
            np.array([0.9]), np.array([0.5]), CONFIG,
        )
        assert np.isclose(value[0], 100_000.0 * 0.10 * 0.80)


class TestNetBenefit:
    def test_net_benefit_is_revenue_minus_loss(self):
        assert np.isclose(net_benefit(np.array([500.0]), np.array([200.0]))[0], 300.0)

    def test_loss_heavy_account_is_negative(self):
        assert net_benefit(np.array([100.0]), np.array([900.0]))[0] < 0

    def test_nulls_are_treated_as_zero(self):
        assert np.isclose(net_benefit(np.array([np.nan]), np.array([100.0]))[0], -100.0)
