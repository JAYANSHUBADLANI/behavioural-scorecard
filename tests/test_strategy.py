"""Tests for decision bands and off policy value estimation."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    AUTO_INCREASE,
    DECREASE_MONITOR,
    HOLD,
    MANUAL_REVIEW,
    StrategyConfig,
    assign_bands,
    band_profile,
    compare_at_depth,
    deteriorating,
    challenger_policy,
    champion_policy,
    compare_policies,
    policy_value,
    realised_value,
    treat_all,
    treat_none,
)

RNG = np.random.default_rng(5)
CONFIG = StrategyConfig(bootstrap_samples=0)


def population(n=4000):
    score = RNG.uniform(455, 790, n)
    return pd.DataFrame({
        "SK_ID_PREV": np.arange(n),
        "score": score,
        "conservative_net_benefit": RNG.uniform(-5_000, 60_000, n),
        "recent_max_dpd": RNG.choice([0, 0, 0, 45], n),
        "recent_max_utilisation": RNG.uniform(0, 1.2, n),
        "observed_revenue": RNG.uniform(0, 30_000, n),
        "baseline_exposure": RNG.uniform(10_000, 200_000, n),
        "target_soft_dpd": RNG.binomial(1, 0.05, n),
        "treated": RNG.binomial(1, 0.3, n),
        "propensity": RNG.uniform(0.15, 0.6, n),
    })


class TestBands:
    def test_every_account_lands_in_exactly_one_band(self):
        frame = population()
        bands = assign_bands(frame, CONFIG)
        assert len(bands) == len(frame)
        assert set(bands.unique()).issubset(
            {AUTO_INCREASE, MANUAL_REVIEW, HOLD, DECREASE_MONITOR}
        )

    def test_arrears_override_a_high_score_and_high_benefit(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [790, 60_000, 45]
        assert assign_bands(frame, CONFIG).iloc[0] == DECREASE_MONITOR

    def test_top_score_and_top_benefit_gets_auto_increase(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [790, 60_000, 0]
        assert assign_bands(frame, CONFIG).iloc[0] == AUTO_INCREASE

    def test_high_score_and_low_benefit_goes_to_review_not_increase(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [790, -5_000, 0]
        assert assign_bands(frame, CONFIG).iloc[0] == MANUAL_REVIEW

    def test_middling_score_holds(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [550, -5_000, 0]
        assert assign_bands(frame, CONFIG).iloc[0] == HOLD

    def test_bottom_decile_score_is_monitored_even_without_arrears(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [455, 60_000, 0]
        assert assign_bands(frame, CONFIG).iloc[0] == DECREASE_MONITOR

    def test_any_recent_arrears_triggers_monitoring(self):
        frame = population()
        frame.loc[0, ["score", "conservative_net_benefit", "recent_max_dpd"]] = [790, 60_000, 5]
        assert assign_bands(frame, CONFIG).iloc[0] == DECREASE_MONITOR

    def test_the_monitor_band_is_reachable_on_a_realistic_population(self):
        frame = population()
        bands = assign_bands(frame, CONFIG)
        assert (bands == DECREASE_MONITOR).sum() > 0

    def test_band_profile_covers_the_whole_population(self):
        frame = population()
        bands = assign_bands(frame, CONFIG)
        profile = band_profile(frame, bands, "target_soft_dpd")
        assert profile["accounts"].sum() == len(frame)
        assert abs(profile["share_pct"].sum() - 100) < 0.1


class TestPolicies:
    def test_champion_targets_only_the_auto_increase_band(self):
        frame = population()
        bands = assign_bands(frame, CONFIG)
        assert champion_policy(frame, CONFIG).sum() == (bands == AUTO_INCREASE).sum()

    def test_challenger_skips_arrears_and_maxed_accounts(self):
        frame = population()
        chosen = challenger_policy(frame, CONFIG).astype(bool)
        assert (frame.loc[chosen, "recent_max_dpd"] < CONFIG.deteriorating_dpd).all()
        assert not deteriorating(frame, CONFIG)[chosen].any()
        assert (frame.loc[chosen, "recent_max_utilisation"] <= CONFIG.challenger_utilisation_high).all()

    def test_challenger_skips_dormant_accounts(self):
        frame = population()
        chosen = challenger_policy(frame, CONFIG).astype(bool)
        assert (frame.loc[chosen, "recent_max_utilisation"] >= CONFIG.challenger_utilisation_low).all()

    def test_treat_all_and_treat_none_are_degenerate(self):
        frame = population()
        assert treat_all(frame, CONFIG).all()
        assert not treat_none(frame, CONFIG).any()


class TestRealisedValue:
    def test_value_is_revenue_less_realised_loss(self):
        frame = pd.DataFrame({
            "observed_revenue": [1_000.0],
            "baseline_exposure": [100_000.0],
            "target_soft_dpd": [1],
        })
        assert np.isclose(realised_value(frame, 0.75)[0], 1_000.0 - 0.75 * 100_000.0)

    def test_good_account_keeps_its_revenue(self):
        frame = pd.DataFrame({
            "observed_revenue": [1_000.0],
            "baseline_exposure": [100_000.0],
            "target_soft_dpd": [0],
        })
        assert np.isclose(realised_value(frame, 0.75)[0], 1_000.0)


class TestPolicyValue:
    def test_only_matching_accounts_contribute(self):
        value = np.array([10.0, 20.0, 30.0, 40.0])
        treated = np.array([1, 0, 1, 0])
        propensity = np.array([0.5, 0.5, 0.5, 0.5])
        recommendation = np.array([1, 1, 0, 0])
        result = policy_value(value, treated, propensity, recommendation)
        assert result["matched_accounts"] == 2

    def test_treat_none_recovers_the_untreated_average(self):
        n = 4000
        treated = RNG.binomial(1, 0.4, n)
        value = RNG.normal(100, 10, n)
        propensity = np.full(n, 0.4)
        result = policy_value(value, treated, propensity, np.zeros(n, dtype=int))
        assert abs(result["value_per_account"] - value[treated == 0].mean()) < 1.0

    def test_weighting_corrects_a_skewed_assignment(self):
        n = 20000
        quality = RNG.uniform(0, 1, n)
        propensity = np.clip(quality, 0.1, 0.9)
        treated = RNG.binomial(1, propensity)
        value = 100 * quality + RNG.normal(0, 1, n)
        weighted = policy_value(value, treated, propensity, np.ones(n, dtype=int))
        naive = value[treated == 1].mean()
        truth = value.mean()
        assert abs(weighted["value_per_account"] - truth) < abs(naive - truth)

    def test_effective_sample_size_is_never_above_matched_count(self):
        frame = population()
        value = realised_value(frame, 0.75)
        result = policy_value(
            value, frame["treated"].to_numpy(), frame["propensity"].to_numpy(),
            champion_policy(frame, CONFIG),
        )
        assert result["effective_sample_size"] <= result["matched_accounts"]

    def test_no_matching_accounts_returns_nan(self):
        value = np.array([1.0, 2.0])
        treated = np.array([1, 1])
        result = policy_value(value, treated, np.array([0.5, 0.5]), np.array([0, 0]))
        assert np.isnan(result["value_per_account"])


class TestComparison:
    def test_every_policy_is_scored_and_ranked(self):
        frame = population()
        table = compare_policies(
            frame,
            {"champion": champion_policy, "challenger": challenger_policy,
             "treat all": treat_all, "treat none": treat_none},
            CONFIG, loss_given_default=0.75, bootstrap=False,
        )
        assert len(table) == 4
        assert table["value_per_account"].is_monotonic_decreasing

    def test_bootstrap_adds_an_interval_around_the_estimate(self):
        frame = population()
        config = StrategyConfig(bootstrap_samples=20)
        table = compare_policies(
            frame, {"champion": champion_policy}, config, 0.75, bootstrap=True
        )
        row = table.iloc[0]
        assert row["ci_low"] <= row["value_per_account"] <= row["ci_high"]


class TestMatchedDepth:
    def test_every_policy_is_scored_at_every_depth(self):
        frame = population()
        table = compare_at_depth(frame, CONFIG, 0.75, depths=(0.10, 0.20))
        assert len(table) == 8
        assert set(table["depth"]) == {0.10, 0.20}

    def test_targeted_share_matches_the_requested_depth(self):
        frame = population()
        table = compare_at_depth(frame, CONFIG, 0.75, depths=(0.20,))
        assert np.allclose(table["treated_share_pct"], 20.0, atol=0.1)

    def test_deteriorating_accounts_rank_last(self):
        frame = population()
        from src.strategy import champion_ranking

        ranking = champion_ranking(frame, CONFIG)
        flagged = deteriorating(frame, CONFIG).to_numpy()
        assert (ranking[flagged] == -1.0).all()
        assert (ranking[~flagged] >= 0).all()
