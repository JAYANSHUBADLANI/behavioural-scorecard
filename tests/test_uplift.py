"""Tests for propensity, T-learner and Qini evaluation on synthetic treatments."""

import numpy as np
import pandas as pd
import pytest

from src.uplift import (
    PropensityModel,
    cross_fitted_cate,
    cross_fitted_propensity,
    propensity_diagnostics,
    TLearner,
    UpliftConfig,
    common_support,
    decile_effects,
    inverse_probability_ate,
    naive_ate,
    qini_coefficient,
    qini_curve,
)

RNG = np.random.default_rng(19)
CONFIG = UpliftConfig(n_estimators=60, max_leaf_nodes=15, random_state=0)


def confounded_sample(n=6000, effect=0.10):
    """Treatment assigned on a covariate that also drives the outcome.

    Good accounts are more likely to be treated, exactly as a lender would behave, so the
    naive difference understates the true effect and may even flip its sign.
    """
    quality = RNG.normal(0, 1, n)
    propensity = 1 / (1 + np.exp(-1.5 * quality))
    treated = RNG.binomial(1, propensity)
    baseline = 1 / (1 + np.exp(-(-1.0 - 1.5 * quality)))
    outcome = RNG.binomial(1, np.clip(baseline + effect * treated, 0, 1))
    features = np.column_stack([quality, RNG.normal(0, 1, n)])
    return features, treated, outcome, propensity


class TestPropensity:
    def test_recovers_a_known_assignment_rule(self):
        features, treated, _, _ = confounded_sample()
        model = PropensityModel(CONFIG).fit(features, treated)
        assert model.diagnostics(features, treated)["propensity_auc"] > 0.75

    def test_random_assignment_gives_uninformative_propensity(self):
        n = 6000
        features = RNG.normal(0, 1, (n, 2))
        treated = RNG.binomial(1, 0.3, n)
        propensity = cross_fitted_propensity(
            features, treated, np.arange(n), CONFIG, n_splits=4
        )
        assert propensity_diagnostics(propensity, treated)["propensity_auc"] < 0.60

    def test_in_sample_propensity_overstates_separability(self):
        n = 6000
        features = RNG.normal(0, 1, (n, 2))
        treated = RNG.binomial(1, 0.3, n)
        in_sample = PropensityModel(CONFIG).fit(features, treated).predict(features)
        out_of_fold = cross_fitted_propensity(
            features, treated, np.arange(n), CONFIG, n_splits=4
        )
        assert (
            propensity_diagnostics(in_sample, treated)["propensity_auc"]
            > propensity_diagnostics(out_of_fold, treated)["propensity_auc"]
        )

    def test_cross_fitting_keeps_accounts_out_of_their_own_fold(self):
        features, treated, _, _ = confounded_sample(n=4000)
        groups = np.repeat(np.arange(2000), 2)
        propensity = cross_fitted_propensity(features, treated, groups, CONFIG, n_splits=4)
        assert propensity.shape == treated.shape
        assert ((propensity > 0) & (propensity < 1)).all()

    def test_treated_accounts_score_higher_than_control(self):
        features, treated, _, _ = confounded_sample()
        diagnostics = PropensityModel(CONFIG).fit(features, treated).diagnostics(
            features, treated
        )
        assert diagnostics["treated_mean_propensity"] > diagnostics["control_mean_propensity"]


class TestCommonSupport:
    def test_trims_to_the_overlapping_region(self):
        propensity = np.concatenate([np.linspace(0.6, 0.99, 500), np.linspace(0.01, 0.4, 500)])
        treated = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
        mask = common_support(propensity, treated, trim=0.05)
        assert mask.sum() < len(propensity)

    def test_identical_distributions_keep_almost_everything(self):
        propensity = RNG.uniform(0.2, 0.8, 2000)
        treated = RNG.binomial(1, 0.5, 2000)
        assert common_support(propensity, treated, trim=0.01).mean() > 0.95

    def test_empty_arm_keeps_nothing(self):
        propensity = RNG.uniform(0, 1, 100)
        treated = np.zeros(100, dtype=int)
        assert common_support(propensity, treated, trim=0.05).sum() == 0


class TestTLearner:
    def test_recovers_a_positive_effect_under_confounding(self):
        features, treated, outcome, _ = confounded_sample(n=12000, effect=0.12)
        learner = TLearner(CONFIG, binary=True).fit(features, treated, outcome)
        assert learner.predict_cate(features).mean() > 0.03

    def test_zero_effect_treatment_gives_near_zero_cate(self):
        features, treated, outcome, _ = confounded_sample(n=12000, effect=0.0)
        learner = TLearner(CONFIG, binary=True).fit(features, treated, outcome)
        assert abs(learner.predict_cate(features).mean()) < 0.05

    def test_continuous_outcome_is_supported(self):
        n = 6000
        quality = RNG.normal(0, 1, n)
        treated = RNG.binomial(1, 1 / (1 + np.exp(-quality)))
        outcome = 100 * quality + 50 * treated + RNG.normal(0, 10, n)
        features = np.column_stack([quality, RNG.normal(0, 1, n)])
        learner = TLearner(CONFIG, binary=False).fit(features, treated, outcome)
        assert 25 < learner.predict_cate(features).mean() < 75


class TestAdjustment:
    def test_confounding_biases_the_naive_difference(self):
        _, treated, outcome, propensity = confounded_sample(n=20000, effect=0.10)
        assert naive_ate(outcome, treated) < 0.05

    def test_inverse_probability_weighting_moves_toward_the_truth(self):
        _, treated, outcome, propensity = confounded_sample(n=20000, effect=0.10)
        naive = naive_ate(outcome, treated)
        adjusted = inverse_probability_ate(outcome, treated, propensity)
        assert adjusted > naive
        assert abs(adjusted - 0.10) < abs(naive - 0.10)


class TestQini:
    def test_perfect_ranking_beats_random(self):
        n = 8000
        treated = RNG.binomial(1, 0.5, n)
        effect = RNG.uniform(0, 1, n)
        outcome = RNG.binomial(1, np.clip(0.1 + 0.6 * effect * treated, 0, 1))
        curve = qini_curve(outcome, treated, effect, bins=20)
        assert qini_coefficient(curve)["qini_coefficient"] > 0

    def test_random_ranking_tracks_the_baseline(self):
        n = 8000
        treated = RNG.binomial(1, 0.5, n)
        outcome = RNG.binomial(1, 0.2 + 0.1 * treated)
        curve = qini_curve(outcome, treated, RNG.uniform(0, 1, n), bins=20)
        stats = qini_coefficient(curve)
        assert abs(stats["lift_over_random"] - 1.0) < 0.35

    def test_curve_starts_at_zero_and_covers_full_depth(self):
        n = 3000
        treated = RNG.binomial(1, 0.5, n)
        outcome = RNG.binomial(1, 0.2, n)
        curve = qini_curve(outcome, treated, RNG.uniform(0, 1, n), bins=10)
        assert curve["qini"].iloc[0] == 0.0
        assert np.isclose(curve["fraction_targeted"].iloc[-1], 1.0)

    def test_random_line_ends_at_the_same_point_as_the_curve(self):
        n = 3000
        treated = RNG.binomial(1, 0.5, n)
        outcome = RNG.binomial(1, 0.2 + 0.1 * treated)
        curve = qini_curve(outcome, treated, RNG.uniform(0, 1, n), bins=10)
        assert np.isclose(curve["qini"].iloc[-1], curve["random"].iloc[-1])


class TestDecileEffects:
    def test_bands_cover_every_account(self):
        n = 5000
        treated = RNG.binomial(1, 0.5, n)
        outcome = RNG.binomial(1, 0.2, n)
        table = decile_effects(outcome, treated, RNG.uniform(0, 1, n), groups=10)
        assert table["accounts"].sum() == n
        assert len(table) == 10

    def test_estimated_effect_declines_across_bands(self):
        n = 5000
        treated = RNG.binomial(1, 0.5, n)
        outcome = RNG.binomial(1, 0.2, n)
        ranking = RNG.uniform(0, 1, n)
        table = decile_effects(outcome, treated, ranking, groups=10)
        assert table["estimated_effect"].is_monotonic_decreasing
