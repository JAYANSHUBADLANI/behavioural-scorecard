"""Tests for points scaling, feature selection and the scorecard table."""

import numpy as np
import pandas as pd

from src.binning import WOETransformer
from src.scorecard import ScalingConfig, Scorecard, select_features

RNG = np.random.default_rng(11)


def fitted_card(n=6000):
    signal = RNG.uniform(0, 1, n)
    probability = 1 / (1 + np.exp(-(4 * (signal - 0.5)))) * 0.3
    y = RNG.binomial(1, probability)
    frame = pd.DataFrame({"signal": signal, "noise": RNG.uniform(0, 1, n)})
    transformer = WOETransformer().fit(frame, y, ["signal", "noise"])
    woe = transformer.transform(frame)
    card = Scorecard().fit(woe, y, ["signal", "noise"])
    return card, transformer, woe, y


class TestScaling:
    def test_factor_matches_points_to_double_the_odds(self):
        scaling = ScalingConfig(pdo=20.0)
        assert np.isclose(scaling.factor * np.log(2), 20.0)

    def test_base_score_lands_at_base_odds(self):
        scaling = ScalingConfig(base_score=600.0, base_odds=50.0, pdo=20.0)
        log_odds_bad = np.log(1 / 50.0)
        assert np.isclose(scaling.offset - scaling.factor * log_odds_bad, 600.0)

    def test_doubling_bad_odds_costs_exactly_pdo_points(self):
        scaling = ScalingConfig(pdo=20.0)
        base = scaling.offset - scaling.factor * np.log(0.02)
        doubled = scaling.offset - scaling.factor * np.log(0.04)
        assert np.isclose(base - doubled, 20.0)


class TestScorecard:
    def test_points_sum_to_the_total_score(self):
        card, _, woe, _ = fitted_card()
        points = card.points(woe)
        assert np.allclose(points["total_score"].to_numpy(), card.score(woe), atol=1e-6)

    def test_higher_score_means_lower_predicted_risk(self):
        card, _, woe, _ = fitted_card()
        score = card.score(woe)
        predicted = card.predict_proba(woe)
        ranks = pd.DataFrame({"score": score, "predicted": predicted}).rank()
        assert np.isclose(ranks.corr().loc["score", "predicted"], -1.0)

    def test_scorecard_table_covers_every_retained_feature(self):
        card, transformer, _, _ = fitted_card()
        table = card.scorecard_table(transformer)
        assert set(table["feature"].unique()) == set(card.features)
        assert table["points"].notna().all()

    def test_coefficient_table_carries_the_intercept(self):
        card, transformer, _, _ = fitted_card()
        table = card.coefficient_table(transformer)
        assert "intercept" in table.attrs
        assert len(table) == len(card.features)


class TestFeatureSelection:
    def test_low_iv_features_are_dropped(self):
        _, transformer, woe, _ = fitted_card()
        kept = select_features(transformer, woe, min_iv=0.02)
        assert "signal" in kept
        assert "noise" not in kept

    def test_correlated_duplicate_is_dropped(self):
        n = 6000
        signal = RNG.uniform(0, 1, n)
        probability = 1 / (1 + np.exp(-(4 * (signal - 0.5)))) * 0.3
        y = RNG.binomial(1, probability)
        frame = pd.DataFrame({"signal": signal, "copy": signal + RNG.normal(0, 1e-4, n)})
        transformer = WOETransformer().fit(frame, y, ["signal", "copy"])
        woe = transformer.transform(frame)
        kept = select_features(transformer, woe, min_iv=0.0, max_correlation=0.75)
        assert len(kept) == 1

    def test_empty_selection_returns_empty_list(self):
        _, transformer, woe, _ = fitted_card()
        assert select_features(transformer, woe, min_iv=99.0) == []
