"""Tests for behavioural feature construction, including outcome window leakage."""

import numpy as np
import pandas as pd

from src.config import WindowConfig
from src.features import build_features, build_feature_table, feature_columns
from src.windows import build_panel

MONTHS = list(range(-36, 0))
VALUE_COLUMNS = (
    "AMT_PAYMENT_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_ATM_CURRENT",
)


def make_contract(contract_id, months, dpd=0, limit=100_000.0, balance=0.0,
                  payment=1_000.0, minimum_due=1_000.0, drawings=500.0, atm=100.0):
    n = len(months)

    def column(value):
        return value if isinstance(value, list) else [value] * n

    return pd.DataFrame({
        "SK_ID_PREV": contract_id,
        "SK_ID_CURR": contract_id * 10,
        "MONTHS_BALANCE": months,
        "SK_DPD": column(dpd),
        "AMT_CREDIT_LIMIT_ACTUAL": column(limit),
        "AMT_BALANCE": column(balance),
        "AMT_PAYMENT_CURRENT": column(payment),
        "AMT_INST_MIN_REGULARITY": column(minimum_due),
        "AMT_DRAWINGS_CURRENT": column(drawings),
        "AMT_DRAWINGS_ATM_CURRENT": column(atm),
    })


CONFIG = WindowConfig(
    outcome_window_months=12,
    dpd_threshold=90,
    min_pre_history_months=12,
    observation_step_months=6,
    earliest_observation_month=-24,
    require_full_outcome_window=True,
    exclude_already_delinquent=True,
    panel_last_month=-1,
)


def panel_from(frames):
    return build_panel(pd.concat(frames, ignore_index=True), MONTHS, VALUE_COLUMNS)


class TestNoLeakage:
    def test_outcome_window_behaviour_does_not_change_features(self):
        clean = make_contract(1, MONTHS)
        dirty_dpd = [0] * len(MONTHS)
        for month in (-5, -4, -3):
            dirty_dpd[MONTHS.index(month)] = 200
        dirty_balance = [0.0] * len(MONTHS)
        for month in (-5, -4, -3):
            dirty_balance[MONTHS.index(month)] = 500_000.0
        dirty = make_contract(2, MONTHS, dpd=dirty_dpd, balance=dirty_balance)

        features = build_features(panel_from([clean, dirty]), -13, CONFIG)
        columns = feature_columns(features)
        first = features.loc[features["SK_ID_PREV"] == 1, columns].reset_index(drop=True)
        second = features.loc[features["SK_ID_PREV"] == 2, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)

    def test_pre_window_behaviour_does_change_features(self):
        clean = make_contract(1, MONTHS)
        dirty_dpd = [0] * len(MONTHS)
        dirty_dpd[MONTHS.index(-20)] = 45
        dirty = make_contract(2, MONTHS, dpd=dirty_dpd)
        features = build_features(panel_from([clean, dirty]), -13, CONFIG)
        values = dict(zip(features["SK_ID_PREV"], features["dpd_max_12m"]))
        assert values[1] == 0
        assert values[2] == 45


class TestFeatureValues:
    def test_utilisation_uses_balance_over_limit(self):
        contract = make_contract(1, MONTHS, balance=25_000.0, limit=100_000.0)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert np.isclose(features["util_at_observation"].iloc[0], 0.25)
        assert np.isclose(features["util_mean_12m"].iloc[0], 0.25)

    def test_rising_utilisation_gives_positive_slope(self):
        balances = [float(i) * 1_000 for i in range(len(MONTHS))]
        contract = make_contract(1, MONTHS, balance=balances)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert features["util_slope_12m"].iloc[0] > 0

    def test_falling_utilisation_gives_negative_slope(self):
        balances = [float(len(MONTHS) - i) * 1_000 for i in range(len(MONTHS))]
        contract = make_contract(1, MONTHS, balance=balances)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert features["util_slope_12m"].iloc[0] < 0

    def test_months_since_arrears_counts_back_from_observation(self):
        dpd = [0] * len(MONTHS)
        dpd[MONTHS.index(-16)] = 20
        contract = make_contract(1, MONTHS, dpd=dpd)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert features["months_since_arrears"].iloc[0] == 3

    def test_never_delinquent_gives_missing_recency(self):
        features = build_features(panel_from([make_contract(1, MONTHS)]), -13, CONFIG)
        assert np.isnan(features["months_since_arrears"].iloc[0])

    def test_atm_share_is_a_proportion_of_total_drawings(self):
        contract = make_contract(1, MONTHS, drawings=1_000.0, atm=250.0)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert np.isclose(features["atm_share_12m"].iloc[0], 0.25)

    def test_payment_ratio_is_payment_over_minimum_due(self):
        contract = make_contract(1, MONTHS, payment=3_000.0, minimum_due=1_000.0)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert np.isclose(features["payment_ratio_mean_12m"].iloc[0], 3.0)

    def test_overlimit_months_are_counted(self):
        balances = [50_000.0] * len(MONTHS)
        for month in (-16, -15):
            balances[MONTHS.index(month)] = 150_000.0
        contract = make_contract(1, MONTHS, balance=balances)
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert features["months_overlimit_12m"].iloc[0] == 2

    def test_tenure_counts_observed_months_up_to_observation(self):
        contract = make_contract(1, list(range(-30, 0)))
        features = build_features(panel_from([contract]), -13, CONFIG)
        assert features["tenure_months"].iloc[0] == 18

    def test_account_with_thin_history_is_not_scored(self):
        contract = make_contract(1, list(range(-20, 0)))
        assert build_features(panel_from([contract]), -13, CONFIG).empty


class TestFeatureTable:
    def test_stacked_table_has_one_row_per_eligible_observation(self):
        table = build_feature_table(panel_from([make_contract(1, MONTHS)]), CONFIG)
        assert sorted(table["observation_month"].unique()) == [-19, -13]

    def test_feature_columns_exclude_identifiers_and_targets(self):
        table = build_feature_table(panel_from([make_contract(1, MONTHS)]), CONFIG)
        columns = feature_columns(table)
        assert "SK_ID_PREV" not in columns
        assert "observation_month" not in columns
        assert "util_mean_12m" in columns

    def test_missing_source_columns_produce_null_features_not_errors(self):
        frame = pd.concat([make_contract(1, MONTHS)], ignore_index=True)
        panel = build_panel(frame, MONTHS, extra_values=())
        features = build_features(panel, -13, CONFIG)
        assert features["payment_ratio_mean_12m"].isna().all()
        assert features["util_mean_12m"].notna().all()


class TestLeakageGuard:
    def test_feature_columns_reads_the_feature_frame_not_a_blacklist(self):
        table = build_feature_table(panel_from([make_contract(1, MONTHS)]), CONFIG)
        columns = feature_columns(table)
        assert "SK_ID_PREV" not in columns
        assert "observation_month" not in columns
        assert set(columns) == set(table.columns) - {"SK_ID_PREV", "observation_month"}

    def test_outcome_columns_raise_rather_than_slip_through(self):
        import pytest

        polluted = pd.DataFrame({
            "SK_ID_PREV": [1], "observation_month": [-13],
            "util_mean_12m": [0.5], "outcome_mean_balance": [100.0],
        })
        with pytest.raises(ValueError, match="outcome_mean_balance"):
            feature_columns(polluted)

    def test_treatment_columns_raise_rather_than_slip_through(self):
        import pytest

        polluted = pd.DataFrame({
            "SK_ID_PREV": [1], "observation_month": [-13],
            "util_mean_12m": [0.5], "limit_increase_amount": [50_000.0],
        })
        with pytest.raises(ValueError, match="limit_increase_amount"):
            feature_columns(polluted)

    def test_feature_offset_moves_the_anchor_earlier(self):
        balances = [float(i) * 1_000 for i in range(len(MONTHS))]
        contract = make_contract(1, MONTHS, balance=balances)
        panel = panel_from([contract])
        late = build_features(panel, -13, CONFIG, feature_offset=0)
        early = build_features(panel, -13, CONFIG, feature_offset=12)
        assert early["util_at_observation"].iloc[0] < late["util_at_observation"].iloc[0]

    def test_offset_beyond_the_panel_returns_empty(self):
        panel = panel_from([make_contract(1, MONTHS)])
        assert build_features(panel, -13, CONFIG, feature_offset=99).empty
