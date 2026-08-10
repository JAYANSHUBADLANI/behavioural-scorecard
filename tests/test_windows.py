"""Tests for the observation point and outcome window logic on synthetic panels."""

import numpy as np
import pandas as pd
import pytest

from src.config import TreatmentConfig, WindowConfig
from src.windows import (
    add_secondary_targets,
    build_labelled_population,
    build_panel,
    label_at,
    limit_increase_matrix,
)

MONTHS = list(range(-24, 0))


def make_contract(contract_id, months, dpd=0, limit=100_000.0, balance=0.0, client_id=None):
    n = len(months)
    return pd.DataFrame(
        {
            "SK_ID_PREV": contract_id,
            "SK_ID_CURR": client_id if client_id is not None else contract_id * 10,
            "MONTHS_BALANCE": months,
            "SK_DPD": dpd if isinstance(dpd, list) else [dpd] * n,
            "AMT_CREDIT_LIMIT_ACTUAL": limit if isinstance(limit, list) else [limit] * n,
            "AMT_BALANCE": balance if isinstance(balance, list) else [balance] * n,
        }
    )


def window_config(**overrides):
    base = dict(
        outcome_window_months=6,
        dpd_threshold=90,
        min_pre_history_months=12,
        observation_step_months=3,
        earliest_observation_month=-12,
        require_full_outcome_window=True,
        exclude_already_delinquent=True,
        require_active_limit_at_observation=True,
        panel_last_month=-1,
    )
    base.update(overrides)
    return WindowConfig(**base)


TREATMENT = TreatmentConfig(lookback_months=12, min_increase_ratio=0.0)


def panel_from(frames):
    return build_panel(pd.concat(frames, ignore_index=True), MONTHS)


def cohort_for(frames, cfg=None, month=-7):
    cfg = cfg or window_config()
    panel = panel_from(frames)
    increases = limit_increase_matrix(panel, TREATMENT.min_increase_ratio)
    return label_at(panel, month, cfg, TREATMENT, increases)


class TestObservationMonths:
    def test_latest_point_leaves_a_full_window(self):
        assert window_config(outcome_window_months=6).latest_observation_month == -7
        assert window_config(outcome_window_months=12).latest_observation_month == -13

    def test_points_step_backwards_from_newest(self):
        assert window_config().observation_months() == [-7, -10]

    def test_window_length_is_config_driven(self):
        wide = window_config(outcome_window_months=12, earliest_observation_month=-19)
        assert wide.observation_months() == [-13, -16, -19]

    def test_impossible_window_is_rejected(self):
        with pytest.raises(ValueError):
            window_config(outcome_window_months=24, earliest_observation_month=-12).observation_months()

    def test_zero_step_is_rejected(self):
        with pytest.raises(ValueError):
            window_config(observation_step_months=0).observation_months()


class TestTargetDefinition:
    def test_delinquency_inside_the_window_is_bad(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-3)] = 120
        cohort, _ = cohort_for([make_contract(1, MONTHS, dpd=dpd)])
        assert cohort["target"].tolist() == [1]

    def test_delinquency_outside_the_window_is_not_bad(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-9)] = 120
        cohort, _ = cohort_for([make_contract(1, MONTHS, dpd=dpd)])
        assert cohort.empty or cohort["target"].tolist() == [0]

    def test_delinquency_below_threshold_is_not_bad(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-3)] = 45
        cohort, _ = cohort_for([make_contract(1, MONTHS, dpd=dpd)])
        assert cohort["target"].tolist() == [0]

    def test_shorter_window_can_flip_a_label_to_good(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-2)] = 120
        contract = make_contract(1, MONTHS, dpd=dpd)
        wide, _ = cohort_for([contract], month=-7)
        narrow, _ = cohort_for(
            [contract], cfg=window_config(outcome_window_months=3), month=-7
        )
        assert wide["target"].tolist() == [1]
        assert narrow["target"].tolist() == [0]

    def test_outcome_uses_only_the_accounts_own_rows(self):
        clean = [0] * 24
        dirty = [0] * 24
        dirty[MONTHS.index(-3)] = 200
        cohort, _ = cohort_for(
            [make_contract(1, MONTHS, dpd=clean), make_contract(2, MONTHS, dpd=dirty)]
        )
        labels = dict(zip(cohort["SK_ID_PREV"], cohort["target"]))
        assert labels == {1: 0, 2: 1}


class TestExclusions:
    def test_account_already_delinquent_is_excluded(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-20)] = 150
        cohort, exclusions = cohort_for([make_contract(1, MONTHS, dpd=dpd)])
        assert cohort.empty
        assert exclusions["already_delinquent_at_observation"] == 1

    def test_thin_pre_history_is_excluded(self):
        cohort, exclusions = cohort_for([make_contract(1, list(range(-10, 0)))])
        assert cohort.empty
        assert exclusions["insufficient_pre_history"] == 1

    def test_missing_observation_row_is_excluded(self):
        cohort, exclusions = cohort_for([make_contract(1, list(range(-24, -7)))])
        assert cohort.empty
        assert exclusions["not_observed_at_observation_point"] == 1

    def test_partial_outcome_window_is_excluded_when_full_required(self):
        months = [m for m in MONTHS if m not in (-1, -2)]
        cohort, exclusions = cohort_for([make_contract(1, months)])
        assert cohort.empty
        assert exclusions["insufficient_outcome_window"] == 1

    def test_partial_outcome_window_is_kept_when_not_required(self):
        months = [m for m in MONTHS if m not in (-1, -2)]
        cfg = window_config(require_full_outcome_window=False)
        cohort, exclusions = cohort_for([make_contract(1, months)], cfg=cfg)
        assert len(cohort) == 1
        assert exclusions["insufficient_outcome_window"] == 0

    def test_account_with_no_credit_limit_is_excluded(self):
        cohort, exclusions = cohort_for([make_contract(1, MONTHS, limit=0.0)])
        assert cohort.empty
        assert exclusions["no_active_credit_limit"] == 1

    def test_limit_rule_looks_at_the_observation_month_only(self):
        limits = [0.0] * 24
        for i in range(MONTHS.index(-12), 24):
            limits[i] = 50_000.0
        cohort, _ = cohort_for([make_contract(1, MONTHS, limit=limits)])
        assert len(cohort) == 1

    def test_limit_rule_can_be_disabled(self):
        cfg = window_config(require_active_limit_at_observation=False)
        cohort, exclusions = cohort_for([make_contract(1, MONTHS, limit=0.0)], cfg=cfg)
        assert len(cohort) == 1
        assert exclusions["no_active_credit_limit"] == 0

    def test_exclusions_partition_the_panel(self):
        frames = [
            make_contract(1, MONTHS),
            make_contract(2, list(range(-10, 0))),
            make_contract(3, list(range(-24, -7))),
        ]
        cohort, exclusions = cohort_for(frames)
        assert sum(exclusions.values()) + len(cohort) == 3

    def test_already_delinquent_exclusion_can_be_disabled(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-20)] = 150
        cfg = window_config(exclude_already_delinquent=False)
        cohort, exclusions = cohort_for([make_contract(1, MONTHS, dpd=dpd)], cfg=cfg)
        assert len(cohort) == 1
        assert exclusions["already_delinquent_at_observation"] == 0


class TestTreatment:
    def test_limit_increase_within_lookback_is_treated(self):
        limits = [100_000.0] * 24
        for i in range(MONTHS.index(-9), 24):
            limits[i] = 200_000.0
        cohort, _ = cohort_for([make_contract(1, MONTHS, limit=limits)])
        assert cohort["treated"].tolist() == [True]

    def test_limit_increase_after_observation_is_not_treated(self):
        limits = [100_000.0] * 24
        for i in range(MONTHS.index(-3), 24):
            limits[i] = 200_000.0
        cohort, _ = cohort_for([make_contract(1, MONTHS, limit=limits)])
        assert cohort["treated"].tolist() == [False]

    def test_limit_decrease_is_not_treatment(self):
        limits = [200_000.0] * 24
        for i in range(MONTHS.index(-9), 24):
            limits[i] = 100_000.0
        cohort, _ = cohort_for([make_contract(1, MONTHS, limit=limits)])
        assert cohort["treated"].tolist() == [False]

    def test_small_increase_filtered_by_minimum_ratio(self):
        limits = [100_000.0] * 24
        for i in range(MONTHS.index(-9), 24):
            limits[i] = 101_000.0
        panel = panel_from([make_contract(1, MONTHS, limit=limits)])
        assert limit_increase_matrix(panel, 0.0).sum() == 1
        assert limit_increase_matrix(panel, 0.5).sum() == 0


class TestStackedPopulation:
    def test_one_row_per_eligible_observation_point(self):
        population, ledger = build_labelled_population(
            panel_from([make_contract(1, MONTHS)]), window_config(), TREATMENT
        )
        assert sorted(population["observation_month"].tolist()) == [-10, -7]
        assert len(ledger) == 2

    def test_ledger_reconciles_to_the_panel(self):
        frames = [make_contract(i, MONTHS) for i in range(1, 4)]
        frames.append(make_contract(9, list(range(-10, 0))))
        panel = panel_from(frames)
        _, ledger = build_labelled_population(panel, window_config(), TREATMENT)
        reasons = list(ledger.columns.intersection(
            ["not_observed_at_observation_point", "insufficient_pre_history",
             "insufficient_outcome_window", "already_delinquent_at_observation"]
        ))
        totals = ledger[reasons].sum(axis=1) + ledger["eligible"]
        assert (totals == len(panel.contract_ids)).all()


class TestSecondaryTargets:
    def test_overlimit_and_soft_dpd_are_flagged(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-3)] = 45
        balances = [10_000.0] * 24
        balances[MONTHS.index(-2)] = 150_000.0
        cohort, _ = cohort_for([make_contract(1, MONTHS, dpd=dpd, balance=balances)])
        enriched = add_secondary_targets(cohort, overlimit_threshold=1.0, soft_dpd_threshold=30)
        assert enriched["target"].tolist() == [0]
        assert enriched["target_soft_dpd"].tolist() == [1]
        assert enriched["target_overlimit"].tolist() == [1]

    def test_already_in_state_flags_are_independent_of_primary_exclusion(self):
        dpd = [0] * 24
        dpd[MONTHS.index(-20)] = 45
        cohort, _ = cohort_for([make_contract(1, MONTHS, dpd=dpd)])
        enriched = add_secondary_targets(cohort, overlimit_threshold=1.0, soft_dpd_threshold=30)
        assert len(enriched) == 1
        assert enriched["already_soft_dpd_at_observation"].tolist() == [1]
