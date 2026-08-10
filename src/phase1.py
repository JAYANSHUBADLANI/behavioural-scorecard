"""Phase 1: validate the raw panel, construct the behavioural target, report exclusions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_io import load_card_panel, profile, validate_and_split, write_report
from .config import Config, WindowConfig
from .windows import (
    add_secondary_targets,
    build_labelled_population,
    build_panel,
    label_at,
    limit_increase_matrix,
)

CARD_FILE = "credit_card_balance.csv"


def prepare_panel(config: Config, force: bool = False) -> pd.DataFrame:
    """Validate the raw card CSV and cache it as Parquet."""
    interim = config.path("interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    parquet = interim / "credit_card_balance.parquet"
    report_path = config.path("reports_dir") / "data_validation.json"

    if parquet.exists() and not force:
        return pd.read_parquet(parquet)

    source = config.path("raw_dir") / CARD_FILE
    if not source.exists():
        raise FileNotFoundError(f"raw card panel not found at {source}")

    validation = validate_and_split(source, interim / f"quarantine_{Path(CARD_FILE).stem}.csv")
    frame = load_card_panel(interim / f"clean_{Path(CARD_FILE).stem}.csv")
    frame.to_parquet(parquet, index=False)

    write_report(
        {"validation": validation.to_dict(), "profile": profile(frame, "credit_card_balance")},
        report_path,
    )
    (interim / f"clean_{Path(CARD_FILE).stem}.csv").unlink()
    return frame


def build_population(config: Config, frame: pd.DataFrame):
    from .features import PANEL_VALUE_COLUMNS

    panel = build_panel(frame, config.panel_months, PANEL_VALUE_COLUMNS)
    population, ledger = build_labelled_population(panel, config.window, config.treatment)
    secondary = config.section("secondary_targets")
    population = add_secondary_targets(
        population, secondary["overlimit_utilisation"], secondary["soft_dpd_threshold"]
    )
    return panel, population, ledger


def summarise(config: Config, panel, population: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    window = config.window
    treated = population["treated"]
    eligible_soft = population["already_soft_dpd_at_observation"] == 0
    eligible_over = population["already_overlimit_at_observation"] == 0

    def cell(mask_target: str, eligible: pd.Series) -> dict:
        sub = population.loc[eligible]
        t = sub["treated"]
        return {
            "rows": int(len(sub)),
            "bads": int(sub[mask_target].sum()),
            "bad_rate_pct": round(sub[mask_target].mean() * 100, 3),
            "treated_rows": int(t.sum()),
            "treated_bads": int(sub.loc[t, mask_target].sum()),
            "treated_bad_rate_pct": round(sub.loc[t, mask_target].mean() * 100, 3),
            "control_bad_rate_pct": round(sub.loc[~t, mask_target].mean() * 100, 3),
        }

    return {
        "window": {
            "outcome_window_months": window.outcome_window_months,
            "dpd_threshold": window.dpd_threshold,
            "min_pre_history_months": window.min_pre_history_months,
            "observation_step_months": window.observation_step_months,
            "observation_points": window.observation_months(),
            "n_observation_points": len(window.observation_months()),
        },
        "panel": {
            "contracts": int(len(panel.contract_ids)),
            "clients": int(pd.unique(panel.client_ids).size),
            "contract_months": int(panel.present.sum()),
            "months_covered": [int(panel.months.min()), int(panel.months.max())],
            "contracts_ever_at_threshold": int(
                (np.nan_to_num(np.nanmax(panel.dpd, axis=1), nan=0) >= window.dpd_threshold).sum()
            ),
        },
        "population": {
            "rows": int(len(population)),
            "unique_contracts": int(population["SK_ID_PREV"].nunique()),
            "rows_per_contract": round(len(population) / population["SK_ID_PREV"].nunique(), 2),
            "bads": int(population["target"].sum()),
            "bad_rate_pct": round(population["target"].mean() * 100, 3),
            "unique_bad_contracts": int(
                population.loc[population["target"] == 1, "SK_ID_PREV"].nunique()
            ),
            "treated_rows": int(treated.sum()),
            "treated_pct": round(treated.mean() * 100, 2),
        },
        "exclusions_total": {
            column: int(ledger[column].sum())
            for column in ledger.columns
            if column not in ("observation_month", "contracts_in_panel")
        },
        "risk_outcomes": {
            "primary_dpd90": cell("target", pd.Series(True, index=population.index)),
            "soft_dpd30": cell("target_soft_dpd", eligible_soft),
            "overlimit": cell("target_overlimit", eligible_over),
        },
    }


def window_sensitivity(config: Config, panel) -> pd.DataFrame:
    """Compare the labelled population under alternative window designs.

    This exists so the sample size claims in the README are reproducible rather than
    asserted. The single observation point rows are the design the project brief
    originally assumed, and they are the reason the stacked design was adopted.
    """
    base = config.window
    treatment = config.treatment
    increases = limit_increase_matrix(panel, treatment.min_increase_ratio)
    rows = []

    def variant(**overrides) -> WindowConfig:
        fields = {
            "outcome_window_months": base.outcome_window_months,
            "dpd_threshold": base.dpd_threshold,
            "min_pre_history_months": base.min_pre_history_months,
            "observation_step_months": base.observation_step_months,
            "earliest_observation_month": base.earliest_observation_month,
            "require_full_outcome_window": base.require_full_outcome_window,
            "exclude_already_delinquent": base.exclude_already_delinquent,
            "panel_last_month": base.panel_last_month,
        }
        fields.update(overrides)
        return WindowConfig(**fields)

    for obs, window_months, threshold in [
        (-7, 6, 90), (-7, 6, 30), (-13, 12, 90), (-25, 12, 90), (-25, 12, 30),
    ]:
        cfg = variant(outcome_window_months=window_months, dpd_threshold=threshold)
        cohort, _ = label_at(panel, obs, cfg, treatment, increases)
        rows.append({
            "design": "single observation point",
            "observation_points": f"{obs}",
            "n_points": 1,
            "outcome_window_months": window_months,
            "dpd_threshold": threshold,
            "rows": len(cohort),
            "unique_contracts": int(cohort["SK_ID_PREV"].nunique()),
            "bads": int(cohort["target"].sum()),
            "bad_rate_pct": round(cohort["target"].mean() * 100, 3) if len(cohort) else 0.0,
            "treated_rows": int(cohort["treated"].sum()),
            "treated_bads": int(cohort.loc[cohort["treated"], "target"].sum()),
        })

    for step, window_months, threshold in [
        (3, 6, 90), (6, 12, 90), (3, 12, 90), (3, 12, 30),
    ]:
        cfg = variant(observation_step_months=step, outcome_window_months=window_months,
                      dpd_threshold=threshold)
        population, _ = build_labelled_population(panel, cfg, treatment)
        points = cfg.observation_months()
        rows.append({
            "design": "stacked observation points",
            "observation_points": f"{points[0]} to {points[-1]} step {step}",
            "n_points": len(points),
            "outcome_window_months": window_months,
            "dpd_threshold": threshold,
            "rows": len(population),
            "unique_contracts": int(population["SK_ID_PREV"].nunique()),
            "bads": int(population["target"].sum()),
            "bad_rate_pct": round(population["target"].mean() * 100, 3),
            "treated_rows": int(population["treated"].sum()),
            "treated_bads": int(population.loc[population["treated"], "target"].sum()),
        })

    return pd.DataFrame(rows)


def make_figures(config: Config, panel, population: pd.DataFrame, ledger: pd.DataFrame) -> None:
    figures = config.path("reports_dir") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    by_month = population.groupby("observation_month")["target"].agg(["mean", "size"])
    ax[0].plot(by_month.index, by_month["mean"] * 100, marker="o", color="#1f4e79")
    ax[0].set_xlabel("observation month")
    ax[0].set_ylabel("bad rate (%)")
    ax[0].set_title("Bad rate by observation point")
    ax[0].grid(alpha=.3)
    ax[1].bar(by_month.index, by_month["size"], width=2, color="#7f9db9")
    ax[1].set_xlabel("observation month")
    ax[1].set_ylabel("eligible accounts")
    ax[1].set_title("Eligible population by observation point")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "population_by_observation_point.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    lengths = panel.present.sum(axis=1)
    ax[0].hist(lengths, bins=48, color="#1f4e79")
    ax[0].set_xlabel("months of panel history")
    ax[0].set_ylabel("contracts")
    ax[0].set_title("Panel length per contract")
    ax[0].grid(alpha=.3, axis="y")
    util = panel.utilisation[panel.present]
    util = util[np.isfinite(util)]
    ax[1].hist(np.clip(util, -0.2, 1.5), bins=60, color="#7f9db9")
    ax[1].axvline(1.0, color="#b03a2e", lw=1.2, ls="--")
    ax[1].set_xlabel("utilisation (clipped at 1.5)")
    ax[1].set_ylabel("contract months")
    ax[1].set_title("Utilisation distribution")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "panel_shape.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    reasons = [c for c in ledger.columns if c not in
               ("observation_month", "contracts_in_panel", "eligible", "bads", "treated")]
    totals = ledger[reasons + ["eligible"]].sum()
    labels = [r.replace("_", " ") for r in totals.index]
    ax.barh(labels, totals.to_numpy(), color=["#b03a2e"] * len(reasons) + ["#1f4e79"])
    ax.set_xlabel("account observation points")
    ax.set_title("Where the panel goes: exclusions vs eligible rows")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures / "exclusion_waterfall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    outcomes = {
        "90+ DPD": ("target", population),
        "30+ DPD": ("target_soft_dpd", population[population["already_soft_dpd_at_observation"] == 0]),
        "overlimit": ("target_overlimit", population[population["already_overlimit_at_observation"] == 0]),
    }
    names, treated_rate, control_rate = [], [], []
    for name, (column, frame) in outcomes.items():
        names.append(name)
        treated_rate.append(frame.loc[frame["treated"], column].mean() * 100)
        control_rate.append(frame.loc[~frame["treated"], column].mean() * 100)
    x = np.arange(len(names))
    bars_c = ax.bar(x - 0.18, control_rate, 0.36, label="no limit increase", color="#7f9db9")
    bars_t = ax.bar(x + 0.18, treated_rate, 0.36, label="limit increased", color="#b03a2e")
    for group in (bars_c, bars_t):
        ax.bar_label(group, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, names)
    ax.set_yscale("log")
    ax.set_ylabel("outcome rate (%, log scale)")
    ax.set_title("Raw treated vs control gaps run in both directions, so both are selection")
    ax.legend(loc="upper left")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "treatment_selection_bias.png")
    plt.close(fig)


def run(config: Config, force: bool = False) -> dict:
    frame = prepare_panel(config, force=force)
    panel, population, ledger = build_population(config, frame)

    processed = config.path("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    population.to_parquet(processed / "labelled_population.parquet", index=False)

    reports = config.path("reports_dir")
    ledger.to_csv(reports / "exclusion_ledger.csv", index=False)
    window_sensitivity(config, panel).to_csv(reports / "window_sensitivity.csv", index=False)
    summary = summarise(config, panel, population, ledger)
    write_report(summary, reports / "phase1_summary.json")
    make_figures(config, panel, population, ledger)
    return summary
