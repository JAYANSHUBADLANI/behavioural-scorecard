"""Phase 4: decision bands, champion against challenger, and the business summary.

The comparison here is deliberately not run on the uplift model's own predictions. Phase 3
found that the uplift ranking does not track observed treated minus control outcomes, so
scoring policies by their predicted benefit would just restate that model's opinion of
itself. Policies are scored by inverse probability weighted realised value instead, which
asks what actually happened to accounts whose real treatment matched each policy.
"""

from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data_io import write_report
from .strategy import (
    AUTO_INCREASE,
    BAND_ORDER,
    DECREASE_MONITOR,
    StrategyConfig,
    assign_bands,
    band_profile,
    challenger_policy,
    champion_policy,
    compare_at_depth,
    compare_policies,
    policy_value,
    random_policy,
    realised_value,
    scorecard_policy,
    treat_all,
    treat_none,
)

RISK_COLUMN = "target_soft_dpd"

POLICIES = {
    "champion, uplift bands": champion_policy,
    "challenger, utilisation rule": challenger_policy,
    "scorecard only": scorecard_policy,
    "random 30 percent": random_policy,
    "treat all": treat_all,
    "treat none": treat_none,
}


def load(config: Config) -> pd.DataFrame:
    processed = config.path("processed_dir")
    uplift_path = processed / "uplift_population.parquet"
    scored_path = processed / "scored_population.parquet"
    for path in (uplift_path, scored_path):
        if not path.exists():
            raise FileNotFoundError(f"run the earlier phases first, {path.name} is missing")

    uplift = pd.read_parquet(uplift_path)
    scored = pd.read_parquet(scored_path)[
        ["SK_ID_PREV", "observation_month", "score", "predicted_bad_rate"]
    ]
    return uplift.merge(scored, on=["SK_ID_PREV", "observation_month"], how="inner")


def exposure_by_band(
    frame: pd.DataFrame, bands: pd.Series, proposed_increase: float, config: Config
) -> pd.DataFrame:
    """Limit granted and exposure created by each band, which is the committee's view."""
    economics = config.section("economics")
    working = frame.assign(band=bands)
    rows = []
    for band in BAND_ORDER:
        subset = working[working["band"] == band]
        if subset.empty:
            continue
        granted = len(subset) * proposed_increase if band == AUTO_INCREASE else 0.0
        rows.append({
            "band": band,
            "accounts": int(len(subset)),
            "limit_granted": granted,
            "incremental_exposure": float(
                subset["incremental_exposure"].sum() if band == AUTO_INCREASE else 0.0
            ),
            "incremental_revenue": float(
                subset["cate_revenue"].sum() if band == AUTO_INCREASE else 0.0
            ),
            "conservative_expected_loss": float(
                subset["conservative_expected_loss"].sum() if band == AUTO_INCREASE else 0.0
            ),
            "existing_exposure": float(subset["baseline_exposure"].sum()),
        })
    return pd.DataFrame(rows)


def run(config: Config, force: bool = False) -> dict:
    strategy = StrategyConfig(**config.section("strategy"))
    economics = config.section("economics")
    reports = config.path("reports_dir")

    frame = load(config)
    bands = assign_bands(frame, strategy)
    profile = band_profile(frame, bands, RISK_COLUMN)
    profile.to_csv(reports / "decision_bands.csv", index=False)

    comparison = compare_policies(
        frame, POLICIES, strategy, economics["loss_given_default"], bootstrap=True
    )
    comparison.to_csv(reports / "policy_comparison.csv", index=False)

    matched = compare_at_depth(frame, strategy, economics["loss_given_default"])
    matched.to_csv(reports / "policy_comparison_matched_depth.csv", index=False)

    treated_increase = frame.loc[frame["treated"], "limit_increase_amount"]
    proposed_increase = float(treated_increase[treated_increase > 0].median())
    exposure = exposure_by_band(frame, bands, proposed_increase, config)
    exposure.to_csv(reports / "exposure_by_band.csv", index=False)

    frame.assign(decision_band=bands).to_parquet(
        config.path("processed_dir") / "strategy_population.parquet", index=False
    )

    champion = comparison[comparison["policy"] == "champion, uplift bands"].iloc[0]
    challenger = comparison[comparison["policy"] == "challenger, utilisation rule"].iloc[0]
    champion_wins = bool(champion["value_per_account"] > challenger["value_per_account"])
    overlapping = bool(
        champion["ci_low"] <= challenger["ci_high"]
        and challenger["ci_low"] <= champion["ci_high"]
    )

    at_twenty = matched[matched["depth"] == 0.20].sort_values(
        "value_per_account", ascending=False
    )
    summary = {
        "decision_bands": profile.to_dict("records"),
        "policy_comparison_matched_depth": matched.to_dict("records"),
        "best_policy_at_20_percent_depth": at_twenty.iloc[0]["policy"],
        "policy_comparison": comparison.to_dict("records"),
        "exposure_by_band": exposure.to_dict("records"),
        "proposed_limit_increase": round(proposed_increase, 2),
        "verdict": {
            "champion_value_per_account": float(champion["value_per_account"]),
            "challenger_value_per_account": float(challenger["value_per_account"]),
            "champion_beats_challenger": champion_wins,
            "confidence_intervals_overlap": overlapping,
            "conclusion": (
                "the uplift policy does not separate from the utilisation rule once the "
                "intervals are taken into account"
                if overlapping else
                ("the uplift policy beats the utilisation rule" if champion_wins else
                 "the utilisation rule beats the uplift policy")
            ),
        },
        "assumptions": economics,
    }
    write_report(summary, reports / "phase4_summary.json")
    make_figures(config, frame, bands, profile, comparison, exposure, matched)
    return summary


def make_figures(config, frame, bands, profile, comparison, exposure, matched) -> None:
    figures = config.path("reports_dir") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    colours = {AUTO_INCREASE: "#1f4e79", "manual review": "#7f9db9",
               "hold": "#c7c7c7", DECREASE_MONITOR: "#b03a2e"}

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bar_colours = [colours.get(b, "#999999") for b in profile["band"]]
    ax[0].bar(profile["band"], profile["share_pct"], color=bar_colours)
    ax[0].bar_label(ax[0].containers[0], fmt="%.1f%%", fontsize=8)
    ax[0].set_ylabel("share of book (%)")
    ax[0].set_title("Where the book falls across decision bands")
    ax[0].tick_params(axis="x", rotation=20)
    ax[0].grid(alpha=.3, axis="y")

    ax[1].bar(profile["band"], profile["observed_risk_rate_pct"], color=bar_colours)
    ax[1].bar_label(ax[1].containers[0], fmt="%.2f%%", fontsize=8)
    ax[1].set_ylabel("observed 30+ DPD rate (%)")
    ax[1].set_title("Observed risk by band, which is the sanity check")
    ax[1].tick_params(axis="x", rotation=20)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "decision_bands.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = comparison.sort_values("value_per_account")
    positions = np.arange(len(ordered))
    highlight = ["#1f4e79" if "champion" in p else "#b03a2e" if "challenger" in p
                 else "#c7c7c7" for p in ordered["policy"]]
    errors = np.vstack([
        ordered["value_per_account"] - ordered["ci_low"],
        ordered["ci_high"] - ordered["value_per_account"],
    ])
    ax.barh(positions, ordered["value_per_account"], xerr=errors, color=highlight,
            error_kw={"ecolor": "#444444", "capsize": 3, "lw": 1})
    ax.set_yticks(positions, ordered["policy"])
    ax.set_xlabel("realised value per account, inverse probability weighted")
    ax.set_title("Champion against challenger, scored on what actually happened")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures / "policy_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    styles = {"champion, uplift net benefit": ("#1f4e79", "o", "-"),
              "challenger, utilisation rule": ("#b03a2e", "s", "-"),
              "scorecard only": ("#7f9db9", "^", "--"),
              "random": ("#999999", "x", ":")}
    for name, group in matched.groupby("policy"):
        colour, marker, line = styles.get(name, ("#333333", "o", "-"))
        group = group.sort_values("depth")
        ax.plot(group["depth"] * 100, group["value_per_account"], color=colour,
                marker=marker, ls=line, label=name)
    ax.set_xlabel("share of book targeted (%)")
    ax.set_ylabel("realised value per account")
    ax.set_title("At equal depth, both simpler policies beat the uplift ranking")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(figures / "policy_comparison_matched_depth.png")
    plt.close(fig)
