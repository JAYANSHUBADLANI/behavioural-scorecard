"""Phase 3: credit limit increase as an uplift problem, with Qini and exposure impact.

The risk arm does not run on the 90 plus delinquency target. Phase 1 measured only 65
treated accounts reaching 90 plus, which cannot support an incremental risk estimate at
any level of modelling effort. That number is reported rather than hidden, and the risk
arm runs on 30 plus delinquency and overlimit breach instead, both of which have enough
treated events to estimate.
"""

from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data_io import write_report
from .economics import (
    EconomicsConfig,
    exposure_at_default,
    incremental_exposure,
    incremental_expected_loss,
    net_benefit,
    revenue,
)
from .features import PANEL_VALUE_COLUMNS, build_feature_table, feature_columns
from .phase1 import prepare_panel
from .uplift import (
    UpliftConfig,
    common_support,
    cross_fitted_arms,
    cross_fitted_propensity,
    decile_effects,
    inverse_probability_ate,
    naive_ate,
    propensity_diagnostics,
    qini_coefficient,
    qini_curve,
)

RISK_OUTCOMES = {
    "dpd90": ("target", None),
    "dpd30": ("target_soft_dpd", "already_soft_dpd_at_observation"),
    "overlimit": ("target_overlimit", "already_overlimit_at_observation"),
}


def prepare(config: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Labelled population joined to covariates measured before the treatment window.

    The scorecard features from Phase 2 cannot be reused here. They are measured up to the
    observation point, which is the same window the limit increase happens in, so they
    encode the treatment: a doubled limit halves every utilisation feature by definition.
    Fitting a propensity model on them returns an AUC of 0.9999 and no overlap at all.
    Covariates are therefore re-measured as at the start of the treatment window.
    """
    from .windows import build_panel

    offset = config.section("uplift")["feature_offset_months"]
    panel_frame = prepare_panel(config)
    panel = build_panel(panel_frame, config.panel_months, PANEL_VALUE_COLUMNS)
    del panel_frame

    features = build_feature_table(
        panel, config.window, config.section("features")["lookbacks"], feature_offset=offset
    )
    population = pd.read_parquet(
        config.path("processed_dir") / "labelled_population.parquet"
    )
    columns = feature_columns(features)
    frame = population.merge(features, on=["SK_ID_PREV", "observation_month"], how="inner")

    economics = EconomicsConfig(**config.section("economics"))
    window_months = config.window.outcome_window_months

    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan)
    frame["observed_revenue"] = revenue(
        frame["outcome_mean_balance"].to_numpy(),
        frame["outcome_total_drawings"].to_numpy(),
        economics,
        window_months,
    )
    frame["baseline_exposure"] = exposure_at_default(
        frame["balance_at_observation"].to_numpy(),
        frame["limit_at_observation"].to_numpy(),
        economics,
    )
    return frame, columns


def treatment_power(frame: pd.DataFrame) -> pd.DataFrame:
    """Treated by outcome cell sizes, which decide what can be estimated at all."""
    rows = []
    for name, (column, already) in RISK_OUTCOMES.items():
        subset = frame if already is None else frame[frame[already] == 0]
        treated = subset["treated"]
        rows.append({
            "risk_outcome": name,
            "eligible_rows": int(len(subset)),
            "events": int(subset[column].sum()),
            "event_rate_pct": round(float(subset[column].mean()) * 100, 3),
            "treated_rows": int(treated.sum()),
            "treated_events": int(subset.loc[treated, column].sum()),
            "control_event_rate_pct": round(float(subset.loc[~treated, column].mean()) * 100, 3),
            "treated_event_rate_pct": round(float(subset.loc[treated, column].mean()) * 100, 3),
            "estimable": bool(subset.loc[treated, column].sum() >= 100),
        })
    return pd.DataFrame(rows)


def estimate_arm(
    frame: pd.DataFrame,
    features: List[str],
    outcome_column: str,
    binary: bool,
    uplift: UpliftConfig,
    propensity: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Cross fitted CATE for one outcome, plus naive and reweighted average effects."""
    matrix = frame[features].to_numpy(dtype="float64")
    treated = frame["treated"].to_numpy().astype(int)
    outcome = frame[outcome_column].to_numpy(dtype="float64")
    groups = frame["SK_ID_PREV"].to_numpy()

    baseline, treated_arm = cross_fitted_arms(
        matrix, treated, outcome, groups, uplift, binary=binary
    )
    cate = treated_arm - baseline
    summary = {
        "naive_ate": round(naive_ate(outcome, treated), 6),
        "ipw_ate": round(inverse_probability_ate(outcome, treated, propensity), 6),
        "mean_cate": round(float(cate.mean()), 6),
        "cate_p10": round(float(np.quantile(cate, 0.10)), 6),
        "cate_p90": round(float(np.quantile(cate, 0.90)), 6),
    }
    return cate, baseline, summary


def run(config: Config, force: bool = False) -> dict:
    uplift = UpliftConfig(**{
        k: v for k, v in config.section("uplift").items()
        if k in UpliftConfig.__dataclass_fields__
    })
    economics = EconomicsConfig(**config.section("economics"))
    reports = config.path("reports_dir")
    processed = config.path("processed_dir")

    frame, features = prepare(config)
    power = treatment_power(frame)
    power.to_csv(reports / "treatment_power.csv", index=False)

    risk_name = "dpd30"
    risk_column, risk_already = RISK_OUTCOMES[risk_name]
    sample = frame[frame[risk_already] == 0].reset_index(drop=True)

    matrix = sample[features].to_numpy(dtype="float64")
    treated = sample["treated"].to_numpy().astype(int)
    groups = sample["SK_ID_PREV"].to_numpy()
    propensity = cross_fitted_propensity(matrix, treated, groups, uplift)
    diagnostics = propensity_diagnostics(propensity, treated)

    support = common_support(propensity, treated, uplift.propensity_trim)
    diagnostics["rows_before_trim"] = int(len(sample))
    diagnostics["rows_after_trim"] = int(support.sum())
    diagnostics["treated_after_trim"] = int(treated[support].sum())

    sample = sample[support].reset_index(drop=True)
    propensity = propensity[support]

    revenue_cate, _, revenue_summary = estimate_arm(
        sample, features, "observed_revenue", False, uplift, propensity
    )
    risk_cate, risk_baseline, risk_summary = estimate_arm(
        sample, features, risk_column, True, uplift, propensity
    )
    balance_cate, _, balance_summary = estimate_arm(
        sample, features, "outcome_mean_balance", False, uplift, propensity
    )

    treated_increase = sample.loc[sample["treated"], "limit_increase_amount"]
    proposed_increase = float(treated_increase[treated_increase > 0].median())

    extra_exposure = incremental_exposure(
        np.full(len(sample), proposed_increase), balance_cate, economics
    )
    extra_loss = incremental_expected_loss(
        sample["baseline_exposure"].to_numpy(),
        extra_exposure,
        risk_baseline,
        risk_cate,
        economics,
    )
    benefit = net_benefit(revenue_cate, extra_loss)

    conservative_risk = np.clip(risk_cate, 0.0, None)
    conservative_loss = incremental_expected_loss(
        sample["baseline_exposure"].to_numpy(), extra_exposure,
        risk_baseline, conservative_risk, economics,
    )
    conservative_benefit = net_benefit(revenue_cate, conservative_loss)

    sample = sample.assign(
        propensity=propensity,
        cate_revenue=revenue_cate,
        cate_risk=risk_cate,
        cate_balance=balance_cate,
        baseline_risk=risk_baseline,
        incremental_exposure=extra_exposure,
        incremental_expected_loss=extra_loss,
        net_benefit=benefit,
        conservative_expected_loss=conservative_loss,
        conservative_net_benefit=conservative_benefit,
    )
    sample.to_parquet(processed / "uplift_population.parquet", index=False)

    curves = {
        "risk_by_risk_cate": qini_curve(
            sample[risk_column].to_numpy(), treated[support], risk_cate, uplift.qini_bins
        ),
        "revenue_by_net_benefit": qini_curve(
            sample["observed_revenue"].to_numpy(), treated[support], benefit, uplift.qini_bins
        ),
        "revenue_by_random": qini_curve(
            sample["observed_revenue"].to_numpy(), treated[support],
            np.random.default_rng(uplift.random_state).uniform(size=len(sample)),
            uplift.qini_bins,
        ),
    }
    for name, curve in curves.items():
        curve.to_csv(reports / f"qini_{name}.csv", index=False)

    deciles = decile_effects(
        sample[risk_column].to_numpy(), treated[support], benefit, groups=10
    )
    deciles.to_csv(reports / "net_benefit_deciles.csv", index=False)

    exposure_table = pd.concat([
        exposure_impact(sample, proposed_increase, "net_benefit",
                        "incremental_expected_loss").assign(scenario="estimated"),
        exposure_impact(sample, proposed_increase, "conservative_net_benefit",
                        "conservative_expected_loss").assign(scenario="conservative"),
    ], ignore_index=True)
    exposure_table.to_csv(reports / "exposure_impact.csv", index=False)

    heterogeneity = heterogeneity_check(deciles)

    summary = {
        "risk_outcome_used": risk_name,
        "risk_outcome_note": (
            "90 plus delinquency has only "
            f"{int(power.loc[power.risk_outcome == 'dpd90', 'treated_events'].iloc[0])} "
            "treated events, which cannot support an incremental risk estimate, so the "
            "risk arm runs on 30 plus delinquency"
        ),
        "treatment_power": power.to_dict("records"),
        "propensity": diagnostics,
        "proposed_limit_increase": round(proposed_increase, 2),
        "effects": {
            "revenue": revenue_summary,
            "risk": risk_summary,
            "balance": balance_summary,
        },
        "qini": {name: qini_coefficient(curve) for name, curve in curves.items()},
        "heterogeneity": heterogeneity,
        "exposure_impact": exposure_table.to_dict("records"),
        "assumptions": config.section("economics"),
    }
    write_report(summary, reports / "phase3_summary.json")
    make_figures(config, sample, curves, deciles, power, exposure_table)
    return summary


def heterogeneity_check(deciles: pd.DataFrame) -> Dict[str, float]:
    """Does the estimated ranking actually track the observed treated minus control gap.

    A uplift model earns its keep by separating accounts that respond from accounts that
    do not. If the estimated effect falls steadily across bands while the observed effect
    stays flat, the ranking is not capturing real heterogeneity and the targeting policy
    built on it is not supported by the data.
    """
    valid = deciles.dropna(subset=["observed_effect", "estimated_effect"])
    if len(valid) < 3:
        return {"bands": int(len(valid)), "rank_correlation": float("nan")}
    correlation = valid["estimated_effect"].corr(valid["observed_effect"], method="spearman")
    return {
        "bands": int(len(valid)),
        "rank_correlation": round(float(correlation), 4),
        "observed_effect_spread": round(
            float(valid["observed_effect"].max() - valid["observed_effect"].min()), 6
        ),
        "estimated_effect_spread": round(
            float(valid["estimated_effect"].max() - valid["estimated_effect"].min()), 2
        ),
    }


def exposure_impact(
    sample: pd.DataFrame, proposed_increase: float,
    benefit_column: str, loss_column: str,
) -> pd.DataFrame:
    """What targeting the top N percent by net benefit does to exposure and loss."""
    ranked = sample.sort_values(benefit_column, ascending=False).reset_index(drop=True)
    rows = []
    for share in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        cut = int(len(ranked) * share)
        top = ranked.iloc[:cut]
        rows.append({
            "targeted_share": share,
            "accounts": int(cut),
            "limit_granted": float(cut * proposed_increase),
            "incremental_revenue": float(top["cate_revenue"].sum()),
            "incremental_exposure": float(top["incremental_exposure"].sum()),
            "incremental_expected_loss": float(top[loss_column].sum()),
            "net_benefit": float(top[benefit_column].sum()),
            "revenue_per_unit_exposure": round(
                float(top["cate_revenue"].sum() / max(top["incremental_exposure"].sum(), 1)), 5
            ),
        })
    return pd.DataFrame(rows)


def make_figures(config, sample, curves, deciles, power, exposure_table) -> None:
    figures = config.path("reports_dir") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    treated_mask = sample["treated"]
    ax[0].hist(sample.loc[~treated_mask, "propensity"], bins=50, density=True,
               histtype="step", lw=1.4, color="#7f9db9", label="no limit increase")
    ax[0].hist(sample.loc[treated_mask, "propensity"], bins=50, density=True,
               histtype="step", lw=1.4, color="#b03a2e", label="limit increased")
    ax[0].set_xlabel("propensity score, out of fold")
    ax[0].set_ylabel("density")
    ax[0].set_title("Overlap after trimming to common support")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    curve = curves["revenue_by_net_benefit"]
    random_curve = curves["revenue_by_random"]
    ax[1].plot(curve["fraction_targeted"], curve["qini"] / 1e6, color="#1f4e79",
               marker="o", ms=3, label="ranked by net benefit")
    ax[1].plot(random_curve["fraction_targeted"], random_curve["random"] / 1e6,
               color="#999999", ls="--", label="random targeting")
    ax[1].set_xlabel("fraction of accounts targeted")
    ax[1].set_ylabel("cumulative incremental revenue (millions)")
    ax[1].set_title("Qini curve, revenue")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(figures / "uplift_overlap_and_qini.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(deciles))
    ax[0].bar(x, deciles["estimated_effect"] * 1e-3, color="#7f9db9",
              label="estimated net benefit (thousands)")
    ax[0].set_xlabel("net benefit band, best first")
    ax[0].set_ylabel("estimated net benefit (thousands)")
    twin = ax[0].twinx()
    twin.plot(x, deciles["observed_effect"] * 100, color="#b03a2e", marker="o", ms=4,
              label="observed risk effect")
    twin.set_ylabel("observed treated minus control (pp)", color="#b03a2e")
    twin.axhline(0, color="#b03a2e", lw=0.7, ls=":")
    ax[0].set_title("Ranking varies, observed effect does not")
    ax[0].grid(alpha=.3, axis="y")

    table = exposure_table[
        (exposure_table["targeted_share"] < 1.0)
        & (exposure_table["scenario"] == "conservative")
    ].reset_index(drop=True)
    width = 0.27
    positions = np.arange(len(table))
    ax[1].bar(positions - width, table["incremental_revenue"] / 1e9, width,
              label="incremental revenue", color="#1f4e79")
    ax[1].bar(positions, table["incremental_exposure"] / 1e9, width,
              label="incremental exposure", color="#7f9db9")
    ax[1].bar(positions + width, table["incremental_expected_loss"] / 1e9, width,
              label="incremental expected loss", color="#b03a2e")
    ax[1].set_xticks(positions, [f"{v:.0%}" for v in table["targeted_share"]])
    ax[1].set_xlabel("share of book targeted")
    ax[1].set_ylabel("billions")
    ax[1].set_title("Conservative scenario: exposure dwarfs revenue")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "net_benefit_and_exposure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = power["risk_outcome"]
    ax.bar(labels, power["treated_events"], color=["#b03a2e", "#1f4e79", "#1f4e79"])
    ax.bar_label(ax.containers[0], fmt="%d", fontsize=8)
    ax.axhline(100, color="#555555", ls="--", lw=1)
    ax.set_ylabel("treated accounts with the event")
    ax.set_yscale("log")
    ax.set_title("Treated event counts, dashed line is the minimum I would estimate on")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "treatment_power.png")
    plt.close(fig)
