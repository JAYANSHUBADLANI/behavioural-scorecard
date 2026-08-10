"""Phase 2: behavioural features, WOE binning, scorecard fit and validation.

Two splits are used rather than one, because they answer different questions. The
development sample is split by contract so that no account appears in both training and
testing, which is the only honest way to measure discrimination on a stacked panel. The
most recent observation points are held out entirely as an out of time sample, which is
where the population shift shows up.
"""

import json
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .binning import BinningConfig, WOETransformer
from .config import Config
from .data_io import write_report
from .features import PANEL_VALUE_COLUMNS, build_feature_table, feature_columns
from .phase1 import prepare_panel
from .scorecard import ScalingConfig, Scorecard, select_features
from .validation import (
    feature_psi,
    population_stability_index,
    score_bands,
    summarise_performance,
)


def assemble(config: Config):
    """Join behavioural features onto the labelled population from Phase 1."""
    from .windows import build_panel

    frame = prepare_panel(config)
    panel = build_panel(frame, config.panel_months, PANEL_VALUE_COLUMNS)
    del frame

    lookbacks = config.section("features")["lookbacks"]
    features = build_feature_table(panel, config.window, lookbacks)

    population_path = config.path("processed_dir") / "labelled_population.parquet"
    if not population_path.exists():
        raise FileNotFoundError("run phase 1 first, labelled_population.parquet is missing")
    population = pd.read_parquet(population_path)

    columns = feature_columns(features)
    merged = population.merge(features, on=["SK_ID_PREV", "observation_month"], how="inner")
    if len(merged) != len(population):
        raise ValueError(
            f"feature join changed the row count from {len(population)} to {len(merged)}, "
            "which means eligibility diverged between target and feature construction"
        )

    merged[columns] = merged[columns].replace([np.inf, -np.inf], np.nan)
    return merged, columns


def split(config: Config, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Development split by contract, plus a recent out of time holdout."""
    validation = config.section("validation")
    split_config = config.section("split")
    cutoff = validation["out_of_time_from_month"]

    out_of_time = frame[frame["observation_month"] >= cutoff]
    development = frame[frame["observation_month"] < cutoff]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=split_config["test_size"],
        random_state=split_config["random_state"],
    )
    train_index, test_index = next(
        splitter.split(development, groups=development[split_config["group_column"]])
    )
    return {
        "train": development.iloc[train_index],
        "test": development.iloc[test_index],
        "out_of_time": out_of_time,
    }


def fit(config: Config, samples: Dict[str, pd.DataFrame], columns: List[str]):
    binning = config.section("binning")
    model = config.section("model")
    scaling = config.section("scaling")

    train = samples["train"]
    transformer = WOETransformer(BinningConfig(**binning)).fit(
        train, train["target"].to_numpy(), columns
    )
    woe_train = transformer.transform(train)
    selected = select_features(
        transformer, woe_train, model["min_iv"], model["max_correlation"]
    )
    if not selected:
        raise ValueError("no feature cleared the information value floor")

    card = Scorecard(ScalingConfig(**scaling), C=model["regularisation_C"]).fit(
        woe_train, train["target"].to_numpy(), selected
    )
    return transformer, card, selected


def evaluate(config: Config, transformer, card, samples: Dict[str, pd.DataFrame]):
    validation = config.section("validation")
    results: Dict[str, dict] = {}
    scores: Dict[str, np.ndarray] = {}
    bands: Dict[str, pd.DataFrame] = {}

    for name, sample in samples.items():
        woe = transformer.transform(sample)
        score = card.score(woe)
        predicted = card.predict_proba(woe)
        target = sample["target"].to_numpy()
        scores[name] = score
        results[name] = summarise_performance(target, score, predicted)
        results[name]["observation_months"] = [
            int(sample["observation_month"].min()), int(sample["observation_month"].max())
        ]
        bands[name] = score_bands(target, score, predicted, validation["score_bands"])

    reference = scores["train"]
    results["stability"] = {
        "score_psi_train_vs_test": round(
            population_stability_index(reference, scores["test"], validation["psi_bins"]), 4
        ),
        "score_psi_train_vs_out_of_time": round(
            population_stability_index(reference, scores["out_of_time"], validation["psi_bins"]), 4
        ),
        "bad_rate_ratio_train_over_out_of_time": round(
            float(samples["train"]["target"].mean() / max(samples["out_of_time"]["target"].mean(), 1e-9)), 2
        ),
    }

    recalibrated = recalibrate(card, transformer, samples)
    results["out_of_time_recalibrated"] = recalibrated
    return results, scores, bands


def segment_performance(transformer, card, samples: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Discrimination inside segments, to show where the headline Gini comes from.

    Two questions an interviewer should ask about a Gini above 0.75 on a behavioural
    scorecard. First, how much of it is just separating dormant accounts from active ones,
    since an account carrying no balance can barely go delinquent. Second, how much is
    arrears carryover, since an account already 60 days late is close to mechanically
    likely to reach 90. Both are answered by scoring the segments separately.
    """
    rows = []
    for name, sample in samples.items():
        woe = transformer.transform(sample)
        score = card.score(woe)
        frame = sample.assign(_score=score)
        segments = {
            "all scoreable": frame,
            "active, balance > 0": frame[frame["balance_at_observation"] > 0],
            "dormant, balance == 0": frame[frame["balance_at_observation"] == 0],
            "no arrears at observation": frame[frame["pre_max_dpd"] == 0],
            "in arrears at observation": frame[frame["pre_max_dpd"] > 0],
        }
        for segment, subset in segments.items():
            target = subset["target"].to_numpy()
            rows.append({
                "sample": name,
                "segment": segment,
                "rows": int(len(subset)),
                "bads": int(target.sum()),
                "bad_rate_pct": round(float(target.mean()) * 100, 4) if len(subset) else np.nan,
                **{k: v for k, v in summarise_performance(
                    target, subset["_score"].to_numpy()
                ).items() if k in ("gini", "ks")},
            })
    return pd.DataFrame(rows)


def recalibrate(card, transformer, samples: Dict[str, pd.DataFrame]) -> dict:
    """Refit the intercept only on the out of time sample.

    Discrimination and calibration fail for different reasons. Shifting the intercept to
    the out of time base rate leaves the ranking untouched, so what remains in the
    calibration error afterwards is genuine model drift rather than a base rate change.
    """
    sample = samples["out_of_time"]
    woe = transformer.transform(sample)
    log_odds = card.log_odds(woe)
    target = sample["target"].to_numpy()

    observed_rate = target.mean()
    predicted_rate = 1 / (1 + np.exp(-log_odds))
    shift = np.log(observed_rate / (1 - observed_rate)) - np.log(
        predicted_rate.mean() / (1 - predicted_rate.mean())
    )
    adjusted = 1 / (1 + np.exp(-(log_odds + shift)))

    from .validation import calibration_error, gini, ks_statistic

    return {
        "intercept_shift": round(float(shift), 4),
        "gini": round(gini(target, -log_odds), 4),
        "ks": round(ks_statistic(target, -log_odds), 4),
        "calibration_error_pct_before": round(
            calibration_error(target, predicted_rate) * 100, 4
        ),
        "calibration_error_pct_after": round(calibration_error(target, adjusted) * 100, 4),
    }


def make_segment_figure(config: Config, segments: pd.DataFrame) -> None:
    figures = config.path("reports_dir") / "figures"
    frame = segments[segments["sample"] == "out_of_time"].copy()
    order = ["all scoreable", "active, balance > 0", "dormant, balance == 0",
             "in arrears at observation", "no arrears at observation"]
    frame = frame.set_index("segment").loc[order].reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    colours = ["#1f4e79", "#7f9db9", "#c7c7c7", "#b03a2e", "#e0a899"]
    ax[0].barh(frame["segment"], frame["gini"], color=colours)
    ax[0].bar_label(ax[0].containers[0], fmt="%.3f", fontsize=8, padding=2)
    ax[0].set_xlabel("Gini")
    ax[0].set_xlim(0, 1.0)
    ax[0].set_title("Out of time Gini by segment")
    ax[0].grid(alpha=.3, axis="x")
    ax[0].invert_yaxis()

    ax[1].barh(frame["segment"], frame["bads"], color=colours)
    ax[1].bar_label(ax[1].containers[0], fmt="%d", fontsize=8, padding=2)
    ax[1].set_xlabel("bad accounts in segment")
    ax[1].set_title("Where the bads actually sit")
    ax[1].grid(alpha=.3, axis="x")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "segment_performance.png")
    plt.close(fig)


def make_figures(config: Config, transformer, card, samples, results, scores, bands) -> None:
    figures = config.path("reports_dir") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    palette = {"train": "#1f4e79", "test": "#7f9db9", "out_of_time": "#b03a2e"}

    iv = transformer.iv_table().head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#1f4e79" if f in card.features else "#c7c7c7" for f in iv["feature"]]
    ax.barh(iv["feature"], iv["iv"], color=colours)
    ax.set_xlabel("information value")
    ax.set_title("Feature information value, shaded bars retained in the card")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures / "information_value.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, score in scores.items():
        ax[0].hist(score, bins=60, histtype="step", density=True,
                   label=name.replace("_", " "), color=palette[name], lw=1.4)
    ax[0].set_xlabel("score")
    ax[0].set_ylabel("density")
    ax[0].set_title("Score distribution by sample")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    names = ["train", "test", "out_of_time"]
    ginis = [results[n]["gini"] for n in names]
    ax[1].bar([n.replace("_", " ") for n in names], ginis,
              color=[palette[n] for n in names])
    ax[1].bar_label(ax[1].containers[0], fmt="%.3f", fontsize=8)
    ax[1].set_ylabel("Gini")
    ax[1].set_title("Discrimination by sample")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "score_and_discrimination.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in names:
        table = bands[name]
        ax[0].plot(range(len(table)), table["actual_bad_rate"] * 100, marker="o",
                   label=name.replace("_", " "), color=palette[name])
    ax[0].set_xlabel("score band, worst to best")
    ax[0].set_ylabel("actual bad rate (%)")
    ax[0].set_yscale("log")
    ax[0].set_title("Bad rate by score band")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    table = bands["out_of_time"]
    ax[1].plot(range(len(table)), table["actual_bad_rate"] * 100, marker="o",
               label="actual", color="#1f4e79")
    ax[1].plot(range(len(table)), table["predicted_bad_rate"] * 100, marker="s",
               label="predicted", color="#b03a2e", ls="--")
    ax[1].set_xlabel("score band, worst to best")
    ax[1].set_ylabel("bad rate (%)")
    ax[1].set_title("Out of time calibration gap")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(figures / "calibration.png")
    plt.close(fig)


def run(config: Config, force: bool = False) -> dict:
    frame, columns = assemble(config)
    samples = split(config, frame)
    transformer, card, selected = fit(config, samples, columns)
    results, scores, bands = evaluate(config, transformer, card, samples)

    reports = config.path("reports_dir")
    processed = config.path("processed_dir")
    transformer.iv_table().to_csv(reports / "information_value.csv", index=False)
    transformer.bin_table().to_csv(reports / "woe_bins.csv", index=False)
    card.scorecard_table(transformer).to_csv(reports / "scorecard.csv", index=False)
    card.coefficient_table(transformer).to_csv(reports / "coefficients.csv", index=False)
    pd.concat(
        [b.assign(sample=name) for name, b in bands.items()], ignore_index=True
    ).to_csv(reports / "score_bands.csv", index=False)

    psi = feature_psi(samples["train"], samples["out_of_time"], columns,
                      config.section("validation")["psi_bins"])
    psi.to_csv(reports / "feature_psi.csv", index=False)

    segments = segment_performance(transformer, card, samples)
    segments.to_csv(reports / "segment_performance.csv", index=False)

    scored = frame[["SK_ID_PREV", "observation_month", "target", "treated"]].copy()
    woe_all = transformer.transform(frame)
    scored["score"] = card.score(woe_all)
    scored["predicted_bad_rate"] = card.predict_proba(woe_all)
    scored.to_parquet(processed / "scored_population.parquet", index=False)

    summary = {
        "features_built": len(columns),
        "features_retained": len(selected),
        "retained": selected,
        "samples": {
            name: {"rows": int(len(sample)), "contracts": int(sample["SK_ID_PREV"].nunique()),
                   "bads": int(sample["target"].sum())}
            for name, sample in samples.items()
        },
        "performance": results,
        "top_feature_psi": psi.head(5).to_dict("records"),
        "segments_out_of_time": segments[segments["sample"] == "out_of_time"].to_dict("records"),
    }
    write_report(summary, reports / "phase2_summary.json")
    make_figures(config, transformer, card, samples, results, scores, bands)
    make_segment_figure(config, segments)
    return summary
