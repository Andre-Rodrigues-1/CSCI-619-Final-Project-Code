from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nyiso_experiments import FUEL_CATEGORIES, TARGET_COLUMNS


MODEL_COLUMNS = {
    "Oracle P-63": "actual",
    "Persistence t-1": "pred_persistence",
    "Soft Repair LP": "pred_soft_repair",
    "Residual-Aware LtO": "pred_residual_aware_lto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final submission time-series comparisons.")
    parser.add_argument("--processed-dir", default="data/processed/final_submission")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--split", default="test_2023_2024", choices=["test_2023_2024", "holdout_2025"])
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-date", default=None, help="Optional UTC/local date string inside the selected split.")
    return parser.parse_args()


def _fuel_key(fuel: str) -> str:
    return fuel.lower().replace(" ", "_")


def _add_persistence_columns(hourly: pd.DataFrame, system_hour: pd.DataFrame) -> pd.DataFrame:
    system = system_hour.sort_values("timestamp_utc").copy()
    for target_col in TARGET_COLUMNS:
        system[f"pred_persistence_{target_col}"] = system[target_col].shift(1)
    persistence_cols = ["timestamp_utc", *[f"pred_persistence_{col}" for col in TARGET_COLUMNS]]
    merged = hourly.merge(system[persistence_cols], on="timestamp_utc", how="left")
    merged["pred_persistence_total_generation_mw"] = merged[
        [f"pred_persistence_{col}" for col in TARGET_COLUMNS]
    ].sum(axis=1)
    return merged


def _select_window(df: pd.DataFrame, days: int, start_date: str | None) -> pd.DataFrame:
    ordered = df.sort_values("timestamp_utc").copy()
    if start_date:
        start = pd.to_datetime(start_date, utc=True)
        ordered = ordered[ordered["timestamp_utc"] >= start]
    if ordered.empty:
        raise ValueError("Selected time-series window is empty.")
    start_ts = ordered["timestamp_utc"].iloc[0]
    end_ts = start_ts + pd.Timedelta(days=days)
    return ordered[(ordered["timestamp_utc"] >= start_ts) & (ordered["timestamp_utc"] < end_ts)].copy()


def _column_for(prefix: str, target_col: str) -> str:
    if prefix == "actual":
        return f"actual_{target_col}"
    return f"{prefix}_{target_col}"


def plot_total_generation(window: pd.DataFrame, split: str, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window["timestamp_utc"], window["actual_total_generation_mw"], label="Oracle P-63", linewidth=2.2)
    ax.plot(window["timestamp_utc"], window["pred_persistence_total_generation_mw"], label="Persistence t-1", alpha=0.8)
    ax.plot(window["timestamp_utc"], window["pred_soft_repair_total_generation_mw"], label="Soft Repair LP", alpha=0.9)
    ax.plot(
        window["timestamp_utc"],
        window["pred_residual_aware_lto_total_generation_mw"],
        label="Residual-Aware LtO",
        alpha=0.9,
    )
    ax.set_title(f"Total Internal Generation: Oracle vs Models ({split})")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = figures_dir / f"final_timeseries_total_generation_{split}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_fuel_panels(window: pd.DataFrame, split: str, figures_dir: Path) -> Path:
    fuels = ["Dual Fuel", "Natural Gas", "Hydro", "Wind"]
    fig, axes = plt.subplots(len(fuels), 1, figsize=(12, 10), sharex=True)
    for ax, fuel in zip(axes, fuels):
        target_col = f"gen_{_fuel_key(fuel)}_mw"
        for label, prefix in MODEL_COLUMNS.items():
            col = _column_for(prefix, target_col)
            ax.plot(window["timestamp_utc"], window[col], label=label, linewidth=1.6 if prefix == "actual" else 1.1)
        ax.set_ylabel("MW")
        ax.set_title(fuel)
        ax.grid(alpha=0.25)
    axes[0].legend(ncols=4, loc="upper left")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Fuel-Level Dispatch Time Series ({split})", y=0.995)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = figures_dir / f"final_timeseries_fuel_panels_{split}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_error_series(df: pd.DataFrame, split: str, figures_dir: Path) -> Path:
    ordered = df.sort_values("timestamp_utc").copy()
    ordered["persistence_total_abs_error_mw"] = (
        ordered["pred_persistence_total_generation_mw"] - ordered["actual_total_generation_mw"]
    ).abs()
    ordered["persistence_overall_abs_error_mw"] = sum(
        (
            ordered[f"pred_persistence_{target_col}"] - ordered[f"actual_{target_col}"]
        ).abs()
        for target_col in TARGET_COLUMNS
    ) / len(TARGET_COLUMNS)
    for col in [
        "persistence_overall_abs_error_mw",
        "soft_repair_overall_abs_error_mw",
        "residual_aware_lto_overall_abs_error_mw",
    ]:
        ordered[f"{col}_rolling"] = ordered[col].rolling(168, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ordered["timestamp_utc"], ordered["persistence_overall_abs_error_mw_rolling"], label="Persistence t-1")
    ax.plot(ordered["timestamp_utc"], ordered["soft_repair_overall_abs_error_mw_rolling"], label="Soft Repair LP")
    ax.plot(
        ordered["timestamp_utc"],
        ordered["residual_aware_lto_overall_abs_error_mw_rolling"],
        label="Residual-Aware LtO",
    )
    ax.set_title(f"168-Hour Rolling Overall Fuel Error ({split})")
    ax.set_ylabel("Overall MAE (MW)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    path = figures_dir / f"final_timeseries_rolling_error_{split}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(processed_dir / "final_submission_hourly.csv")
    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    system_hour = pd.read_parquet(processed_dir / "nyiso_system_hour.parquet")
    system_hour["timestamp_utc"] = pd.to_datetime(system_hour["timestamp_utc"], utc=True)

    hourly = _add_persistence_columns(hourly, system_hour)
    split_df = hourly[hourly["split"] == args.split].copy()
    if split_df.empty:
        raise ValueError(f"No hourly rows found for split: {args.split}")
    window = _select_window(split_df, args.days, args.start_date)

    paths = [
        plot_total_generation(window, args.split, figures_dir),
        plot_fuel_panels(window, args.split, figures_dir),
        plot_error_series(split_df, args.split, figures_dir),
    ]
    print("Saved time-series figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
