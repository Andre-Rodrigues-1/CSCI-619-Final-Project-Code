from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyiso_experiments import (
    COMPACT_METRIC_COLUMNS,
    ExperimentConfig,
    FUEL_CATEGORIES,
    LAG1_BALANCE_COLUMN,
    LOSS_EXPERIMENT_BALANCE_COLUMN,
    TARGET_COLUMNS,
    _predict_residual_aware_cost_lp_with_costs,
    add_loss_experiment_balance_column,
    add_solver_balance_targets,
    compute_available_capacities,
    compute_empirical_ramp_limits,
    format_results_table,
    load_system_hour,
    main_project_split,
    make_aligned_test_frame,
    metrics,
    persistence_baseline,
    solve_soft_repair_dispatch_many,
    train_direct_model,
    train_residual_aware_cost_model,
)
from nyiso_processing import build_and_validate


FINAL_MODELS = [
    "residual_soft_repair_lp_with_weather_p0p1",
    "residual_aware_fenchel_young_with_weather",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean final NYISO submission experiment.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--processed-dir", default="data/processed/final_submission")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--direct-epochs", type=int, default=40)
    parser.add_argument("--structured-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--soft-balance-penalty", type=float, default=0.1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--max-eval-hours",
        type=int,
        default=0,
        help="Optional quick-run cap per evaluation split. Leave at 0 for the submission run.",
    )
    return parser.parse_args()


def _require_fixed_year_splits(splits: dict[str, pd.DataFrame]) -> None:
    sizes = {name: int(len(frame)) for name, frame in splits.items()}
    missing = [name for name, size in sizes.items() if size == 0]
    if missing:
        raise ValueError(
            "Final submission requires non-empty fixed-year splits. "
            f"Missing {missing}; sizes={sizes}. Use --start 2020-01-01 --end 2025-12-31."
        )


def _fuel_key(fuel: str) -> str:
    return fuel.lower().replace(" ", "_")


def _add_prefixed_generation_columns(out: pd.DataFrame, prefix: str, values: np.ndarray) -> None:
    for idx, fuel in enumerate(FUEL_CATEGORIES):
        out[f"{prefix}_{TARGET_COLUMNS[idx]}"] = values[:, idx]


def _hourly_predictions_frame(
    split_name: str,
    solve_df: pd.DataFrame,
    target: np.ndarray,
    soft_pred: np.ndarray,
    residual_aware_pred: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "split": split_name,
            "timestamp_utc": solve_df["timestamp_utc"].to_numpy(),
            "timestamp_local": solve_df["timestamp_local"].to_numpy(),
        }
    )
    _add_prefixed_generation_columns(out, "actual", target)
    _add_prefixed_generation_columns(out, "pred_soft_repair", soft_pred)
    _add_prefixed_generation_columns(out, "pred_residual_aware_lto", residual_aware_pred)
    out["actual_total_generation_mw"] = target.sum(axis=1)
    out["pred_soft_repair_total_generation_mw"] = soft_pred.sum(axis=1)
    out["pred_residual_aware_lto_total_generation_mw"] = residual_aware_pred.sum(axis=1)
    out["soft_repair_overall_abs_error_mw"] = np.mean(np.abs(soft_pred - target), axis=1)
    out["residual_aware_lto_overall_abs_error_mw"] = np.mean(np.abs(residual_aware_pred - target), axis=1)
    out["soft_repair_total_abs_error_mw"] = np.abs(soft_pred.sum(axis=1) - target.sum(axis=1))
    out["residual_aware_lto_total_abs_error_mw"] = np.abs(residual_aware_pred.sum(axis=1) - target.sum(axis=1))
    return out


def _evaluate_final_split(
    split_name: str,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    capacities: np.ndarray,
    ramp_limits,
    config: ExperimentConfig,
    soft_balance_penalty: float,
    max_eval_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_eval_hours > 0:
        eval_df = eval_df.sort_values("timestamp_utc").head(max_eval_hours).copy()
    history_df = history_df.sort_values("timestamp_utc").copy()
    eval_df = eval_df.sort_values("timestamp_utc").copy()

    rows: list[dict[str, object]] = []
    target = eval_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    persistence_pred = persistence_baseline(history_df, eval_df)
    rows.append({"split": split_name, "model": "persistence_previous_hour", **metrics(persistence_pred, target)})

    test_context = make_aligned_test_frame(history_df, eval_df, config.resolved_history_window)
    direct_pred, direct_target, direct_solve_df = train_direct_model(
        train_df,
        test_context,
        use_weather=True,
        config=config,
        use_residual=True,
        return_solve_frame=True,
    )
    rows.append(
        {
            "split": split_name,
            "model": "direct_neural_residual_with_weather_with_lag",
            **metrics(direct_pred, direct_target),
        }
    )

    soft_pred, soft_diag = solve_soft_repair_dispatch_many(
        direct_solve_df,
        direct_pred,
        capacities,
        config,
        soft_balance_penalty=soft_balance_penalty,
        ramp_limits=ramp_limits,
        load_column=LAG1_BALANCE_COLUMN,
    )
    rows.append(
        {
            "split": split_name,
            "model": "residual_soft_repair_lp_with_weather_p0p1",
            **metrics(soft_pred, direct_target, soft_diag),
        }
    )

    loss_config = ExperimentConfig(
        history_window_hours=config.history_window_hours,
        feature_mode=config.feature_mode,
        load_balance_mode="forecast_load",
        solver_load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        use_ramp_constraints=config.use_ramp_constraints,
        ramp_quantile=config.ramp_quantile,
        direct_epochs=config.direct_epochs,
        structured_epochs=config.structured_epochs,
        structured_lr=config.structured_lr,
        hidden_dim=config.hidden_dim,
        candidate_count=config.candidate_count,
        ens_penalty=config.ens_penalty,
        interface_penalty=config.interface_penalty,
        transfer_penalty=config.transfer_penalty,
        flow_match_penalty=config.flow_match_penalty,
        residual_penalty=config.residual_penalty,
        spill_penalty=config.spill_penalty,
        seed=config.seed,
        include_lags=True,
        soft_balance_penalty_grid=[soft_balance_penalty],
    )
    residual_model = train_residual_aware_cost_model(
        train_df,
        use_weather=True,
        capacities=capacities,
        config=loss_config,
        ramp_limits=ramp_limits,
        load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        soft_balance_penalty=soft_balance_penalty,
    )
    residual_aware_model, residual_history = residual_model
    residual_aware_pred, residual_aware_target, residual_aware_diag, _, residual_solve_df = (
        _predict_residual_aware_cost_lp_with_costs(
            residual_aware_model,
            test_context,
            use_weather=True,
            capacities=capacities,
            config=loss_config,
            ramp_limits=ramp_limits,
            load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            soft_balance_penalty=soft_balance_penalty,
        )
    )
    residual_aware_diag["structured_train_loss"] = residual_history[-1] if residual_history else np.nan
    rows.append(
        {
            "split": split_name,
            "model": "residual_aware_fenchel_young_with_weather",
            **metrics(residual_aware_pred, residual_aware_target, residual_aware_diag),
        }
    )

    hourly = _hourly_predictions_frame(
        split_name,
        residual_solve_df,
        residual_aware_target,
        soft_pred,
        residual_aware_pred,
    )
    return pd.DataFrame(rows), hourly


def _plot_model_mae(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = metrics_df[metrics_df["model"].isin(["persistence_previous_hour", *FINAL_MODELS])].copy()
    pivot = plot_df.pivot(index="model", columns="split", values="overall_mae")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=25)
    ax.set_ylabel("Overall MAE (MW)")
    ax.set_title("Final Model Overall MAE")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_submission_model_mae.png", dpi=180)
    plt.close()


def _plot_fuel_mae(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    final_test = metrics_df[(metrics_df["split"] == "test_2023_2024") & metrics_df["model"].isin(FINAL_MODELS)]
    rows = []
    for _, row in final_test.iterrows():
        for fuel in FUEL_CATEGORIES:
            rows.append({"model": row["model"], "fuel": fuel, "mae": row[f"mae_{_fuel_key(fuel)}"]})
    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="fuel", columns="model", values="mae")
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=30)
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Fuel-Specific MAE on 2023-2024 Test Split")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_submission_fuel_mae.png", dpi=180)
    plt.close()


def _plot_total_generation_error(hourly_df: pd.DataFrame, figures_dir: Path) -> None:
    test = hourly_df[hourly_df["split"] == "test_2023_2024"].copy()
    test["timestamp_utc"] = pd.to_datetime(test["timestamp_utc"], utc=True)
    test = test.sort_values("timestamp_utc")
    test["soft_rolling"] = test["soft_repair_total_abs_error_mw"].rolling(168, min_periods=1).mean()
    test["lto_rolling"] = test["residual_aware_lto_total_abs_error_mw"].rolling(168, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test["timestamp_utc"], test["soft_rolling"], label="soft repair p=0.1")
    ax.plot(test["timestamp_utc"], test["lto_rolling"], label="residual-aware LtO")
    ax.set_ylabel("168-hour rolling total abs error (MW)")
    ax.set_title("Total Generation Error on 2023-2024 Test Split")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "final_submission_total_generation_error.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_build:
        print(f"Skipping build; using processed tables in {processed_dir}")
    else:
        outputs, checks = build_and_validate(
            start=args.start,
            end=args.end,
            processed_dir=processed_dir,
            include_interface_hour=False,
        )
        print("Processed final-submission tables:")
        for name, frame in outputs.items():
            print(f"{name}\trows={frame.shape[0]}\tcols={frame.shape[1]}")
        print("\nKey checks:")
        for key in [
            "weather_zones",
            "weather_duplicate_zone_hours",
            "weather_remaining_ghi_sentinel",
            "system_duplicate_hours",
            "system_rows",
            "target_negative_values",
            "zone_hour_zones",
            "zone_hour_duplicate_zone_hours",
            "zone_system_forecast_load_max_abs_diff",
            "zonal_capacity_rows",
        ]:
            if key in checks:
                print(f"{key}\t{checks[key]}")

    df = load_system_hour(processed_dir / "nyiso_system_hour.parquet")
    splits = main_project_split(df)
    _require_fixed_year_splits(splits)
    train_df, val_df, test_df, holdout_df = add_solver_balance_targets(
        splits["train"],
        splits["train"],
        splits["validation"],
        splits["test"],
        splits["holdout"],
    )
    train_df, val_df, test_df, holdout_df = add_loss_experiment_balance_column(
        train_df,
        val_df,
        test_df,
        holdout_df,
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": int(len(frame)),
                "start_utc": frame["timestamp_utc"].min(),
                "end_utc": frame["timestamp_utc"].max(),
            }
            for name, frame in [
                ("train", train_df),
                ("validation", val_df),
                ("test", test_df),
                ("holdout", holdout_df),
            ]
        ]
    )
    split_summary.to_csv(processed_dir / "final_submission_split_summary.csv", index=False)

    config = ExperimentConfig(
        direct_epochs=args.direct_epochs,
        structured_epochs=args.structured_epochs,
        candidate_count=4,
        hidden_dim=args.hidden_dim,
        soft_balance_penalty_grid=[args.soft_balance_penalty],
    )
    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None

    test_metrics, test_hourly = _evaluate_final_split(
        "test_2023_2024",
        train_df,
        pd.concat([train_df, val_df], ignore_index=True).sort_values("timestamp_utc"),
        test_df,
        capacities=capacities,
        ramp_limits=ramp_limits,
        config=config,
        soft_balance_penalty=args.soft_balance_penalty,
        max_eval_hours=args.max_eval_hours,
    )
    holdout_metrics, holdout_hourly = _evaluate_final_split(
        "holdout_2025",
        train_df,
        pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("timestamp_utc"),
        holdout_df,
        capacities=capacities,
        ramp_limits=ramp_limits,
        config=config,
        soft_balance_penalty=args.soft_balance_penalty,
        max_eval_hours=args.max_eval_hours,
    )

    metrics_df = pd.concat([test_metrics, holdout_metrics], ignore_index=True)
    hourly_df = pd.concat([test_hourly, holdout_hourly], ignore_index=True)
    metrics_df.to_csv(processed_dir / "final_submission_metrics.csv", index=False)
    hourly_df.to_csv(processed_dir / "final_submission_hourly.csv", index=False)

    _plot_model_mae(metrics_df, figures_dir)
    _plot_fuel_mae(metrics_df, figures_dir)
    _plot_total_generation_error(hourly_df, figures_dir)

    print("\nSplit row counts:")
    print(format_results_table(split_summary, ["split", "rows", "start_utc", "end_utc"]))

    print("\nFinal submission metrics:")
    print(format_results_table(metrics_df, ["split", *COMPACT_METRIC_COLUMNS]))

    print("\nSaved outputs:")
    print(f"{processed_dir / 'final_submission_metrics.csv'}")
    print(f"{processed_dir / 'final_submission_hourly.csv'}")
    print(f"{figures_dir / 'final_submission_model_mae.png'}")
    print(f"{figures_dir / 'final_submission_fuel_mae.png'}")
    print(f"{figures_dir / 'final_submission_total_generation_error.png'}")


if __name__ == "__main__":
    main()
