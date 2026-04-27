# NYISO Final Submission Runner

This submission path is intentionally separate from the exploratory research runners. It evaluates the two final selected models plus two reviewer-facing baselines on the year-aligned NYISO split:

```text
train:      2020-2021
validation: 2022
test:       2023-2024
holdout:    2025
```

## Models

The final runner reports:

```text
persistence_previous_hour
direct_neural_residual_with_weather_with_lag
residual_soft_repair_lp_with_weather_p0p1
residual_aware_fenchel_young_with_weather
```

The two final models are:

```text
residual_soft_repair_lp_with_weather_p0p1
residual_aware_fenchel_young_with_weather
```

## Reviewer Commands

If processed parquet files are included:

```bash
python3 run_final_submission.py --skip-build
```

If raw data is included and the reviewer wants to rebuild:

```bash
python3 run_final_submission.py
```

For a quick smoke run on an existing processed directory:

```bash
python3 run_final_submission.py --skip-build --processed-dir data/processed/final_submission --max-eval-hours 24
```

## Outputs

The runner writes:

```text
data/processed/final_submission/final_submission_metrics.csv
data/processed/final_submission/final_submission_hourly.csv
data/processed/final_submission/final_submission_split_summary.csv
figures/final_submission_model_mae.png
figures/final_submission_fuel_mae.png
figures/final_submission_total_generation_error.png
```

`final_submission_hourly.csv` contains actual P-63 generation and hourly predicted generation for both final models, by fuel category.

## Data To Include

Preferred reliable submission package:

```text
nyiso_processing.py
nyiso_experiments.py
run_final_submission.py
requirements.txt
SUBMISSION_README.md
design_descision.md
Project Proposal.pdf
data/processed/final_submission/nyiso_system_hour.parquet
data/processed/final_submission/nyiso_zone_hour.parquet
data/processed/final_submission/nyiso_weather_zone_hour.parquet
data/processed/final_submission/nyiso_zonal_capacity.parquet
```

The final selected models are system-level, so the submission runner does not require `nyiso_interface_hour.parquet`.

If processed data is not included, the raw NYISO and weather files needed by `nyiso_processing.py` must be included instead.
