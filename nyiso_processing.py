from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import list_datatype_csvs

NY_TZ = "America/New_York"

ZONE_CODE_TO_NAME = {
    "A": "WEST",
    "B": "GENESE",
    "C": "CENTRL",
    "D": "NORTH",
    "E": "MHK VL",
    "F": "CAPITL",
    "G": "HUD VL",
    "H": "MILLWD",
    "I": "DUNWOD",
    "J": "N.Y.C.",
    "K": "LONGIL",
}

ZONE_NAME_TO_CODE = {
    "WEST": "A",
    "GENESE": "B",
    "GENESEE": "B",
    "CENTRL": "C",
    "CENTRAL": "C",
    "NORTH": "D",
    "MHK VL": "E",
    "MOHAWK VALLEY": "E",
    "CAPITL": "F",
    "CAPITAL": "F",
    "HUD VL": "G",
    "HUDSON VALLEY": "G",
    "MILLWD": "H",
    "MILLWOOD": "H",
    "DUNWOD": "I",
    "DUNWOODIE": "I",
    "N.Y.C.": "J",
    "NYC": "J",
    "N.Y.C": "J",
    "LONGIL": "K",
    "LONG ISLAND": "K",
}

P7_ZONE_COLUMNS = {
    "West": "A",
    "Genese": "B",
    "Centrl": "C",
    "North": "D",
    "Mhk Vl": "E",
    "Capitl": "F",
    "Hud Vl": "G",
    "Millwd": "H",
    "Dunwod": "I",
    "N.Y.C.": "J",
    "Longil": "K",
}

FUEL_CATEGORIES = [
    "Dual Fuel",
    "Natural Gas",
    "Hydro",
    "Nuclear",
    "Wind",
    "Other Renewables",
    "Other Fossil Fuels",
]

FUEL_TARGET_COLUMNS = {
    fuel: "gen_" + fuel.lower().replace(" ", "_") + "_mw" for fuel in FUEL_CATEGORIES
}

INTERFACE_SENTINEL_ABS_LIMIT = 9000.0

CAUSAL_LAG_BASE_COLUMNS = [
    "actual_load_mw",
    "rt_lbmp_mean",
    "rt_lbmp_max",
    "rt_congestion_mean",
    "rt_constraint_count",
    "rt_constraint_cost_mean",
    "rt_constraint_abs_cost_max",
    *FUEL_TARGET_COLUMNS.values(),
]

CAUSAL_LAGS = [1, 24]

AVAILABILITY_FUEL_COLUMNS = {
    fuel: fuel.lower().replace(" ", "_") + "_capacity_mw" for fuel in FUEL_CATEGORIES
}


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path = Path("data/raw")
    weather_dir: Path = Path("data/weather")
    processed_dir: Path = Path("data/processed")
    start: str | date | None = None
    end: str | date | None = None
    include_interface_hour: bool = True


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _date_from_file(path: Path) -> date | None:
    prefix = path.name[:8]
    try:
        return datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError:
        return None


def _filter_csvs(data_dir: Path, start: str | date | None, end: str | date | None) -> list[Path]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    files = list_datatype_csvs(data_dir)
    if start_date is None and end_date is None:
        return files

    selected: list[Path] = []
    for path in files:
        file_date = _date_from_file(path)
        if file_date is None:
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        selected.append(path)
    return selected


def _read_csvs(paths: list[Path], *, usecols: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _first_column(data: pd.DataFrame, column: str) -> pd.Series:
    values = data[column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _normalize_zone_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return ZONE_NAME_TO_CODE.get(text)


def _localize_nyiso_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    try:
        local = parsed.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        local = parsed.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return local


def _timestamp_frame(series: pd.Series, *, floor_hour: bool) -> pd.DataFrame:
    timestamp_local = _localize_nyiso_timestamp(series)
    if floor_hour:
        try:
            timestamp_local = timestamp_local.dt.floor("h", ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            timestamp_local = timestamp_local.dt.floor("h", ambiguous="NaT", nonexistent="shift_forward")
    timestamp_utc = timestamp_local.dt.tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp_local": timestamp_local,
            "timestamp_utc": timestamp_utc,
        }
    )


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_weather_zone_hour(
    weather_dir: str | Path = "data/weather",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    weather_dir = Path(weather_dir)
    frames = []
    for path in sorted(weather_dir.glob("*_weather.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No weather CSV files found in {weather_dir}")

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(
        columns={
            "datetime": "timestamp_utc",
            "zone": "zone_code",
            "ALLSKY_SFC_SW_DWN": "ghi_wh_m2",
            "WS50M": "wind_speed_50m_mps",
            "WD50M": "wind_dir_50m_deg",
            "T2M": "temp_2m_c",
        }
    )
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    data["zone_code"] = data["zone_code"].astype(str).str.strip().str.upper()
    data = data.drop(columns=[c for c in ["PRECTOT", "PRECTOTCORR"] if c in data.columns])

    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        data = data[data["timestamp_utc"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date + timedelta(days=1), tz="UTC")
        data = data[data["timestamp_utc"] < end_ts]

    data = data.sort_values(["zone_code", "timestamp_utc"]).reset_index(drop=True)
    data.loc[data["ghi_wh_m2"] == -999, "ghi_wh_m2"] = np.nan

    def clean_zone(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group = group.set_index("timestamp_utc")
        group["ghi_wh_m2"] = group["ghi_wh_m2"].interpolate(method="time", limit_direction="both")
        hour_of_year = group.index.dayofyear * 24 + group.index.hour
        median_by_hoy = group.groupby(hour_of_year)["ghi_wh_m2"].transform("median")
        group["ghi_wh_m2"] = group["ghi_wh_m2"].fillna(median_by_hoy).fillna(0.0)
        group["ghi_rolling_3h"] = group["ghi_wh_m2"].rolling(3, min_periods=1).mean()
        group["temp_rolling_24h"] = group["temp_2m_c"].rolling(24, min_periods=1).mean()
        return group.reset_index()

    data = data.groupby("zone_code", group_keys=False).apply(clean_zone)
    radians = np.deg2rad(data["wind_dir_50m_deg"].astype(float))
    data["wind_dir_sin"] = np.sin(radians)
    data["wind_dir_cos"] = np.cos(radians)
    data["wind_speed_cubed"] = data["wind_speed_50m_mps"].astype(float) ** 3
    data["zone_name"] = data["zone_code"].map(ZONE_CODE_TO_NAME)

    columns = [
        "timestamp_utc",
        "zone_code",
        "zone_name",
        "ghi_wh_m2",
        "wind_speed_50m_mps",
        "wind_dir_50m_deg",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_speed_cubed",
        "temp_2m_c",
        "ghi_rolling_3h",
        "temp_rolling_24h",
    ]
    return data[columns].sort_values(["timestamp_utc", "zone_code"]).reset_index(drop=True)


def load_static_capacity(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / "existing_generating_facilities.csv"
    facilities = pd.read_csv(path, encoding="utf-8-sig")
    facilities["capacity_mw"] = pd.to_numeric(
        facilities["name_plate_rating_MW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    facilities["zone_code"] = facilities["zone"].astype(str).str.strip().str.upper()
    facilities["fuel_category"] = facilities.apply(_facility_fuel_category, axis=1)
    grouped = (
        facilities[facilities["zone_code"].isin(ZONE_CODE_TO_NAME)]
        .groupby(["zone_code", "fuel_category"], as_index=False)["capacity_mw"]
        .sum()
    )
    grouped["zone_name"] = grouped["zone_code"].map(ZONE_CODE_TO_NAME)
    return grouped[["zone_code", "zone_name", "fuel_category", "capacity_mw"]]


def build_zonal_capacity_table(capacity: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [sorted(ZONE_CODE_TO_NAME), FUEL_CATEGORIES],
        names=["zone_code", "fuel_category"],
    )
    complete = (
        capacity.set_index(["zone_code", "fuel_category"])[["capacity_mw"]]
        .reindex(index)
        .fillna(0.0)
        .reset_index()
    )
    complete["zone_name"] = complete["zone_code"].map(ZONE_CODE_TO_NAME)
    total_by_fuel = complete.groupby("fuel_category")["capacity_mw"].transform("sum")
    complete["capacity_share_by_fuel"] = np.divide(
        complete["capacity_mw"],
        total_by_fuel,
        out=np.zeros(len(complete), dtype=np.float64),
        where=total_by_fuel.to_numpy(dtype=np.float64) > 0,
    )
    return complete[["zone_code", "zone_name", "fuel_category", "capacity_mw", "capacity_share_by_fuel"]]


def _normalized_wind_power_curve(wind_speed_mps: pd.Series) -> pd.Series:
    speed = pd.to_numeric(wind_speed_mps, errors="coerce").fillna(0.0).clip(lower=0.0)
    cut_in = 3.0
    rated = 12.0
    cut_out = 25.0
    cubic = ((speed**3 - cut_in**3) / (rated**3 - cut_in**3)).clip(lower=0.0, upper=1.0)
    return cubic.where(speed < rated, 1.0).where(speed <= cut_out, 0.0)


def add_weather_availability_features(zone_hour: pd.DataFrame, zonal_capacity: pd.DataFrame) -> pd.DataFrame:
    zone_hour = zone_hour.copy()
    capacity_wide = (
        zonal_capacity.pivot(index="zone_code", columns="fuel_category", values="capacity_mw")
        .reindex(sorted(ZONE_CODE_TO_NAME))
        .fillna(0.0)
        .rename(columns=AVAILABILITY_FUEL_COLUMNS)
        .reset_index()
    )
    zone_hour = zone_hour.merge(capacity_wide, on="zone_code", how="left")
    capacity_cols = [col for col in AVAILABILITY_FUEL_COLUMNS.values() if col in zone_hour.columns]
    zone_hour[capacity_cols] = zone_hour[capacity_cols].fillna(0.0)

    wind_cf = _normalized_wind_power_curve(zone_hour["wind_speed_50m_mps"])
    ghi_cf = pd.to_numeric(zone_hour["ghi_wh_m2"], errors="coerce").fillna(0.0).clip(lower=0.0).div(900.0).clip(upper=1.0)
    other_renewable_cap = zone_hour.get("other_renewables_capacity_mw", pd.Series(0.0, index=zone_hour.index))
    hydro_cap = zone_hour.get("hydro_capacity_mw", pd.Series(0.0, index=zone_hour.index))
    nuclear_cap = zone_hour.get("nuclear_capacity_mw", pd.Series(0.0, index=zone_hour.index))

    zone_hour["wind_available_mw"] = zone_hour.get("wind_capacity_mw", 0.0) * wind_cf
    zone_hour["solar_available_mw"] = other_renewable_cap * ghi_cf
    zone_hour["other_renewables_available_mw"] = (other_renewable_cap * (0.20 + 0.80 * ghi_cf)).clip(
        lower=0.0,
        upper=other_renewable_cap,
    )
    zone_hour["hydro_available_mw"] = hydro_cap
    zone_hour["nuclear_available_mw"] = nuclear_cap
    return zone_hour


def _facility_fuel_category(row: pd.Series) -> str:
    fuel = str(row.get("fuel_type_1", "")).strip().upper()
    dual = str(row.get("Dual", "")).strip().upper()
    if dual == "YES":
        return "Dual Fuel"
    if fuel == "NG":
        return "Natural Gas"
    if fuel == "WAT":
        return "Hydro"
    if fuel == "UR":
        return "Nuclear"
    if fuel in {"LBW", "OSW"}:
        return "Wind"
    if fuel in {"SUN", "MTE", "REF", "WD", "BAT", "FW"}:
        return "Other Renewables"
    return "Other Fossil Fuels"


def load_actual_load_zone_hour(
    raw_dir: str | Path = "data/raw",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(Path(raw_dir) / "P-58C Integrated Real Time Load", start, end)
    data = _read_csvs(paths, usecols=["Time Stamp", "Name", "PTID", "Integrated Load"])
    ts = _timestamp_frame(data["Time Stamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1)
    data["zone_code"] = data["Name"].map(_normalize_zone_name)
    data = data.dropna(subset=["timestamp_utc", "zone_code"])
    data["actual_load_mw"] = pd.to_numeric(data["Integrated Load"], errors="coerce")
    data["zone_name"] = data["zone_code"].map(ZONE_CODE_TO_NAME)
    return (
        data.groupby(["timestamp_utc", "timestamp_local", "zone_code", "zone_name"], as_index=False)[
            "actual_load_mw"
        ]
        .mean()
        .sort_values(["timestamp_utc", "zone_code"])
    )


def load_forecast_load_zone_hour(
    raw_dir: str | Path = "data/raw",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    start_date = _parse_date(start)
    file_start = start_date - timedelta(days=1) if start_date is not None else None
    paths = _filter_csvs(Path(raw_dir) / "P-7 ISO Load Forecast", file_start, end)
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        source_date = _date_from_file(path)
        frame["_forecast_issue_date"] = pd.Timestamp(source_date) if source_date else pd.NaT
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone_code", "forecast_load_mw", "zone_name"])
    ts = _timestamp_frame(data["Time Stamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1)
    target_local_date = data["timestamp_local"].dt.date
    issue_date = data["_forecast_issue_date"].dt.date
    day_ahead_mask = target_local_date == (issue_date + timedelta(days=1))
    data = data[day_ahead_mask].copy()
    if start_date is not None:
        data = data[target_local_date[day_ahead_mask] >= start_date]
    end_date = _parse_date(end)
    if end_date is not None:
        data = data[data["timestamp_local"].dt.date <= end_date]
    records = []
    for col, zone_code in P7_ZONE_COLUMNS.items():
        records.append(
            pd.DataFrame(
                {
                    "timestamp_utc": data["timestamp_utc"],
                    "zone_code": zone_code,
                    "forecast_load_mw": pd.to_numeric(data[col], errors="coerce"),
                }
            )
        )
    melted = pd.concat(records, ignore_index=True)
    melted["zone_name"] = melted["zone_code"].map(ZONE_CODE_TO_NAME)
    return melted.dropna(subset=["timestamp_utc", "zone_code"]).sort_values(["timestamp_utc", "zone_code"])


def load_lbmp_zone_hour(
    raw_dir: str | Path,
    dataset: str,
    *,
    value_prefix: str,
    start: str | date | None = None,
    end: str | date | None = None,
    floor_hour: bool = True,
) -> pd.DataFrame:
    paths = _filter_csvs(Path(raw_dir) / dataset, start, end)
    data = _read_csvs(paths)
    data = data.rename(columns={"Marginal Cost Congestion ($/MWH": "Marginal Cost Congestion ($/MWHr)"})
    ts = _timestamp_frame(_first_column(data, "Time Stamp"), floor_hour=floor_hour)
    data = pd.concat([data, ts], axis=1)
    data["zone_code"] = _first_column(data, "Name").map(_normalize_zone_name)
    data = data.dropna(subset=["timestamp_utc", "zone_code"])
    data[f"{value_prefix}_lbmp"] = pd.to_numeric(_first_column(data, "LBMP ($/MWHr)"), errors="coerce")
    data[f"{value_prefix}_losses"] = pd.to_numeric(_first_column(data, "Marginal Cost Losses ($/MWHr)"), errors="coerce")
    data[f"{value_prefix}_congestion"] = pd.to_numeric(
        _first_column(data, "Marginal Cost Congestion ($/MWHr)"), errors="coerce"
    )
    return (
        data.groupby(["timestamp_utc", "zone_code"], as_index=False)[
            [f"{value_prefix}_lbmp", f"{value_prefix}_losses", f"{value_prefix}_congestion"]
        ]
        .mean()
        .sort_values(["timestamp_utc", "zone_code"])
    )


def load_fuel_mix_hour(
    raw_dir: str | Path = "data/raw",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(Path(raw_dir) / "P-63 Real-Time Fuel Mix (Generation)", start, end)
    data = _read_csvs(paths)
    gen_col = "Gen MW" if "Gen MW" in data.columns else "Gen MWh"
    if "Gen MW" in data.columns and "Gen MWh" in data.columns:
        data["gen_mw"] = pd.to_numeric(data["Gen MW"].fillna(data["Gen MWh"]), errors="coerce")
    else:
        data["gen_mw"] = pd.to_numeric(data[gen_col], errors="coerce")
    ts = _timestamp_frame(data["Time Stamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1)
    data = data[data["Fuel Category"].isin(FUEL_CATEGORIES)]
    grouped = (
        data.groupby(["timestamp_utc", "Fuel Category"], as_index=False)["gen_mw"]
        .mean()
        .pivot(index="timestamp_utc", columns="Fuel Category", values="gen_mw")
        .reset_index()
    )
    for fuel in FUEL_CATEGORIES:
        if fuel not in grouped.columns:
            grouped[fuel] = 0.0
    grouped = grouped.rename(columns=FUEL_TARGET_COLUMNS)
    target_cols = list(FUEL_TARGET_COLUMNS.values())
    grouped[target_cols] = grouped[target_cols].fillna(0.0).clip(lower=0.0)
    grouped["oracle_total_generation_mw"] = grouped[target_cols].sum(axis=1)
    return grouped[["timestamp_utc", *target_cols, "oracle_total_generation_mw"]].sort_values("timestamp_utc")


def load_interface_features_hour(
    raw_dir: str | Path = "data/raw",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(
        Path(raw_dir) / "P-32 Internal External Transmission Interface Limits and Flows",
        start,
        end,
    )
    data = _read_csvs(paths)
    ts = _timestamp_frame(data["Timestamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1)
    flow = pd.to_numeric(data["Flow (MWH)"], errors="coerce")
    positive = pd.to_numeric(data["Positive Limit (MWH)"], errors="coerce")
    negative = pd.to_numeric(data["Negative Limit (MWH)"], errors="coerce").abs()
    positive = positive.where(positive < 9000)
    negative = negative.where(negative < 9000)
    limit = np.where(flow >= 0, positive, negative)
    stress = np.abs(flow) / np.where(np.asarray(limit) > 0, limit, np.nan)
    data["interface_stress"] = pd.Series(stress).replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    data["interface_is_stressed"] = data["interface_stress"] >= 0.9
    return (
        data.groupby("timestamp_utc", as_index=False)
        .agg(
            interface_stress_mean=("interface_stress", "mean"),
            interface_stress_max=("interface_stress", "max"),
            interface_stressed_count=("interface_is_stressed", "sum"),
        )
        .sort_values("timestamp_utc")
    )


def load_interface_hour(
    raw_dir: str | Path = "data/raw",
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(
        Path(raw_dir) / "P-32 Internal External Transmission Interface Limits and Flows",
        start,
        end,
    )
    data = _read_csvs(
        paths,
        usecols=[
            "Timestamp",
            "Interface Name",
            "Point ID",
            "Flow (MWH)",
            "Positive Limit (MWH)",
            "Negative Limit (MWH)",
        ],
    )
    if data.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "interface_name",
                "point_id",
                "flow_mw_mean",
                "flow_mw_min",
                "flow_mw_max",
                "upper_limit_mw",
                "lower_limit_mw",
                "upper_is_finite",
                "lower_is_finite",
                "active_side_sentinel_count",
                "stress_mean",
                "stress_max",
            ]
        )

    ts = _timestamp_frame(data["Timestamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1).dropna(subset=["timestamp_utc"])
    data["interface_name"] = data["Interface Name"].astype(str).str.strip()
    data["point_id"] = pd.to_numeric(data["Point ID"], errors="coerce").astype("Int64")
    data["flow_mw"] = pd.to_numeric(data["Flow (MWH)"], errors="coerce")
    raw_positive = pd.to_numeric(data["Positive Limit (MWH)"], errors="coerce")
    raw_negative = pd.to_numeric(data["Negative Limit (MWH)"], errors="coerce")

    data["upper_limit_row_mw"] = raw_positive.where(
        (raw_positive > 0.0) & (raw_positive < INTERFACE_SENTINEL_ABS_LIMIT)
    )
    data["lower_limit_row_mw"] = raw_negative.where(
        (raw_negative < 0.0) & (raw_negative > -INTERFACE_SENTINEL_ABS_LIMIT)
    )
    data["upper_is_finite_row"] = data["upper_limit_row_mw"].notna()
    data["lower_is_finite_row"] = data["lower_limit_row_mw"].notna()

    active_limit = np.where(
        data["flow_mw"] >= 0.0,
        data["upper_limit_row_mw"],
        -data["lower_limit_row_mw"],
    )
    active_is_sentinel = np.where(
        data["flow_mw"] >= 0.0,
        ~data["upper_is_finite_row"],
        ~data["lower_is_finite_row"],
    )
    data["active_side_sentinel"] = active_is_sentinel.astype(float)
    data["stress"] = (
        data["flow_mw"].abs()
        / pd.Series(active_limit, index=data.index).where(pd.Series(active_limit, index=data.index) > 0.0)
    ).replace([np.inf, -np.inf], np.nan)

    hourly = (
        data.groupby(["timestamp_utc", "interface_name", "point_id"], as_index=False)
        .agg(
            flow_mw_mean=("flow_mw", "mean"),
            flow_mw_min=("flow_mw", "min"),
            flow_mw_max=("flow_mw", "max"),
            upper_limit_mw=("upper_limit_row_mw", "min"),
            lower_limit_mw=("lower_limit_row_mw", "max"),
            upper_is_finite=("upper_is_finite_row", "max"),
            lower_is_finite=("lower_is_finite_row", "max"),
            active_side_sentinel_count=("active_side_sentinel", "sum"),
            stress_mean=("stress", "mean"),
            stress_max=("stress", "max"),
        )
        .sort_values(["timestamp_utc", "interface_name"])
        .reset_index(drop=True)
    )
    hourly["upper_is_finite"] = hourly["upper_is_finite"].astype(bool)
    hourly["lower_is_finite"] = hourly["lower_is_finite"].astype(bool)
    return hourly


def load_constraint_features_hour(
    raw_dir: str | Path,
    dataset: str,
    *,
    prefix: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(Path(raw_dir) / dataset, start, end)
    data = _read_csvs(paths)
    if data.empty:
        return pd.DataFrame(columns=["timestamp_utc"])
    ts = _timestamp_frame(data["Time Stamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1)
    data["constraint_cost"] = pd.to_numeric(data["Constraint Cost($)"], errors="coerce")
    data["constraint_abs_cost"] = data["constraint_cost"].abs()
    return (
        data.groupby("timestamp_utc", as_index=False)
        .agg(
            **{
                f"{prefix}_constraint_count": ("constraint_cost", "size"),
                f"{prefix}_constraint_cost_mean": ("constraint_cost", "mean"),
                f"{prefix}_constraint_abs_cost_max": ("constraint_abs_cost", "max"),
            }
        )
        .sort_values("timestamp_utc")
    )


def load_outage_features_hour(
    raw_dir: str | Path,
    dataset: str,
    *,
    prefix: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    paths = _filter_csvs(Path(raw_dir) / dataset, start, end)
    data = _read_csvs(paths)
    if data.empty:
        return pd.DataFrame(columns=["timestamp_utc", f"{prefix}_outage_count"])
    ts = _timestamp_frame(data["Timestamp"], floor_hour=True)
    data = pd.concat([data, ts], axis=1).dropna(subset=["timestamp_utc"])
    data["equipment_name"] = data["Equipment Name"].astype(str).str.upper()
    data["is_345kv"] = data["equipment_name"].str.contains("345", regex=False)
    interface_terms = ["CENTRAL", "DYSINGER", "MOSES", "CONED", "DUN", "SPR", "MARCY", "RAMAPO"]
    data["has_interface_hint"] = data["equipment_name"].apply(lambda text: any(term in text for term in interface_terms))
    return (
        data.groupby("timestamp_utc", as_index=False)
        .agg(
            **{
                f"{prefix}_outage_count": ("PTID", "size"),
                f"{prefix}_outage_ptid_count": ("PTID", "nunique"),
                f"{prefix}_outage_equipment_count": ("equipment_name", "nunique"),
                f"{prefix}_outage_345kv_count": ("is_345kv", "sum"),
                f"{prefix}_outage_by_interface_hint": ("has_interface_hint", "sum"),
            }
        )
        .sort_values("timestamp_utc")
    )


def _system_weather_aggregates(
    weather: pd.DataFrame,
    capacity: pd.DataFrame,
    zone_load: pd.DataFrame,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    wind_weights = _fuel_capacity_weights(capacity, ["Wind"])
    solar_weights = _raw_fuel_capacity_weights(raw_dir, ["SUN"])
    if solar_weights.empty:
        solar_weights = _equal_zone_weights()

    load_share = (
        zone_load.groupby("zone_code", as_index=False)["actual_load_mw"].mean()
        if not zone_load.empty
        else pd.DataFrame(columns=["zone_code", "actual_load_mw"])
    )
    if load_share.empty or load_share["actual_load_mw"].sum() <= 0:
        temp_weights = _equal_zone_weights()
    else:
        load_share["weight"] = load_share["actual_load_mw"] / load_share["actual_load_mw"].sum()
        temp_weights = load_share[["zone_code", "weight"]]

    pieces = []
    for name, col, weights in [
        ("system_wind_speed_50m_mps", "wind_speed_50m_mps", wind_weights),
        ("system_wind_speed_cubed", "wind_speed_cubed", wind_weights),
        ("system_ghi_wh_m2", "ghi_wh_m2", solar_weights),
        ("system_ghi_rolling_3h", "ghi_rolling_3h", solar_weights),
        ("system_temp_2m_c", "temp_2m_c", temp_weights),
        ("system_temp_rolling_24h", "temp_rolling_24h", temp_weights),
    ]:
        merged = weather[["timestamp_utc", "zone_code", col]].merge(weights, on="zone_code", how="left")
        merged["weight"] = merged["weight"].fillna(0.0)
        feature = (
            merged.assign(weighted=merged[col] * merged["weight"])
            .groupby("timestamp_utc", as_index=False)["weighted"]
            .sum()
            .rename(columns={"weighted": name})
        )
        pieces.append(feature)

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on="timestamp_utc", how="outer")
    return result.sort_values("timestamp_utc")


def _fuel_capacity_weights(capacity: pd.DataFrame, fuels: list[str]) -> pd.DataFrame:
    weights = capacity[capacity["fuel_category"].isin(fuels)].groupby("zone_code", as_index=False)["capacity_mw"].sum()
    total = weights["capacity_mw"].sum()
    if total <= 0:
        return _equal_zone_weights()
    weights["weight"] = weights["capacity_mw"] / total
    return weights[["zone_code", "weight"]]


def _raw_fuel_capacity_weights(raw_dir: str | Path, fuel_codes: list[str]) -> pd.DataFrame:
    facilities_path = Path(raw_dir) / "existing_generating_facilities.csv"
    if not facilities_path.exists():
        return _equal_zone_weights()
    facilities = pd.read_csv(facilities_path, encoding="utf-8-sig")
    facilities["capacity_mw"] = pd.to_numeric(
        facilities["name_plate_rating_MW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    facilities["zone_code"] = facilities["zone"].astype(str).str.strip().str.upper()
    facilities["fuel_type_1"] = facilities["fuel_type_1"].astype(str).str.strip().str.upper()
    weights = (
        facilities[facilities["fuel_type_1"].isin(fuel_codes) & facilities["zone_code"].isin(ZONE_CODE_TO_NAME)]
        .groupby("zone_code", as_index=False)["capacity_mw"]
        .sum()
    )
    total = weights["capacity_mw"].sum()
    if total <= 0:
        return _equal_zone_weights()
    weights["weight"] = weights["capacity_mw"] / total
    return weights[["zone_code", "weight"]]


def _equal_zone_weights() -> pd.DataFrame:
    zones = sorted(ZONE_CODE_TO_NAME)
    return pd.DataFrame({"zone_code": zones, "weight": 1.0 / len(zones)})


def build_processed_datasets(config: BuildConfig) -> dict[str, pd.DataFrame]:
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    weather = load_weather_zone_hour(config.weather_dir, start=config.start, end=config.end)
    capacity = load_static_capacity(raw_dir)
    zonal_capacity = build_zonal_capacity_table(capacity)
    actual_load = load_actual_load_zone_hour(raw_dir, start=config.start, end=config.end)
    forecast_load = load_forecast_load_zone_hour(raw_dir, start=config.start, end=config.end)
    da_lbmp = load_lbmp_zone_hour(
        raw_dir,
        "P-2A Day-Ahead Market LBMP",
        value_prefix="da",
        start=config.start,
        end=config.end,
        floor_hour=True,
    )
    rt_lbmp = load_lbmp_zone_hour(
        raw_dir,
        "P-24A Real-Time Market LBMP",
        value_prefix="rt",
        start=config.start,
        end=config.end,
        floor_hour=True,
    )

    zone_hour = (
        actual_load.merge(forecast_load, on=["timestamp_utc", "zone_code", "zone_name"], how="outer")
        .merge(da_lbmp, on=["timestamp_utc", "zone_code"], how="left")
        .merge(rt_lbmp, on=["timestamp_utc", "zone_code"], how="left")
        .merge(weather, on=["timestamp_utc", "zone_code", "zone_name"], how="left")
    )
    zone_hour["load_forecast_error_mw"] = zone_hour["actual_load_mw"] - zone_hour["forecast_load_mw"]
    zone_hour = zone_hour.sort_values(["timestamp_utc", "zone_code"]).reset_index(drop=True)
    zone_hour = add_weather_availability_features(zone_hour, zonal_capacity)

    fuel_mix = load_fuel_mix_hour(raw_dir, start=config.start, end=config.end)
    interface_hour = (
        load_interface_hour(raw_dir, start=config.start, end=config.end)
        if config.include_interface_hour
        else pd.DataFrame()
    )
    interface = load_interface_features_hour(raw_dir, start=config.start, end=config.end)
    rt_constraints = load_constraint_features_hour(
        raw_dir,
        "P-33 Limiting Constraints",
        prefix="rt",
        start=config.start,
        end=config.end,
    )
    da_constraints = load_constraint_features_hour(
        raw_dir,
        "P-511A Day-Ahead Limiting Constraints",
        prefix="da",
        start=config.start,
        end=config.end,
    )
    scheduled_outages = load_outage_features_hour(
        raw_dir,
        "P-54A Scheduled Outages",
        prefix="scheduled",
        start=config.start,
        end=config.end,
    )
    realtime_outages = load_outage_features_hour(
        raw_dir,
        "P-54B Real-Time Outages",
        prefix="realtime",
        start=config.start,
        end=config.end,
    )

    system_load = (
        zone_hour.groupby("timestamp_utc", as_index=False)
        .agg(
            actual_load_mw=("actual_load_mw", "sum"),
            forecast_load_mw=("forecast_load_mw", "sum"),
            da_lbmp_mean=("da_lbmp", "mean"),
            da_lbmp_max=("da_lbmp", "max"),
            da_congestion_mean=("da_congestion", "mean"),
            rt_lbmp_mean=("rt_lbmp", "mean"),
            rt_lbmp_max=("rt_lbmp", "max"),
            rt_congestion_mean=("rt_congestion", "mean"),
        )
        .sort_values("timestamp_utc")
    )
    system_load["load_forecast_error_mw"] = system_load["actual_load_mw"] - system_load["forecast_load_mw"]
    weather_system = _system_weather_aggregates(weather, capacity, zone_hour, raw_dir)

    system_hour = system_load.merge(fuel_mix, on="timestamp_utc", how="inner")
    system_hour["target_internal_generation_mw"] = system_hour[list(FUEL_TARGET_COLUMNS.values())].sum(axis=1)
    system_hour["forecast_net_import_residual_mw"] = (
        system_hour["forecast_load_mw"] - system_hour["target_internal_generation_mw"]
    )
    system_hour["actual_net_import_residual_mw"] = (
        system_hour["actual_load_mw"] - system_hour["target_internal_generation_mw"]
    )
    for feature in [weather_system, interface, rt_constraints, da_constraints, scheduled_outages, realtime_outages]:
        system_hour = system_hour.merge(feature, on="timestamp_utc", how="left")

    system_hour = _add_calendar_features(system_hour)
    system_hour = system_hour.sort_values("timestamp_utc").reset_index(drop=True)
    fill_zero_cols = [
        "interface_stressed_count",
        "rt_constraint_count",
        "da_constraint_count",
        "scheduled_outage_count",
        "realtime_outage_count",
        "scheduled_outage_ptid_count",
        "scheduled_outage_equipment_count",
        "scheduled_outage_345kv_count",
        "scheduled_outage_by_interface_hint",
        "realtime_outage_ptid_count",
        "realtime_outage_equipment_count",
        "realtime_outage_345kv_count",
        "realtime_outage_by_interface_hint",
    ]
    for col in fill_zero_cols:
        if col in system_hour.columns:
            system_hour[col] = system_hour[col].fillna(0.0)
    numeric_cols = system_hour.select_dtypes(include=[np.number]).columns
    system_hour[numeric_cols] = system_hour[numeric_cols].ffill().bfill()

    capacity_system = (
        capacity.groupby("fuel_category", as_index=False)["capacity_mw"]
        .sum()
        .rename(columns={"capacity_mw": "static_capacity_mw"})
    )
    for fuel, col in FUEL_TARGET_COLUMNS.items():
        cap = capacity_system.loc[capacity_system["fuel_category"] == fuel, "static_capacity_mw"]
        system_hour[f"static_capacity_{col.removeprefix('gen_').removesuffix('_mw')}_mw"] = float(cap.iloc[0]) if len(cap) else 0.0

    system_hour = add_causal_lag_features(system_hour)

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "weather_zone_hour": weather,
        "zone_hour": zone_hour,
        "interface_hour": interface_hour,
        "zonal_capacity": zonal_capacity,
        "system_hour": system_hour,
    }
    _write_table(weather, processed_dir / "nyiso_weather_zone_hour.parquet")
    _write_table(zone_hour, processed_dir / "nyiso_zone_hour.parquet")
    if config.include_interface_hour:
        _write_table(interface_hour, processed_dir / "nyiso_interface_hour.parquet")
    _write_table(zonal_capacity, processed_dir / "nyiso_zonal_capacity.parquet")
    _write_table(system_hour, processed_dir / "nyiso_system_hour.parquet")
    return outputs


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_local = ts_utc.dt.tz_convert(NY_TZ)
    df["timestamp_local"] = ts_local
    df["hour"] = ts_local.dt.hour
    df["dayofweek"] = ts_local.dt.dayofweek
    df["month"] = ts_local.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_causal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    for base_col in CAUSAL_LAG_BASE_COLUMNS:
        if base_col not in df.columns:
            continue
        for lag in CAUSAL_LAGS:
            lag_col = f"{base_col}_lag_{lag}"
            df[lag_col] = df[base_col].shift(lag)
    return df


def validate_processed_tables(
    weather: pd.DataFrame,
    system_hour: pd.DataFrame,
    *,
    zone_hour: pd.DataFrame | None = None,
    interface_hour: pd.DataFrame | None = None,
    zonal_capacity: pd.DataFrame | None = None,
    expected_weather_hours_per_zone: int | None = None,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["weather_zones"] = sorted(weather["zone_code"].dropna().unique().tolist())
    checks["weather_duplicate_zone_hours"] = int(weather.duplicated(["zone_code", "timestamp_utc"]).sum())
    checks["weather_remaining_ghi_sentinel"] = int((weather["ghi_wh_m2"] == -999).sum())
    checks["weather_nulls"] = weather.isna().sum().to_dict()
    if expected_weather_hours_per_zone is not None:
        counts = weather.groupby("zone_code")["timestamp_utc"].nunique()
        checks["weather_expected_hours_per_zone"] = expected_weather_hours_per_zone
        checks["weather_bad_hour_counts"] = counts[counts != expected_weather_hours_per_zone].to_dict()
    checks["system_duplicate_hours"] = int(system_hour.duplicated(["timestamp_utc"]).sum())
    checks["system_rows"] = int(len(system_hour))
    target_cols = list(FUEL_TARGET_COLUMNS.values())
    checks["target_negative_values"] = int((system_hour[target_cols] < 0).sum().sum())
    checks["target_nulls"] = system_hour[target_cols].isna().sum().to_dict()
    checks["load_nulls"] = system_hour[["actual_load_mw", "forecast_load_mw"]].isna().sum().to_dict()
    if zone_hour is not None:
        checks["zone_hour_zones"] = sorted(zone_hour["zone_code"].dropna().unique().tolist())
        checks["zone_hour_duplicate_zone_hours"] = int(zone_hour.duplicated(["zone_code", "timestamp_utc"]).sum())
        zone_system_load = zone_hour.groupby("timestamp_utc")["forecast_load_mw"].sum()
        system_load = system_hour.set_index("timestamp_utc")["forecast_load_mw"]
        common = zone_system_load.index.intersection(system_load.index)
        checks["zone_system_forecast_load_max_abs_diff"] = (
            float((zone_system_load.loc[common] - system_load.loc[common]).abs().max()) if len(common) else np.nan
        )
    if interface_hour is not None and not interface_hour.empty:
        finite_upper = interface_hour.loc[interface_hour["upper_is_finite"], "upper_limit_mw"].abs()
        finite_lower = interface_hour.loc[interface_hour["lower_is_finite"], "lower_limit_mw"].abs()
        checks["interface_rows"] = int(len(interface_hour))
        checks["interface_names"] = sorted(interface_hour["interface_name"].dropna().unique().tolist())
        checks["interface_finite_limit_abs_ge_9000"] = int(
            (finite_upper >= INTERFACE_SENTINEL_ABS_LIMIT).sum()
            + (finite_lower >= INTERFACE_SENTINEL_ABS_LIMIT).sum()
        )
    if zonal_capacity is not None:
        checks["zonal_capacity_zones"] = sorted(zonal_capacity["zone_code"].dropna().unique().tolist())
        checks["zonal_capacity_fuels"] = sorted(zonal_capacity["fuel_category"].dropna().unique().tolist())
        checks["zonal_capacity_rows"] = int(len(zonal_capacity))
    return checks


def build_and_validate(
    *,
    raw_dir: str | Path = "data/raw",
    weather_dir: str | Path = "data/weather",
    processed_dir: str | Path = "data/processed",
    start: str | date | None = None,
    end: str | date | None = None,
    include_interface_hour: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    config = BuildConfig(Path(raw_dir), Path(weather_dir), Path(processed_dir), start, end, include_interface_hour)
    outputs = build_processed_datasets(config)
    expected = None
    if start is None and end is None:
        expected = 89112
    checks = validate_processed_tables(
        outputs["weather_zone_hour"],
        outputs["system_hour"],
        zone_hour=outputs["zone_hour"],
        interface_hour=outputs["interface_hour"],
        zonal_capacity=outputs["zonal_capacity"],
        expected_weather_hours_per_zone=expected,
    )
    return outputs, checks


if __name__ == "__main__":
    outputs, checks = build_and_validate()
    print("Wrote processed datasets:")
    for name, frame in outputs.items():
        print(f"  {name}: {frame.shape}")
    print("Checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
