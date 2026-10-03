"""Cleaning and feature engineering for RHF / mill / water-box process data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def load_raw(path=None) -> pd.DataFrame:
    path = path or cfg.RAW_DIR / "process_data.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned frame, report of what was changed)."""
    df = df.copy()
    report = {"rows_in": len(df)}

    df = df.drop_duplicates()
    report["duplicates_dropped"] = report["rows_in"] - len(df)

    # a stuck-at-zero pressure/flow sensor is a fault, not a real operating point
    for col in ("wb_pressure_bar", "wb_flow_m3h", "mill_speed_mps"):
        df.loc[df[col] == 0, col] = np.nan

    flagged = {}
    for col, (lo, hi) in cfg.VALID_RANGES.items():
        if col in df.columns:
            bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            flagged[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
    report["out_of_range_set_nan"] = {k: v for k, v in flagged.items() if v}

    # unlabeled rows cannot be used for supervised learning
    n = len(df)
    df = df.dropna(subset=[cfg.TARGET]).reset_index(drop=True)
    report["missing_target_dropped"] = n - len(df)

    # impute process sensors with grade+diameter-aware medians, fall back to global median
    num = [c for c in cfg.NUMERIC_FEATURES if c != "ceq"]
    report["imputed_cells"] = int(df[num].isna().sum().sum())
    grp = df.groupby(["grade", "bar_dia_mm"])
    for col in num:
        df[col] = df[col].fillna(grp[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # chemistry-derived, recomputed so it stays consistent after imputation
    df["ceq"] = df["c_pct"] + df["mn_pct"] / 6 + df["si_pct"] / 24 + df["v_pct"] / 14

    report["rows_out"] = len(df)
    return df, report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physically motivated features; every term is computable online from plant signals."""
    df = df.copy()
    df["inv_speed"] = 1.0 / df["mill_speed_mps"]  # dwell time in water box ~ 1/speed
    df["surface_to_volume"] = 4.0 / df["bar_dia_mm"]
    df["quench_index"] = (
        np.sqrt(df["wb_pressure_bar"]) * df["wb_flow_m3h"] * (1 - 0.012 * (df["wb_water_temp_c"] - 30))
        * df["inv_speed"] / df["bar_dia_mm"] ** 1.4
    )
    df["wb_power_index"] = df["wb_pressure_bar"] * df["wb_flow_m3h"]
    df["soak_to_finish_drop_c"] = df["soaking_temp_c"] - df["finishing_temp_c"]
    df["zone_gradient_c"] = df["soaking_zone_temp_c"] - df["preheat_zone_temp_c"]
    df["microalloy_sum"] = df["v_pct"] + df["nb_pct"]
    df["mn_c_ratio"] = df["mn_pct"] / df["c_pct"]
    df["grade_min_ys"] = df["grade"].map({g: s["min_ys"] for g, s in cfg.GRADES.items()})
    return df


def feature_columns(include_grade_dummies: bool = True) -> list[str]:
    return cfg.NUMERIC_FEATURES + cfg.ENGINEERED


def prepare(df: pd.DataFrame | None = None, save: bool = True):
    df = load_raw() if df is None else df
    df, report = clean(df)
    df = add_features(df)
    if save:
        cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.PROCESSED_DIR / "clean_features.csv", index=False)
    return df, report


def main() -> None:
    df, report = prepare()
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"features: {len(feature_columns())} -> {cfg.PROCESSED_DIR / 'clean_features.csv'}")


if __name__ == "__main__":
    main()
