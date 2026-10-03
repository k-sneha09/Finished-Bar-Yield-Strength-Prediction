"""Physics-informed synthetic data generator for RHF + rolling mill + water-box (Tempcore) data.

No plant data is available yet, so this produces a realistic stand-in with the same schema
the real ISP Burnpur data would have. Replace `data/raw/process_data.csv` with real data
(same column names) and the rest of the pipeline works unchanged.

Metallurgical logic encoded (simplified):
  * Core strength: solid solution (Mn, Si), carbon/pearlite, V/Nb precipitation + grain size.
  * Grain size: coarser with higher soaking temperature / residence time, finer with lower
    finishing temperature (Hall-Petch).
  * Quenched rim: water box removes heat ~ pressure^0.5 * flow / (bar surface-area/speed term);
    martensitic rim fraction grows with cooling intensity and saturates. Strong self-tempering
    (high final surface temperature) softens the rim.
  * Zone temperatures drive discharge temperature; a cold/under-soaked billet raises rolling
    load and lowers the finishing temperature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# grade: (C, Mn, Si, V, Nb) target means; larger = stronger/more alloyed
_GRADE_CHEM = {
    "Fe500": dict(c=0.22, mn=0.95, si=0.20, v=0.000, nb=0.000),
    "Fe500D": dict(c=0.21, mn=1.00, si=0.20, v=0.015, nb=0.000),
    "Fe550D": dict(c=0.22, mn=1.02, si=0.22, v=0.025, nb=0.008),
    "Fe600": dict(c=0.23, mn=1.08, si=0.24, v=0.035, nb=0.012),
}
_GRADE_MIX = {"Fe500": 0.25, "Fe500D": 0.40, "Fe550D": 0.25, "Fe600": 0.10}
_DIAMETERS = np.array([8, 10, 12, 16, 20, 25, 28, 32])
_DIA_PROB = np.array([0.04, 0.10, 0.20, 0.25, 0.18, 0.12, 0.07, 0.04])


def _ceq(c, mn, si, v):
    return c + mn / 6 + si / 24 + v / 14  # simplified IIW-like (Cr/Mo/Ni/Cu ~ residual)


def generate(n: int = 5000, seed: int = cfg.RANDOM_STATE, messy: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    grade = rng.choice(list(_GRADE_MIX), size=n, p=list(_GRADE_MIX.values()))
    chem = {k: np.array([_GRADE_CHEM[g][k] for g in grade]) for k in ("c", "mn", "si", "v", "nb")}
    c = chem["c"] + rng.normal(0, 0.012, n)
    mn = chem["mn"] + rng.normal(0, 0.05, n)
    si = chem["si"] + rng.normal(0, 0.02, n)
    v = np.clip(chem["v"] + rng.normal(0, 0.004, n) * (chem["v"] > 0), 0, None)
    nb = np.clip(chem["nb"] + rng.normal(0, 0.002, n) * (chem["nb"] > 0), 0, None)
    s = np.clip(rng.normal(0.030, 0.006, n), 0.008, 0.045)
    p = np.clip(rng.normal(0.028, 0.006, n), 0.008, 0.045)
    ceq = _ceq(c, mn, si, v)

    # ---- reheating furnace ------------------------------------------------------------
    billet = rng.choice([125, 130, 150], size=n, p=[0.3, 0.3, 0.4])
    preheat = rng.normal(900, 25, n)
    heating = rng.normal(1180, 20, n)
    soaking_zone = rng.normal(1225, 18, n)
    residence = np.clip(rng.normal(95, 15, n) * (billet / 130) ** 0.5, 55, 150)
    # billet soaking temperature lags the zone setpoint when residence is short
    soak_lag = 25 * np.exp(-(residence - 55) / 35)
    soaking = soaking_zone - 40 - soak_lag + rng.normal(0, 8, n)
    discharge = soaking - rng.normal(15, 4, n)

    # ---- rolling mill -----------------------------------------------------------------
    dia = rng.choice(_DIAMETERS, size=n, p=_DIA_PROB).astype(float)
    # thinner bars are rolled faster (m/s); noise from operator practice
    speed = np.clip(22 * (8 / dia) ** 0.55 * 1.0 + rng.normal(0, 0.6, n), 2.0, 22)
    ambient = rng.normal(30, 7, n)
    # finishing temperature falls with lower discharge temp, thicker bars hold heat longer
    fin = 0.66 * discharge + 215 + 0.4 * (dia - 16) - 0.9 * (ambient - 30) + rng.normal(0, 10, n)
    fin = np.clip(fin, 880, 1090)

    # ---- water box (Tempcore) ---------------------------------------------------------
    wb_press = np.clip(rng.normal(8.0, 1.6, n) + 0.03 * (dia - 16), 2.0, 14.0)
    wb_flow = np.clip(rng.normal(220, 35, n) + 1.5 * (dia - 16), 80, 400)
    wb_wtemp = np.clip(rng.normal(32, 4, n) + 0.3 * (ambient - 30), 18, 45)
    wb_valve = np.clip(55 + 4.0 * (wb_press - 8) + rng.normal(0, 5, n), 20, 100)

    # ---- strength model ---------------------------------------------------------------
    # grain size (um): coarsens with soaking T / residence, refines with low finishing T
    d_um = 12 + 0.045 * (soaking - 1150) + 0.04 * (residence - 90) + 0.035 * (fin - 980)
    d_um = np.clip(d_um - 350 * nb, 4, None)
    hall_petch = 18.0 / np.sqrt(d_um)  # MPa per sqrt(um)^-1 scaled
    core = (
        275 + 280 * c + 25 * mn + 70 * si
        + 1800 * v * np.exp(-(fin - 900) / 400)  # V(C,N) precipitation, strain-ind. helps at low T
        + 2200 * nb
        + 22 * hall_petch
        + 450 * np.clip(c - 0.15, 0, None) * 0.3
    )

    # cooling intensity of water box vs bar thermal mass and dwell time in box
    dwell = 1.8 / speed  # s in the quenching line (fixed box length)
    q = (wb_press ** 0.5) * (wb_flow / 200) * (1 - 0.012 * (wb_wtemp - 30)) * dwell / (dia / 16) ** 1.4
    rim_frac = np.clip(0.9 * q ** 1.5, 0, 0.45)
    # rim hardness (martensite/bainite) increases with carbon equivalent
    rim_strength = 690 + 500 * (ceq - 0.40)
    # self-tempering softens rim when the core retains lots of heat (high finishing T, thick bar)
    temper_loss = 55 * (rim_frac / 0.2) * np.clip((fin - 960) / 100, 0, 1.5) * np.clip(dia / 25, 0.5, 1.3)
    ys = (1 - rim_frac) * core + rim_frac * rim_strength - temper_loss
    # mild interaction: very high soaking temp (>1240) causes excess scale / decarb -> small drop
    ys -= 0.08 * np.clip(soaking - 1210, 0, None) ** 1.2
    ys += rng.normal(0, 7.5, n)  # lab test scatter + unmodelled variation

    # UTS/YS ratio (ductility proxy): a harder martensitic rim lifts YS but erodes the ratio
    uts_ys = 1.14 + 0.30 * (0.20 - rim_frac) + 0.05 * (ceq - 0.40) / 0.2
    uts = ys * uts_ys + rng.normal(0, 4, n)
    uts = np.maximum(uts, ys + 10)

    df = pd.DataFrame({
        "heat_no": [f"H{250000 + i // 6}" for i in range(n)],
        "grade": grade,
        "c_pct": c, "mn_pct": mn, "si_pct": si, "s_pct": s, "p_pct": p, "v_pct": v, "nb_pct": nb,
        "ceq": ceq,
        "billet_size_mm": billet,
        "preheat_zone_temp_c": preheat, "heating_zone_temp_c": heating,
        "soaking_zone_temp_c": soaking_zone, "soaking_temp_c": soaking,
        "rhf_residence_min": residence, "discharge_temp_c": discharge,
        "bar_dia_mm": dia, "mill_speed_mps": speed, "finishing_temp_c": fin, "ambient_temp_c": ambient,
        "wb_pressure_bar": wb_press, "wb_flow_m3h": wb_flow, "wb_water_temp_c": wb_wtemp,
        "wb_valve_open_pct": wb_valve,
        cfg.TARGET: ys, cfg.SECONDARY_TARGET: uts,
    })
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=n, freq="37min"))

    if messy:
        df = _inject_plant_noise(df, rng)
    num = df.select_dtypes('number').columns
    df[num] = df[num].round(4)
    return df


def _inject_plant_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic data-quality issues: sensor dropouts, spikes, and a few bad lab records."""
    n = len(df)
    sensor_cols = ["soaking_temp_c", "finishing_temp_c", "wb_pressure_bar", "wb_flow_m3h",
                   "mill_speed_mps", "wb_water_temp_c"]
    for col in sensor_cols:
        df.loc[rng.random(n) < 0.012, col] = np.nan
    spike_idx = rng.choice(n, size=max(1, n // 400), replace=False)
    df.loc[spike_idx, "finishing_temp_c"] *= rng.choice([0.4, 1.8], size=len(spike_idx))
    df.loc[rng.random(n) < 0.003, "wb_pressure_bar"] = 0.0  # sensor stuck at zero
    bad_lab = rng.choice(n, size=max(1, n // 500), replace=False)
    df.loc[bad_lab, cfg.TARGET] = np.nan  # missing test result
    return df


def main() -> None:
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    out = cfg.RAW_DIR / "process_data.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {out}")
    print(df[cfg.TARGET].describe().round(1).to_string())
    print(df.groupby("grade")[cfg.TARGET].agg(["count", "mean", "std", "min", "max"]).round(1))


if __name__ == "__main__":
    main()
