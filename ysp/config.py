"""Central configuration: paths, feature groups, grade definitions and spec limits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "yield_strength_mpa"
SECONDARY_TARGET = "uts_mpa"

# Grade -> (min YS, min UTS/YS ratio) per IS 1786 style TMT specifications.
GRADES = {
    "Fe500": {"min_ys": 500, "min_uts_ys": 1.08},
    "Fe500D": {"min_ys": 500, "min_uts_ys": 1.10},
    "Fe550D": {"min_ys": 550, "min_uts_ys": 1.10},
    "Fe600": {"min_ys": 600, "min_uts_ys": 1.06},
}
# Upper agreed band on YS to avoid over-strength (and wasted alloy / poor ductility).
YS_UPPER_MARGIN = 80  # MPa above min_ys

CHEMISTRY = ["c_pct", "mn_pct", "si_pct", "s_pct", "p_pct", "v_pct", "nb_pct", "ceq"]
RHF = [
    "billet_size_mm", "preheat_zone_temp_c", "heating_zone_temp_c", "soaking_zone_temp_c",
    "soaking_temp_c", "rhf_residence_min", "discharge_temp_c",
]
MILL = [
    "bar_dia_mm", "mill_speed_mps", "finishing_temp_c", "ambient_temp_c",
]
WATER_BOX = [
    "wb_pressure_bar", "wb_flow_m3h", "wb_water_temp_c", "wb_valve_open_pct",
]
CATEGORICAL = ["grade"]
NUMERIC_FEATURES = CHEMISTRY + RHF + MILL + WATER_BOX

# Physically plausible sensor ranges; values outside are treated as faulty readings (-> NaN).
VALID_RANGES = {
    "c_pct": (0.05, 0.40), "mn_pct": (0.3, 1.8), "si_pct": (0.05, 0.6),
    "s_pct": (0.0, 0.06), "p_pct": (0.0, 0.06), "v_pct": (0.0, 0.15), "nb_pct": (0.0, 0.06),
    "preheat_zone_temp_c": (700, 1050), "heating_zone_temp_c": (1000, 1300),
    "soaking_zone_temp_c": (1100, 1320), "soaking_temp_c": (1050, 1300),
    "rhf_residence_min": (30, 200), "discharge_temp_c": (1050, 1300),
    "bar_dia_mm": (6, 40), "mill_speed_mps": (1.0, 30.0), "finishing_temp_c": (850, 1120),
    "ambient_temp_c": (-10, 60),
    "wb_pressure_bar": (0.5, 20.0), "wb_flow_m3h": (30, 500),
    "wb_water_temp_c": (5, 60), "wb_valve_open_pct": (0, 100),
    "yield_strength_mpa": (250, 900), "uts_mpa": (300, 1000),
}
ENGINEERED = [
    "quench_index", "inv_speed", "surface_to_volume", "soak_to_finish_drop_c",
    "zone_gradient_c", "microalloy_sum", "mn_c_ratio", "wb_power_index", "grade_min_ys",
]

# Agreed error band for the model vs lab-tested YS (MPa); tune with plant metallurgists.
ERROR_BAND_MPA = 15
HOLDOUT_FRACTION = 0.2  # most recent heats are held out as the "lab benchmark"
