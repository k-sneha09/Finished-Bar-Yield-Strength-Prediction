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
