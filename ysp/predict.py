"""Single-heat prediction helpers shared by the prototype app and scripts."""
from __future__ import annotations

import pandas as pd

from . import config as cfg
from .models import load_model, time_split
from .optimize import load_or_train_uts, predict_pair
from .preprocess import add_features, prepare


def build_row(heat: dict) -> pd.DataFrame:
    row = pd.DataFrame([dict(heat)])
    row["ceq"] = row["c_pct"] + row["mn_pct"] / 6 + row["si_pct"] / 24 + row["v_pct"] / 14
    return add_features(row)


def assess(heat: dict, ys_bundle=None, uts_bundle=None) -> dict:
    """Predicted YS/UTS for one heat and its status against the grade spec."""
    ys_bundle = ys_bundle or load_model()
    if uts_bundle is None:
        df, _ = prepare(save=False)
        uts_bundle = load_or_train_uts(time_split(df)[0])
    row = build_row(heat)
    ys, uts = (float(v[0]) for v in predict_pair(row, ys_bundle, uts_bundle))
    spec = cfg.GRADES[heat["grade"]]
    lo, hi = spec["min_ys"], spec["min_ys"] + cfg.YS_UPPER_MARGIN
    ratio = uts / ys
    if ys < lo:
        status = "BELOW MIN YS"
    elif ys > hi:
        status = "ABOVE UPPER BAND"
    elif ratio < spec["min_uts_ys"]:
        status = "LOW UTS/YS"
    elif ys < lo + cfg.ERROR_BAND_MPA:
        status = "MARGINAL (close to min YS)"
    else:
        status = "OK"
    return {"ys": ys, "uts": uts, "ratio": ratio, "min_ys": lo, "upper_ys": hi,
            "min_ratio": spec["min_uts_ys"], "status": status}
