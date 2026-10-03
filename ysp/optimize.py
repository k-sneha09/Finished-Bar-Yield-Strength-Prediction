"""Operating-window optimisation by grade (water-box pressure/flow, mill speed).

For every heat the controllable settings (water-box pressure, flow, mill speed) are searched on a
grid inside the range actually used on the line for that bar diameter. Chemistry, RHF
temperatures, finishing temperature and bar size are held at the heat's own values.

A candidate is feasible when, with a safety margin for model error,
    min_ys + margin <= predicted YS <= min_ys + YS_UPPER_MARGIN - margin
    predicted UTS / predicted YS >= grade minimum + ratio margin
Among feasible candidates, those within TOLERANCE of the lowest expected off-spec probability are
kept (robust, centred in the band); the fastest mill speed then wins (throughput), then lowest flow.
Windows per (grade, diameter) are the P10-P90 of the per-heat recommendations.

The predicted off-spec rate is a model-based estimate (Gaussian lab/model scatter around the
prediction); it must be confirmed with plant trials before being quoted as a rejection reduction.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as cfg
from .models import load_model, make_models, model_columns, time_split
from .preprocess import add_features, prepare

CONTROLS = ["wb_pressure_bar", "wb_flow_m3h", "mill_speed_mps"]
RATIO_MARGIN = 0.01
GRID_POINTS = 7
TOLERANCE = 0.01  # accepted extra off-spec probability in exchange for throughput


def train_uts_model(train: pd.DataFrame):
    num, cat = model_columns()
    pipe = make_models()["gradient_boosting"].fit(train[num + cat], train[cfg.SECONDARY_TARGET])
    return {"model": pipe, "features": num + cat, "name": "gradient_boosting"}


def load_or_train_uts(train: pd.DataFrame) -> dict:
    path = cfg.MODEL_DIR / "uts_model.joblib"
    if path.exists():
        return joblib.load(path)
    bundle = train_uts_model(train)
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def control_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """P5-P95 of each control per bar diameter: what the line has actually run."""
    g = df.groupby("bar_dia_mm")[CONTROLS]
    lo, hi = g.quantile(0.05), g.quantile(0.95)
    return lo.join(hi, lsuffix="_lo", rsuffix="_hi")


def _with_controls(rows: pd.DataFrame, p, f, s) -> pd.DataFrame:
    out = rows.copy()
    out["wb_pressure_bar"], out["wb_flow_m3h"], out["mill_speed_mps"] = p, f, s
    # valve opening follows pressure on this line (see data generator / plant practice)
    out["wb_valve_open_pct"] = np.clip(55 + 4.0 * (out["wb_pressure_bar"] - 8), 20, 100)
    return add_features(out)


def predict_pair(rows: pd.DataFrame, ys_bundle: dict, uts_bundle: dict):
    ys = ys_bundle["model"].predict(rows[ys_bundle["features"]])
    uts = uts_bundle["model"].predict(rows[uts_bundle["features"]])
    return ys, uts


def spec_limits(grades: pd.Series):
    lo = grades.map({g: s["min_ys"] for g, s in cfg.GRADES.items()}).to_numpy(float)
    ratio = grades.map({g: s["min_uts_ys"] for g, s in cfg.GRADES.items()}).to_numpy(float)
    return lo, lo + cfg.YS_UPPER_MARGIN, ratio


def optimise_heats(rows: pd.DataFrame, ys_bundle: dict, uts_bundle: dict, bounds: pd.DataFrame,
                   safety: float = cfg.ERROR_BAND_MPA, n: int = GRID_POINTS,
                   sigma_ys: float = 8.5, sigma_ratio: float = 0.011) -> pd.DataFrame:
    """Return one recommended (pressure, flow, speed) per input row (NaN when nothing is feasible)."""
    rows = rows.reset_index(drop=True)
    results = []
    for dia, sub in rows.groupby("bar_dia_mm"):
        b = bounds.loc[dia]
        ps = np.linspace(b["wb_pressure_bar_lo"], b["wb_pressure_bar_hi"], n)
        fs = np.linspace(b["wb_flow_m3h_lo"], b["wb_flow_m3h_hi"], n)
        ss = np.linspace(b["mill_speed_mps_lo"], b["mill_speed_mps_hi"], n)
        grid = np.array(np.meshgrid(ps, fs, ss, indexing="ij")).reshape(3, -1).T  # (n^3, 3)
        k = len(grid)
        big = sub.loc[sub.index.repeat(k)]
        cand = _with_controls(big, np.tile(grid[:, 0], len(sub)), np.tile(grid[:, 1], len(sub)),
                              np.tile(grid[:, 2], len(sub)))
        ys, uts = predict_pair(cand, ys_bundle, uts_bundle)
        lo, hi, ratio = spec_limits(cand["grade"])
        ok = (ys >= lo + safety) & (ys <= hi - safety) & (uts / ys >= ratio + RATIO_MARGIN)
        res = pd.DataFrame({
            "row": np.repeat(sub.index.to_numpy(), k), "p": cand["wb_pressure_bar"].to_numpy(),
            "f": cand["wb_flow_m3h"].to_numpy(), "s": cand["mill_speed_mps"].to_numpy(),
            "ys": ys, "uts": uts, "ok": ok,
            "p_off": expected_offspec(ys, uts, cand["grade"], sigma_ys, sigma_ratio),
        })
        res = res[res["ok"]]
        if len(res):
            res = res[res["p_off"] <= res.groupby("row")["p_off"].transform("min") + TOLERANCE]
            best = res.sort_values(["row", "s", "f"], ascending=[True, False, True]).groupby("row").head(1)
            results.append(best)
    rec = pd.concat(results).set_index("row").reindex(rows.index) if results else pd.DataFrame(index=rows.index)
    return rec.rename(columns={"p": "rec_pressure_bar", "f": "rec_flow_m3h", "s": "rec_speed_mps",
                               "ys": "rec_ys_pred", "uts": "rec_uts_pred"}).drop(columns=["ok", "p_off"], errors="ignore")


def expected_offspec(ys, uts, grades: pd.Series, sigma_ys: float, sigma_ratio: float) -> np.ndarray:
    """P(off-spec) per row assuming N(pred, sigma) scatter on YS and on the UTS/YS ratio."""
    lo, hi, ratio = spec_limits(grades)
    p_ys = norm.cdf((lo - ys) / sigma_ys) + norm.sf((hi - ys) / sigma_ys)
    p_ratio = norm.cdf((ratio - uts / ys) / sigma_ratio)
    return np.clip(p_ys + p_ratio - p_ys * p_ratio, 0, 1)


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df, _ = prepare()
    train, test = time_split(df)
    ys_bundle = load_model()
    uts_bundle = load_or_train_uts(train)
    bounds = control_bounds(train)

    # model scatter on held-out heats -> noise level for the expected-off-spec estimate
    ys_te, uts_te = predict_pair(test, ys_bundle, uts_bundle)
    sigma_ys = float(np.std(ys_te - test[cfg.TARGET]))
    sigma_ratio = float(np.std(uts_te / ys_te - test[cfg.SECONDARY_TARGET] / test[cfg.TARGET]))

    rec = optimise_heats(test, ys_bundle, uts_bundle, bounds, sigma_ys=sigma_ys, sigma_ratio=sigma_ratio)
    res = test.reset_index(drop=True).join(rec)
    feasible = res["rec_speed_mps"].notna()

    base_p = expected_offspec(ys_te, uts_te, test["grade"].reset_index(drop=True), sigma_ys, sigma_ratio)
    # where nothing is feasible keep the operator's current setting (no change)
    new_ys = np.where(feasible, res["rec_ys_pred"], ys_te)
    new_uts = np.where(feasible, res["rec_uts_pred"], uts_te)
    new_p = expected_offspec(new_ys, new_uts, res["grade"], sigma_ys, sigma_ratio)

    lo, _, _ = spec_limits(res["grade"])
    actual_below = (test[cfg.TARGET].reset_index(drop=True) < lo)
    summary = {
        "n_heats": len(res), "feasible_pct": float(feasible.mean() * 100),
        "sigma_ys_mpa": sigma_ys, "sigma_uts_ys_ratio": sigma_ratio,
        "lab_below_min_ys_pct_actual": float(actual_below.mean() * 100),
        "expected_offspec_pct_current": float(base_p.mean() * 100),
        "expected_offspec_pct_recommended": float(new_p.mean() * 100),
        "mill_speed_mean_current": float(res["mill_speed_mps"].mean()),
        "mill_speed_mean_recommended": float(np.where(feasible, res["rec_speed_mps"], res["mill_speed_mps"]).mean()),
        "per_grade": {},
    }
    summary["offspec_reduction_pct_points"] = summary["expected_offspec_pct_current"] - summary["expected_offspec_pct_recommended"]
    summary["mill_speed_gain_pct"] = (summary["mill_speed_mean_recommended"] / summary["mill_speed_mean_current"] - 1) * 100
    for g in cfg.GRADES:
        m = (res["grade"] == g).to_numpy()
        if m.any():
            summary["per_grade"][g] = {
                "n": int(m.sum()), "feasible_pct": float(feasible[m].mean() * 100),
                "expected_offspec_pct_current": float(base_p[m].mean() * 100),
                "expected_offspec_pct_recommended": float(new_p[m].mean() * 100),
            }

    windows = _windows(res[feasible])
    out = cfg.ROOT / "reports"
    out.mkdir(exist_ok=True)
    windows.to_csv(out / "operating_windows.csv", index=False)
    (out / "optimisation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(windows, summary, out / "operating_windows.md")
    _plots(summary, res, ys_bundle, uts_bundle, bounds, sigma_ys)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_grade"}, indent=2))
    return {"summary": summary, "windows": windows, "recommendations": res}


def _windows(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, d), sub in res.groupby(["grade", "bar_dia_mm"]):
        if len(sub) < 5:
            continue
        row = {"grade": g, "bar_dia_mm": int(d), "n_heats": len(sub),
               "ys_target_lo": cfg.GRADES[g]["min_ys"] + cfg.ERROR_BAND_MPA,
               "ys_target_hi": cfg.GRADES[g]["min_ys"] + cfg.YS_UPPER_MARGIN - cfg.ERROR_BAND_MPA}
        for name, col in (("pressure_bar", "rec_pressure_bar"), ("flow_m3h", "rec_flow_m3h"),
                          ("speed_mps", "rec_speed_mps")):
            row[f"{name}_p10"], row[f"{name}_set"], row[f"{name}_p90"] = sub[col].quantile([0.1, 0.5, 0.9]).round(2)
        row["current_speed_mps"] = round(sub["mill_speed_mps"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(w: pd.DataFrame, s: dict, path) -> None:
    lines = ["# Recommended operating windows", "",
             "Setpoint = median of per-heat optima, window = P10-P90. Derived from a model trained on "
             "**synthetic** data; validate on plant trials before use.", "",
             f"- Expected off-spec rate: {s['expected_offspec_pct_current']:.1f}% -> "
             f"{s['expected_offspec_pct_recommended']:.1f}% (model-based, held-out heats)",
             f"- Mean mill speed change: {s['mill_speed_gain_pct']:+.1f}%", "",
             "| Grade | Dia (mm) | n | Pressure bar (window / set) | Flow m3/h (window / set) | Speed m/s (window / set) | Current speed |",
             "|---|---|---|---|---|---|---|"]
    for _, r in w.iterrows():
        lines.append(
            f"| {r.grade} | {r.bar_dia_mm} | {r.n_heats} | {r.pressure_bar_p10}-{r.pressure_bar_p90} / {r.pressure_bar_set} "
            f"| {r.flow_m3h_p10}-{r.flow_m3h_p90} / {r.flow_m3h_set} "
            f"| {r.speed_mps_p10}-{r.speed_mps_p90} / {r.speed_mps_set} | {r.current_speed_mps} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(summary, res, ys_bundle, uts_bundle, bounds, sigma_ys) -> None:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    gs = list(summary["per_grade"])
    cur = [summary["per_grade"][g]["expected_offspec_pct_current"] for g in gs]
    rec = [summary["per_grade"][g]["expected_offspec_pct_recommended"] for g in gs]
    x = np.arange(len(gs))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, cur, 0.4, label="Current practice", color="#c0392b")
    ax.bar(x + 0.2, rec, 0.4, label="Recommended", color="#16a085")
    ax.set_xticks(x, gs), ax.set_ylabel("Expected off-spec (%)")
    ax.set_title("Model-based off-spec rate, held-out heats"), ax.legend()
    fig.tight_layout(), fig.savefig(cfg.FIG_DIR / "11_offspec_before_after.png", dpi=130), plt.close(fig)

    # YS surface over pressure x flow for a median Fe500D 16 mm heat at current speed
    sub = res[(res["grade"] == "Fe500D") & (res["bar_dia_mm"] == 16)]
    if len(sub):
        base = sub.iloc[[len(sub) // 2]]
        b = bounds.loc[16]
        ps = np.linspace(b["wb_pressure_bar_lo"], b["wb_pressure_bar_hi"], 25)
        fs = np.linspace(b["wb_flow_m3h_lo"], b["wb_flow_m3h_hi"], 25)
        P, F = np.meshgrid(ps, fs)
        rows = base.loc[base.index.repeat(P.size)]
        cand = _with_controls(rows, P.ravel(), F.ravel(), np.full(P.size, float(base["mill_speed_mps"].iloc[0])))
        ys, _ = predict_pair(cand, ys_bundle, uts_bundle)
        lo, hi = cfg.GRADES["Fe500D"]["min_ys"], cfg.GRADES["Fe500D"]["min_ys"] + cfg.YS_UPPER_MARGIN
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(P, F, ys.reshape(P.shape), levels=20, cmap="viridis")
        ax.contour(P, F, ys.reshape(P.shape), levels=[lo, hi], colors=["red", "orange"], linewidths=2)
        fig.colorbar(cs, label="Predicted YS (MPa)")
        ax.set_xlabel("Water-box pressure (bar)"), ax.set_ylabel("Water-box flow (m3/h)")
        ax.set_title("Fe500D, 16 mm: YS vs water box (red = min spec, orange = upper band)")
        fig.tight_layout(), fig.savefig(cfg.FIG_DIR / "12_ys_water_box_surface.png", dpi=130), plt.close(fig)


def recommend_for(heat: dict | pd.Series, ys_bundle=None, uts_bundle=None, bounds=None) -> dict:
    """Recommend water-box and mill-speed setpoints for one heat (used by the prototype app)."""
    ys_bundle = ys_bundle or load_model()
    if uts_bundle is None or bounds is None:
        df, _ = prepare(save=False)
        train, _ = time_split(df)
        uts_bundle = uts_bundle or load_or_train_uts(train)
        bounds = bounds if bounds is not None else control_bounds(train)
    row = pd.DataFrame([dict(heat)])
    row["ceq"] = row["c_pct"] + row["mn_pct"] / 6 + row["si_pct"] / 24 + row["v_pct"] / 14
    dia = float(row["bar_dia_mm"].iloc[0])
    nearest = bounds.index[np.abs(bounds.index - dia).argmin()]
    bnd = bounds.loc[[nearest]].rename(index={nearest: dia})
    rec = optimise_heats(add_features(row), ys_bundle, uts_bundle, bnd)
    return rec.iloc[0].to_dict()


if __name__ == "__main__":
    run()
