"""Prototype: real-time YS prediction and setpoint recommendation (ISP Burnpur, RHF + mill)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ysp import config as cfg  # noqa: E402
from ysp.models import load_model, time_split  # noqa: E402
from ysp.optimize import control_bounds, load_or_train_uts, predict_pair, recommend_for  # noqa: E402
from ysp.predict import assess, build_row  # noqa: E402
from ysp.preprocess import prepare  # noqa: E402

st.set_page_config(page_title="Finished Bar YS Predictor", layout="wide")


@st.cache_resource
def load_assets():
    df, _ = prepare(save=False)
    train, _ = time_split(df)
    return df, load_model(), load_or_train_uts(train), control_bounds(train)


df, ys_bundle, uts_bundle, bounds = load_assets()

st.title("Finished Bar Yield Strength Predictor")
st.caption("IISCO Steel Plant, Burnpur | RHF + rolling mill + water box. "
           "Model trained on synthetic stand-in data until plant data is supplied.")

# ---- sidebar: heat inputs, defaulted to the typical heat for the chosen grade and bar size
with st.sidebar:
    st.header("Heat inputs")
    grade = st.selectbox("Grade", list(cfg.GRADES), index=1)
    dia = st.selectbox("Bar diameter (mm)", sorted(df["bar_dia_mm"].unique().astype(int)), index=3)
    ref = df[(df["grade"] == grade) & (df["bar_dia_mm"] == dia)]
    ref = ref if len(ref) else df[df["grade"] == grade]
    med = ref.median(numeric_only=True)

    def num(col, step, fmt="%.2f"):
        lo, hi = cfg.VALID_RANGES[col]
        return st.number_input(col, float(lo), float(hi), float(round(med[col], 3)), step, format=fmt,
                               key=f"{col}_{grade}_{dia}")

    with st.expander("Chemistry (wt %)"):
        chem = {c: num(c, 0.005, "%.3f") for c in ["c_pct", "mn_pct", "si_pct", "s_pct", "p_pct", "v_pct", "nb_pct"]}
    with st.expander("Reheating furnace"):
        rhf = {c: num(c, 1.0, "%.1f") for c in [
            "preheat_zone_temp_c", "heating_zone_temp_c", "soaking_zone_temp_c", "soaking_temp_c",
            "rhf_residence_min", "discharge_temp_c"]}
        billet = st.selectbox("Billet size (mm)", [125, 130, 150], index=1)
    with st.expander("Rolling mill", expanded=True):
        mill = {"finishing_temp_c": num("finishing_temp_c", 1.0, "%.1f"),
                "ambient_temp_c": num("ambient_temp_c", 0.5, "%.1f")}
        speed = num("mill_speed_mps", 0.1, "%.2f")
    with st.expander("Water box", expanded=True):
        press = num("wb_pressure_bar", 0.1, "%.2f")
        flow = num("wb_flow_m3h", 1.0, "%.1f")
        wtemp = num("wb_water_temp_c", 0.5, "%.1f")

heat = {"grade": grade, "bar_dia_mm": float(dia), "billet_size_mm": billet, **chem, **rhf, **mill,
        "mill_speed_mps": speed, "wb_pressure_bar": press, "wb_flow_m3h": flow, "wb_water_temp_c": wtemp,
        "wb_valve_open_pct": float(np.clip(55 + 4 * (press - 8), 20, 100))}

tab_pred, tab_rec, tab_whatif, tab_model = st.tabs(
    ["Prediction", "Recommended setpoints", "What-if sweep", "Model & windows"])

# ---- prediction
res = assess(heat, ys_bundle, uts_bundle)
with tab_pred:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted YS (MPa)", f"{res['ys']:.0f}", f"±{cfg.ERROR_BAND_MPA} band", delta_color="off")
    c2.metric("Predicted UTS (MPa)", f"{res['uts']:.0f}")
    c3.metric("UTS/YS", f"{res['ratio']:.3f}", f"min {res['min_ratio']:.2f}", delta_color="off")
    c4.metric("Status", res["status"])
    st.write(f"Spec for **{grade}**: YS {res['min_ys']:.0f}-{res['upper_ys']:.0f} MPa, "
             f"UTS/YS >= {res['min_ratio']:.2f}. Prediction band: "
             f"{res['ys'] - cfg.ERROR_BAND_MPA:.0f}-{res['ys'] + cfg.ERROR_BAND_MPA:.0f} MPa.")
    if res["status"] == "OK":
        st.success("Predicted to meet spec with margin.")
    else:
        st.warning("Check the Recommended setpoints tab for adjustments.")

# ---- recommendation
with tab_rec:
    st.write("Searches water-box pressure, flow and mill speed within ranges the line has run for this bar "
             "size, keeping chemistry, furnace and finishing temperatures as entered.")
    if st.button("Recommend setpoints", type="primary"):
        rec = recommend_for(heat, ys_bundle, uts_bundle, bounds)
        if pd.isna(rec.get("rec_speed_mps")):
            st.error("No setting in the observed range meets spec with margin for this heat. "
                     "Review chemistry / finishing temperature (RHF and mill) first.")
        else:
            a, b, c = st.columns(3)
            a.metric("Water-box pressure (bar)", f"{rec['rec_pressure_bar']:.2f}",
                     f"{rec['rec_pressure_bar'] - press:+.2f}", delta_color="off")
            b.metric("Water-box flow (m3/h)", f"{rec['rec_flow_m3h']:.0f}",
                     f"{rec['rec_flow_m3h'] - flow:+.0f}", delta_color="off")
            c.metric("Mill speed (m/s)", f"{rec['rec_speed_mps']:.2f}",
                     f"{rec['rec_speed_mps'] - speed:+.2f}", delta_color="off")
            st.write(f"Predicted with these settings: YS **{rec['rec_ys_pred']:.0f}** MPa, "
                     f"UTS **{rec['rec_uts_pred']:.0f}** MPa.")

# ---- what-if
with tab_whatif:
    knob = st.selectbox("Parameter to sweep", ["wb_pressure_bar", "wb_flow_m3h", "mill_speed_mps",
                                                "finishing_temp_c", "soaking_temp_c"])
    lo, hi = df[knob].quantile([0.02, 0.98])
    xs = np.linspace(lo, hi, 40)
    sweep = pd.concat([pd.DataFrame([heat])] * len(xs), ignore_index=True)
    sweep[knob] = xs
    if knob == "wb_pressure_bar":
        sweep["wb_valve_open_pct"] = np.clip(55 + 4 * (xs - 8), 20, 100)
    rows = pd.concat([build_row(r.to_dict()) for _, r in sweep.iterrows()], ignore_index=True)
    ys, _ = predict_pair(rows, ys_bundle, uts_bundle)
    chart = pd.DataFrame({knob: xs, "Predicted YS": ys, "Min YS": res["min_ys"], "Upper band": res["upper_ys"]})
    st.line_chart(chart, x=knob, y=["Predicted YS", "Min YS", "Upper band"])

# ---- model and windows
with tab_model:
    mpath = cfg.ROOT / "reports" / "model_metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        h = m["results"][m["best_model"]]["holdout"]
        band = m["error_band_mpa"]
        st.subheader(f"Held-out benchmark vs lab ({m['best_model']})")
        a, b, c, d = st.columns(4)
        a.metric("MAE (MPa)", f"{h['mae']:.1f}")
        b.metric("RMSE (MPa)", f"{h['rmse']:.1f}")
        c.metric("R2", f"{h['r2']:.3f}")
        d.metric(f"Within ±{band} MPa", f"{h['within_%s_mpa_pct' % band]:.1f}%")
    for name in ("06_pred_vs_lab.png", "08_permutation_importance.png"):
        f = cfg.FIG_DIR / name
        if f.exists():
            st.image(str(f))
    wpath = cfg.ROOT / "reports" / "operating_windows.csv"
    if wpath.exists():
        st.subheader("Recommended operating windows")
        w = pd.read_csv(wpath)
        st.dataframe(w[w["grade"] == grade], width="stretch")
