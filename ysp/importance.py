"""Ranking of RHF / mill / water-box parameters by influence on YS.

Uses a model trained on the *raw* process parameters (no engineered features) so that
importance is attributed to quantities operators actually set or measure, not to derived indices.
Two complementary views:
  * permutation importance on held-out heats (rise in MAE when a parameter is shuffled)
  * local sensitivity: mean change in predicted YS for a +1 std change of one parameter, others fixed
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as cfg
from .models import time_split
from .preprocess import prepare

PROCESS_GROUPS = {
    "RHF": cfg.RHF, "Mill": cfg.MILL, "Water box": cfg.WATER_BOX, "Chemistry": cfg.CHEMISTRY,
}
GROUP_OF = {c: g for g, cols in PROCESS_GROUPS.items() for c in cols}


def fit_raw_model(train: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    feats = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL
    prep = ColumnTransformer(
        [("num", "passthrough", cfg.NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.CATEGORICAL)]
    )
    pipe = Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=cfg.RANDOM_STATE))])
    pipe.fit(train[feats], train[cfg.TARGET])
    return pipe, feats


def permutation_ranking(pipe, test: pd.DataFrame, feats: list[str], n_repeats: int = 15) -> pd.DataFrame:
    res = permutation_importance(
        pipe, test[feats], test[cfg.TARGET], n_repeats=n_repeats, random_state=cfg.RANDOM_STATE,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    out = pd.DataFrame({"parameter": feats, "mae_increase_mpa": res.importances_mean,
                        "std": res.importances_std})
    out["group"] = out["parameter"].map(GROUP_OF).fillna("Grade")
    return out.sort_values("mae_increase_mpa", ascending=False).reset_index(drop=True)


def sensitivity(pipe, test: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Mean predicted dYS for +1 std of each numeric parameter (and per natural unit)."""
    base = pipe.predict(test[feats])
    rows = []
    for c in cfg.NUMERIC_FEATURES:
        sd = test[c].std()
        shifted = test[feats].copy()
        shifted[c] = shifted[c] + sd
        d = pipe.predict(shifted) - base
        rows.append({"parameter": c, "group": GROUP_OF[c], "std_unit": sd,
                     "dys_per_std_mpa": d.mean(), "dys_per_unit_mpa": d.mean() / sd})
    return pd.DataFrame(rows).sort_values("dys_per_std_mpa", key=np.abs, ascending=False).reset_index(drop=True)


def _plots(perm: pd.DataFrame, sens: pd.DataFrame, pipe, test, feats) -> None:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"RHF": "#e67e22", "Mill": "#2980b9", "Water box": "#16a085", "Chemistry": "#7f8c8d", "Grade": "#8e44ad"}
    top = perm.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["parameter"], top["mae_increase_mpa"], xerr=top["std"], color=[colors[g] for g in top["group"]])
    ax.set_xlabel("MAE increase when shuffled (MPa)")
    ax.set_title("Permutation importance (held-out heats)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=list(colors), loc="lower right")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "08_permutation_importance.png", dpi=130)
    plt.close(fig)

    s = sens.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(s["parameter"], s["dys_per_std_mpa"], color=np.where(s["dys_per_std_mpa"] > 0, "#2980b9", "#c0392b"))
    ax.set_xlabel("Change in predicted YS for +1 std (MPa)")
    ax.set_title("Sensitivity of YS to each parameter")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "09_sensitivity.png", dpi=130)
    plt.close(fig)

    ctrl = perm[perm["group"].isin(["RHF", "Mill", "Water box"])]["parameter"].head(6).tolist()
    fig, ax = plt.subplots(figsize=(12, 7))
    PartialDependenceDisplay.from_estimator(pipe, test[feats], ctrl, ax=ax, n_cols=3, kind="average",
                                            subsample=1500, random_state=cfg.RANDOM_STATE)
    fig.suptitle("Partial dependence of YS on top controllable parameters")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "10_partial_dependence.png", dpi=130)
    plt.close(fig)


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df, _ = prepare()
    train, test = time_split(df)
    pipe, feats = fit_raw_model(train)
    perm = permutation_ranking(pipe, test, feats)
    sens = sensitivity(pipe, test, feats)
    _plots(perm, sens, pipe, test, feats)

    ctrl = perm[perm["group"].isin(["RHF", "Mill", "Water box"])].reset_index(drop=True)
    ctrl.insert(0, "rank", ctrl.index + 1)
    ctrl = ctrl.merge(sens[["parameter", "dys_per_std_mpa", "dys_per_unit_mpa"]], on="parameter")
    out = cfg.ROOT / "reports"
    out.mkdir(exist_ok=True)
    perm.to_csv(out / "permutation_importance.csv", index=False)
    ctrl.to_csv(out / "parameter_ranking.csv", index=False)

    lines = ["# Parameter influence ranking (controllable RHF / mill / water-box parameters)", "",
             "| Rank | Parameter | Group | MAE increase if shuffled (MPa) | dYS per +1 std (MPa) | dYS per unit (MPa) |",
             "|---|---|---|---|---|---|"]
    for _, r in ctrl.iterrows():
        lines.append(f"| {r['rank']} | {r['parameter']} | {r['group']} | {r['mae_increase_mpa']:.2f} "
                     f"| {r['dys_per_std_mpa']:+.1f} | {r['dys_per_unit_mpa']:+.2f} |")
    lines += ["", "Chemistry and grade are ranked separately in `permutation_importance.csv`.", ""]
    (out / "parameter_ranking.md").write_text("\n".join(lines), encoding="utf-8")
    print(ctrl[["rank", "parameter", "group", "mae_increase_mpa", "dys_per_std_mpa"]].round(2).to_string(index=False))
    return {"permutation": perm, "sensitivity": sens, "controllable": ctrl}


if __name__ == "__main__":
    run()
