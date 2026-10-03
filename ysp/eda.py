"""Exploratory analysis: figures + a short markdown summary in reports/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .preprocess import add_features, clean, feature_columns, load_raw


def _save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / name, dpi=130)
    plt.close(fig)


def run() -> pd.DataFrame:
    raw = load_raw()
    df, report = clean(raw)
    df = add_features(df)
    t = cfg.TARGET

    # 1. missingness before cleaning
    miss = raw.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    miss.plot.barh(ax=ax, color="#c0392b")
    ax.set_xlabel("% missing (raw)")
    ax.set_title("Missing data by column")
    _save(fig, "01_missingness.png")

    # 2. YS by grade with spec limits
    grades = list(cfg.GRADES)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df.loc[df.grade == g, t] for g in grades], tick_labels=grades)
    for i, g in enumerate(grades, 1):
        ms = cfg.GRADES[g]["min_ys"]
        ax.hlines([ms, ms + cfg.YS_UPPER_MARGIN], i - 0.4, i + 0.4, colors=["r", "orange"], linestyles="--")
    ax.set_ylabel("Yield strength (MPa)")
    ax.set_title("YS by grade (red = min spec, orange = upper band)")
    _save(fig, "02_ys_by_grade.png")

    # 3. correlation with target (within-grade centred so grade chemistry doesn't dominate)
    cols = [c for c in feature_columns() if c != "grade_min_ys"]
    centred = df[cols + [t]] - df.groupby("grade")[cols + [t]].transform("mean")
    corr = centred.corr()[t].drop(t).sort_values()
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(corr.index, corr.values, color=np.where(corr.values > 0, "#2980b9", "#c0392b"))
    ax.set_title("Within-grade correlation with YS")
    _save(fig, "03_target_correlation.png")

    # 4. key driver scatter plots
    drivers = ["finishing_temp_c", "soaking_temp_c", "quench_index", "mill_speed_mps", "wb_pressure_bar", "ceq"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, c in zip(axes.ravel(), drivers):
        ax.scatter(df[c], df[t], s=3, alpha=0.3, c=df["grade_min_ys"], cmap="viridis")
        ax.set_xlabel(c)
        ax.set_ylabel("YS (MPa)")
    fig.suptitle("Key drivers vs YS (colour = grade)")
    _save(fig, "04_driver_scatter.png")

    # 5. feature collinearity
    fc = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(fc, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    fig.colorbar(im)
    ax.set_title("Feature correlation")
    _save(fig, "05_feature_correlation.png")

    # summary
    lines = ["# EDA summary", "", "## Cleaning report", ""]
    lines += [f"- {k}: {v}" for k, v in report.items()]
    lines += ["", "## Spec compliance in data (YS)", "", "| Grade | n | mean | std | below min % | above band % |", "|---|---|---|---|---|---|"]
    for g, s in cfg.GRADES.items():
        x = df.loc[df.grade == g, t]
        lines.append(
            f"| {g} | {len(x)} | {x.mean():.1f} | {x.std():.1f} | {(x < s['min_ys']).mean()*100:.1f} "
            f"| {(x > s['min_ys'] + cfg.YS_UPPER_MARGIN).mean()*100:.1f} |"
        )
    lines += ["", "## Top within-grade correlations with YS", ""]
    for c in corr.abs().sort_values(ascending=False).index[:8]:
        lines.append(f"- {c}: {corr[c]:+.2f}")
    (cfg.ROOT / "reports" / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


if __name__ == "__main__":
    run()
    print("EDA figures and reports/eda_summary.md written")
