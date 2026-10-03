"""Model training, grouped cross-validation and benchmarking against lab-tested YS."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .preprocess import feature_columns, prepare


def model_columns() -> tuple[list[str], list[str]]:
    return feature_columns(), cfg.CATEGORICAL


def make_models() -> dict[str, Pipeline]:
    num, cat = model_columns()
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    linear_prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", onehot, cat)])
    tree_prep = ColumnTransformer([("num", "passthrough", num), ("cat", onehot, cat)])
    return {
        "ridge": Pipeline([("prep", linear_prep), ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))]),
        "random_forest": Pipeline([
            ("prep", tree_prep),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                            random_state=cfg.RANDOM_STATE)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", tree_prep),
            ("model", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=cfg.RANDOM_STATE)),
        ]),
    }


def time_split(df: pd.DataFrame, frac: float = cfg.HOLDOUT_FRACTION):
    """Chronological split on heats so no heat straddles train/test (no leakage via heat chemistry)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    heats = df["heat_no"].drop_duplicates().tolist()
    cut = set(heats[int(len(heats) * (1 - frac)):])
    test = df["heat_no"].isin(cut)
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)


def metrics(y_true, y_pred, grades=None, band: float = cfg.ERROR_BAND_MPA) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        f"within_{band}_mpa_pct": float((np.abs(err) <= band).mean() * 100),
        "bias": float(err.mean()),
    }
    if grades is not None:
        # off-spec detection: can the model flag bars that fail min YS?
        min_ys = np.array([cfg.GRADES[g]["min_ys"] for g in grades])
        actual, pred = np.asarray(y_true) < min_ys, np.asarray(y_pred) < min_ys
        tp = int((actual & pred).sum())
        out["offspec_recall"] = tp / max(int(actual.sum()), 1)
        out["offspec_precision"] = tp / max(int(pred.sum()), 1)
        out["offspec_actual_n"] = int(actual.sum())
    return out


def train_and_benchmark(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        df, _ = prepare()
    train, test = time_split(df)
    num, cat = model_columns()
    X_tr, y_tr, X_te, y_te = train[num + cat], train[cfg.TARGET], test[num + cat], test[cfg.TARGET]

    results, fitted = {}, {}
    gkf = GroupKFold(n_splits=5)
    for name, pipe in make_models().items():
        cv_pred = cross_val_predict(pipe, X_tr, y_tr, cv=gkf, groups=train["heat_no"])
        cv = metrics(y_tr, cv_pred)
        pipe.fit(X_tr, y_tr)
        te = metrics(y_te, pipe.predict(X_te), test["grade"])
        results[name] = {"cv": cv, "holdout": te}
        fitted[name] = pipe
        print(f"{name:18s} CV MAE {cv['mae']:.2f} | holdout MAE {te['mae']:.2f} RMSE {te['rmse']:.2f} "
              f"R2 {te['r2']:.3f} within±{cfg.ERROR_BAND_MPA}: {te[f'within_{cfg.ERROR_BAND_MPA}_mpa_pct']:.1f}%")

    best = min(results, key=lambda k: results[k]["cv"]["mae"])
    pred = fitted[best].predict(X_te)
    per_grade = {g: metrics(y_te[m], pred[m]) for g in cfg.GRADES if (m := (test["grade"] == g).to_numpy()).any()}
    summary = {
        "best_model": best, "n_train": len(train), "n_holdout": len(test),
        "error_band_mpa": cfg.ERROR_BAND_MPA, "results": results, "best_holdout_per_grade": per_grade,
    }
    if save:
        cfg.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": fitted[best], "features": num + cat, "name": best}, cfg.MODEL_DIR / "ys_model.joblib")
        (cfg.ROOT / "reports").mkdir(exist_ok=True)
        (cfg.ROOT / "reports" / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _plots(test, pred, best)
    print(f"best model (by CV MAE): {best}")
    return summary


def _plots(test: pd.DataFrame, pred: np.ndarray, name: str) -> None:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    y = test[cfg.TARGET].to_numpy()
    b = cfg.ERROR_BAND_MPA
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, s=6, alpha=0.4)
    lim = [y.min() - 10, y.max() + 10]
    ax.plot(lim, lim, "k-", lw=1)
    ax.fill_between(lim, [l - b for l in lim], [l + b for l in lim], color="green", alpha=0.12, label=f"±{b} MPa band")
    ax.set_xlabel("Lab YS (MPa)"), ax.set_ylabel("Predicted YS (MPa)")
    ax.set_title(f"Holdout: predicted vs lab ({name})")
    ax.legend()
    fig.tight_layout(), fig.savefig(cfg.FIG_DIR / "06_pred_vs_lab.png", dpi=130), plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    err = pred - y
    axes[0].hist(err, bins=50, color="#2980b9")
    axes[0].axvline(-b, c="g", ls="--"), axes[0].axvline(b, c="g", ls="--")
    axes[0].set_title("Residuals (pred - lab)"), axes[0].set_xlabel("MPa")
    grades = [g for g in cfg.GRADES if (test["grade"] == g).any()]
    axes[1].boxplot([err[(test["grade"] == g).to_numpy()] for g in grades], tick_labels=grades)
    axes[1].axhline(0, c="k", lw=0.8), axes[1].set_title("Residuals by grade")
    fig.tight_layout(), fig.savefig(cfg.FIG_DIR / "07_residuals.png", dpi=130), plt.close(fig)


def load_model(path=None) -> dict:
    return joblib.load(path or cfg.MODEL_DIR / "ys_model.joblib")


def predict(df: pd.DataFrame, bundle: dict | None = None) -> np.ndarray:
    bundle = bundle or load_model()
    return bundle["model"].predict(df[bundle["features"]])


if __name__ == "__main__":
    train_and_benchmark()
