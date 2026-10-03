import numpy as np

from ysp import config as cfg
from ysp.data_generator import generate
from ysp.models import make_models, model_columns, time_split
from ysp.optimize import (control_bounds, expected_offspec, optimise_heats, predict_pair,
                          spec_limits, train_uts_model)
from ysp.preprocess import add_features, clean


def _setup():
    df, _ = clean(generate(n=2500, seed=31))
    df = add_features(df)
    train, test = time_split(df)
    num, cat = model_columns()
    ys = {"model": make_models()["gradient_boosting"].fit(train[num + cat], train[cfg.TARGET]),
          "features": num + cat}
    return train, test, ys, train_uts_model(train)


def test_recommendations_are_feasible_and_within_bounds():
    train, test, ys, uts = _setup()
    sample = test.head(40)
    bounds = control_bounds(train)
    rec = optimise_heats(sample, ys, uts, bounds, n=4)
    ok = rec["rec_speed_mps"].notna()
    assert ok.mean() > 0.5
    r = rec[ok].join(sample.reset_index(drop=True)[["grade", "bar_dia_mm"]])
    lo, hi, _ = spec_limits(r["grade"])
    assert (r["rec_ys_pred"] >= lo + cfg.ERROR_BAND_MPA - 1e-6).all()
    assert (r["rec_ys_pred"] <= hi - cfg.ERROR_BAND_MPA + 1e-6).all()
    for _, row in r.iterrows():
        b = bounds.loc[row["bar_dia_mm"]]
        assert b["mill_speed_mps_lo"] - 1e-6 <= row["rec_speed_mps"] <= b["mill_speed_mps_hi"] + 1e-6


def test_expected_offspec_behaviour():
    import pandas as pd
    g = pd.Series(["Fe500D", "Fe500D"])
    p = expected_offspec(np.array([545.0, 400.0]), np.array([545 * 1.15, 400 * 1.15]), g, 8.5, 0.011)
    assert p[0] < 0.01 and p[1] > 0.99
