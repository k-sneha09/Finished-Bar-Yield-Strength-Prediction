import numpy as np

from ysp import config as cfg
from ysp.data_generator import generate
from ysp.preprocess import add_features, clean, feature_columns


def test_clean_removes_all_nans_and_missing_targets():
    df, rep = clean(generate(n=1500, seed=2))
    assert not df[cfg.NUMERIC_FEATURES + [cfg.TARGET]].isna().any().any()
    assert rep["rows_out"] == len(df)


def test_stuck_zero_and_spike_are_handled():
    raw = generate(n=500, seed=4, messy=False)
    raw.loc[0, "wb_pressure_bar"] = 0.0
    raw.loc[1, "finishing_temp_c"] = 300.0
    df, _ = clean(raw)
    assert (df["wb_pressure_bar"] > 0).all()
    assert df["finishing_temp_c"].between(*cfg.VALID_RANGES["finishing_temp_c"]).all()


def test_features_finite_and_complete():
    df, _ = clean(generate(n=800, seed=6))
    df = add_features(df)
    X = df[feature_columns()]
    assert X.shape[1] == len(cfg.NUMERIC_FEATURES) + len(cfg.ENGINEERED)
    assert np.isfinite(X.to_numpy()).all()


def test_quench_index_decreases_with_speed():
    df, _ = clean(generate(n=300, seed=8, messy=False))
    fast = add_features(df.assign(mill_speed_mps=df["mill_speed_mps"] * 1.5))
    base = add_features(df)
    assert (fast["quench_index"] < base["quench_index"]).all()
