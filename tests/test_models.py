import numpy as np

from ysp import config as cfg
from ysp.data_generator import generate
from ysp.models import make_models, metrics, model_columns, time_split
from ysp.preprocess import add_features, clean


def _df(n=1200):
    df, _ = clean(generate(n=n, seed=11))
    return add_features(df)


def test_time_split_has_no_heat_overlap_and_is_chronological():
    train, test = time_split(_df())
    assert set(train["heat_no"]).isdisjoint(test["heat_no"])
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_metrics_perfect_and_band():
    y = np.array([500.0, 520.0, 540.0])
    m = metrics(y, y)
    assert m["mae"] == 0 and m["r2"] == 1
    assert metrics(y, y + 30)[f"within_{cfg.ERROR_BAND_MPA}_mpa_pct"] == 0


def test_models_beat_mean_baseline():
    train, test = time_split(_df(2000))
    num, cat = model_columns()
    baseline = np.abs(test[cfg.TARGET] - train[cfg.TARGET].mean()).mean()
    pipe = make_models()["gradient_boosting"].fit(train[num + cat], train[cfg.TARGET])
    mae = np.abs(pipe.predict(test[num + cat]) - test[cfg.TARGET]).mean()
    assert mae < 0.5 * baseline
