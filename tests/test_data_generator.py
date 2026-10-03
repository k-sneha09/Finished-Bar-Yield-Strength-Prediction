import numpy as np

from ysp import config as cfg
from ysp.data_generator import generate


def test_schema_and_size():
    df = generate(n=500, seed=1)
    assert len(df) == 500
    for col in cfg.NUMERIC_FEATURES + cfg.CATEGORICAL + [cfg.TARGET, cfg.SECONDARY_TARGET]:
        assert col in df.columns


def test_reproducible():
    a, b = generate(n=200, seed=7), generate(n=200, seed=7)
    assert a.equals(b)


def test_clean_mode_has_no_missing():
    df = generate(n=300, seed=3, messy=False)
    assert not df.isna().any().any()


def test_strength_in_realistic_range():
    df = generate(n=2000, seed=5, messy=False)
    assert 380 < df[cfg.TARGET].min() and df[cfg.TARGET].max() < 720
    assert (df["uts_mpa"] > df[cfg.TARGET]).all()


def test_stronger_quench_raises_strength():
    df = generate(n=6000, seed=9, messy=False)
    X = np.column_stack([df["wb_pressure_bar"], df["bar_dia_mm"], df["mill_speed_mps"],
                         df["finishing_temp_c"], df["ceq"], np.ones(len(df))])
    coef, *_ = np.linalg.lstsq(X, df[cfg.TARGET].to_numpy(), rcond=None)
    assert coef[0] > 1.0  # MPa per bar, holding other drivers fixed
