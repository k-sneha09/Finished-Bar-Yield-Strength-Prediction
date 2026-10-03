from pathlib import Path

import pytest

from ysp import config as cfg
from ysp.predict import assess

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(not (cfg.MODEL_DIR / "ys_model.joblib").exists(),
                                 reason="run `python -m ysp.models` first")


def _typical(grade="Fe500D"):
    import pandas as pd
    df = pd.read_csv(cfg.PROCESSED_DIR / "clean_features.csv")
    return df[(df.grade == grade) & (df.bar_dia_mm == 16)].median(numeric_only=True).to_dict() | {"grade": grade}


@needs_model
def test_assess_typical_heat_is_sensible():
    heat = _typical()
    out = assess(heat)
    assert 450 < out["ys"] < 650 and out["uts"] > out["ys"]
    assert out["status"] in {"OK", "MARGINAL (close to min YS)", "LOW UTS/YS", "BELOW MIN YS", "ABOVE UPPER BAND"}


@needs_model
def test_assess_flags_weak_heat():
    heat = _typical() | {"wb_pressure_bar": 2.5, "wb_flow_m3h": 110.0, "finishing_temp_c": 1060.0}
    assert assess(heat)["status"] != "OK"


@needs_model
def test_streamlit_app_runs_headless():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
