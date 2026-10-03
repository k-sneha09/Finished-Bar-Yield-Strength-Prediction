from ysp import config as cfg
from ysp.data_generator import generate
from ysp.importance import fit_raw_model, permutation_ranking, sensitivity
from ysp.models import time_split
from ysp.preprocess import add_features, clean


def test_ranking_finds_known_drivers():
    df, _ = clean(generate(n=3000, seed=21))
    train, test = time_split(add_features(df))
    pipe, feats = fit_raw_model(train)
    perm = permutation_ranking(pipe, test, feats, n_repeats=3)
    assert {"finishing_temp_c", "bar_dia_mm"} & set(perm["parameter"].head(4))
    # irrelevant sensor should rank near the bottom
    assert perm.set_index("parameter").loc["preheat_zone_temp_c", "mae_increase_mpa"] < 1.0


def test_sensitivity_signs():
    df, _ = clean(generate(n=3000, seed=21))
    train, test = time_split(add_features(df))
    pipe, feats = fit_raw_model(train)
    s = sensitivity(pipe, test, feats).set_index("parameter")
    assert s.loc["finishing_temp_c", "dys_per_std_mpa"] < 0   # hotter finish -> softer
    assert s.loc["wb_pressure_bar", "dys_per_std_mpa"] > 0    # harder quench -> stronger
