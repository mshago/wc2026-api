# tests/test_xgb_features.py
import numpy as np
import pandas as pd
import xgb


def _df():
    rows = [
        ("2021-01-01", "A", "B", 2, 0, False, "Friendly", "A"),
        ("2021-02-01", "B", "A", 1, 1, False, "Friendly", "B"),
        ("2021-03-01", "A", "C", 3, 1, True, "FIFA World Cup", "Qatar"),
    ]
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score",
                                       "away_score", "neutral", "tournament", "country"]
                        ).assign(date=lambda d: pd.to_datetime(d.date))


def test_feature_columns_and_labels():
    f = xgb.compute_features(_df())
    assert set(xgb.FEATURES).issubset(f.columns)
    assert list(f["y"]) == [0, 1, 0]          # A win, draw, A win
    assert len(f) == 3


def test_first_match_has_no_form_history():
    f = xgb.compute_features(_df())
    # first match: neither side has prior games -> form/rest are NaN, elo == BASE-derived diff 0
    assert np.isnan(f.iloc[0]["form_gf_home"])
    assert f.iloc[0]["elo_diff"] == 0.0


def test_features_are_leak_free():
    f = xgb.compute_features(_df())
    # A scored 2 then drew 1: by match 3 A's recent goals-for avg uses matches 1-2 only (=1.5),
    # never match 3's 3 goals.
    assert f.iloc[2]["home_team"] == "A"
    assert f.iloc[2]["form_gf_home"] == 1.5


def test_fixture_features_returns_feature_dict():
    fx = xgb.fixture_features(_df(), "A", "B")
    assert set(fx.keys()) == set(xgb.FEATURES)
