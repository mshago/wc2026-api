import numpy as np
import pandas as pd
import xgb


def _df(n=120):
    # synthetic: "Strong" beats "Weak" most of the time so the model has signal
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, a = ("Strong", "Weak") if i % 2 == 0 else ("Weak", "Strong")
        strong_home = h == "Strong"
        hg = rng.integers(2, 5) if strong_home else rng.integers(0, 2)
        ag = rng.integers(0, 2) if strong_home else rng.integers(2, 5)
        rows.append((f"2021-{(i % 11) + 1:02d}-01", h, a, hg, ag, True, "Friendly", "Qatar"))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score",
                                       "away_score", "neutral", "tournament", "country"]
                        ).assign(date=lambda d: pd.to_datetime(d.date))


def test_predict_proba_sums_to_one():
    df = _df()
    model = xgb.train_xgb(xgb.compute_features(df))
    p = xgb.predict_proba(model, xgb.fixture_features(df, "Strong", "Weak"))
    assert set(p) == {"home_win", "draw", "away_win"}
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_predict_fixtures_covers_directed_pairs():
    df = _df()
    model = xgb.train_xgb(xgb.compute_features(df))
    fx = xgb.predict_fixtures(model, df, ["Strong", "Weak"])
    assert "Strong|Weak" in fx and "Weak|Strong" in fx
    assert "Strong|Strong" not in fx


def test_assemble_artifact_schema():
    sb = {"holdout": "x", "n_matches": 1, "models": {"bayesian": {}, "xgboost": {}}}
    art = xgb.assemble_artifact({"Strong|Weak": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3}}, sb)
    assert art["neutral"] is True
    assert "generated" in art
    assert art["scoreboard"] == sb
    assert "Strong|Weak" in art["fixtures"]
