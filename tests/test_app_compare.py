# tests/test_app_compare.py
from fastapi.testclient import TestClient
import app as APP
import compare as C


def _client(monkeypatch):
    monkeypatch.setattr(C, "DATA", {
        "fixtures": {"Mexico|South Korea": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3}},
        "scoreboard": {"holdout": "x", "n_matches": 3,
                       "models": {"bayesian": {"hit_rate": 0.5}, "xgboost": {"hit_rate": 0.55}}},
    })
    return TestClient(APP.app)


def test_compare_ok_shape(monkeypatch):
    r = _client(monkeypatch).get("/compare", params={"home": "Mexico", "away": "South Korea"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["xgboost"]) == {"home_win", "draw", "away_win"}
    assert set(body["bayesian"]) == {"home_win", "draw", "away_win"}
    assert abs(sum(body["xgboost"].values()) - 1.0) < 1e-6


def test_compare_non_wc_pair_404(monkeypatch):
    # both are valid teams, but the pair isn't in the precomputed fixtures
    r = _client(monkeypatch).get("/compare", params={"home": "Mexico", "away": "Brazil"})
    assert r.status_code == 404


def test_compare_unknown_team_404(monkeypatch):
    r = _client(monkeypatch).get("/compare", params={"home": "Atlantis", "away": "Mexico"})
    assert r.status_code == 404


def test_scoreboard_ok(monkeypatch):
    r = _client(monkeypatch).get("/compare/scoreboard")
    assert r.status_code == 200
    assert r.json()["models"]["xgboost"]["hit_rate"] == 0.55
