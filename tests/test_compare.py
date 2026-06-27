import json
import pytest
import compare


_ART = {"fixtures": {"Mexico|South Korea": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3}},
        "scoreboard": {"holdout": "x", "n_matches": 3,
                       "models": {"bayesian": {}, "xgboost": {}}}}


def test_fixture_probs_returns_pair():
    assert compare.fixture_probs(_ART, "Mexico", "South Korea")["home_win"] == 0.5


def test_fixture_probs_missing_raises_keyerror():
    with pytest.raises(KeyError):
        compare.fixture_probs(_ART, "Mexico", "Brazil")


def test_scoreboard_passthrough():
    assert compare.scoreboard(_ART)["n_matches"] == 3


def test_load_missing_file_is_empty_valid(tmp_path):
    d = compare.load(tmp_path / "nope.json")
    assert d["fixtures"] == {} and "models" in d["scoreboard"]


def test_load_reads_file(tmp_path):
    p = tmp_path / "a.json"; p.write_text(json.dumps(_ART))
    assert compare.load(p)["fixtures"]["Mexico|South Korea"]["draw"] == 0.2
