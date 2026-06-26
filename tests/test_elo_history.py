import pandas as pd
import elo


def _df():
    return pd.DataFrame([
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "home_score": 2, "away_score": 0, "neutral": False, "tournament": "Friendly"},
        {"date": "2020-02-01", "home_team": "A", "away_team": "C",
         "home_score": 1, "away_score": 1, "neutral": True, "tournament": "Friendly"},
    ]).assign(date=lambda d: pd.to_datetime(d.date))


def test_first_match_pre_ratings_are_base():
    final, pre = elo.compute_elo_history(_df())
    assert list(pre.columns) == ["date", "home_team", "away_team", "elo_home", "elo_away"]
    assert pre.iloc[0]["elo_home"] == elo.BASE
    assert pre.iloc[0]["elo_away"] == elo.BASE


def test_pre_ratings_are_leak_free_and_update():
    final, pre = elo.compute_elo_history(_df())
    # A won match 1, so A's pre-rating for match 2 must be above BASE (no leak from match 2).
    assert pre.iloc[1]["home_team"] == "A"
    assert pre.iloc[1]["elo_home"] > elo.BASE


def test_compute_elo_matches_history_final():
    final_a = elo.compute_elo(_df())
    final_b, _ = elo.compute_elo_history(_df())
    assert final_a == final_b
