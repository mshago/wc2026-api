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


def test_elo_alignment_stable_sort():
    """Regression test for C1: Elo features must be aligned to the correct match row.

    Design: build a frame where two matches occur on the SAME date (2021-06-01),
    but one team (Strong) has many prior wins and the other (Weak) has many prior
    losses, so their pre-match Elos diverge far from BASE before the shared date.
    If the internal sort in compute_elo_history uses an unstable sort, tied-date rows
    can be reordered relative to the outer sort in compute_features, causing
    elo_home[i] to belong to the OTHER same-date match — Strong's row would get
    Weak's (below-BASE) elo_home and vice versa.  With the stable-sort fix both
    sorts preserve the same insertion order for tied dates, so identities hold.

    The test would FAIL under the old bug when the pandas default (quicksort)
    happened to swap the two same-date rows between the two independent sort calls.
    With stable sort the relative order of equal-key rows is always preserved, so
    the Elo values stay attached to the correct match.
    """
    from elo import BASE

    # Give Strong 6 prior wins (all on distinct dates before the shared date)
    # and Weak 6 prior losses, so their Elo ratings diverge clearly from BASE.
    prior_rows = []
    for i in range(6):
        date = f"2021-0{i+1}-01"
        # Strong beats Dummy1 each month -> builds high Elo
        prior_rows.append((date, "Strong", "Dummy1", 2, 0, False, "Friendly", "Neutral"))
        # Weak loses to Dummy2 each month -> builds low Elo
        prior_rows.append((date, "Dummy2", "Weak",   2, 0, False, "Friendly", "Neutral"))

    # Two matches on the SAME shared date: Strong vs Dummy3, Weak vs Dummy4
    # Row order: Strong first, then Weak (both on "2021-07-01")
    shared_date = "2021-07-01"
    prior_rows.append((shared_date, "Strong", "Dummy3", 1, 0, False, "Friendly", "Neutral"))
    prior_rows.append((shared_date, "Weak",   "Dummy4", 0, 2, False, "Friendly", "Neutral"))

    df = pd.DataFrame(prior_rows,
                      columns=["date", "home_team", "away_team", "home_score",
                               "away_score", "neutral", "tournament", "country"])
    df["date"] = pd.to_datetime(df["date"])

    feat = xgb.compute_features(df)

    # Identify the two shared-date rows in the feature output
    shared = feat[feat["date"] == pd.Timestamp(shared_date)]
    assert len(shared) == 2, "Expected exactly 2 rows for the shared date"

    strong_row = shared[shared["home_team"] == "Strong"]
    weak_row   = shared[shared["home_team"] == "Weak"]

    assert len(strong_row) == 1, "Strong's shared-date row not found"
    assert len(weak_row)   == 1, "Weak's shared-date row not found"

    # With correct alignment, Strong (many wins) must have elo_home > BASE,
    # and Weak (many losses) must have elo_home < BASE.
    # Under the old unstable-sort bug the elo columns could be swapped between
    # the two same-date rows, making these assertions fail.
    assert strong_row.iloc[0]["elo_home"] > BASE, (
        f"Strong's elo_home {strong_row.iloc[0]['elo_home']:.1f} should be > BASE {BASE} "
        "(Elo misaligned — unstable-sort bug not fixed)"
    )
    assert weak_row.iloc[0]["elo_home"] < BASE, (
        f"Weak's elo_home {weak_row.iloc[0]['elo_home']:.1f} should be < BASE {BASE} "
        "(Elo misaligned — unstable-sort bug not fixed)"
    )
