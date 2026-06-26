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

    Prior bug: compute_elo_history and compute_features each independently called
    sort_values("date") with the default (unstable) sort.  For a tied-date block
    larger than ~16 rows, numpy/pandas' quicksort does NOT fall back to stable
    insertion sort, so the two sort passes could produce different orderings of the
    tied block.  That misaligns elo_home[i] (from compute_elo_history's ordering)
    to the wrong row in compute_features' iteration.  Fix: both calls use
    kind="stable" (mergesort).

    Design: 20 prior matches (distinct dates) establish divergent Elo for "Strong"
    (10 wins → elo > BASE) and "Weak" (10 losses → elo < BASE).  Then 40 matches
    on a single shared date are appended, with Strong's match first and Weak's last
    in the tied block.  The 40-row tied block exceeds the 16-element insertion-sort
    threshold, so an unstable sort CAN permute Strong and Weak's positions between
    the two sort calls, attaching the wrong elo_home to each row.  With stable sort
    both calls preserve relative order and the assertions hold.

    To verify sensitivity: temporarily revert both sort_values calls in elo.py and
    xgb.py to sort_values("date") (no kind= arg) and confirm this test FAILS.
    """
    from elo import BASE

    # --- Prior history: Strong wins 10 games, Weak loses 10; all on distinct dates ---
    prior_rows = []
    for i in range(10):
        month = f"{i + 1:02d}"
        prior_rows.append((f"2021-{month}-01", "Strong", f"Victim{i}",
                           3, 0, False, "Friendly", "Neutral"))
        prior_rows.append((f"2021-{month}-02", f"Beater{i}", "Weak",
                           3, 0, False, "Friendly", "Neutral"))
    # 20 prior rows → Strong's Elo well above BASE; Weak's well below BASE.

    # --- 40 matches all on the same date (beats the 16-element insertion-sort limit) ---
    # Strong's match is at index 0 of the tied block; Weak's is at index 39.
    # Stable sort preserves this order in both calls → correct Elo alignment.
    # Unstable quicksort on 40 equal-key rows can permute Strong and Weak's
    # positions between the two independent sort passes → Elo misalignment.
    SHARED = "2022-06-01"
    shared_rows = [
        (SHARED, "Strong", "FillerOpp_S", 1, 0, False, "Friendly", "Neutral"),
    ]
    for k in range(38):
        shared_rows.append(
            (SHARED, f"Filler{k:02d}", f"FillerOpp{k:02d}", 1, 1, False, "Friendly", "Neutral")
        )
    shared_rows.append(
        (SHARED, "Weak", "FillerOpp_W", 0, 1, False, "Friendly", "Neutral")
    )
    # 40 shared-date rows total.

    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "neutral", "tournament", "country"]
    df = pd.DataFrame(prior_rows + shared_rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    # 60 rows total; 40 share the same date.

    feat = xgb.compute_features(df)

    shared = feat[feat["date"] == pd.Timestamp(SHARED)]
    assert len(shared) == 40, f"Expected 40 shared-date rows, got {len(shared)}"

    strong_row = shared[shared["home_team"] == "Strong"]
    weak_row = shared[shared["home_team"] == "Weak"]
    assert len(strong_row) == 1, "Strong's shared-date row missing from features"
    assert len(weak_row) == 1, "Weak's shared-date row missing from features"

    # With correct (stable-sort) alignment:
    #   Strong (10 wins before shared date) → elo_home > BASE
    #   Weak   (10 losses before shared date) → elo_home < BASE
    # Under the old unstable-sort bug, the two sort passes produce different orderings
    # of the 40 tied rows, so Strong and Weak pick up each other's (or a filler's)
    # Elo value — making one or both of these assertions fail.
    assert strong_row.iloc[0]["elo_home"] > BASE, (
        f"Strong's elo_home {strong_row.iloc[0]['elo_home']:.1f} should be > BASE {BASE}: "
        "Elo likely misaligned between compute_elo_history and compute_features "
        "(unstable-sort bug not fixed)"
    )
    assert weak_row.iloc[0]["elo_home"] < BASE, (
        f"Weak's elo_home {weak_row.iloc[0]['elo_home']:.1f} should be < BASE {BASE}: "
        "Elo likely misaligned between compute_elo_history and compute_features "
        "(unstable-sort bug not fixed)"
    )
