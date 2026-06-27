"""
World Football Elo — OFFLINE helper for training-time priors only.

Computed from the full results.csv history (back to 1872), Elo gives every team
a strength estimate that (a) covers all teams, (b) uses out-of-window history the
2021+ training window doesn't see, and (c) propagates strength transitively
(beating strong teams counts more). train.py turns these into informative priors
on attack/defense, which mainly helps the low-data teams whose flat priors are
otherwise uninformative. NOT used at serving time — predict.py never imports this.
"""
import numpy as np
import pandas as pd

BASE = 1500.0          # starting rating for an unseen team
HFA = 65.0             # home-field advantage in Elo points (0 at neutral venues)


def _k_base(tournament: str) -> float:
    """Tournament-importance weight on the update step (World Football Elo style)."""
    t = str(tournament)
    if t == "Friendly":
        return 20.0
    if "qualification" in t:
        return 40.0
    if t in ("FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations"):
        return 60.0
    return 35.0


def _g_mult(goal_diff: int) -> float:
    """Margin-of-victory multiplier (standard WFE formula)."""
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8.0


def compute_elo_history(df: pd.DataFrame):
    """Chronological Elo over played matches in `df`, also returning the
    PRE-match ratings for every match (leak-free feature input).

    Returns (final_elo: dict, pre: pd.DataFrame). `pre` has one row per played
    match in date order with columns date, home_team, away_team, elo_home,
    elo_away — the ratings BEFORE that match (unseen team -> BASE).
    """
    d = df[df.home_score.notna()].sort_values("date", kind="stable")
    hs = d.home_score.astype(int).to_numpy()
    as_ = d.away_score.astype(int).to_numpy()
    ht = d.home_team.to_numpy()
    at = d.away_team.to_numpy()
    neu = d.neutral.to_numpy()
    tour = d.tournament.to_numpy()
    dates = d.date.to_numpy()
    elo: dict = {}
    rows = []
    for i in range(len(d)):
        h, a = ht[i], at[i]
        rh = elo.get(h, BASE)
        ra = elo.get(a, BASE)
        rows.append((dates[i], h, a, rh, ra))   # pre-match snapshot
        adv = 0.0 if neu[i] else HFA
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - rh - adv) / 400.0))
        diff = int(hs[i] - as_[i])
        w = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
        delta = _k_base(tour[i]) * _g_mult(diff) * (w - exp_h)
        elo[h] = rh + delta
        elo[a] = ra - delta
    pre = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "elo_home", "elo_away"])
    return elo, pre


def compute_elo(df: pd.DataFrame) -> dict:
    """Chronological Elo over the played matches in `df`.

    `df` needs columns: date, home_team, away_team, home_score, away_score,
    neutral (bool), tournament. Returns {team: final_rating}. Pass a date-filtered
    frame in a backtest so ratings use only information available before the cutoff.
    """
    final, _ = compute_elo_history(df)
    return final


def z_scores(elo: dict, teams) -> np.ndarray:
    """Standardized (mean 0, std 1) Elo aligned to `teams`, for use as a prior mean.

    Teams missing from `elo` fall back to the population mean (z=0) so they simply
    get the uninformative flat prior rather than a degenerate value.
    """
    raw = np.array([elo.get(str(t), np.nan) for t in teams], dtype=float)
    mu = np.nanmean(raw)
    sd = np.nanstd(raw)
    raw = np.where(np.isfinite(raw), raw, mu)
    return (raw - mu) / sd
