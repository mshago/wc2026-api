"""
OFFLINE XGBoost comparison model — never runs in production.
Trains a 1X2 outcome model on leak-free features and writes a committed static
artifact (model/xgb_compare.json) that the API serves. Uses training deps;
run in the same venv as train.py:

    pip install -r requirements-train.txt
    python xgb.py
"""
import numpy as np
import pandas as pd
import elo as ELO
import geo

FORM_K = 10  # rolling window: each team's last K matches feed the form features

FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "form_gf_home", "form_ga_home", "form_win_home",
    "form_gf_away", "form_ga_away", "form_win_away",
    "rest_home", "rest_away", "support", "k_imp",
]


def _avg(seq):
    return float(np.mean(seq)) if seq else np.nan


def _support(home, away, country, neutral):
    s = geo.support(home, away, country)
    if s is None:
        s = 0.0 if neutral else 1.0
    return float(s)


def _new_state():
    # per-team rolling history; elo comes from elo.compute_elo_history (pre-match)
    return {"gf": {}, "ga": {}, "win": {}, "last": {}}


def _state_features(st, team, elo_val):
    return {
        "elo": elo_val,
        "gf": _avg(st["gf"].get(team)),
        "ga": _avg(st["ga"].get(team)),
        "win": _avg(st["win"].get(team)),
        "last": st["last"].get(team),
    }


def _update_state(st, team, gf, ga, result, date):
    for key, val in (("gf", gf), ("ga", ga), ("win", result)):
        st[key].setdefault(team, [])
        st[key][team].append(val)
        st[key][team] = st[key][team][-FORM_K:]
    st["last"][team] = date


def _rest(prev_date, date):
    return float((date - prev_date).days) if prev_date is not None else np.nan


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """One leak-free feature row per played match (date order). See FEATURES."""
    d = df[df.home_score.notna()].sort_values("date").reset_index(drop=True)
    _, pre = ELO.compute_elo_history(d)
    elo_home = pre["elo_home"].to_numpy()
    elo_away = pre["elo_away"].to_numpy()
    st = _new_state()
    rows = []
    for i, r in enumerate(d.itertuples(index=False)):
        eh, ea = float(elo_home[i]), float(elo_away[i])
        sh = _state_features(st, r.home_team, eh)
        sa = _state_features(st, r.away_team, ea)
        y = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append({
            "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
            "form_gf_home": sh["gf"], "form_ga_home": sh["ga"], "form_win_home": sh["win"],
            "form_gf_away": sa["gf"], "form_ga_away": sa["ga"], "form_win_away": sa["win"],
            "rest_home": _rest(sh["last"], r.date), "rest_away": _rest(sa["last"], r.date),
            "support": _support(r.home_team, r.away_team, r.country, r.neutral),
            "k_imp": ELO._k_base(r.tournament),
            "y": y, "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
        })
        hw = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        _update_state(st, r.home_team, r.home_score, r.away_score, hw, r.date)
        _update_state(st, r.away_team, r.away_score, r.home_score, 1.0 - hw, r.date)
    return pd.DataFrame(rows)


def fixture_features(df: pd.DataFrame, home: str, away: str) -> dict:
    """Latest-state neutral-venue feature dict for an unplayed fixture."""
    d = df[df.home_score.notna()].sort_values("date").reset_index(drop=True)
    final_elo, _ = ELO.compute_elo_history(d)
    st = _new_state()
    for r in d.itertuples(index=False):
        hw = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        _update_state(st, r.home_team, r.home_score, r.away_score, hw, r.date)
        _update_state(st, r.away_team, r.away_score, r.home_score, 1.0 - hw, r.date)
    eh = final_elo.get(home, ELO.BASE); ea = final_elo.get(away, ELO.BASE)
    sh = _state_features(st, home, eh); sa = _state_features(st, away, ea)
    last = d["date"].max()
    return {
        "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
        "form_gf_home": sh["gf"], "form_ga_home": sh["ga"], "form_win_home": sh["win"],
        "form_gf_away": sa["gf"], "form_ga_away": sa["ga"], "form_win_away": sa["win"],
        "rest_home": _rest(sh["last"], last), "rest_away": _rest(sa["last"], last),
        "support": 0.0, "k_imp": ELO._k_base("FIFA World Cup"),
    }
