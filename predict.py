"""
Runtime predictor — pure NumPy/SciPy, no PyMC.
Loads the posterior draws produced offline by train.py and computes
posterior-predictive scoreline probabilities (with the Dixon-Coles
low-score correction baked in).
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import poisson
import geo

MAXG = 10
_PATH = Path(__file__).parent / "model" / "wc2026_poisson_dc_posterior.npz"
_d = np.load(_PATH, allow_pickle=True)
TEAMS = sorted(map(str, _d["teams"]))
_idx = {str(t): i for i, t in enumerate(_d["teams"])}
_intc, _hom, _atk, _dfn, _rho = _d["intc"], _d["homv"], _d["atk"], _d["dfn"], _d["rhov"]

_VERSION_PATH = Path(__file__).parent / "model" / "version.json"


def _load_version():
    """Model build metadata written by train.py; '{}' if not present yet
    (e.g. a deploy made before the first automated refresh)."""
    try:
        with open(_VERSION_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# build metadata; `latest_match_date` is the frontend's cache key
VERSION = _load_version()

# score-matrix index helpers (home goals = row i, away goals = col j)
_I, _J = np.indices((MAXG + 1, MAXG + 1))
_HOME_WIN, _DRAW, _AWAY_WIN = _I > _J, _I == _J, _I < _J
_TOTAL = _I + _J
EXTRAS = {"markets", "margin", "uncertainty"}


def _ci(arr, nd=4):
    """mean + 90% credible interval of a per-draw quantity."""
    return {"mean": round(float(arr.mean()), nd),
            "p05": round(float(np.percentile(arr, 5)), nd),
            "p95": round(float(np.percentile(arr, 95)), nd)}


def predict(home: str, away: str, neutral: bool = True, venue: str = None,
            top_n: int = 10, matrix_size: int = 7, extras=None):
    if home not in _idx:
        raise KeyError(home)
    if away not in _idx:
        raise KeyError(away)
    hh, aa = _idx[home], _idx[away]
    extras = set(extras or [])

    # crowd-support in [-1, +1]; from venue geography when a known `venue`
    # is given, else the binary fallback (neutral -> 0, home game -> 1).
    s = geo.support(home, away, venue) if venue else None
    if s is None:
        s = 0.0 if neutral else 1.0

    # posterior expected goals per draw
    lh = np.exp(_intc + _hom * s + _atk[:, hh] - _dfn[:, aa])
    la = np.exp(_intc + _atk[:, aa] - _dfn[:, hh])

    g = np.arange(MAXG + 1)
    ph = poisson.pmf(g[None, :], lh[:, None])
    pa = poisson.pmf(g[None, :], la[:, None])
    j = np.einsum("si,sj->sij", ph, pa)                       # (S, G+1, G+1)
    # Dixon-Coles low-score correction, per draw
    j[:, 0, 0] *= (1 - lh * la * _rho); j[:, 0, 1] *= (1 + lh * _rho)
    j[:, 1, 0] *= (1 + la * _rho);      j[:, 1, 1] *= (1 - _rho)
    j = np.clip(j, 0, None)
    mat = j.mean(0); mat /= mat.sum()                         # average over posterior

    p_home = float(mat[_HOME_WIN].sum())
    p_draw = float(mat[_DRAW].sum())
    p_away = float(mat[_AWAY_WIN].sum())

    flat = sorted(((i, k, float(mat[i, k])) for i in range(MAXG + 1) for k in range(MAXG + 1)),
                  key=lambda x: -x[2])
    top = [{"score": f"{i}-{k}", "home": i, "away": k, "prob": round(p, 4)}
           for i, k, p in flat[:top_n]]
    bi, bk, bp = flat[0]

    out = {
        "home": home, "away": away, "neutral": neutral,
        "venue": venue, "support": round(float(s), 4),
        "expected_goals": {"home": round(float(lh.mean()), 3),
                           "away": round(float(la.mean()), 3)},
        "outcome": {"home_win": round(p_home, 4),
                    "draw": round(p_draw, 4),
                    "away_win": round(p_away, 4)},
        "most_likely_score": {"home": bi, "away": bk, "prob": round(bp, 4)},
        "top_scores": top,
        "score_matrix": [[round(float(mat[i, k]), 5) for k in range(matrix_size)]
                         for i in range(matrix_size)],
    }

    if "markets" in extras:
        out["markets"] = {
            "over_under": {f"{ln}": {"over": round(float(mat[_TOTAL > ln].sum()), 4),
                                     "under": round(float(mat[_TOTAL < ln].sum()), 4)}
                           for ln in (0.5, 1.5, 2.5, 3.5)},
            "btts": round(float(mat[(_I >= 1) & (_J >= 1)].sum()), 4),
            "clean_sheet": {"home": round(float(mat[_J == 0].sum()), 4),
                            "away": round(float(mat[_I == 0].sum()), 4)},
        }
    if "margin" in extras:
        # goal difference (home - away); tails folded into +-5 so it sums to 1
        cd = np.clip(_I - _J, -5, 5)
        out["margin"] = [{"diff": d, "prob": round(float(mat[cd == d].sum()), 4)}
                         for d in range(-5, 6)]
    if "uncertainty" in extras:
        jn = j / j.sum((1, 2), keepdims=True)                 # per-draw normalized
        out["uncertainty"] = {
            "outcome": {"home_win": _ci(jn[:, _HOME_WIN].sum(1)),
                        "draw": _ci(jn[:, _DRAW].sum(1)),
                        "away_win": _ci(jn[:, _AWAY_WIN].sum(1))},
            "expected_goals": {"home": _ci(lh, 3), "away": _ci(la, 3)},
        }
    return out


def ratings():
    """Per-team attack/defense strength with posterior uncertainty, ranked.
    Enables a strength leaderboard and an attack-vs-defense scatter."""
    teams = []
    for t, i in _idx.items():
        atk, dfn = _atk[:, i], _dfn[:, i]
        teams.append({"team": str(t),
                      "attack": _ci(atk, 3), "defense": _ci(dfn, 3),
                      "rating": round(float(atk.mean() + dfn.mean()), 3)})
    teams.sort(key=lambda x: -x["rating"])
    return {"count": len(teams), "teams": teams}
