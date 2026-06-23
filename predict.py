"""
Runtime predictor — pure NumPy/SciPy, no PyMC.
Loads the posterior draws produced offline by train.py and computes
posterior-predictive scoreline probabilities (with the Dixon-Coles
low-score correction baked in).
"""
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


def predict(home: str, away: str, neutral: bool = True, venue: str = None,
            top_n: int = 10, matrix_size: int = 7):
    if home not in _idx:
        raise KeyError(home)
    if away not in _idx:
        raise KeyError(away)
    hh, aa = _idx[home], _idx[away]

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

    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    flat = sorted(((i, k, float(mat[i, k])) for i in range(MAXG + 1) for k in range(MAXG + 1)),
                  key=lambda x: -x[2])
    top = [{"score": f"{i}-{k}", "home": i, "away": k, "prob": round(p, 4)}
           for i, k, p in flat[:top_n]]
    bi, bk, bp = flat[0]

    return {
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
