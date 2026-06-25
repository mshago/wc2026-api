# XGBoost Comparison Model — Design

**Date:** 2026-06-25
**Status:** Approved (pending spec review)
**Repo:** wc2026-api

## Goal

Add an XGBoost outcome model as an independent comparison to the existing
Bayesian Poisson–Dixon-Coles model, so we can answer **"which model picks
winners better"** and surface a side-by-side comparison on the frontend page.

The XGBoost model predicts **1X2 outcome probabilities** (P home win / draw /
away win) — directly comparable to the Bayesian model's `outcome` block and the
metric most accuracy scores (log-loss / Brier) operate on.

## Guiding constraint (the "one rule")

Everything XGBoost lives **offline**. Production serving stays pure-NumPy:

- Runtime `requirements.txt` is **unchanged** (no xgboost in production).
- The Bayesian model (`train.py`, `predict.py`, the committed `.npz`) is
  **untouched**.
- The API only ever reads a committed **static JSON artifact**; it never runs
  XGBoost inference, never imports xgboost.

This mirrors the existing train/serve split: heavy model work happens offline on
a dev machine / CI, the host only ships and reads repo contents.

## Architecture / data flow

```
OFFLINE (requirements-train.txt, += xgboost)         RUNTIME (no new deps)
┌─────────────────────────────────────────┐         ┌────────────────────────┐
│ xgb.py                                   │         │ app.py                 │
│  • build leak-free per-match features    │         │  GET /compare?home&away│
│  • train XGBoost (multi:softprob, 1X2)   │  write  │  GET /compare/scoreboard│
│  • predict all WC team-pair fixtures ────┼────────▶│        ▲               │
│                                          │ model/  │        │ reads          │
│ backtest.py (+ "xgb" arm)                │ xgb_    │ compare.py (json loader)│
│  • score XGB on SAME time-holdout  ──────┼────────▶│  stdlib/json only      │
│    as Bayesian → scoreboard metrics      │ compare │                        │
└──────────────────────────────────────────┘ .json  └────────────────────────┘
```

The committed `model/xgb_compare.json` holds two blocks:

1. **`fixtures`** — XGBoost 1X2 probabilities for every directed WC team pair
   (neutral venue).
2. **`scoreboard`** — both models' hit-rate / log-loss / Brier on the shared
   time-holdout, produced by `backtest.py`.

## Files

### New: `xgb.py` (offline, training-only)

Responsibilities:

- **Feature builder** — per-match, leak-free, as-of-match features (see below).
- **Trainer** — XGBoost `multi:softprob`, 3 classes (home win / draw / away win),
  on the same `2021-01-01+` WC-relevant window as `train.py`.
- **Fixture predictor** — predict every directed pair among the WC teams at a
  neutral venue.
- **JSON writer** — assemble `model/xgb_compare.json` (`fixtures` block here;
  `scoreboard` block pulled from `backtest.py` — see Build flow).

Imports `geo` and `elo` (offline helpers). Imports `xgboost`. Never imported by
the runtime path.

### New: `compare.py` (runtime, pure stdlib/json)

- Loads `model/xgb_compare.json` once at import (mirrors how `predict.py` loads
  its `.npz`).
- Exposes a lookup for a fixture's XGBoost 1X2 and the scoreboard block.
- **No xgboost import.** stdlib `json` + `pathlib` only. Safe in the request path.

### Changed: `app.py`

Two new GET routes (additive — existing routes untouched):

```
GET /compare?home=<team>&away=<team>
  → {
      "home": "Mexico", "away": "South Korea",
      "bayesian": { "home_win": 0.48, "draw": 0.27, "away_win": 0.25 },
      "xgboost":  { "home_win": 0.53, "draw": 0.19, "away_win": 0.28 }
    }
  - bayesian: computed live via predict.py (reuse existing predictor)
  - xgboost:  static lookup from compare.py
  - 404 { detail: "..." } if the pair is not in the precomputed set
    (e.g. a non-WC team), or unknown team (mirror /predict semantics)
  - team-name query params get the same max_length / validation treatment as
    /predict

GET /compare/scoreboard
  → {
      "holdout": "2025-06-01..2025-12-01 (+92d windows)",
      "n_matches": 412,
      "models": {
        "bayesian": { "hit_rate": 0.542, "log_loss": 0.982, "brier": 0.612 },
        "xgboost":  { "hit_rate": 0.558, "log_loss": 0.961, "brier": 0.598 }
      }
    }
```

(Numbers above are illustrative.)

### Changed: `backtest.py`

Add an `"xgb"` arm alongside the existing `"base"` (production Bayesian) and
`"flat"` (ablation) arms, scored on the **identical** time-holdout cutoffs and
test windows. This is what makes the scoreboard a fair, apples-to-apples number
rather than two models evaluated on different data.

The arm also emits the aggregate metrics (hit-rate / log-loss / Brier per model)
that `xgb.py` reads into the `scoreboard` block of the JSON artifact. Hit-rate =
fraction of test matches where the model's argmax class matches the actual
outcome.

### Changed: `requirements-train.txt`

`+= xgboost`. Runtime `requirements.txt` unchanged.

## Features (XGBoost input)

All computable from `results.csv` at each match's date, **no leakage** (only
information available before kickoff):

| Feature | Source | Notes |
|---|---|---|
| `elo_home`, `elo_away`, `elo_diff` | `elo.py` | **As-of-match** Elo — snapshot the chronological loop, not final ratings |
| recent goals-for / goals-against / win-rate (last K) | rolling over `results.csv` | per side; K to be set in implementation (e.g. 10) |
| rest days | `results.csv` | days since each side's previous match |
| venue support / neutral | `geo.py` | continuous support value; 0 at neutral |
| tournament importance | `elo._k_base` | competition weight |

This is a deliberately *different* signal mix from the Bayesian model so the
comparison is meaningful (not two views of the same posterior).

**As-of Elo:** `elo.compute_elo` currently returns only final ratings. The
feature builder needs each team's rating *as of* each match date. Implementation
snapshots ratings during the chronological pass (or recomputes per cutoff like
`backtest.py` already does at line 50). No leakage: ratings reflect only matches
strictly before the row's date.

## Scope boundaries

- **WC fixtures only.** The per-fixture endpoint covers directed pairs among the
  ~48 WC teams (≈2.2k small entries), neutral venue. Arbitrary non-WC matchups →
  404. This matches what the frontend page displays.
- **Neutral venue** for all precomputed fixtures (WC games are largely neutral).
  Directed pairs are kept (home/away listing) for generality and to mirror
  `/predict`'s `home`/`away` arguments.
- **Frontend not in this repo.** This work delivers the API + artifact + offline
  training/eval. Wiring the side-by-side picks and the scoreboard into the page
  is done separately against the contract above.
- Existing `/`, `/teams`, `/predict` contracts are **unchanged**. The two new
  routes are purely additive.

## Build / regeneration flow

Offline, in the training venv (like `train.py` / `backtest.py`):

```bash
pip install -r requirements-train.txt   # now includes xgboost
python backtest.py                       # scores base / flat / xgb arms
python xgb.py                            # trains XGB, writes model/xgb_compare.json
                                         #   (fixtures + scoreboard blocks)
git add -f model/xgb_compare.json        # commit the artifact (like the .npz)
```

`model/xgb_compare.json` is committed to git, same as the `.npz`, so the host
ships it.

## Risks & open considerations

- **Calibration.** Raw `multi:softprob` can be overconfident, which would
  unfairly inflate XGBoost's log-loss. If the backtest shows miscalibration, add
  simple probability calibration (isotonic on a held-out validation fold) before
  declaring a winner. Finding-dependent — not committed to v1 scope, but the
  backtest arm is the gate that decides.
- **Honest expectation.** With these features XGBoost is plausibly competitive
  but not obviously superior — the Bayesian model's Elo-informed priors already
  capture much of the same signal. The deliverable's value is the *measured*
  head-to-head, not a presumed winner.
- **Artifact staleness.** `xgb_compare.json` is a snapshot like the `.npz`; it
  must be regenerated when `results.csv` gains matches (same cadence as a model
  refresh). Out of scope to automate here.

## Testing

- `compare.py`: loads the artifact; returns a valid 1X2 (sums to ~1) for a known
  WC pair; raises / signals "not found" for an absent pair.
- `app.py` routes via FastAPI `TestClient`: `/compare` 200 shape (both
  `bayesian` and `xgboost` blocks present, each summing to ~1), 404 for a
  non-WC / unknown team; `/compare/scoreboard` 200 shape (both models present).
- `xgb.py` / `backtest.py` are offline-only; no runtime tests, but the feature
  builder should have a leak-free assertion (no feature uses a date ≥ the row's
  date).
