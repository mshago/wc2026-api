# XGBoost Comparison Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an XGBoost 1X2 outcome model, trained offline and served as a committed static JSON, to compare head-to-head with the existing Bayesian model via two new API endpoints.

**Architecture:** All XGBoost work is offline (training venv only). The API reads a committed `model/xgb_compare.json` (per-fixture 1X2 + a backtest scoreboard) through a pure-stdlib loader; production keeps its NumPy/SciPy-only runtime and the Bayesian model untouched.

**Tech Stack:** Offline — Python, pandas, xgboost, pymc (existing), plus the repo's `elo.py`/`geo.py` helpers. Runtime — FastAPI, stdlib `json` (no new runtime deps).

## Global Constraints

- Runtime `requirements.txt` stays **exactly** as is — no `xgboost`, `pandas`, `pymc` added. xgboost goes only in `requirements-train.txt`.
- `compare.py` (runtime) imports **stdlib + nothing heavy** — `json`, `pathlib` only. It must NOT import `xgboost`, `pandas`, `elo`, or `geo`.
- `predict.py`, `train.py`, and the committed `.npz` are **not modified**. Existing routes `/`, `/teams`, `/ratings`, `/predict` keep their exact contracts.
- Features must be **leak-free**: every feature for a match uses only data strictly before that match's date.
- Comparison fixtures are computed at **neutral venue** (`neutral=True`), directed pairs among WC teams. The Bayesian side of `/compare` is also computed with `neutral=True` so the two are comparable.
- 404 error detail for unknown teams mirrors `/predict`: `f"Unknown team {e}. Check /teams for valid names."`
- The committed artifact is `model/xgb_compare.json`. Schema:

```json
{
  "generated": "2026-06-25",
  "neutral": true,
  "scoreboard": {
    "holdout": "cutoffs 2025-06-01/2025-09-01/2025-12-01, +92d windows",
    "n_matches": 412,
    "models": {
      "bayesian": {"hit_rate": 0.542, "log_loss": 0.982, "brier": 0.612},
      "xgboost":  {"hit_rate": 0.558, "log_loss": 0.961, "brier": 0.598}
    }
  },
  "fixtures": {
    "Mexico|South Korea": {"home_win": 0.53, "draw": 0.19, "away_win": 0.28}
  }
}
```

Fixture key format: `f"{home}|{away}"`. Probabilities in each fixture/model sum to ~1.

**Dev/test environment:** Run tests in the training venv (`pip install -r requirements-train.txt`), plus `pip install pytest httpx`. (`httpx` is needed by FastAPI's `TestClient`.)

---

### Task 1: Elo pre-match history helper

Add a function to `elo.py` that exposes the **pre-match** rating for every played match (the leak-free input the feature builder needs), refactoring `compute_elo` to share the single chronological pass.

**Files:**
- Modify: `elo.py` (add `compute_elo_history`, refactor `compute_elo` to call it)
- Test: `tests/test_elo_history.py`

**Interfaces:**
- Consumes: existing `elo.BASE`, `elo.HFA`, `elo._k_base`, `elo._g_mult`.
- Produces: `elo.compute_elo_history(df) -> (final_elo: dict, pre: pd.DataFrame)` where `pre` has columns `date, home_team, away_team, elo_home, elo_away`, one row per played match in date order, and `elo_home`/`elo_away` are the ratings **before** that match (unseen team → `elo.BASE`). `elo.compute_elo(df) -> dict` keeps its existing return.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elo_history.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elo_history.py -v`
Expected: FAIL with `AttributeError: module 'elo' has no attribute 'compute_elo_history'`

- [ ] **Step 3: Refactor `elo.py` to add the history helper**

Replace the body of `compute_elo` with a thin wrapper and add `compute_elo_history`:

```python
def compute_elo_history(df: pd.DataFrame):
    """Chronological Elo over played matches in `df`, also returning the
    PRE-match ratings for every match (leak-free feature input).

    Returns (final_elo: dict, pre: pd.DataFrame). `pre` has one row per played
    match in date order with columns date, home_team, away_team, elo_home,
    elo_away — the ratings BEFORE that match (unseen team -> BASE).
    """
    d = df[df.home_score.notna()].sort_values("date")
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
    """Chronological Elo over the played matches in `df`. Returns {team: final_rating}."""
    final, _ = compute_elo_history(df)
    return final
```

(Delete the old `compute_elo` loop body — `compute_elo_history` is now the single source of the update math.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_elo_history.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add elo.py tests/test_elo_history.py
git commit -m "feat: elo pre-match history helper for leak-free features"
```

---

### Task 2: XGBoost feature builder

Build the per-match, leak-free feature table in a new `xgb.py`. No xgboost import yet — this task is pure pandas/numpy + the Task 1 helper.

**Files:**
- Create: `xgb.py`
- Test: `tests/test_xgb_features.py`

**Interfaces:**
- Consumes: `elo.compute_elo_history`, `elo._k_base`, `geo.support`, `geo.CENTROIDS`.
- Produces:
  - `xgb.FEATURES: list[str]` — ordered feature column names.
  - `xgb.compute_features(df) -> pd.DataFrame` — one row per played match (date order) with the `FEATURES` columns plus `y` (0=home win,1=draw,2=away win), `date`, `home_team`, `away_team`. No-history values are `NaN` (xgboost handles natively).
  - `xgb.fixture_features(df, home, away) -> dict` — latest-state feature dict for an unplayed neutral fixture (uses final Elo + each team's most recent form/rest as of the last match in `df`).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xgb_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xgb'`

- [ ] **Step 3: Write `xgb.py` feature builder**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xgb_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add xgb.py tests/test_xgb_features.py
git commit -m "feat: leak-free feature builder for xgboost comparison model"
```

---

### Task 3: XGBoost train + predict + artifact assembly

Add the xgboost-dependent functions to `xgb.py`: train a 3-class model, predict a fixture, and assemble the artifact dict.

**Files:**
- Modify: `xgb.py` (add `train_xgb`, `predict_proba`, `predict_fixtures`, `assemble_artifact`)
- Modify: `requirements-train.txt` (add `xgboost`)
- Test: `tests/test_xgb_model.py`

**Interfaces:**
- Consumes: `xgb.FEATURES`, `xgb.compute_features`, `xgb.fixture_features`.
- Produces:
  - `xgb.train_xgb(feat_df) -> booster` — fits `XGBClassifier(objective="multi:softprob", num_class=3)` on `feat_df[FEATURES]` → `feat_df["y"]`.
  - `xgb.predict_proba(model, feat_dict) -> dict` — `{"home_win","draw","away_win"}` summing to 1.
  - `xgb.predict_fixtures(model, df, teams) -> dict` — `{f"{h}|{a}": {1X2}}` for all directed pairs, neutral venue.
  - `xgb.assemble_artifact(fixtures, scoreboard) -> dict` — full schema dict (see Global Constraints), with `generated` (today ISO) and `neutral=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xgb_model.py
import numpy as np
import pandas as pd
import xgb


def _df(n=120):
    # synthetic: "Strong" beats "Weak" most of the time so the model has signal
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, a = ("Strong", "Weak") if i % 2 == 0 else ("Weak", "Strong")
        strong_home = h == "Strong"
        hg = rng.integers(2, 5) if strong_home else rng.integers(0, 2)
        ag = rng.integers(0, 2) if strong_home else rng.integers(2, 5)
        rows.append((f"2021-{(i % 11) + 1:02d}-01", h, a, hg, ag, True, "Friendly", "Qatar"))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score",
                                       "away_score", "neutral", "tournament", "country"]
                        ).assign(date=lambda d: pd.to_datetime(d.date))


def test_predict_proba_sums_to_one():
    df = _df()
    model = xgb.train_xgb(xgb.compute_features(df))
    p = xgb.predict_proba(model, xgb.fixture_features(df, "Strong", "Weak"))
    assert set(p) == {"home_win", "draw", "away_win"}
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_predict_fixtures_covers_directed_pairs():
    df = _df()
    model = xgb.train_xgb(xgb.compute_features(df))
    fx = xgb.predict_fixtures(model, df, ["Strong", "Weak"])
    assert "Strong|Weak" in fx and "Weak|Strong" in fx
    assert "Strong|Strong" not in fx


def test_assemble_artifact_schema():
    sb = {"holdout": "x", "n_matches": 1, "models": {"bayesian": {}, "xgboost": {}}}
    art = xgb.assemble_artifact({"Strong|Weak": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3}}, sb)
    assert art["neutral"] is True
    assert "generated" in art
    assert art["scoreboard"] == sb
    assert "Strong|Weak" in art["fixtures"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xgb_model.py -v`
Expected: FAIL with `AttributeError: module 'xgb' has no attribute 'train_xgb'`

- [ ] **Step 3: Add model functions to `xgb.py` and the dep**

Append to `xgb.py`:

```python
import datetime as _dt
from xgboost import XGBClassifier

_CLASS_KEYS = ("home_win", "draw", "away_win")


def train_xgb(feat_df: pd.DataFrame):
    """Fit a 3-class (home/draw/away) softprob model on the feature table."""
    X = feat_df[FEATURES]
    y = feat_df["y"].to_numpy()
    model = XGBClassifier(objective="multi:softprob", num_class=3,
                          n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          eval_metric="mlogloss")
    model.fit(X, y)
    return model


def predict_proba(model, feat_dict: dict) -> dict:
    """1X2 probabilities for one feature dict, ordered to match FEATURES."""
    X = pd.DataFrame([[feat_dict[f] for f in FEATURES]], columns=FEATURES)
    p = model.predict_proba(X)[0]
    return {k: float(p[i]) for i, k in enumerate(_CLASS_KEYS)}


def predict_fixtures(model, df: pd.DataFrame, teams) -> dict:
    """1X2 for every directed pair among `teams`, neutral venue."""
    out = {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            out[f"{h}|{a}"] = predict_proba(model, fixture_features(df, h, a))
    return out


def assemble_artifact(fixtures: dict, scoreboard: dict) -> dict:
    return {
        "generated": _dt.date.today().isoformat(),
        "neutral": True,
        "scoreboard": scoreboard,
        "fixtures": fixtures,
    }
```

Add to `requirements-train.txt` (new line):

```
xgboost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xgb_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add xgb.py requirements-train.txt tests/test_xgb_model.py
git commit -m "feat: xgboost train/predict/artifact-assembly + train dep"
```

---

### Task 4: Backtest `xgb` arm + scoreboard metrics

Add XGBoost to the time-holdout backtest so it's scored on the identical folds as the Bayesian model, add hit-rate, and emit `model/xgb_scoreboard.json`.

**Files:**
- Modify: `backtest.py` (add `xgb` scoring, `hit_rate`, scoreboard writer)
- Test: `tests/test_backtest_metrics.py`

**Interfaces:**
- Consumes: `xgb.compute_features`, `xgb.train_xgb`, `xgb.predict_proba`, `xgb.fixture_features`; existing `backtest.fit`, `backtest.outcome_probs`.
- Produces:
  - `backtest.hit_rate(probs, y) -> bool` helper — argmax(probs)==y.
  - `backtest.metrics(records) -> dict` — `{"hit_rate","log_loss","brier"}` (rounded 4dp) from a list of `{"p":(p0,p1,p2),"y":int}`.
  - A `model/xgb_scoreboard.json` written by `backtest.main()` with `{"holdout","n_matches","models":{"bayesian":{...},"xgboost":{...}}}` (bayesian = `base` arm).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_metrics.py
import backtest


def test_hit_rate_true_when_argmax_matches():
    assert backtest.hit_rate((0.6, 0.3, 0.1), 0) is True
    assert backtest.hit_rate((0.2, 0.3, 0.5), 0) is False


def test_metrics_shape_and_values():
    recs = [{"p": (0.7, 0.2, 0.1), "y": 0}, {"p": (0.2, 0.2, 0.6), "y": 2}]
    m = backtest.metrics(recs)
    assert set(m) == {"hit_rate", "log_loss", "brier"}
    assert m["hit_rate"] == 1.0           # both argmax-correct
    assert m["log_loss"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_metrics.py -v`
Expected: FAIL with `AttributeError: module 'backtest' has no attribute 'hit_rate'`

- [ ] **Step 3: Add metrics helpers + xgb arm to `backtest.py`**

Add these helpers near the top of `backtest.py` (after imports):

```python
import json

def hit_rate(probs, y) -> bool:
    return int(np.argmax(probs)) == int(y)


def metrics(records) -> dict:
    """Aggregate hit-rate / log-loss / brier from [{'p':(p0,p1,p2),'y':int}, ...]."""
    n = len(records)
    hits = sum(hit_rate(r["p"], r["y"]) for r in records)
    ll = np.mean([-np.log(np.clip(r["p"][r["y"]], 1e-12, 1)) for r in records])
    br = np.mean([sum((r["p"][k] - (k == r["y"])) ** 2 for k in range(3)) for r in records])
    return {"hit_rate": round(hits / n, 4), "log_loss": round(float(ll), 4),
            "brier": round(float(br), 4)}
```

In `main()`, fit an xgb arm per cutoff and collect per-model records. Inside the cutoff loop, after `arms = {"base": fit(train, True), "flat": fit(train, False)}`, add:

```python
        import xgb as XGB
        xgb_model = XGB.train_xgb(XGB.compute_features(train))
        sb_records = {"bayesian": [], "xgboost": []}
```

Then in the per-test-row loop, alongside the existing arm scoring, append scoreboard records:

```python
            pv_base = np.clip(outcome_probs(arms["base"], r.home_team, r.away_team, bool(r.neutral)), 1e-12, 1)
            sb_records["bayesian"].append({"p": tuple(pv_base), "y": y})
            fx = XGB.fixture_features(train, r.home_team, r.away_team)
            pv_xgb = XGB.predict_proba(xgb_model, fx)
            sb_records["xgboost"].append({"p": (pv_xgb["home_win"], pv_xgb["draw"], pv_xgb["away_win"]), "y": y})
```

(Hold `sb_records` across cutoffs by initializing `all_sb = {"bayesian": [], "xgboost": []}` before the cutoff loop and extending it: `all_sb["bayesian"].extend(sb_records["bayesian"])` etc. at the end of each cutoff.)

After the cutoff loop, write the scoreboard:

```python
    scoreboard = {
        "holdout": f"cutoffs {'/'.join(CUTOFFS)}, +{HORIZON}d windows",
        "n_matches": len(all_sb["bayesian"]),
        "models": {"bayesian": metrics(all_sb["bayesian"]),
                   "xgboost": metrics(all_sb["xgboost"])},
    }
    Path("model").mkdir(exist_ok=True)
    with open("model/xgb_scoreboard.json", "w") as f:
        json.dump(scoreboard, f, indent=2)
    print("Wrote model/xgb_scoreboard.json:", scoreboard["models"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest_metrics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest.py tests/test_backtest_metrics.py
git commit -m "feat: xgb arm + hit-rate/scoreboard metrics in backtest"
```

---

### Task 5: Offline orchestrator — generate the committed artifact

Add `main()` to `xgb.py` that runs the full offline build and writes `model/xgb_compare.json`, then run it and commit the artifact.

**Files:**
- Modify: `xgb.py` (add `main()` + `if __name__ == "__main__"`)
- Create (generated): `model/xgb_compare.json`, `model/xgb_scoreboard.json`

**Interfaces:**
- Consumes: `xgb.compute_features`, `xgb.train_xgb`, `xgb.predict_fixtures`, `xgb.assemble_artifact`; `model/xgb_scoreboard.json` from Task 4.

- [ ] **Step 1: Add `main()` to `xgb.py`**

```python
import os, json, urllib.request

WINDOW_START = "2021-01-01"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def main():
    if not os.path.exists("results.csv"):
        print("Downloading results.csv ..."); urllib.request.urlretrieve(DATA_URL, "results.csv")
    df = pd.read_csv("results.csv"); df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    wc26 = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
    teams = sorted(set(wc26.home_team) | set(wc26.away_team))
    train = df[(df.home_score.notna()) & (df.date >= WINDOW_START) &
               ((df.home_team.isin(teams)) | (df.away_team.isin(teams)))].copy()
    train["home_score"] = train.home_score.astype(int)
    train["away_score"] = train.away_score.astype(int)
    print(f"Training XGBoost on {len(train)} matches, {len(teams)} WC teams ...")
    model = train_xgb(compute_features(train))
    fixtures = predict_fixtures(model, train, teams)
    sb_path = "model/xgb_scoreboard.json"
    if not os.path.exists(sb_path):
        raise SystemExit("Run `python backtest.py` first to produce model/xgb_scoreboard.json")
    with open(sb_path) as f:
        scoreboard = json.load(f)
    artifact = assemble_artifact(fixtures, scoreboard)
    os.makedirs("model", exist_ok=True)
    with open("model/xgb_compare.json", "w") as f:
        json.dump(artifact, f)
    print(f"Saved model/xgb_compare.json ({len(fixtures)} fixtures)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the offline build**

Run (training venv):
```bash
pip install -r requirements-train.txt && pip install pytest httpx
python backtest.py        # writes model/xgb_scoreboard.json
python xgb.py             # writes model/xgb_compare.json
```
Expected: `Saved model/xgb_compare.json (N fixtures)` where N ≈ teams×(teams−1).

- [ ] **Step 3: Sanity-check the artifact**

Run:
```bash
python -c "import json; d=json.load(open('model/xgb_compare.json')); k=next(iter(d['fixtures'])); p=d['fixtures'][k]; print(k, p, round(sum(p.values()),4)); print(d['scoreboard']['models'])"
```
Expected: a fixture line whose probs sum to ~1.0, and both `bayesian`/`xgboost` metric dicts printed.

- [ ] **Step 4: Commit the artifact (force-add; mirrors the .npz)**

```bash
git add -f model/xgb_compare.json model/xgb_scoreboard.json
git add xgb.py
git commit -m "feat: offline xgb orchestrator + committed comparison artifact"
```

---

### Task 6: Runtime loader `compare.py`

Pure-stdlib loader the API uses to read the artifact. No heavy imports.

**Files:**
- Create: `compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces:
  - `compare.load(path=_PATH) -> dict` — loads the artifact; missing file → empty-but-valid `{"fixtures":{}, "scoreboard":{...}}`.
  - `compare.fixture_probs(data, home, away) -> dict` — 1X2 for a directed pair; `KeyError` if absent.
  - `compare.scoreboard(data) -> dict`.
  - `compare.DATA` — module-level loaded artifact.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare'`

- [ ] **Step 3: Write `compare.py`**

```python
"""
Runtime loader for the offline XGBoost comparison artifact — pure stdlib,
NO heavy deps (no xgboost/pandas). Safe in the request path.

Reads model/xgb_compare.json (per-fixture 1X2 + backtest scoreboard) produced
offline by xgb.py. Missing artifact -> empty-but-valid structure so the API
still boots; /compare then 404s and the scoreboard is empty.
"""
import json
from pathlib import Path

_PATH = Path(__file__).parent / "model" / "xgb_compare.json"
_EMPTY = {"fixtures": {}, "scoreboard": {"holdout": None, "n_matches": 0, "models": {}}}


def load(path=_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"fixtures": {}, "scoreboard": dict(_EMPTY["scoreboard"])}


def fixture_probs(data: dict, home: str, away: str) -> dict:
    key = f"{home}|{away}"
    if key not in data["fixtures"]:
        raise KeyError(key)
    return data["fixtures"][key]


def scoreboard(data: dict) -> dict:
    return data["scoreboard"]


DATA = load()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add compare.py tests/test_compare.py
git commit -m "feat: pure-stdlib runtime loader for xgb comparison artifact"
```

---

### Task 7: `/compare` and `/compare/scoreboard` endpoints

Wire the two additive routes into `app.py`.

**Files:**
- Modify: `app.py` (import `compare`, add two routes)
- Test: `tests/test_app_compare.py`

**Interfaces:**
- Consumes: `predict.predict` (via existing `P`), `compare.fixture_probs`, `compare.scoreboard`, `compare.DATA`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_compare.py -v`
Expected: FAIL — `/compare` returns 404/405 or `KeyError` (routes don't exist yet)

- [ ] **Step 3: Add routes to `app.py`**

Add the import beside `import predict as P`:

```python
import compare as C
```

Add at the end of `app.py`:

```python
@app.get("/compare")
def compare_match(
    home: str = Query(..., max_length=64, description="Home team (WC teams only)"),
    away: str = Query(..., max_length=64, description="Away team (WC teams only)"),
):
    """Side-by-side 1X2: Bayesian model (live) vs XGBoost (precomputed, neutral)."""
    try:
        bayes = P.predict(home, away, True)["outcome"]
    except KeyError as e:
        raise HTTPException(status_code=404,
                            detail=f"Unknown team {e}. Check /teams for valid names.")
    try:
        xgb = C.fixture_probs(C.DATA, home, away)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"No XGBoost prediction for '{home}' vs '{away}'. "
                                   f"The comparison covers World Cup teams only.")
    return {"home": home, "away": away, "bayesian": bayes, "xgboost": xgb}


@app.get("/compare/scoreboard")
def compare_scoreboard():
    """Backtest accuracy of both models on the shared time-holdout."""
    return C.scoreboard(C.DATA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_compare.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite + commit**

Run: `pytest -v`
Expected: all tests PASS.

```bash
git add app.py tests/test_app_compare.py
git commit -m "feat: /compare and /compare/scoreboard endpoints"
```

---

## Notes for the implementer

- The runtime app must still boot if `model/xgb_compare.json` is absent (Task 6's `load` fallback). But the real artifact IS committed in Task 5, so production serves real numbers.
- Do not add `xgboost`/`pandas` to `requirements.txt`. If you find yourself importing either in `compare.py` or `app.py`, stop — that breaks the train/serve split.
- `/compare` uses `neutral=True` on the Bayesian side deliberately, to match the neutral-venue precomputed XGBoost fixtures. Don't "fix" this to read a venue.
- The honest expected outcome is a close race; the deliverable is the measured scoreboard, not an XGBoost win. If XGBoost looks badly miscalibrated (much worse log-loss despite similar hit-rate), that's the calibration risk from the spec — raise it before drawing conclusions.
