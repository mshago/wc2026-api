# WC2026 Poisson Predictor — API

Bayesian hierarchical Poisson (Dixon-Coles) model that predicts international
football match outcomes and scorelines. Team attack/defense strengths use
**Elo-informed priors**, so low-data teams borrow signal from their longer
match history instead of defaulting to an uninformative flat prior. Trained
offline; served as a tiny FastAPI app that runs in pure NumPy/SciPy (no PyMC
at runtime).

## How it's split

- **`train.py`** — offline. Fits the model in PyMC and writes the posterior to
  `model/wc2026_poisson_dc_posterior.npz`. Needs `requirements-train.txt`.
  Never runs on Railway.
- **`app.py` + `predict.py`** — runtime. Load the `.npz` and serve predictions.
  Needs only `requirements.txt` (fastapi, uvicorn, numpy, scipy).
- **`elo.py`** — offline helper used by `train.py`. Computes World Football Elo
  from the full `results.csv` history and turns it into the priors above. Never
  imported at runtime.
- **`backtest.py`** — offline model evaluation: time-holdout log-loss / Brier,
  split by low-data-team subset, with a prior-ablation arm. The gate for model
  changes. Uses `requirements-train.txt`; never runs on Railway.

The trained artifact (`model/*.npz`, ~4.7 MB) is committed to the repo, so the
deploy needs no training step.

## API

| Route | Description |
|---|---|
| `GET /` | health + usage |
| `GET /teams` | list of valid team names |
| `GET /predict?home=France&away=Iraq&neutral=true` | full prediction |
| `GET /ratings` | per-team attack/defense strength, ranked, with 90% CIs |

`/predict` also accepts two optional query params:

- `venue=<country>` — derives continuous crowd-support from venue geography and
  **overrides** `neutral` (e.g. `&venue=United States`).
- `extras=markets,margin,uncertainty` (or `extras=all`) — adds opt-in blocks:
  over/under & BTTS markets, the goal-margin distribution, and credible-interval
  bands. Omit for the lean default payload.

Example response (`/predict?home=Argentina&away=Austria`):

```json
{
  "home": "Argentina", "away": "Austria", "neutral": true,
  "venue": null, "support": 0.0,
  "expected_goals": { "home": 1.533, "away": 0.701 },
  "outcome": { "home_win": 0.559, "draw": 0.274, "away_win": 0.167 },
  "most_likely_score": { "home": 1, "away": 0, "prob": 0.156 },
  "top_scores": [ { "score": "1-0", "home": 1, "away": 0, "prob": 0.156 }, ... ],
  "score_matrix": [[...]]  // P(home goals i, away goals j), i=row, j=col
}
```

`score_matrix[i][j]` is the probability of the exact score `i-j` — feed it
straight into a heatmap component on the client.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
# http://127.0.0.1:8000/predict?home=France&away=Iraq&neutral=true
# docs at http://127.0.0.1:8000/docs
```

## Deploy to Railway

**Option A — GitHub (recommended)**
1. Push this folder to a GitHub repo.
2. Railway → New Project → Deploy from GitHub repo → pick it.
3. Railway auto-detects Python (Nixpacks), installs `requirements.txt`, and
   starts via the `Procfile`. No env vars needed — Railway injects `$PORT`.
4. Settings → Networking → **Generate Domain** to get a public URL.

**Option B — CLI**
```bash
npm i -g @railway/cli
railway login
railway init        # from this folder
railway up
railway domain      # expose a public URL
```

Test it:
```bash
curl "https://<your-app>.up.railway.app/predict?home=France&away=Iraq&neutral=true"
```

## Call it from React / React Native

```ts
const base = "https://<your-app>.up.railway.app";

export async function predictMatch(home: string, away: string, neutral = true) {
  const url = `${base}/predict?home=${encodeURIComponent(home)}`
            + `&away=${encodeURIComponent(away)}&neutral=${neutral}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return res.json();
}
```

## Refreshing the model

As more matches are played, re-run training and redeploy:

```bash
pip install -r requirements-train.txt
rm -f results.csv          # force a fresh download
python train.py            # regenerates model/*.npz (Elo priors included)
python backtest.py         # optional: validate before committing
git commit -am "refresh model" && git push   # Railway redeploys
```

> CORS is currently open (`*`) for convenience. Restrict `allow_origins` in
> `app.py` to your app's domain before going to production.
