# WC2026 Poisson Predictor — API

Bayesian hierarchical Poisson (Dixon-Coles) model that predicts international
football match outcomes and scorelines. Trained offline; served as a tiny
FastAPI app that runs in pure NumPy/SciPy (no PyMC at runtime).

## How it's split

- **`train.py`** — offline. Fits the model in PyMC and writes the posterior to
  `model/wc2026_poisson_dc_posterior.npz`. Needs `requirements-train.txt`.
  Never runs on Railway.
- **`app.py` + `predict.py`** — runtime. Load the `.npz` and serve predictions.
  Needs only `requirements.txt` (fastapi, uvicorn, numpy, scipy).

The trained artifact (`model/*.npz`, ~4.7 MB) is committed to the repo, so the
deploy needs no training step.

## API

| Route | Description |
|---|---|
| `GET /` | health + usage |
| `GET /teams` | list of valid team names |
| `GET /predict?home=France&away=Iraq&neutral=true` | full prediction |

Example response (`/predict?home=Argentina&away=Austria`):

```json
{
  "home": "Argentina", "away": "Austria", "neutral": true,
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
python train.py            # regenerates model/*.npz
git commit -am "refresh model" && git push   # Railway redeploys
```

> CORS is currently open (`*`) for convenience. Restrict `allow_origins` in
> `app.py` to your app's domain before going to production.
