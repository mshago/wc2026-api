"""
WC2026 Poisson Predictor — FastAPI service.
Run locally:  uvicorn app:app --reload
"""
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import predict as P
import compare as C
import fixtures as F

app = FastAPI(title="WC2026 Poisson Predictor", version="1.0.0",
              description="Bayesian hierarchical Poisson (Dixon-Coles) match predictor.")

# CORS — open for now so a React/RN client can call it.
# In production, replace ["*"] with your app's origin(s).
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET"], allow_headers=["*"])


@app.get("/")
def root():
    return {"status": "ok",
            "model": "bayesian-poisson-dixon-coles",
            "model_version": P.VERSION.get("latest_match_date"),
            "model_trained_at": P.VERSION.get("trained_at"),
            "teams_available": len(P.TEAMS),
            "usage": "/predict?home=France&away=Iraq&neutral=true"}


@app.get("/teams")
def teams():
    """All team names the model knows (valid inputs)."""
    return {"count": len(P.TEAMS), "teams": P.TEAMS}


@app.get("/ratings")
def ratings():
    """Per-team attack/defense strength (mean + 90% CI), ranked by overall."""
    return P.ratings()


@app.get("/predict")
def predict_match(
    home: str = Query(..., description="Home/first team name (see /teams)"),
    away: str = Query(..., description="Away/second team name (see /teams)"),
    neutral: bool = Query(True, description="True for a neutral venue (most WC games)"),
    venue: str = Query(None, description="Optional venue country (e.g. 'United States'); "
                                         "derives continuous crowd-support and overrides `neutral`"),
    extras: str = Query(None, description="Comma-separated optional blocks: "
                                          "markets, margin, uncertainty (or 'all')"),
):
    ex = None
    if extras:
        ex = {x.strip() for x in extras.split(",") if x.strip()}
        if "all" in ex:
            ex = set(P.EXTRAS)
        bad = ex - P.EXTRAS
        if bad:
            raise HTTPException(status_code=400,
                                detail=f"Unknown extras {sorted(bad)}. "
                                       f"Allowed: {sorted(P.EXTRAS)} (or 'all').")
    try:
        return P.predict(home, away, neutral, venue, extras=ex)
    except KeyError as e:
        raise HTTPException(status_code=404,
                            detail=f"Unknown team {e}. Check /teams for valid names.")


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


@app.get("/fixtures")
def fixtures(
    status: str = Query(None, max_length=64,
                        description="football-data status filter, e.g. SCHEDULED,TIMED,IN_PLAY"),
    stage: str = Query(None, max_length=32,
                       description="Stage filter, e.g. GROUP_STAGE, LAST_16, FINAL"),
):
    """World Cup fixtures from football-data.org, with team names normalized to
    /teams names and a derived neutral flag — ready for /predict and /compare."""
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise HTTPException(status_code=503,
                            detail="Schedule unavailable: FOOTBALL_DATA_API_KEY not configured.")
    try:
        return F.get_fixtures(P.TEAMS, {"status": status, "stage": stage}, key)
    except F.UpstreamError as e:
        raise HTTPException(status_code=502, detail=f"Schedule provider error: {e}")
