"""
WC2026 Poisson Predictor — FastAPI service.
Run locally:  uvicorn app:app --reload
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import predict as P

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
            "teams_available": len(P.TEAMS),
            "usage": "/predict?home=France&away=Iraq&neutral=true"}


@app.get("/teams")
def teams():
    """All team names the model knows (valid inputs)."""
    return {"count": len(P.TEAMS), "teams": P.TEAMS}


@app.get("/predict")
def predict_match(
    home: str = Query(..., description="Home/first team name (see /teams)"),
    away: str = Query(..., description="Away/second team name (see /teams)"),
    neutral: bool = Query(True, description="True for a neutral venue (most WC games)"),
):
    try:
        return P.predict(home, away, neutral)
    except KeyError as e:
        raise HTTPException(status_code=404,
                            detail=f"Unknown team {e}. Check /teams for valid names.")
