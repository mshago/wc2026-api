"""
OFFLINE training — run this locally (NOT on Railway) to (re)generate
model/wc2026_poisson_dc_posterior.npz, then commit the new .npz.

    pip install -r requirements-train.txt
    python train.py

Re-run whenever results.csv updates (e.g. as more WC games are played)
to refresh team strengths, then redeploy.
"""
import os, urllib.request
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt

RNG = 42; WINDOW_START = "2021-01-01"; DECAY = 0.40; S = 1500
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

if not os.path.exists("results.csv"):
    print("Downloading results.csv ..."); urllib.request.urlretrieve(DATA_URL, "results.csv")

df = pd.read_csv("results.csv"); df["date"] = pd.to_datetime(df["date"])
df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
wc26 = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
WC = sorted(set(wc26.home_team) | set(wc26.away_team))

m = df[(df.home_score.notna()) & (df.date >= WINDOW_START) &
       ((df.home_team.isin(WC)) | (df.away_team.isin(WC)))].copy()
m["home_score"] = m.home_score.astype(int); m["away_score"] = m.away_score.astype(int)
teams = sorted(set(m.home_team) | set(m.away_team)); idx = {t: i for i, t in enumerate(teams)}
n = len(teams)
hi = m.home_team.map(idx).to_numpy(); ai = m.away_team.map(idx).to_numpy()
hg = m.home_score.to_numpy(); ag = m.away_score.to_numpy()
neu = m.neutral.to_numpy().astype(float)
w = np.exp(-DECAY * ((m.date.max() - m.date).dt.days.to_numpy() / 365.25))
print(f"Training on {len(m)} matches, {n} teams.")

with pm.Model():
    intercept = pm.Normal("intercept", 0, 1); home = pm.Normal("home", 0.25, 0.25)
    sa = pm.HalfNormal("sa", 1); sd = pm.HalfNormal("sd", 1)
    ar = pm.Normal("ar", 0, sa, shape=n); dr = pm.Normal("dr", 0, sd, shape=n)
    attack = pm.Deterministic("attack", ar - ar.mean())
    defense = pm.Deterministic("defense", dr - dr.mean())
    rho = pm.Normal("rho", 0, 0.1)
    lh = pm.math.exp(intercept + home * (1 - neu) + attack[hi] - defense[ai])
    la = pm.math.exp(intercept + attack[ai] - defense[hi])
    pm.Potential("lh", (w * pm.logp(pm.Poisson.dist(mu=lh), hg)).sum())
    pm.Potential("la", (w * pm.logp(pm.Poisson.dist(mu=la), ag)).sum())
    tau = pt.ones_like(lh)
    tau = pt.switch((hg == 0) & (ag == 0), 1 - lh * la * rho, tau)
    tau = pt.switch((hg == 0) & (ag == 1), 1 + lh * rho, tau)
    tau = pt.switch((hg == 1) & (ag == 0), 1 + la * rho, tau)
    tau = pt.switch((hg == 1) & (ag == 1), 1 - rho, tau)
    pm.Potential("dc", (w * pt.log(pt.clip(tau, 1e-6, np.inf))).sum())
    idata = pm.sample(1000, tune=1000, chains=2, cores=1, target_accept=0.9,
                      random_seed=RNG, progressbar=True)

p = idata.posterior
sel = np.random.default_rng(RNG).integers(0, p.sizes["chain"] * p.sizes["draw"], S)
os.makedirs("model", exist_ok=True)
np.savez_compressed("model/wc2026_poisson_dc_posterior.npz",
                    intc=p["intercept"].values.reshape(-1)[sel],
                    homv=p["home"].values.reshape(-1)[sel],
                    atk=p["attack"].values.reshape(-1, n)[sel],
                    dfn=p["defense"].values.reshape(-1, n)[sel],
                    rhov=p["rho"].values.reshape(-1)[sel],
                    teams=np.array(teams))
print("Saved model/wc2026_poisson_dc_posterior.npz")
