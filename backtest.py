"""
OFFLINE backtest harness — model evaluation, never runs in production.
Uses training deps (pymc); run it in the same venv as train.py:

    python backtest.py

Time-holdout evaluation of outcome (home/draw/away) forecasts: for each cutoff
date it fits on matches BEFORE the cutoff and scores the matches in the window
after it, reporting multiclass log-loss + Brier. Results are split out for the
low-data-team subset, since that is where this model is estimation-limited and
where prior/structure changes are meant to help.

It fits two arms per cutoff so changes can be compared apples-to-apples:
  - "base": current production model (Elo-informed priors, see train.py/elo.py)
  - "flat": ablation with the Elo priors switched off (flat Normal priors)
This is the gate that steered shipping the Elo priors (~7% OOS log-loss gain) and
rejecting xG. To evaluate a NEW change, add a third arm in `fit()` and report it.

Sampling is intentionally light (500/500, 2 chains) for turnaround; both arms use
identical settings so the comparison is fair even if noisier than train.py.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt
import geo, elo as ELO

RNG = 42; WINDOW_START = "2021-01-01"; DECAY = 0.10; MAXG = 10
LOWDATA_MAX = 64                       # <= this many in-fold games == "low-data"
CUTOFFS = ["2025-06-01", "2025-09-01", "2025-12-01"]
HORIZON = 92                           # days of test matches after each cutoff

_CSV = Path(__file__).parent / "results.csv"
df = pd.read_csv(_CSV); df["date"] = pd.to_datetime(df["date"])
df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
wc26 = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
WC = sorted(set(wc26.home_team) | set(wc26.away_team))


def fit(train, use_elo):
    """Fit the model on `train`; mirrors train.py. use_elo toggles the Elo priors."""
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    hi = train.home_team.map(idx).to_numpy(); ai = train.away_team.map(idx).to_numpy()
    hg = train.home_score.astype(int).to_numpy(); ag = train.away_score.astype(int).to_numpy()
    sup = geo.support_array(train.home_team.to_numpy(), train.away_team.to_numpy(),
                            train.country.to_numpy())
    w = np.exp(-DECAY * ((train.date.max() - train.date).dt.days.to_numpy() / 365.25))
    # Elo from history available before the cutoff only (no leakage).
    z = (ELO.z_scores(ELO.compute_elo(df[df.date <= train.date.max()]), teams)
         if use_elo else np.zeros(n))
    with pm.Model():
        intercept = pm.Normal("intercept", 0, 1); home = pm.Normal("home", 0.25, 0.25)
        sa = pm.HalfNormal("sa", 1); sd = pm.HalfNormal("sd", 1)
        ar_raw = pm.Normal("ar_raw", 0, 1, shape=n); dr_raw = pm.Normal("dr_raw", 0, 1, shape=n)
        if use_elo:
            ba = pm.Normal("ba", 0, 1); bd = pm.Normal("bd", 0, 1)
            ar = ba * z + ar_raw * sa; dr = bd * z + dr_raw * sd
        else:
            ar = ar_raw * sa; dr = dr_raw * sd
        attack = pm.Deterministic("attack", ar - ar.mean())
        defense = pm.Deterministic("defense", dr - dr.mean())
        rho = pm.Normal("rho", 0, 0.1)
        lh = pm.math.exp(intercept + home * sup + attack[hi] - defense[ai])
        la = pm.math.exp(intercept + attack[ai] - defense[hi])
        pm.Potential("lh", (w * pm.logp(pm.Poisson.dist(mu=lh), hg)).sum())
        pm.Potential("la", (w * pm.logp(pm.Poisson.dist(mu=la), ag)).sum())
        tau = pt.ones_like(lh)
        tau = pt.switch((hg == 0) & (ag == 0), 1 - lh * la * rho, tau)
        tau = pt.switch((hg == 0) & (ag == 1), 1 + lh * rho, tau)
        tau = pt.switch((hg == 1) & (ag == 0), 1 + la * rho, tau)
        tau = pt.switch((hg == 1) & (ag == 1), 1 - rho, tau)
        pm.Potential("dc", (w * pt.log(pt.clip(tau, 1e-6, np.inf))).sum())
        idata = pm.sample(500, tune=500, chains=2, cores=1, target_accept=0.9,
                          random_seed=RNG, progressbar=False)
    p = idata.posterior
    return (idx, p["intercept"].values.reshape(-1), p["home"].values.reshape(-1),
            p["attack"].values.reshape(-1, n), p["defense"].values.reshape(-1, n),
            p["rho"].values.reshape(-1))


def outcome_probs(model, home, away, neutral):
    """(home_win, draw, away_win) from the fitted posterior — mirrors predict.py."""
    from scipy.stats import poisson
    idx, intc, hom, atk, dfn, rho = model
    hh, aa = idx[home], idx[away]
    s = 0.0 if neutral else 1.0
    lh = np.exp(intc + hom * s + atk[:, hh] - dfn[:, aa])
    la = np.exp(intc + atk[:, aa] - dfn[:, hh])
    g = np.arange(MAXG + 1)
    ph = poisson.pmf(g[None, :], lh[:, None]); pa = poisson.pmf(g[None, :], la[:, None])
    j = np.einsum("si,sj->sij", ph, pa)
    j[:, 0, 0] *= (1 - lh * la * rho); j[:, 0, 1] *= (1 + lh * rho)
    j[:, 1, 0] *= (1 + la * rho);      j[:, 1, 1] *= (1 - rho)
    j = np.clip(j, 0, None); mat = j.mean(0); mat /= mat.sum()
    I, J = np.indices((MAXG + 1, MAXG + 1))
    return float(mat[I > J].sum()), float(mat[I == J].sum()), float(mat[I < J].sum())


def main():
    rows = []
    for co in CUTOFFS:
        co = pd.Timestamp(co)
        train = df[(df.home_score.notna()) & (df.date >= WINDOW_START) & (df.date < co) &
                   ((df.home_team.isin(WC)) | (df.away_team.isin(WC)))].copy()
        teams_tr = set(train.home_team) | set(train.away_team)
        ng = pd.concat([train.home_team, train.away_team]).value_counts()
        lowdata = {t for t in teams_tr if ng.get(t, 0) <= LOWDATA_MAX}
        test = df[(df.home_score.notna()) & (df.date >= co) &
                  (df.date < co + pd.Timedelta(days=HORIZON)) &
                  (df.home_team.isin(teams_tr)) & (df.away_team.isin(teams_tr))].copy()
        if test.empty:
            continue
        print(f"cutoff {co.date()}: fit on {len(train)} matches, score {len(test)} ...")
        arms = {"base": fit(train, True), "flat": fit(train, False)}
        for _, r in test.iterrows():
            y = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            low = (r.home_team in lowdata) or (r.away_team in lowdata)
            for arm, mdl in arms.items():
                pv = np.clip(outcome_probs(mdl, r.home_team, r.away_team, bool(r.neutral)), 1e-12, 1)
                rows.append({"arm": arm, "low": low, "ll": -np.log(pv[y]),
                             "br": float(sum((pv[k] - (k == y)) ** 2 for k in range(3)))})

    R = pd.DataFrame(rows)
    print(f"\n=== {len(R[R.arm=='base'])} held-out matches across {len(CUTOFFS)} cutoffs ===\n")
    base_arm = "base"
    for label, sub in [("ALL matches", R), ("LOW-DATA-team matches", R[R.low]),
                       ("high-data-only matches", R[~R.low])]:
        g = sub.groupby("arm").agg(ll=("ll", "mean"), br=("br", "mean"))
        n = len(sub[sub.arm == base_arm])
        print(f"{label} (n={n}):")
        for arm in g.index:
            tag = "  (production)" if arm == base_arm else ""
            print(f"  {arm:5s} log-loss={g.loc[arm,'ll']:.4f}  brier={g.loc[arm,'br']:.4f}{tag}")
        if {"base", "flat"} <= set(g.index):
            dll = 100 * (g.loc["flat", "ll"] - g.loc["base", "ll"]) / g.loc["flat", "ll"]
            print(f"        base vs flat: log-loss {dll:+.2f}%\n")


if __name__ == "__main__":
    main()
