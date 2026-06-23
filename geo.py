"""
Geography helpers for the venue *support* term — pure NumPy, NO training deps.

Shared by BOTH train.py (offline) and predict.py (runtime) so the support
math is defined in exactly one place and cannot drift between training and
serving. Safe to import in the request path (numpy + stdlib only).

`support` replaces the old binary home/neutral switch with a continuous value
in [-1, +1] derived from how close the match venue is to each team's home:

    support = (d_away - d_home) / (d_away + d_home + EPS)

  +1  venue in the home-listed team's country   (true home game)
   0  venue equidistant from both                (genuinely neutral)
  -1  venue in the away team's country           (home team is the visitor)
"""
import csv
from pathlib import Path
import numpy as np

_CSV = Path(__file__).parent / "data" / "country_centroids.csv"
EPS = 1.0          # km — guards the support denominator against 0/0
_EARTH_KM = 6371.0088


def _load_centroids():
    out = {}
    with open(_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = (float(row["lat"]), float(row["lon"]))
    return out


CENTROIDS = _load_centroids()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalar or numpy-array inputs."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return _EARTH_KM * 2 * np.arcsin(np.sqrt(a))


def _support_from_d(d_home, d_away):
    return (d_away - d_home) / (d_away + d_home + EPS)


def _coords(names):
    """Names -> (lat, lon) arrays; NaN where the name has no centroid."""
    lat = np.array([CENTROIDS[n][0] if n in CENTROIDS else np.nan for n in names])
    lon = np.array([CENTROIDS[n][1] if n in CENTROIDS else np.nan for n in names])
    return lat, lon


def support_array(home, away, venue):
    """Vectorized support for training. Unknown centroid -> 0.0 (neutral)."""
    hlat, hlon = _coords(home)
    alat, alon = _coords(away)
    vlat, vlon = _coords(venue)
    d_home = haversine_km(hlat, hlon, vlat, vlon)
    d_away = haversine_km(alat, alon, vlat, vlon)
    s = _support_from_d(d_home, d_away)
    return np.where(np.isnan(s), 0.0, s)


def support(home, away, venue):
    """Scalar support for one fixture, or None if any centroid is unknown
    (the caller then falls back to the binary neutral flag)."""
    if home not in CENTROIDS or away not in CENTROIDS or venue not in CENTROIDS:
        return None
    hlat, hlon = CENTROIDS[home]
    alat, alon = CENTROIDS[away]
    vlat, vlon = CENTROIDS[venue]
    d_home = haversine_km(hlat, hlon, vlat, vlon)
    d_away = haversine_km(alat, alon, vlat, vlon)
    return float(_support_from_d(d_home, d_away))
