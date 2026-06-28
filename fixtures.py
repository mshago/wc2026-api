"""
Schedule proxy — fetches World Cup fixtures from football-data.org server-side
and returns them with team names normalized to our canonical /teams names, so
the frontend can feed them straight into /predict and /compare.

Runtime-safe: stdlib only (json, urllib, time). The football-data API key is
read from the FOOTBALL_DATA_API_KEY env var by the caller and never leaves the
server. Upstream responses are TTL-cached to respect the free tier's rate limit.
"""
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

_DATA = Path(__file__).parent / "data"
with open(_DATA / "team_name_map.json", encoding="utf-8") as _f:
    _TEAM_MAP = json.load(_f)
with open(_DATA / "wc2026_venues.json", encoding="utf-8") as _f:
    _VENUES = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}

HOST_COUNTRIES = {"United States", "Canada", "Mexico"}
_BASE = "https://api.football-data.org/v4/competitions/WC/matches"
_TTL = 60.0                 # seconds; football-data free tier is ~10 req/min
_cache: dict = {}           # params-key -> (fetched_at, payload)


class UpstreamError(Exception):
    """football-data.org was unreachable or returned a non-200 status."""


def map_name(name):
    """football-data team name -> canonical name. None stays None (TBD slots).
    Unmapped names pass through unchanged (caller flags them via *_known)."""
    if name is None:
        return None
    return _TEAM_MAP.get(name, name)


def transform_match(m: dict, teams) -> dict:
    """One football-data match -> a clean fixture with canonical names, the
    venue's host country, a derived neutral flag, and the result if played."""
    home = map_name((m.get("homeTeam") or {}).get("name"))
    away = map_name((m.get("awayTeam") or {}).get("name"))
    venue_country = _VENUES.get(m.get("venue"))
    # home advantage only when a host nation plays in its own country
    host_playing = venue_country in HOST_COUNTRIES and venue_country in (home, away)
    status = m.get("status")
    played = status == "FINISHED"
    ft = (m.get("score") or {}).get("fullTime") or {}
    result = {"home": ft.get("home"), "away": ft.get("away")} if played else None
    return {
        "id": m.get("id"),
        "utc_date": m.get("utcDate"),
        "status": status,
        "stage": m.get("stage"),
        "group": m.get("group"),
        "home": home, "away": away,
        "home_known": home in teams,
        "away_known": away in teams,
        "venue": m.get("venue"),
        "venue_country": venue_country,
        "neutral": not host_playing,
        "played": played,
        "result": result,
    }


def _http_get(params: dict, key: str) -> dict:
    """Raw GET against football-data.org. Raises UpstreamError on any failure."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
    url = _BASE + (f"?{query}" if query else "")
    req = urllib.request.Request(url, headers={"X-Auth-Token": key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        raise UpstreamError(str(e)) from e


def _fetch(params: dict, key: str) -> dict:
    """TTL-cached upstream fetch, keyed on the request params."""
    ckey = tuple(sorted((k, v) for k, v in params.items() if v))
    hit = _cache.get(ckey)
    now = time.monotonic()
    if hit and now - hit[0] < _TTL:
        return hit[1]
    payload = _http_get(params, key)
    _cache[ckey] = (now, payload)
    return payload


def get_fixtures(teams, params: dict, key: str) -> dict:
    """Fetch (cached) and normalize WC fixtures for the given query params."""
    payload = _fetch(params, key)
    out = [transform_match(m, teams) for m in payload.get("matches", [])]
    return {"count": len(out), "fixtures": out}
