import pytest
from fastapi.testclient import TestClient

import fixtures
import app as APP


TEAMS = {"South Korea", "United States", "Brazil", "Senegal", "Mexico", "Canada"}


def _match(**kw):
    m = {"id": 1, "utcDate": "2026-06-28T18:00:00Z", "status": "SCHEDULED",
         "stage": "LAST_16", "group": None,
         "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Senegal"},
         "venue": "SoFi Stadium",
         "score": {"winner": None, "fullTime": {"home": None, "away": None}}}
    m.update(kw)
    return m


def test_maps_team_names_to_canonical():
    f = fixtures.transform_match(
        _match(homeTeam={"name": "Korea Republic"}, awayTeam={"name": "USA"}), TEAMS)
    assert f["home"] == "South Korea" and f["home_known"] is True
    assert f["away"] == "United States" and f["away_known"] is True


def test_unmapped_name_flagged_not_known():
    f = fixtures.transform_match(_match(homeTeam={"name": "Wakanda"}), TEAMS)
    assert f["home"] == "Wakanda" and f["home_known"] is False


def test_tbd_null_team():
    f = fixtures.transform_match(_match(homeTeam=None), TEAMS)
    assert f["home"] is None and f["home_known"] is False


def test_host_at_home_is_not_neutral():
    # USA playing at a US venue -> home advantage, not neutral
    f = fixtures.transform_match(
        _match(homeTeam={"name": "USA"}, awayTeam={"name": "Brazil"},
               venue="SoFi Stadium"), TEAMS)
    assert f["venue_country"] == "United States"
    assert f["neutral"] is False


def test_non_host_match_is_neutral():
    # two non-host teams at a US venue -> still neutral (no home crowd for either)
    f = fixtures.transform_match(
        _match(homeTeam={"name": "Brazil"}, awayTeam={"name": "Senegal"},
               venue="MetLife Stadium"), TEAMS)
    assert f["venue_country"] == "United States"
    assert f["neutral"] is True


def test_unknown_venue_is_neutral():
    f = fixtures.transform_match(_match(venue="Some Unlisted Ground"), TEAMS)
    assert f["venue_country"] is None
    assert f["neutral"] is True


def test_finished_match_carries_result():
    f = fixtures.transform_match(
        _match(status="FINISHED",
               score={"winner": "HOME_TEAM", "fullTime": {"home": 2, "away": 1}}), TEAMS)
    assert f["played"] is True
    assert f["result"] == {"home": 2, "away": 1}


def test_get_fixtures_transforms_and_caches(monkeypatch):
    calls = {"n": 0}

    def fake_http(params, key):
        calls["n"] += 1
        return {"matches": [_match(), _match(homeTeam={"name": "Korea Republic"})]}

    monkeypatch.setattr(fixtures, "_http_get", fake_http)
    fixtures._cache.clear()
    out = fixtures.get_fixtures(TEAMS, {"stage": "LAST_16"}, "k")
    assert out["count"] == 2
    assert out["fixtures"][1]["home"] == "South Korea"
    # second identical call is served from cache (no second upstream hit)
    fixtures.get_fixtures(TEAMS, {"stage": "LAST_16"}, "k")
    assert calls["n"] == 1


def test_live_football_data_names_resolve_to_known_teams():
    # names football-data.org actually returns for WC2026 that differ from ours
    import predict as P
    known = set(P.TEAMS)
    for fd_name in ["Bosnia-Herzegovina", "Cape Verde Islands", "Korea Republic",
                    "USA", "IR Iran", "Czechia", "Congo DR"]:
        canonical = fixtures.map_name(fd_name)
        assert canonical in known, f"{fd_name!r} -> {canonical!r} not in /teams"


def test_route_503_without_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with TestClient(APP.app) as client:
        r = client.get("/fixtures")
    assert r.status_code == 503


def test_route_502_on_upstream_error(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "k")

    def boom(*a, **k):
        raise fixtures.UpstreamError("football-data down")

    monkeypatch.setattr(fixtures, "get_fixtures", boom)
    with TestClient(APP.app) as client:
        r = client.get("/fixtures")
    assert r.status_code == 502
