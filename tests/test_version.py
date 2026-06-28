import json

from fastapi.testclient import TestClient

import app as APP
import predict as P


def test_load_version_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_VERSION_PATH", tmp_path / "nope.json")
    assert P._load_version() == {}


def test_load_version_reads_file(monkeypatch, tmp_path):
    p = tmp_path / "version.json"
    p.write_text(json.dumps({"latest_match_date": "2026-06-27", "csv_rows": 49512}))
    monkeypatch.setattr(P, "_VERSION_PATH", p)
    assert P._load_version()["latest_match_date"] == "2026-06-27"


def test_root_exposes_model_version(monkeypatch):
    monkeypatch.setattr(P, "VERSION",
                        {"latest_match_date": "2026-06-27", "trained_at": "2026-06-28T06:00:00Z"})
    with TestClient(APP.app) as client:
        body = client.get("/").json()
    assert body["model_version"] == "2026-06-27"
    assert body["model_trained_at"] == "2026-06-28T06:00:00Z"
    # existing health-check fields stay intact
    assert body["status"] == "ok"
    assert body["teams_available"] == len(P.TEAMS)


def test_root_tolerates_absent_version(monkeypatch):
    monkeypatch.setattr(P, "VERSION", {})
    with TestClient(APP.app) as client:
        body = client.get("/").json()
    assert body["model_version"] is None
    assert body["model_trained_at"] is None
