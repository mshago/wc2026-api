"""
Runtime loader for the offline XGBoost comparison artifact — pure stdlib,
NO heavy deps (no xgboost/pandas). Safe in the request path.

Reads model/xgb_compare.json (per-fixture 1X2 + backtest scoreboard) produced
offline by xgb.py. Missing artifact -> empty-but-valid structure so the API
still boots; /compare then 404s and the scoreboard is empty.
"""
import json
from pathlib import Path

_PATH = Path(__file__).parent / "model" / "xgb_compare.json"
_EMPTY = {"fixtures": {}, "scoreboard": {"holdout": None, "n_matches": 0, "models": {}}}


def load(path=_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"fixtures": {}, "scoreboard": dict(_EMPTY["scoreboard"])}


def fixture_probs(data: dict, home: str, away: str) -> dict:
    key = f"{home}|{away}"
    if key not in data["fixtures"]:
        raise KeyError(key)
    return data["fixtures"][key]


def scoreboard(data: dict) -> dict:
    return data["scoreboard"]


DATA = load()
