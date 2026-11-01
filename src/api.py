"""JSON API handlers. Each function takes the DataStore and a query dict and returns a dict.

Errors are raised as ApiError(status, message) and turned into JSON responses by app.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import analytics as an
from src import model as model_mod
from src.features import FEATURES, state_frame
from src.strategy import analyse


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _need_data(store):
    if not store.ready:
        raise ApiError(503, "No data loaded. Upload a Cricsheet T20 ZIP on the Data page, or place it in data/raw/ and restart.")


def _filters(q: dict) -> an.Filters:
    def g(name):
        v = (q.get(name) or "").strip()
        return v or None
    return an.Filters(match_type=g("match_type"), season=g("season"),
                      event=g("event"), team=g("team"))


def _int(q, name, default=None, lo=None, hi=None) -> int | None:
    raw = q.get(name)
    if raw in (None, ""):
        if default is None:
            raise ApiError(400, f"Missing parameter '{name}'.")
        return default
    try:
        v = int(float(raw))
    except ValueError:
        raise ApiError(400, f"Parameter '{name}' must be a number.")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ApiError(400, f"Parameter '{name}' must be between {lo} and {hi}.")
    return v


def _filtered(store, f: an.Filters):
    matches = an.filter_matches(store.matches, f)
    deliveries = an.filter_deliveries(store.deliveries, store.matches, f)
    return matches, deliveries


def _summary(store, f: an.Filters, deliveries):
    return store.cached(("summary",) + f.key(), lambda: an.player_summary(deliveries))


def _season_map(store):
    return store.cached("season_map", lambda: store.matches.set_index(store.matches["match_id"].astype(str))["season"])


def _sanitize_rows(df: pd.DataFrame) -> list[dict]:
    """Records as plain Python values, with NaN/inf turned into null for valid JSON."""
    clean = df.astype(object).where(pd.notna(df), None)
    rows = clean.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and not np.isfinite(v):
                row[k] = None
            elif isinstance(v, np.generic):
                row[k] = v.item()
    return rows


# ---- endpoints -------------------------------------------------------------
def filters(store, q):
    _need_data(store)
    def build():
        m = store.matches
        players = pd.unique(pd.concat([store.deliveries["batter"].astype(str),
                                       store.deliveries["bowler"].astype(str)]))
        return {
            "seasons": sorted(m["season"].dropna().unique().tolist()),
            "match_types": sorted(m["match_type"].dropna().unique().tolist()),
            "events": sorted([e for e in m["event"].dropna().unique() if e]),
            "teams": sorted(pd.unique(pd.concat([m["team1"], m["team2"]]).dropna()).tolist()),
            "players": sorted(players.tolist()),
        }
    return store.cached("filters", build)


def overview(store, q):
    _need_data(store)
    f = _filters(q)
    matches, deliveries = _filtered(store, f)
    base = an.overview(matches, deliveries)
    summary = _summary(store, f, deliveries)
    batters = summary[summary["balls"] >= config.MIN_BALLS_RANKING].sort_values("runs", ascending=False).head(10)
    bowlers = summary[summary["legal_balls"] >= config.MIN_BALLS_RANKING].sort_values("wickets", ascending=False).head(10)
    teams = an.team_table(matches).head(10)
    base.update({
        "top_batters": _sanitize_rows(batters[["player", "runs", "strike_rate", "average", "balls"]]),
        "top_bowlers": _sanitize_rows(bowlers[["player", "wickets", "economy", "bowling_average", "legal_balls"]]),
        "top_teams": _sanitize_rows(teams[["team", "played", "wins", "win_pct", "chase_win_pct"]]),
        "filters": {"match_type": f.match_type, "season": f.season, "event": f.event, "team": f.team},
        "ranking_rule": f"Players ranked with at least {config.MIN_BALLS_RANKING} balls in the selected data.",
    })
    return base


def matches(store, q):
    _need_data(store)
    f = _filters(q)
    m = an.filter_matches(store.matches, f).sort_values(["date", "match_id"], ascending=False)
    limit = _int(q, "limit", 50, 1, 500)
    offset = _int(q, "offset", 0, 0)
    page = m.iloc[offset:offset + limit]
    cols = ["match_id", "date", "season", "event", "match_type", "team1", "team2", "winner",
            "inn1_team", "inn1_runs", "inn1_wickets", "inn2_team", "inn2_runs", "inn2_wickets", "venue"]
    return {"total": int(len(m)), "offset": offset, "limit": limit,
            "rows": _sanitize_rows(page[cols])}


def match(store, q):
    _need_data(store)
    mid = (q.get("id") or "").strip()
    if not mid:
        raise ApiError(400, "Missing parameter 'id'.")
    detail = an.match_detail(store.matches, store.deliveries, mid)
    if detail is None:
        raise ApiError(404, f"Match {mid} not found.")
    detail["winprob"] = _match_winprob(store, mid, detail)
    return detail


def _match_winprob(store, mid, detail):
    """Model win probability after each legal ball of the chase innings (if a model exists)."""
    if not any(i["innings"] == 2 for i in detail["innings"]):
        return {"available": False, "reason": "Match has no second innings."}
    if model_mod.load_model() is None:
        return {"available": False, "reason": "Model not trained yet."}
    d2 = store.deliveries[(store.deliveries["match_id"].astype(str) == mid) & (store.deliveries["innings"] == 2)
                          & (store.deliveries["legal"] == 1)]
    if d2.empty:
        return {"available": False, "reason": "No chase deliveries recorded."}
    target = int(d2["target"].iloc[0])
    feats = state_frame(
        runs_needed=target - d2["runs_before"].to_numpy(),
        balls_left=config.T20_BALLS - d2["legal_before"].to_numpy(),
        wickets_left=config.MAX_WICKETS - d2["wickets_before"].to_numpy(),
        runs_scored=d2["runs_before"].to_numpy(),
        balls_bowled=d2["legal_before"].to_numpy(),
        target=np.full(len(d2), target),
    )
    valid = (feats["runs_needed"] > 0).to_numpy() & (feats["balls_left"] > 0).to_numpy() & (feats["wickets_left"] > 0).to_numpy()
    probs = model_mod.predict_win_probability(feats[valid])
    balls = (d2["legal_before"].to_numpy() + 1)[valid]
    return {"available": True, "target": target,
            "points": [{"ball": int(b), "over": round(b / config.BALLS_PER_OVER, 2), "probability": round(float(p), 4)}
                       for b, p in zip(balls, probs)],
            "note": "Model estimate for the chasing side's win probability at each legal ball."}


def strategy(store, q):
    _need_data(store)
    runs_needed = _int(q, "runs_needed", lo=0, hi=1000)
    balls_left = _int(q, "balls_left", lo=0, hi=config.T20_BALLS)
    wickets_left = _int(q, "wickets_left", lo=0, hi=config.MAX_WICKETS)
    target = _int(q, "target", default=runs_needed + 1, lo=1, hi=1000)
    if runs_needed > target:
        raise ApiError(400, "Runs needed cannot exceed the target.")
    feats = state_frame(runs_needed=[runs_needed], balls_left=[balls_left],
                        wickets_left=[wickets_left], target=[target])
    prob = model_mod.predict_win_probability(feats)
    p = float(prob[0]) if prob is not None else None
    result = analyse({"runs_needed": runs_needed, "balls_left": balls_left,
                      "wickets_left": wickets_left, "target": target},
                     store.dists, store.states, p)
    result["model_available"] = p is not None
    result["simulator_calibration"] = simulator_calibration()
    return result


def simulator_calibration() -> dict | None:
    path = config.MODELS_DIR / "simulator_validation.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())["summary"]
    return {**summary,
            "interpretation": ("The simulator's absolute win probabilities run low compared with realised outcomes "
                               "in matching historical states. Use scenario differences, not absolute values.")}


def winprob_grid(store, q):
    """Win probability (model) over runs needed x balls left for a given wicket count and target."""
    _need_data(store)
    if model_mod.load_model() is None:
        raise ApiError(503, "Model not trained yet. Build the model on the Data page.")
    wickets_left = _int(q, "wickets_left", default=10, lo=1, hi=10)
    target = _int(q, "target", default=160, lo=20, hi=400)
    balls_axis = [120, 102, 90, 78, 66, 54, 42, 30, 18, 12, 6]
    runs_axis = sorted({int(round(x)) for x in np.linspace(max(1, int(target * 0.25)),
                                                           min(target - 1, int(target * 0.95)), 9)})
    n = len(runs_axis) * len(balls_axis)
    feats = state_frame(
        runs_needed=np.repeat(runs_axis, len(balls_axis)),
        balls_left=np.tile(balls_axis, len(runs_axis)),
        wickets_left=np.full(n, wickets_left),
        target=np.full(n, target),
    )
    probs = model_mod.predict_win_probability(feats).reshape(len(runs_axis), len(balls_axis))
    grid = [[round(float(x), 4) for x in row] for row in probs]
    return {"wickets_left": wickets_left, "target": target,
            "runs_needed": runs_axis, "balls_left": balls_axis, "probability": grid,
            "note": "Model estimate. Rows are runs still needed; columns are balls left."}


def player_search(store, q):
    _need_data(store)
    term = (q.get("q") or "").strip().lower()
    names = filters(store, {})["players"]
    hits = [n for n in names if term and term in n.lower()][:25]
    return {"players": hits}


def player(store, q):
    _need_data(store)
    name = (q.get("name") or "").strip()
    if not name:
        raise ApiError(400, "Missing parameter 'name'.")
    f = _filters(q)
    _, deliveries = _filtered(store, f)
    summary = _summary(store, f, deliveries)
    profile = an.player_profile(deliveries, name, summary)
    if profile.get("found"):
        d = deliveries[(deliveries["batter"].astype(str) == name) & (deliveries["extra_type"].astype(object) != "wides")]
        seasons = _season_map(store)
        by_season = d.assign(season=d["match_id"].astype(str).map(seasons)) \
                     .groupby("season", observed=True).agg(balls=("runs_batter", "size"), runs=("runs_batter", "sum"))
        by_season = by_season.reset_index()
        by_season["strike_rate"] = (by_season["runs"] * 100 / by_season["balls"]).round(2)
        profile["by_season"] = _sanitize_rows(by_season.sort_values("season"))
    return profile


def compare(store, q):
    _need_data(store)
    a = (q.get("a") or "").strip()
    b = (q.get("b") or "").strip()
    if not a or not b:
        raise ApiError(400, "Provide two players: 'a' and 'b'.")
    f = _filters(q)
    _, deliveries = _filtered(store, f)
    return an.compare_players(deliveries, a, b, _summary(store, f, deliveries))


def matchup(store, q):
    _need_data(store)
    batter = (q.get("batter") or "").strip()
    bowler = (q.get("bowler") or "").strip()
    f = _filters(q)
    _, deliveries = _filtered(store, f)
    if batter and not bowler:
        return {"batter": batter, "bowlers": an.matchup_candidates(deliveries, batter)}
    if not batter or not bowler:
        raise ApiError(400, "Provide 'batter' and 'bowler'.")
    return an.matchup(deliveries, batter, bowler)


def teams(store, q):
    _need_data(store)
    f = _filters(q)
    # Team filter is not applied here: the table compares all teams.
    matches = an.filter_matches(store.matches, an.Filters(match_type=f.match_type, season=f.season, event=f.event))
    table = an.team_table(matches)
    team = (q.get("team") or "").strip()
    result = {"table": _sanitize_rows(table),
              "head_to_head": an.head_to_head(matches)}
    if team:
        row = table[table["team"] == team]
        result["selected"] = _sanitize_rows(row)[0] if not row.empty else None
        result["season_trend"] = an.season_trend(matches, team)
    return result


def explorer(store, q):
    _need_data(store)
    table = (q.get("table") or "matches").strip()
    limit = _int(q, "limit", 50, 1, 500)
    offset = _int(q, "offset", 0, 0)
    if table == "matches":
        return matches(store, {**q, "limit": str(limit), "offset": str(offset)})
    if table == "deliveries":
        mid = (q.get("match_id") or "").strip()
        if not mid:
            raise ApiError(400, "Deliveries are shown per match: provide 'match_id'.")
        d = store.deliveries[store.deliveries["match_id"].astype(str) == mid]
        cols = ["innings", "over", "batting_team", "bowling_team", "batter", "bowler", "non_striker",
                "legal", "runs_batter", "runs_extras", "runs_total", "extra_type", "wicket_count",
                "player_out", "wicket_kind"]
        page = d.iloc[offset:offset + limit][cols].copy()
        for c in ["batting_team", "bowling_team", "batter", "bowler", "non_striker", "extra_type", "player_out", "wicket_kind"]:
            page[c] = page[c].astype(object)
        return {"total": int(len(d)), "offset": offset, "limit": limit, "rows": _sanitize_rows(page)}
    raise ApiError(400, "table must be 'matches' or 'deliveries'.")


def model_report(store, q):
    if not config.EVAL_FILE.exists():
        raise ApiError(404, "No model evaluation yet. Build the model on the Data page.")
    return json.loads(config.EVAL_FILE.read_text())


def status(store, q):
    return store.status()


def csv_matches(store) -> str:
    _need_data(store)
    cols = ["match_id", "date", "season", "event", "match_type", "team1", "team2", "winner",
            "inn1_team", "inn1_runs", "inn1_wickets", "inn2_team", "inn2_runs", "inn2_wickets", "venue", "city"]
    return store.matches[cols].to_csv(index=False)
