"""Player, matchup, team and match analytics computed from ball-by-ball data.

All functions take the tidy tables produced by src/ingest.py. Definitions:
  Batting  : balls faced exclude wides; runs = runs off the bat; dismissals exclude
             'retired hurt' / 'retired not out'.
  Bowling  : legal balls only (no wides/no-balls); runs conceded exclude byes and
             leg-byes; wickets credited only for bowler-credited dismissal kinds.
  Dot ball : a legal ball on which no runs were scored off the bat (batting) or
             no runs were conceded at all (bowling).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class Filters:
    match_type: str | None = None
    season: str | None = None
    event: str | None = None
    team: str | None = None

    def key(self) -> tuple:
        return (self.match_type, self.season, self.event, self.team)


def filter_matches(matches: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = matches
    if f.match_type:
        m = m[m["match_type"] == f.match_type]
    if f.season:
        m = m[m["season"] == f.season]
    if f.event:
        m = m[m["event"] == f.event]
    if f.team:
        m = m[(m["team1"] == f.team) | (m["team2"] == f.team)]
    return m


def filter_deliveries(deliveries: pd.DataFrame, matches: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if not (f.match_type or f.season or f.event or f.team):
        return deliveries
    ids = filter_matches(matches, f)["match_id"].astype(str).unique()
    return deliveries[deliveries["match_id"].isin(ids)]


def _text(v):
    """Missing text fields come back as None instead of NaN."""
    return None if v is None or pd.isna(v) else str(v)


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def _dismissed(d: pd.DataFrame) -> pd.DataFrame:
    kinds = d["wicket_kind"].astype(object)
    return d[d["player_out"].notna() & ~kinds.isin(config.NOT_DISMISSAL_KINDS)]


def batting_table(d: pd.DataFrame) -> pd.DataFrame:
    faced = d[d["extra_type"].astype(object) != "wides"]
    g = faced.groupby("batter", observed=True)
    out = pd.DataFrame({
        "matches": d.groupby("batter", observed=True)["match_id"].nunique(),
        "balls": g.size(),
        "runs": d.groupby("batter", observed=True)["runs_batter"].sum(),
        "fours": faced[faced["runs_batter"] == 4].groupby("batter", observed=True).size(),
        "sixes": faced[faced["runs_batter"] == 6].groupby("batter", observed=True).size(),
        "dots": faced[faced["runs_batter"] == 0].groupby("batter", observed=True).size(),
        "dismissals": _dismissed(d).groupby("player_out", observed=True).size(),
    })
    out.index = out.index.astype(str)
    out = out.fillna(0)
    out["boundaries"] = out["fours"] + out["sixes"]
    out["strike_rate"] = _safe_div(out["runs"] * 100, out["balls"])
    out["average"] = _safe_div(out["runs"], out["dismissals"])
    out["boundary_pct"] = _safe_div(out["boundaries"] * 100, out["balls"])
    out["dot_pct"] = _safe_div(out["dots"] * 100, out["balls"])
    return out


def bowling_table(d: pd.DataFrame) -> pd.DataFrame:
    legal = d[d["legal"] == 1]
    conceded = d["runs_total"] - d["runs_byes_legbyes"]
    d2 = d.assign(conceded=conceded)
    kinds = d["wicket_kind"].astype(object)
    credited = d[d["player_out"].notna() & kinds.isin(config.BOWLER_CREDIT_KINDS)]
    out = pd.DataFrame({
        "bowl_matches": d.groupby("bowler", observed=True)["match_id"].nunique(),
        "legal_balls": legal.groupby("bowler", observed=True).size(),
        "runs_conceded": d2.groupby("bowler", observed=True)["conceded"].sum(),
        "wickets": credited.groupby("bowler", observed=True).size(),
        "bowl_dots": legal[legal["runs_total"] == 0].groupby("bowler", observed=True).size(),
    })
    out.index = out.index.astype(str)
    out = out.fillna(0)
    out["overs"] = out["legal_balls"] / config.BALLS_PER_OVER
    out["economy"] = _safe_div(out["runs_conceded"] * config.BALLS_PER_OVER, out["legal_balls"])
    out["bowling_strike_rate"] = _safe_div(out["legal_balls"], out["wickets"])
    out["bowling_average"] = _safe_div(out["runs_conceded"], out["wickets"])
    out["bowling_dot_pct"] = _safe_div(out["bowl_dots"] * 100, out["legal_balls"])
    return out


def player_summary(d: pd.DataFrame, min_balls: int = 0) -> pd.DataFrame:
    bat = batting_table(d)
    bowl = bowling_table(d)
    out = bat.join(bowl, how="outer").fillna(0)
    out.index.name = "player"
    if min_balls:
        out = out[(out["balls"] >= min_balls) | (out["legal_balls"] >= min_balls)]
    return out.reset_index()


def player_profile(d: pd.DataFrame, player: str, summary: pd.DataFrame | None = None) -> dict:
    """Full metric set for one player, using the already-filtered deliveries."""
    if summary is None:
        summary = player_summary(d)
    row = summary[summary["player"] == player]
    if row.empty:
        return {"player": player, "found": False}
    r = row.iloc[0]
    return {
        "player": player, "found": True,
        "matches": int(max(r.get("matches", 0), r.get("bowl_matches", 0))),
        "batting": {
            "runs": int(r["runs"]), "balls_faced": int(r["balls"]),
            "dismissals": int(r["dismissals"]),
            "average": _num(r["average"]), "strike_rate": _num(r["strike_rate"]),
            "fours": int(r["fours"]), "sixes": int(r["sixes"]),
            "boundary_pct": _num(r["boundary_pct"]), "dot_pct": _num(r["dot_pct"]),
        },
        "bowling": {
            "legal_balls": int(r["legal_balls"]), "overs": _num(r["overs"], 1),
            "wickets": int(r["wickets"]), "runs_conceded": int(r["runs_conceded"]),
            "economy": _num(r["economy"]), "bowling_strike_rate": _num(r["bowling_strike_rate"]),
            "bowling_average": _num(r["bowling_average"]), "dot_pct": _num(r["bowling_dot_pct"]),
        },
        "pressure": pressure_index(d, player),
    }


def _num(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), nd)


def compare_players(d: pd.DataFrame, a: str, b: str, summary: pd.DataFrame | None = None) -> dict:
    if summary is None:
        summary = player_summary(d)
    pa, pb = player_profile(d, a, summary), player_profile(d, b, summary)
    radar_keys = [
        ("Strike rate", "batting", "strike_rate"),
        ("Average", "batting", "average"),
        ("Boundary %", "batting", "boundary_pct"),
        ("Economy (lower better)", "bowling", "economy"),
        ("Wickets", "bowling", "wickets"),
        ("Dot % (bowling)", "bowling", "dot_pct"),
    ]
    radar = []
    for label, group, key in radar_keys:
        va = (pa.get(group) or {}).get(key) if pa.get("found") else None
        vb = (pb.get(group) or {}).get(key) if pb.get("found") else None
        va = va or 0.0
        vb = vb or 0.0
        top = max(va, vb) or 1.0
        inverse = "lower" in label
        radar.append({
            "axis": label,
            "a_raw": va, "b_raw": vb,
            "a": round((top - va if inverse else va) / top, 3) if top else 0,
            "b": round((top - vb if inverse else vb) / top, 3) if top else 0,
        })
    return {"a": pa, "b": pb, "radar": radar}


def pressure_index(d: pd.DataFrame, player: str) -> dict:
    """Custom Pressure Performance Index (NOT an official cricket statistic).

    For each pressure context C with at least MIN_CONTEXT_BALLS balls faced:
        ratio_C = strike_rate_in_C / career_strike_rate
    PI = 100 * mean(ratio_C over valid contexts). 100 = same scoring rate under
    pressure as overall; above 100 = scores faster under pressure.
    Contexts: death overs (16-20); chase with required run rate >= 10;
    batting with 4+ wickets down; batting in a chase the side went on to win.
    """
    faced = d[(d["batter"].astype(str) == player) & (d["extra_type"].astype(object) != "wides")]
    career_balls = len(faced)
    career_runs = int(faced["runs_batter"].sum())
    career_sr = (career_runs * 100 / career_balls) if career_balls else None
    contexts = {}
    if career_sr is None or career_balls == 0:
        return {"index": None, "career_strike_rate": None, "contexts": contexts,
                "note": "No balls faced in this selection."}

    death = faced["phase"].astype(str) == "death"
    rrr = pd.to_numeric(faced["rrr_before"], errors="coerce")
    high_rrr = (faced["innings"] == 2) & (rrr >= 10)
    wkts = faced["wickets_before"] >= 4
    won_chase = (faced["innings"] == 2) & (faced["won_innings"] == 1)
    masks = {"death_overs": death, "high_rrr_chase": high_rrr,
             "four_plus_wickets_down": wkts, "won_chases": won_chase}
    ratios = []
    for name, m in masks.items():
        sub = faced[m.fillna(False).to_numpy()]
        balls = len(sub)
        if balls < config.MIN_CONTEXT_BALLS:
            contexts[name] = {"balls": int(balls), "strike_rate": None, "used": False}
            continue
        sr = sub["runs_batter"].sum() * 100 / balls
        ratios.append(sr / career_sr)
        contexts[name] = {"balls": int(balls), "strike_rate": round(float(sr), 2), "used": True}
    index = round(100 * float(np.mean(ratios)), 1) if ratios else None
    note = None if index is not None else "Not enough balls in pressure contexts to compute the index."
    return {"index": index, "career_strike_rate": round(float(career_sr), 2),
            "career_balls": int(career_balls), "contexts": contexts, "note": note}


def sample_warning(balls: int) -> str:
    if balls < config.MATCHUP_LOW_SAMPLE:
        return "Very small sample: not enough balls to draw any conclusion."
    if balls < config.MATCHUP_OK_SAMPLE:
        return "Small sample: treat these numbers as indicative only."
    return "Adequate sample for descriptive comparison (still historical, not predictive)."


def matchup(d: pd.DataFrame, batter: str, bowler: str) -> dict:
    sub = d[(d["batter"].astype(str) == batter) & (d["bowler"].astype(str) == bowler)]
    faced = sub[sub["extra_type"].astype(object) != "wides"]
    balls = len(faced)
    runs = int(faced["runs_batter"].sum())
    dots = int((faced["runs_batter"] == 0).sum())
    boundaries = int(faced["runs_batter"].isin([4, 6]).sum())
    kinds = sub["wicket_kind"].astype(object)
    out_by_bowler = sub[(sub["player_out"].astype(str) == batter) & kinds.isin(config.BOWLER_CREDIT_KINDS)]
    return {
        "batter": batter, "bowler": bowler,
        "balls": int(balls), "runs": runs,
        "dismissals": int(len(out_by_bowler)),
        "strike_rate": _num(runs * 100 / balls) if balls else None,
        "dot_pct": _num(dots * 100 / balls) if balls else None,
        "boundary_pct": _num(boundaries * 100 / balls) if balls else None,
        "matches": int(sub["match_id"].nunique()),
        "warning": sample_warning(balls),
    }


def matchup_candidates(d: pd.DataFrame, batter: str, min_balls: int = 1) -> list[dict]:
    """Bowlers who have bowled to a batter, ordered by balls faced."""
    sub = d[(d["batter"].astype(str) == batter) & (d["extra_type"].astype(object) != "wides")]
    counts = sub.groupby("bowler", observed=True).size().sort_values(ascending=False)
    counts = counts[counts >= min_balls]
    return [{"bowler": str(k), "balls": int(v)} for k, v in counts.head(200).items()]


def team_table(matches: pd.DataFrame) -> pd.DataFrame:
    decided = matches[matches["winner"].notna()]
    teams = pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna())
    rows = []
    for t in teams:
        played = matches[(matches["team1"] == t) | (matches["team2"] == t)]
        won = decided[decided["winner"] == t]
        chase = decided[decided["inn2_team"] == t]
        chase_won = chase[chase["winner"] == t]
        defend = decided[decided["inn1_team"] == t]
        defend_won = defend[defend["winner"] == t]
        first = matches[(matches["inn1_team"] == t) & matches["inn1_runs"].notna()]
        toss = decided[decided["toss_winner"] == t]
        rows.append({
            "team": t,
            "played": int(len(played)),
            "decided": int(len(played[played["winner"].notna()])),
            "wins": int(len(won)),
            "win_pct": _num(len(won) * 100 / len(played[played["winner"].notna()]), 1) if len(played[played["winner"].notna()]) else None,
            "chases": int(len(chase)),
            "chase_wins": int(len(chase_won)),
            "chase_win_pct": _num(len(chase_won) * 100 / len(chase), 1) if len(chase) else None,
            "bat_first": int(len(defend)),
            "defend_wins": int(len(defend_won)),
            "defend_win_pct": _num(len(defend_won) * 100 / len(defend), 1) if len(defend) else None,
            "avg_first_innings": _num(first["inn1_runs"].mean(), 1) if len(first) else None,
            "toss_win_match_win_pct": _num(len(toss[toss["winner"] == t]) * 100 / len(toss), 1) if len(toss) else None,
        })
    return pd.DataFrame(rows).sort_values("wins", ascending=False).reset_index(drop=True)


def head_to_head(matches: pd.DataFrame, top_n: int = 12) -> dict:
    decided = matches[matches["winner"].notna()]
    counts = pd.concat([decided["team1"], decided["team2"]]).value_counts()
    teams = counts.head(top_n).index.tolist()
    grid = []
    for a in teams:
        row = []
        for b in teams:
            if a == b:
                row.append(None)
                continue
            pair = decided[((decided["team1"] == a) & (decided["team2"] == b)) |
                           ((decided["team1"] == b) & (decided["team2"] == a))]
            n = len(pair)
            wins = int((pair["winner"] == a).sum())
            row.append({"played": int(n), "wins": wins,
                        "win_pct": round(wins * 100 / n, 1) if n else None})
        grid.append(row)
    return {"teams": teams, "grid": grid}


def season_trend(matches: pd.DataFrame, team: str) -> list[dict]:
    decided = matches[matches["winner"].notna() & ((matches["team1"] == team) | (matches["team2"] == team))]
    out = []
    for season, grp in decided.groupby("season"):
        wins = int((grp["winner"] == team).sum())
        out.append({"season": str(season), "played": int(len(grp)), "wins": wins,
                    "win_pct": round(wins * 100 / len(grp), 1)})
    return sorted(out, key=lambda r: r["season"])


def match_detail(matches: pd.DataFrame, deliveries: pd.DataFrame, match_id: str) -> dict | None:
    row = matches[matches["match_id"].astype(str) == str(match_id)]
    if row.empty:
        return None
    m = row.iloc[0]
    d = deliveries[deliveries["match_id"].astype(str) == str(match_id)]
    innings = []
    for inn_no in sorted(d["innings"].unique()):
        di = d[d["innings"] == inn_no]
        per_over = di.groupby("over", observed=True)["runs_total"].sum().sort_index()
        wkts_over = di.groupby("over", observed=True)["wicket_count"].sum().reindex(per_over.index, fill_value=0)
        cumulative = per_over.cumsum()
        innings.append({
            "innings": int(inn_no),
            "team": str(di["batting_team"].iloc[0]),
            "runs": int(di["runs_total"].sum()),
            "wickets": int(di["wicket_count"].sum()),
            "legal_balls": int(di["legal"].sum()),
            "manhattan": [{"over": int(o) + 1, "runs": int(r), "wickets": int(w)}
                          for o, r, w in zip(per_over.index, per_over.values, wkts_over.values)],
            "worm": [{"over": int(o) + 1, "runs": int(c)}
                     for o, c in zip(per_over.index, cumulative.values)],
        })
    return {
        "match_id": str(match_id),
        "date": _text(m["date"]), "season": _text(m["season"]), "event": _text(m["event"]),
        "match_type": _text(m["match_type"]), "venue": _text(m["venue"]), "city": _text(m["city"]),
        "teams": [m["team1"], m["team2"]],
        "toss": {"winner": m["toss_winner"], "decision": m["toss_decision"]},
        "winner": None if pd.isna(m["winner"]) else m["winner"],
        "result": {
            "by_runs": None if pd.isna(m["win_by_runs"]) else int(m["win_by_runs"]),
            "by_wickets": None if pd.isna(m["win_by_wickets"]) else int(m["win_by_wickets"]),
            "result": None if pd.isna(m["result"]) else str(m["result"]),
            "method": None if pd.isna(m["method"]) else str(m["method"]),
        },
        "innings": innings,
    }


def overview(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict:
    decided = matches[matches["winner"].notna()]
    chases = decided[decided["inn2_team"].notna()]
    chase_win_rate = (chases["winner"] == chases["inn2_team"]).mean() if len(chases) else None
    seasons = matches.groupby("season").size().sort_index()
    return {
        "matches": int(len(matches)),
        "decided_matches": int(len(decided)),
        "deliveries": int(len(deliveries)),
        "teams": int(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()).size),
        "date_range": [matches["date"].min(), matches["date"].max()],
        "chase_win_rate": _num(chase_win_rate * 100, 1) if chase_win_rate is not None else None,
        "avg_first_innings": _num(matches["inn1_runs"].mean(), 1),
        "matches_per_season": [{"season": str(k), "matches": int(v)} for k, v in seasons.items()],
        "match_types": matches["match_type"].value_counts().to_dict(),
    }
