"""Ingest Cricsheet T20 JSON ZIP archives into tidy, analysis-ready tables.

Input : one or more Cricsheet "json" ZIP files (https://cricsheet.org/downloads/),
        e.g. t20s_json.zip. Format reference: https://cricsheet.org/format/json/
Output: data/processed/matches.csv       one row per match
        data/processed/deliveries.pkl    one row per ball (pickled pandas DataFrame)
        data/processed/quality_report.json  what was kept, skipped and why

Cleaning rules (all counted in the quality report):
  * super-over innings are excluded from ball-level tables (they are not part of the match)
  * matches are kept even without a winner, but flagged; win/loss statistics ignore them
  * deliveries missing a batter/bowler are dropped
  * duplicate match ids across several uploaded ZIPs are kept once
"""
from __future__ import annotations

import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

import config

BALL_COLUMNS = [
    "match_id", "innings", "batting_team", "bowling_team", "over",
    "batter", "bowler", "non_striker", "legal",
    "runs_batter", "runs_extras", "runs_total", "runs_byes_legbyes", "extra_type",
    "wicket_count", "player_out", "wicket_kind",
    "legal_before", "runs_before", "wickets_before",
]


def _iter_json_members(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".json"):
                continue
            yield name, json.loads(archive.read(name))


def parse_match(name: str, data: dict, report: Counter):
    """Return (match_row, ball_rows) for one Cricsheet match, or (None, []) if unusable."""
    info = data.get("info") or {}
    match_id = Path(name).stem
    outcome = info.get("outcome") or {}
    by = outcome.get("by") or {}
    toss = info.get("toss") or {}
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    event = info.get("event") or {}

    innings_list = []
    for inn in data.get("innings") or []:
        if inn.get("super_over"):
            report["super_over_innings_excluded"] += 1
            continue
        innings_list.append(inn)

    match_row = {
        "match_id": match_id,
        "date": dates[0] if dates else None,
        "season": str(info.get("season", "")),
        "event": event.get("name", "") or "",
        "match_type": info.get("match_type", "") or "",
        "gender": info.get("gender", "") or "",
        "team_type": info.get("team_type", "") or "",
        "overs": info.get("overs"),
        "venue": info.get("venue", "") or "",
        "city": info.get("city", "") or "",
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"),
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "result": outcome.get("result"),
        "method": outcome.get("method"),
    }

    if len(innings_list) == 0:
        report["matches_without_innings"] += 1
        return None, []

    ball_rows = []
    summaries = []
    for inn_no, inn in enumerate(innings_list[:2], start=1):
        batting = inn.get("team")
        if len(teams) == 2:
            bowling = teams[1] if batting == teams[0] else teams[0]
        else:
            bowling = None
        legal_before = runs_before = wickets_before = 0
        for over_obj in inn.get("overs") or []:
            over_no = over_obj.get("over")
            for dlv in over_obj.get("deliveries") or []:
                batter, bowler = dlv.get("batter"), dlv.get("bowler")
                if not batter or not bowler:
                    report["deliveries_missing_player_dropped"] += 1
                    continue
                runs = dlv.get("runs") or {}
                extras = dlv.get("extras") or {}
                total = int(runs.get("total", 0) or 0)
                bat_runs = int(runs.get("batter", 0) or 0)
                extras_runs = int(runs.get("extras", total - bat_runs) or 0)
                legal = not ("wides" in extras or "noballs" in extras)
                wickets = dlv.get("wickets") or []
                first = wickets[0] if wickets else {}
                ball_rows.append((
                    match_id, inn_no, batting, bowling, over_no,
                    batter, bowler, dlv.get("non_striker"), int(legal),
                    bat_runs, extras_runs, total,
                    int(extras.get("byes", 0) or 0) + int(extras.get("legbyes", 0) or 0),
                    next(iter(extras), ""),
                    len(wickets), first.get("player_out"), first.get("kind"),
                    legal_before, runs_before, wickets_before,
                ))
                legal_before += int(legal)
                runs_before += total
                wickets_before += len(wickets)
        summaries.append({"team": batting, "runs": runs_before,
                          "wickets": wickets_before, "legal_balls": legal_before})

    if summaries:
        match_row["inn1_team"] = summaries[0]["team"]
        match_row["inn1_runs"] = summaries[0]["runs"]
        match_row["inn1_wickets"] = summaries[0]["wickets"]
        match_row["inn1_balls"] = summaries[0]["legal_balls"]
    if len(summaries) == 2:
        match_row["inn2_team"] = summaries[1]["team"]
        match_row["inn2_runs"] = summaries[1]["runs"]
        match_row["inn2_wickets"] = summaries[1]["wickets"]
        match_row["inn2_balls"] = summaries[1]["legal_balls"]
    else:
        report["matches_with_single_innings"] += 1

    if match_row["winner"] is None:
        report["matches_without_winner"] += 1
    if match_row.get("method"):
        report["matches_with_method_adjustment_dl"] += 1
    if (match_row.get("overs") or config.T20_OVERS) != config.T20_OVERS:
        report["matches_not_20_overs"] += 1
    report["matches_parsed"] += 1
    return match_row, ball_rows


def build_dataset(zip_paths: list[Path], out_dir: Path = config.PROCESSED_DIR) -> dict:
    """Parse all ZIPs and write processed tables. Returns the quality report."""
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Counter = Counter()
    match_rows: dict[str, dict] = {}
    ball_chunks: list[list[tuple]] = []
    seen_ids: set[str] = set()

    for zip_path in zip_paths:
        for name, data in _iter_json_members(zip_path):
            report["json_files_read"] += 1
            try:
                row, balls = parse_match(name, data, report)
            except (KeyError, TypeError, ValueError, AttributeError):
                report["malformed_match_skipped"] += 1
                continue
            if row is None:
                continue
            if row["match_id"] in seen_ids:
                report["duplicate_matches_ignored"] += 1
                continue
            seen_ids.add(row["match_id"])
            match_rows[row["match_id"]] = row
            ball_chunks.append(balls)

    matches = pd.DataFrame(list(match_rows.values()))
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    deliveries = pd.DataFrame(
        [b for chunk in ball_chunks for b in chunk], columns=BALL_COLUMNS
    )

    # Target score for the chasing side (second innings), used for required-run-rate.
    target = (matches.set_index("match_id")["inn1_runs"] + 1)
    deliveries["target"] = deliveries["match_id"].map(target).where(deliveries["innings"] == 2)
    balls_left = config.T20_BALLS - deliveries["legal_before"]
    runs_needed = deliveries["target"] - deliveries["runs_before"]
    deliveries["rrr_before"] = (runs_needed * config.BALLS_PER_OVER / balls_left.where(balls_left > 0)).where(deliveries["innings"] == 2)
    deliveries["phase"] = deliveries["over"].map(config.phase_of_over)
    winners = matches.set_index("match_id")["winner"]
    deliveries["won_innings"] = (
        deliveries["batting_team"] == deliveries["match_id"].map(winners)
    ).astype("int8")

    for col in ["match_id", "batting_team", "bowling_team", "batter", "bowler",
                "non_striker", "extra_type", "player_out", "wicket_kind", "phase"]:
        deliveries[col] = deliveries[col].astype("category")
    for col in ["innings", "over", "legal", "runs_batter", "runs_extras", "runs_total",
                "runs_byes_legbyes", "wicket_count", "legal_before", "runs_before",
                "wickets_before"]:
        deliveries[col] = pd.to_numeric(deliveries[col], downcast="integer")

    matches.to_csv(config.PROCESSED_MATCHES, index=False)
    deliveries.to_pickle(config.PROCESSED_DELIVERIES)

    summary = {
        "source_files": [p.name for p in zip_paths],
        "matches_total": int(len(matches)),
        "matches_with_winner": int(matches["winner"].notna().sum()),
        "matches_without_winner": int(matches["winner"].isna().sum()),
        "deliveries_total": int(len(deliveries)),
        "legal_deliveries": int(deliveries["legal"].sum()),
        "match_types": matches["match_type"].value_counts().to_dict(),
        "seasons": sorted(matches["season"].unique().tolist()),
        "date_range": [matches["date"].min(), matches["date"].max()],
        "build_seconds": round(time.time() - started, 1),
        "counters": dict(report),
        "limitations": [
            "Cricsheet data is ball-by-ball as recorded by the scorers; mistakes in the source remain.",
            "Super-over deliveries are excluded from ball-level statistics.",
            "Dismissal kinds are the first wicket recorded on a delivery; rare multi-wicket deliveries are counted in totals but only the first kind is stored.",
            "Duckworth-Lewis / rain-shortened matches are kept but flagged via 'method'; chase modelling uses full 20-over matches only.",
            "Player names are Cricsheet registry names; different spellings of the same person may not be merged.",
            "Wides are not balls faced by the batter and are not balls bowled by the bowler.",
        ],
    }
    config.QUALITY_REPORT.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def find_raw_zips() -> list[Path]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in config.RAW_DIR.glob("*.zip") if p.is_file())


if __name__ == "__main__":
    zips = find_raw_zips()
    if not zips:
        raise SystemExit(f"No .zip files found in {config.RAW_DIR}. Add a Cricsheet ZIP there.")
    result = build_dataset(zips)
    print(json.dumps({k: v for k, v in result.items() if k != "limitations"}, indent=2, default=str))
