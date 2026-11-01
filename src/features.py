"""Chase-state features for win-probability modelling.

Each row is the state of a T20 run chase *before* a legal delivery:
what the chasing side knows at that moment. The label is whether the
chasing side eventually won the match. Features use only information
available before that ball, so there is no look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

FEATURES = [
    "runs_needed",
    "balls_left",
    "wickets_left",
    "crr",
    "rrr",
    "rrr_minus_crr",
    "target",
]


def state_frame(runs_needed, balls_left, wickets_left, runs_scored=None,
                balls_bowled=None, target=None) -> pd.DataFrame:
    """Build the feature matrix from raw chase-state values (vectorised)."""
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_left = np.asarray(balls_left, dtype=float)
    wickets_left = np.asarray(wickets_left, dtype=float)
    if balls_bowled is None:
        balls_bowled = config.T20_BALLS - balls_left
    balls_bowled = np.asarray(balls_bowled, dtype=float)
    if runs_scored is None:
        runs_scored = (target - runs_needed) if target is not None else np.zeros_like(runs_needed)
    runs_scored = np.asarray(runs_scored, dtype=float)
    if target is None:
        target = runs_scored + runs_needed
    target = np.asarray(target, dtype=float)

    crr = np.where(balls_bowled > 0, runs_scored * config.BALLS_PER_OVER / np.maximum(balls_bowled, 1), 0.0)
    rrr = np.where(balls_left > 0, runs_needed * config.BALLS_PER_OVER / np.maximum(balls_left, 1), np.inf)
    rrr = np.minimum(rrr, 99.0)
    return pd.DataFrame({
        "runs_needed": runs_needed,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "crr": crr,
        "rrr": rrr,
        "rrr_minus_crr": rrr - crr,
        "target": target,
    })


def adjusted_result_mask(matches: pd.DataFrame) -> pd.Series:
    """True for matches with a non-empty outcome 'method' (D/L, Awarded, ...).

    Their chase targets may have been revised, so they are excluded from win-probability
    training. The Cricsheet innings target field is not used because the safer rule is method-based.
    """
    m = matches["method"]
    return m.notna() & (m.astype(str).str.strip() != "")


def _chase_candidate_mask(matches: pd.DataFrame) -> pd.Series:
    """Decided, complete, 20-over chases (before the adjusted-result exclusion)."""
    return (
        matches["inn1_runs"].notna()
        & matches["inn2_team"].notna()
        & matches["winner"].notna()
        & (matches["overs"].fillna(config.T20_OVERS) == config.T20_OVERS)
        & ((matches["winner"] == matches["inn1_team"]) | (matches["winner"] == matches["inn2_team"]))
    )


def chase_training_exclusions(matches: pd.DataFrame) -> dict:
    """Counts for the report: how many candidate chases were excluded for an adjusted result."""
    candidates = matches[_chase_candidate_mask(matches)]
    adjusted = candidates[adjusted_result_mask(candidates)]
    return {
        "candidate_decided_20_over_chases": int(len(candidates)),
        "excluded_adjusted_result_matches": int(len(adjusted)),
        "excluded_by_method": {str(k): int(v) for k, v in adjusted["method"].value_counts().items()},
        "included_matches": int(len(candidates) - len(adjusted)),
        "rule": "Matches with a non-empty outcome 'method' (D/L, Awarded) are excluded from chase-state training.",
    }


def build_chase_states(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per legal ball in a complete, decided, unadjusted 20-over chase."""
    eligible = matches[_chase_candidate_mask(matches) & ~adjusted_result_mask(matches)]
    usable = eligible[["match_id", "date", "season", "inn1_runs", "inn2_team", "winner", "event", "match_type"]]
    usable = usable.copy()
    usable["target"] = usable["inn1_runs"].astype(int) + 1
    usable["label"] = (usable["winner"] == usable["inn2_team"]).astype(int)

    second = deliveries[(deliveries["innings"] == 2) & (deliveries["legal"] == 1)]
    second = second[second["match_id"].isin(usable["match_id"])]
    frame = second[["match_id", "legal_before", "runs_before", "wickets_before"]].copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame = frame.merge(usable, on="match_id", how="inner")

    frame = frame[(frame["runs_before"] < frame["target"]) &
                  (frame["wickets_before"] < config.MAX_WICKETS) &
                  (frame["legal_before"] < config.T20_BALLS)]
    features = state_frame(
        runs_needed=frame["target"] - frame["runs_before"],
        balls_left=config.T20_BALLS - frame["legal_before"],
        wickets_left=config.MAX_WICKETS - frame["wickets_before"],
        runs_scored=frame["runs_before"],
        balls_bowled=frame["legal_before"],
        target=frame["target"],
    )
    out = pd.concat([
        frame[["match_id", "date", "season", "event", "match_type", "label"]].reset_index(drop=True),
        features.reset_index(drop=True),
    ], axis=1)
    out["match_id"] = out["match_id"].astype("category")
    return out


def chronological_split(states: pd.DataFrame):
    """Match-grouped, chronological split: test = latest matches, validation = latest of the rest.

    Returns (fit_mask, val_mask, test_mask) boolean arrays aligned to `states`.
    No match appears in more than one partition.
    """
    match_dates = states.groupby("match_id", observed=True)["date"].first().sort_values(kind="stable")
    ids = match_dates.index.to_numpy()
    n = len(ids)
    n_test = max(1, int(round(n * config.TEST_FRACTION)))
    n_train = n - n_test
    n_val = max(1, int(round(n_train * config.VALIDATION_FRACTION)))
    test_ids = set(ids[n_train:])
    val_ids = set(ids[n_train - n_val:n_train])
    fit_ids = set(ids[:n_train - n_val])

    mid = states["match_id"].astype(str).to_numpy()
    test_mask = np.isin(mid, list(test_ids))
    val_mask = np.isin(mid, list(val_ids))
    fit_mask = np.isin(mid, list(fit_ids))
    return fit_mask, val_mask, test_mask
