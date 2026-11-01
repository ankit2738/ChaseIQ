"""Run-chase strategy engine.

Three components, each labelled clearly in the UI:

1. Historical analogues (data-grounded): among past chases in the uploaded data
   with the same wickets left and a similar balls-left and required-run-rate, what
   fraction were won? This is an observed frequency, with the sample size shown.

2. Model probability (data-trained): the saved win-probability model's estimate
   for the current state (see src/model.py for validation results).

3. Scenario simulator (assumption-based): Monte Carlo simulation of the remaining
   balls. Each ball's outcome (0-5 runs, or a wicket) is drawn from the historical
   second-innings distribution for its phase (powerplay / middle / death) and its
   wickets-fallen group, then scaled by the scenario multipliers in config.SCENARIOS.
   The multipliers are ASSUMPTIONS, not estimates from data. Use the scenarios to
   compare decisions relative to each other. The simulator's agreement with
   realised outcomes is measured in tests/validate_simulator.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

PHASES = ["powerplay", "middle", "death"]
# Wickets-fallen groups: 0-2, 3-5, 6-7, 8-9 fallen.
WICKET_EDGES = [3, 6, 8]
WICKET_GROUP_LABELS = ["0-2 fallen", "3-5 fallen", "6-7 fallen", "8-9 fallen"]
N_RUN_VALUES = 6   # 0..5 runs per legal ball; 6+ is rare and folded into 5


def wicket_group(fallen):
    return np.digitize(np.asarray(fallen), WICKET_EDGES)


def phase_distributions(deliveries: pd.DataFrame) -> dict:
    """Empirical second-innings probabilities by (phase, wickets-fallen group).

    Returns {"cdf": array[3 phases, 4 groups, 6 run values],
             "wicket_prob": array[3, 4], "balls": array[3, 4], "phases": PHASES}
    """
    legal = deliveries[(deliveries["legal"] == 1) & (deliveries["innings"] == 2)]
    groups = wicket_group(legal["wickets_before"].to_numpy())
    phase_arr = legal["phase"].astype(str).to_numpy()
    runs = np.clip(legal["runs_total"].to_numpy(), 0, N_RUN_VALUES - 1)
    wicket = (legal["wicket_count"].to_numpy() > 0)

    probs = np.zeros((len(PHASES), len(WICKET_GROUP_LABELS), N_RUN_VALUES))
    wprob = np.zeros((len(PHASES), len(WICKET_GROUP_LABELS)))
    balls = np.zeros((len(PHASES), len(WICKET_GROUP_LABELS)), dtype=int)
    for i, phase in enumerate(PHASES):
        for g in range(len(WICKET_GROUP_LABELS)):
            m = (phase_arr == phase) & (groups == g)
            n = int(m.sum())
            if n == 0:
                raise ValueError(f"No second-innings deliveries for phase '{phase}', group {g}.")
            counts = np.bincount(runs[m], minlength=N_RUN_VALUES)[:N_RUN_VALUES].astype(float)
            probs[i, g] = counts / counts.sum()
            wprob[i, g] = wicket[m].mean()
            balls[i, g] = n
    weights = balls / balls.sum()
    base_runs_per_ball = float((probs @ np.arange(N_RUN_VALUES) * weights).sum())
    return {"cdf": np.cumsum(probs, axis=2), "probs": probs, "wicket_prob": wprob,
            "balls": balls, "phases": PHASES, "base_runs_per_ball": base_runs_per_ball}


PACE_LIMITS = (0.5, 2.0)
# A scenario must beat balanced by at least this many points before it is recommended.
MATERIAL_DIFFERENCE_PP = 2.0


def pace_factor(crr: float, dists: dict) -> float:
    """How fast this chase is scoring compared with the historical second-innings average.

    1.0 = league-average scoring; 1.3 = scoring 30% faster. Clamped to PACE_LIMITS, and 1.0
    at the start of a chase (no scoring rate yet).
    """
    base = dists.get("base_runs_per_ball") or 0.0
    if crr <= 0 or base <= 0:
        return 1.0
    return float(np.clip((crr / config.BALLS_PER_OVER) / base, *PACE_LIMITS))


def simulate_chase(runs_needed: int, balls_left: int, wickets_left: int,
                   balls_bowled: int, dists: dict, scenario: dict, pace: float = 1.0,
                   n_sims: int = config.SIM_RUNS, seed: int = config.SIM_SEED) -> dict:
    """Monte Carlo of the rest of a chase under one scenario.

    `pace` scales historical run outcomes to the chasing side's current scoring rate
    (see pace_factor). Scenario multipliers are applied on top of it.
    """
    rng = np.random.default_rng(seed)
    need = np.full(n_sims, float(runs_needed))
    wk = np.full(n_sims, int(wickets_left), dtype=int)
    start_wk = int(wickets_left)
    scored = np.zeros(n_sims)
    status = np.zeros(n_sims, dtype=np.int8)     # 0 running, 1 won, 2 lost
    if runs_needed <= 0:
        status[:] = 1
    elif wickets_left <= 0 or balls_left <= 0:
        status[:] = 2

    phase_index = {p: i for i, p in enumerate(dists["phases"])}
    for b in range(max(0, balls_left)):
        idx = np.nonzero(status == 0)[0]
        if idx.size == 0:
            break
        over_index = (balls_bowled + b) // config.BALLS_PER_OVER
        pi = phase_index[config.phase_of_over(over_index)]
        groups = wicket_group(config.MAX_WICKETS - wk[idx])
        cdf_rows = dists["cdf"][pi][groups]                  # (k, 6)
        u = rng.random(idx.size)
        base = (u[:, None] > cdf_rows).sum(axis=1).clip(0, N_RUN_VALUES - 1).astype(float)
        scaled = base * scenario["run_multiplier"] * pace
        floor = np.floor(scaled)
        runs = floor + (rng.random(idx.size) < (scaled - floor))       # stochastic rounding
        wprob = np.minimum(1.0, dists["wicket_prob"][pi][groups] * scenario["wicket_multiplier"])
        is_wicket = rng.random(idx.size) < wprob
        need[idx] -= runs
        scored[idx] += runs
        wk[idx] -= is_wicket.astype(int)
        won = need[idx] <= 0
        lost = (~won) & (wk[idx] <= 0)
        status[idx[won]] = 1
        status[idx[lost]] = 2
    status[status == 0] = 2  # balls ran out
    return {
        "win_probability": round(float((status == 1).mean()), 4),
        "avg_runs_scored": round(float(scored.mean()), 1),
        "avg_wickets_lost": round(float((start_wk - wk).clip(0, start_wk).mean()), 2),
        "simulations": int(n_sims),
    }


def historical_analogues(states: pd.DataFrame, wickets_left: int, balls_left: int, rrr: float,
                         crr: float | None = None, balls_tol: int = 6, rrr_tol: float = 0.75,
                         crr_tol: float = 1.0) -> dict:
    """Observed win rate for past chase states that look like this one.

    Matches on wickets left, balls left, required run rate and (if given) current run rate,
    so the comparison is like for like: a fast-scoring chase and a stalled one are different.
    """
    if states is None or len(states) == 0:
        return {"available": False, "reason": "No historical chase data loaded."}
    m = (states["wickets_left"] == wickets_left) \
        & ((states["balls_left"] - balls_left).abs() <= balls_tol) \
        & ((states["rrr"] - rrr).abs() <= rrr_tol)
    criteria = f"wickets left = {wickets_left}, balls left within ±{balls_tol}, required run rate within ±{rrr_tol}"
    if crr is not None:
        m = m & ((states["crr"] - crr).abs() <= crr_tol)
        criteria += f", current run rate within ±{crr_tol}"
    n = int(m.sum())
    if n == 0:
        return {"available": True, "sample_size": 0, "win_rate": None,
                "note": "No comparable historical chase states."}
    matches = int(states.loc[m, "match_id"].nunique())
    return {"available": True, "sample_size": n, "matches": matches,
            "win_rate": round(float(states.loc[m, "label"].mean()), 4),
            "criteria": criteria}


def pressure_level(rrr: float, crr: float, wickets_left: int) -> str:
    """Heuristic pressure label: required run rate bands, raised one level when <=3 wickets remain."""
    if rrr <= crr and rrr <= 8:
        level = 0
    elif rrr <= 10:
        level = 1
    elif rrr <= 12:
        level = 2
    else:
        level = 3
    if wickets_left <= 3:
        level = min(3, level + 1)
    return ["Low", "Medium", "High", "Very high"][level]


def analyse(state: dict, dists: dict, states: pd.DataFrame | None, model_probability: float | None) -> dict:
    """Full strategy report for one chase state (runs_needed, balls_left, wickets_left, target)."""
    runs_needed = int(state["runs_needed"])
    balls_left = int(state["balls_left"])
    wickets_left = int(state["wickets_left"])
    target = int(state.get("target", runs_needed))
    balls_bowled = config.T20_BALLS - balls_left
    runs_scored = target - runs_needed
    crr = runs_scored * config.BALLS_PER_OVER / balls_bowled if balls_bowled > 0 else 0.0
    rrr = runs_needed * config.BALLS_PER_OVER / balls_left if balls_left > 0 else 99.0
    rrr = min(rrr, 99.0)

    pace = pace_factor(crr, dists)
    scenarios = {}
    for key, sc in config.SCENARIOS.items():
        result = simulate_chase(runs_needed, balls_left, wickets_left, balls_bowled, dists, sc, pace=pace)
        scenarios[key] = {"label": sc["label"], "description": sc["description"],
                          "assumptions": {"run_multiplier": sc["run_multiplier"],
                                          "wicket_multiplier": sc["wicket_multiplier"],
                                          "pace_factor": round(pace, 3)},
                          **result}
    balanced = scenarios["balanced"]["win_probability"]
    for key in scenarios:
        scenarios[key]["delta_vs_balanced_pp"] = round((scenarios[key]["win_probability"] - balanced) * 100, 1)

    best = max(scenarios, key=lambda k: scenarios[k]["win_probability"])
    runner_up = max(scenarios[k]["win_probability"] for k in scenarios if k != best)
    margin_pp = (scenarios[best]["win_probability"] - runner_up) * 100
    if balls_left <= 0 or runs_needed <= 0:
        approach = "Chase is decided or can no longer be completed; no batting approach to recommend."
    elif margin_pp < MATERIAL_DIFFERENCE_PP:
        approach = (f"No scenario clearly wins under the stated assumptions (the best beats the next option by "
                    f"{margin_pp:.1f} points, below the {MATERIAL_DIFFERENCE_PP:.0f}-point threshold). "
                    "Keep the scoring rate close to the required rate and protect wickets.")
    else:
        gap = rrr - crr
        if best == "aggressive":
            advice = ("Under the stated assumptions, attacking gives the best simulated chance. "
                      "Take boundary options on ordinary deliveries, but keep the set batters accountable for finishing.")
        elif best == "conservative":
            advice = ("Under the stated assumptions, rotating strike gives the best simulated chance: "
                      "the chase is more likely to be lost to wickets than to runs, so protect wickets while scoring steadily.")
        else:
            advice = ("Under the stated assumptions, a balanced approach works best: keep the scoring rate close to the "
                      "required rate and take boundary chances on bad balls.")
        if wickets_left <= 2 and gap > 0:
            advice += " With two or fewer wickets left, one dismissal can end the chase, so avoid unnecessary risk early."
        approach = advice

    return {
        "state": {"target": target, "runs_needed": runs_needed, "balls_left": balls_left,
                  "wickets_left": wickets_left, "runs_scored": runs_scored,
                  "balls_bowled": balls_bowled},
        "crr": round(crr, 2),
        "rrr": round(rrr, 2),
        "rrr_minus_crr": round(rrr - crr, 2),
        "pressure_level": pressure_level(rrr, crr, wickets_left),
        "model_win_probability": None if model_probability is None else round(float(model_probability), 4),
        "historical": historical_analogues(states, wickets_left, balls_left, rrr, crr=crr),
        "scenarios": scenarios,
        "best_scenario": best,
        "recommended_approach": approach,
        "pace_factor": round(pace, 3),
        "assumption_note": ("Scenario results come from a simulator that draws each ball from historical "
                            "second-innings outcomes for the same phase and wickets-fallen group, scaled to this "
                            "chase's current scoring rate (pace factor) and by multipliers that are assumptions. "
                            "Compare scenarios relative to each other."),
    }


def distribution_summary(dists: dict) -> dict:
    out = {}
    for i, phase in enumerate(dists["phases"]):
        out[phase] = {}
        for g, label in enumerate(WICKET_GROUP_LABELS):
            out[phase][label] = {
                "balls": int(dists["balls"][i, g]),
                "wicket_prob_per_ball": round(float(dists["wicket_prob"][i, g]), 4),
                "run_probs": [round(float(x), 4) for x in dists["probs"][i, g]],
            }
    return out
