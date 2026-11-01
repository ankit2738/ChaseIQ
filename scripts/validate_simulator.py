"""Check the strategy simulator and the ML model against realised outcomes.

For a grid of chase states (wickets left, balls left, required run rate), compare:
  * realised: share of historical chases in matching states that were won
  * model:    the saved win-probability model's estimate for the same state
  * simulator (balanced scenario): Monte Carlo estimate with historical ball outcomes

Writes models/simulator_validation.json. Run:  python -m scripts.validate_simulator
Note: the realised rates use the same states the model was trained on (no hold-out),
so this is a sanity check of the simulator, not an out-of-sample accuracy claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.features import state_frame  # noqa: E402
from src.model import predict_win_probability  # noqa: E402
from src.store import DataStore  # noqa: E402
from src.strategy import historical_analogues, pace_factor, simulate_chase  # noqa: E402


def run(store) -> dict:
    """Run the check on a loaded DataStore and write models/simulator_validation.json."""
    if store.states is None or store.dists is None:
        raise ValueError("Store must have chase states and distributions loaded.")
    rows = []
    for wk in [3, 5, 7]:
        for balls in [18, 30, 48, 66]:
            for rrr in [7.0, 9.0, 11.0, 13.0]:
                runs_needed = int(round(rrr * balls / config.BALLS_PER_OVER))
                hist = historical_analogues(store.states, wk, balls, rrr, crr=8.0, balls_tol=6, rrr_tol=0.5, crr_tol=1.0)
                if not hist.get("sample_size") or hist["sample_size"] < 100:
                    continue
                balls_bowled = config.T20_BALLS - balls
                sim = simulate_chase(runs_needed, balls, wk, balls_bowled, store.dists,
                                     config.SCENARIOS["balanced"],
                                     pace=pace_factor(8.0, store.dists), n_sims=4000)
                # The model also uses the current run rate; the grid fixes it at 8.0 (stated assumption).
                runs_scored = int(round(8.0 * balls_bowled / config.BALLS_PER_OVER))
                feats = state_frame(runs_needed=[runs_needed], balls_left=[balls], wickets_left=[wk],
                                    runs_scored=[runs_scored], balls_bowled=[balls_bowled],
                                    target=[runs_needed + runs_scored])
                model_p = predict_win_probability(feats)
                rows.append({
                    "wickets_left": wk, "balls_left": balls, "rrr": rrr,
                    "runs_needed": runs_needed,
                    "historical_sample": hist["sample_size"],
                    "realised_win_rate": hist["win_rate"],
                    "model_probability": None if model_p is None else round(float(model_p[0]), 4),
                    "simulator_balanced": sim["win_probability"],
                })

    if not rows:
        raise ValueError("No states had enough historical samples.")
    real = np.array([r["realised_win_rate"] for r in rows])
    sim = np.array([r["simulator_balanced"] for r in rows])
    summary = {
        "states_compared": len(rows),
        "simulator_mean_abs_error": round(float(np.mean(np.abs(sim - real))), 4),
        "simulator_max_abs_error": round(float(np.max(np.abs(sim - real))), 4),
        "simulator_bias": round(float(np.mean(sim - real)), 4),
        "model_mean_abs_error": None,
        "note": ("Errors are against realised historical win rates in matching states. "
                 "Model rows are in-sample for these states, so the model error is not an out-of-sample measure."),
    }
    if rows[0]["model_probability"] is not None:
        mod = np.array([r["model_probability"] for r in rows])
        summary["model_mean_abs_error"] = round(float(np.mean(np.abs(mod - real))), 4)
    out = {"summary": summary, "rows": rows}
    path = config.MODELS_DIR / "simulator_validation.json"
    path.write_text(json.dumps(out, indent=2))
    return out


def main() -> dict:
    store = DataStore()
    if not store.load() or store.states is None:
        raise SystemExit("Build the data first (python app.py --build).")
    out = run(store)
    print(json.dumps(out["summary"], indent=2))
    return out


if __name__ == "__main__":
    main()
