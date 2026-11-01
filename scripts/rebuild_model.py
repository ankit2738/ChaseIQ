"""Rebuild ChaseIQ from the original Cricsheet ZIP(s) in data/raw/.

Steps: parse every match -> clean -> build ball-level and chase-state tables ->
train and evaluate models with a fixed random seed -> validate the simulator ->
write data/processed/ and models/.

Usage (from the project root):
    python scripts/rebuild_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.store import DataStore  # noqa: E402


def main() -> int:
    zips = sorted(config.RAW_DIR.glob("*.zip"))
    if not zips:
        print(f"No Cricsheet ZIP found in {config.RAW_DIR}.")
        print("Download t20s_json.zip from https://cricsheet.org/downloads/ and place it there.")
        return 1
    print(f"Source ZIP(s): {', '.join(z.name for z in zips)}")
    store = DataStore()
    result = store.build(log=print)
    m = result["model"]
    print("\nRebuild complete.")
    print(f"Selected model : {m['selected_model']}")
    print(f"Test accuracy  : {m['test_metrics']['accuracy']:.4f}")
    print(f"Test ROC-AUC   : {m['test_metrics']['roc_auc']:.4f}")
    print(f"Test matches   : {m['test_matches']}")
    print(f"Evaluation     : {config.EVAL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
