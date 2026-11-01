"""In-memory data store: loads processed tables once, runs the full build on demand."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

import config
from src import model as model_mod
from src.features import build_chase_states, chase_training_exclusions
from src.ingest import build_dataset, find_raw_zips
from src.strategy import phase_distributions

TEXT_COLUMNS = {
    "match_id": str, "season": str, "event": str, "venue": str, "city": str,
    "team1": str, "team2": str, "toss_winner": str, "toss_decision": str,
    "winner": str, "result": str, "method": str, "match_type": str,
    "gender": str, "team_type": str, "inn1_team": str, "inn2_team": str,
}


def read_matches(path: Path = config.PROCESSED_MATCHES) -> pd.DataFrame:
    return pd.read_csv(path, dtype=TEXT_COLUMNS, keep_default_na=True)


class DataStore:
    def __init__(self):
        self._lock = threading.RLock()
        self.matches: pd.DataFrame | None = None
        self.deliveries: pd.DataFrame | None = None
        self.states: pd.DataFrame | None = None
        self.dists: dict | None = None
        self.build_log: list[str] = []
        self._cache: dict = {}

    # ---- state ------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.matches is not None and self.deliveries is not None

    def status(self) -> dict:
        zips = [p.name for p in find_raw_zips()]
        quality = None
        if config.QUALITY_REPORT.exists():
            quality = json.loads(config.QUALITY_REPORT.read_text())
        evaluation = None
        if config.EVAL_FILE.exists():
            evaluation = json.loads(config.EVAL_FILE.read_text())
        return {
            "data_loaded": self.ready,
            "raw_zips": zips,
            "matches": int(len(self.matches)) if self.ready else 0,
            "deliveries": int(len(self.deliveries)) if self.ready else 0,
            "chase_states": int(len(self.states)) if self.states is not None else 0,
            "model_available": config.MODEL_FILE.exists(),
            "model_name": (evaluation or {}).get("selected_model"),
            "quality": quality,
            "build_log": self.build_log[-40:],
        }

    def load(self) -> bool:
        """Load processed tables from disk. Returns True if data is ready."""
        with self._lock:
            if not (config.PROCESSED_MATCHES.exists() and config.PROCESSED_DELIVERIES.exists()):
                return False
            self.matches = read_matches()
            self.deliveries = pd.read_pickle(config.PROCESSED_DELIVERIES)
            self.states = pd.read_pickle(config.PROCESSED_CHASE_STATES) \
                if config.PROCESSED_CHASE_STATES.exists() else None
            self.dists = phase_distributions(self.deliveries)
            model_mod.clear_model_cache()
            self._cache.clear()
            return True

    def build(self, log=None) -> dict:
        """Ingest all raw ZIPs, build chase states, train the model, then reload."""
        def _log(msg: str):
            self.build_log.append(msg)
            if log:
                log(msg)

        with self._lock:
            zips = find_raw_zips()
            if not zips:
                raise FileNotFoundError(f"No Cricsheet .zip files found in {config.RAW_DIR}")
            _log(f"Ingesting {len(zips)} ZIP file(s): {', '.join(z.name for z in zips)}")
            ingest_summary = build_dataset(zips)
            _log(f"Parsed {ingest_summary['matches_total']} matches, {ingest_summary['deliveries_total']} deliveries")

            matches = read_matches()
            deliveries = pd.read_pickle(config.PROCESSED_DELIVERIES)
            states = build_chase_states(deliveries, matches)
            states.to_pickle(config.PROCESSED_CHASE_STATES)
            exclusions = chase_training_exclusions(matches)
            quality = json.loads(config.QUALITY_REPORT.read_text()) if config.QUALITY_REPORT.exists() else {}
            quality["chase_training"] = {**exclusions, "chase_states": int(len(states)),
                                         "chase_matches_in_states": int(states["match_id"].nunique())}
            config.QUALITY_REPORT.write_text(json.dumps(quality, indent=2, default=str))
            _log(f"Built {len(states)} chase states from {states['match_id'].nunique()} decided 20-over chases; "
                 f"excluded {exclusions['excluded_adjusted_result_matches']} adjusted-result (D/L/Awarded) matches")

            _log("Training and evaluating win-probability models...")
            report = model_mod.train(states, log=_log)
            _log(f"Selected model: {report['selected_model']}")
            self.load()
            from scripts.validate_simulator import run as run_validation
            _log("Checking simulator against historical outcomes...")
            run_validation(self)
            return {"ingest": ingest_summary,
                    "model": {"selected_model": report["selected_model"],
                              "test_metrics": report["models"][report["selected_model"]]["test"],
                              "baseline": report["models"]["baseline_rrr_only"]["test"],
                              "test_matches": report["data"]["test_matches"]}}

    # ---- cache helper -----------------------------------------------------
    def cached(self, key, fn):
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = fn()
        return self._cache[key]
