"""Central configuration for ChaseIQ.

Everything that is an *assumption* (scenario multipliers, thresholds) lives
here so it can be read, changed and documented in one place.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # drop Cricsheet *.zip files here
PROCESSED_DIR = DATA_DIR / "processed"     # pipeline output (generated)
MODELS_DIR = PROJECT_ROOT / "models"       # trained model + evaluation report
STATIC_DIR = PROJECT_ROOT / "static"

HOST = "127.0.0.1"
PORT = 8000

# ---- T20 match geometry -------------------------------------------------
T20_OVERS = 20
BALLS_PER_OVER = 6
T20_BALLS = T20_OVERS * BALLS_PER_OVER     # 120
MAX_WICKETS = 10

# Phases by 0-based over index (T20 convention).
POWERPLAY_LAST_OVER = 5    # overs 1-6
DEATH_FIRST_OVER = 15      # overs 16-20


def phase_of_over(over_index: int) -> str:
    if over_index <= POWERPLAY_LAST_OVER:
        return "powerplay"
    if over_index >= DEATH_FIRST_OVER:
        return "death"
    return "middle"


# ---- Cricket rules used for statistics ----------------------------------
# Dismissals credited to the bowler.
BOWLER_CREDIT_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
# Wicket kinds that are not dismissals of the batter at all.
NOT_DISMISSAL_KINDS = {"retired hurt", "retired not out"}

# ---- Sample-size rules ---------------------------------------------------
MIN_BALLS_RANKING = 100        # players ranked only above this many balls
MIN_CONTEXT_BALLS = 30         # pressure-context sample needed for PI
MATCHUP_LOW_SAMPLE = 12        # below this: "very small sample"
MATCHUP_OK_SAMPLE = 30         # below this: "small sample, treat with caution"

# ---- Strategy simulator assumptions (NOT historical estimates) ----------
# Multipliers applied to the empirical run outcomes and wicket probability
# drawn from the historical data for each phase. They are explicit
# hypotheses, shown in the UI, and can be changed here.
SCENARIOS = {
    "conservative": {
        "label": "Conservative",
        "run_multiplier": 0.85,
        "wicket_multiplier": 0.80,
        "description": "Rotate strike, avoid high-risk shots; fewer runs, fewer wickets.",
    },
    "balanced": {
        "label": "Balanced",
        "run_multiplier": 1.00,
        "wicket_multiplier": 1.00,
        "description": "Continue at the current scoring pace, with historical wicket rates.",
    },
    "aggressive": {
        "label": "Aggressive",
        "run_multiplier": 1.20,
        "wicket_multiplier": 1.30,
        "description": "Attack from the start; more boundaries and more dismissals.",
    },
}
SIM_RUNS = 4000
SIM_SEED = 7

# ---- Modelling -----------------------------------------------------------
TEST_FRACTION = 0.20           # latest 20% of matches (by date) = test set
VALIDATION_FRACTION = 0.15     # latest 15% of the remaining training matches
MODEL_FILE = MODELS_DIR / "chaseiq_win_model.joblib"
EVAL_FILE = MODELS_DIR / "evaluation.json"
RF_TRAIN_SAMPLE = 200_000      # random forest is trained on a row subsample
RANDOM_STATE = 42

PROCESSED_MATCHES = PROCESSED_DIR / "matches.csv"
PROCESSED_DELIVERIES = PROCESSED_DIR / "deliveries.pkl"
PROCESSED_CHASE_STATES = PROCESSED_DIR / "chase_states.pkl"
QUALITY_REPORT = PROCESSED_DIR / "quality_report.json"
