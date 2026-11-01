# ChaseIQ — Technical Report

## 1. Summary

ChaseIQ analyses T20 cricket from ball-by-ball Cricsheet data. It provides a validated
win-probability model for run chases, a strategy engine that combines historical analogues with
a simulator, player and matchup analytics, and a browser dashboard. The application is a
single Python process: a standard-library HTTP server, JSON endpoints, and a static
HTML/CSS/JavaScript front end with SVG charts.

## 2. Data

| Item | Value |
|---|---|
| Source | Cricsheet T20 JSON archive (`t20s_json.zip`), format v1.2.0 / v1.3.0 |
| Matches parsed | 5,845 of 5,845 files |
| Matches with a winner | 5,676 |
| Matches without a winner (ties, no results) | 169 |
| Deliveries (balls) | 1,319,141 |
| Super-over innings excluded | 96 |
| Matches with an outcome method (D/L 194, Awarded 4) | 198 (kept for descriptive analytics) |
| Decided 20-over chases (before exclusion) | 5,657 |
| **Excluded from win-probability training: adjusted result (D/L 192, Awarded 2)** | **194** |
| Chases used for training and evaluation | 5,463 |
| Chase states (one per legal ball) | 552,963 |

The other 4 of the 198 method-flagged matches are not complete decided 20-over chases, so they were never in the chase set. Cricsheet also provides innings-level adjusted targets (`innings[].target`), which ChaseIQ does not parse; instead, any match with a non-empty outcome `method` is excluded because its target may have been revised. 11 matches without a method have an innings target that differs from first-innings runs + 1; they remain in training (a known limitation).

Cleaning rules are applied in `src/ingest.py` and reported in `data/processed/quality_report.json`:
super-over innings are excluded; matches without a winner are kept but excluded from win/loss
statistics; matches with an adjusted result (`method`) are kept for descriptive analytics but excluded from chase-state training (`src/features.py`, counts in `quality_report.json` under `chase_training`); deliveries without a batter or bowler are dropped; duplicate match IDs across ZIPs are kept once.

## 3. Features and labels

Each row is the state of the chase before a **legal** ball (wides and no-balls are not states).
Only information available before that ball is used.

Features: `runs_needed`, `balls_left`, `wickets_left`, `crr` (current run rate), `rrr` (required run rate),
`rrr_minus_crr`, `target`. Label: 1 if the chasing side won the match.

Excluded on purpose: team identity, venue, season and player identity. Those would need careful
handling to avoid leakage and to keep the model honest about what it knows.

## 4. Validation protocol

* **Split:** match-grouped and chronological. Test = the latest 20% of matches (1,093 matches,
  107,391 ball-states, 2025-10-09 to 2026-10-25). Validation = the latest 15% of the remaining matches
  (656 matches, 65,543 ball-states); fit = 3,714 matches, 380,029 ball-states. No match appears in
  more than one partition.
* **Selection:** candidate models are fitted on the fit partition and compared by validation log loss.
  The selected model is refitted on all training matches. The test set is scored once.
* **Unit of evaluation:** every metric below is computed across all 107,391 test ball-states pooled together, from 1,093 test matches. Metrics are not averaged per match, so matches with more legal balls in the chase contribute more rows.
* **Baselines:** majority class (chase win rate of the training data) and a logistic regression on the
  required run rate alone.

## 5. Results (`models/evaluation.json`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier ↓ | Log loss ↓ |
|---|---|---|---|---|---|---|---|
| **Logistic regression (selected)** | **0.849** | 0.804 | 0.846 | 0.824 | 0.935 | 0.103 | 0.321 |
| Random forest | 0.851 | 0.811 | 0.842 | 0.826 | 0.936 | 0.102 | 0.318 |
| Gradient boosting (HistGB) | 0.850 | 0.811 | 0.837 | 0.824 | 0.934 | 0.103 | 0.323 |
| Baseline: RRR only | 0.818 | 0.761 | 0.825 | 0.792 | 0.909 | 0.123 | 0.378 |
| Baseline: majority | 0.581 | 0.000 | 0.000 | 0.000 | 0.500 | 0.245 | 0.682 |

Logistic regression was selected because it had the lowest validation log loss (0.3533, against 0.3544 for the random forest and 0.3578 for gradient boosting). On the test set the random forest is marginally better on every headline metric; the three models are within about 0.002 accuracy of each other. The selection was fixed on validation data and was not changed after seeing the test scores.

Accuracy, precision, recall and F1 use a 0.5 threshold. The test chase win rate is 41.9%, so the
majority baseline is a weak comparison and the RRR-only baseline is the meaningful one.

**Calibration (test set, 10 bins):** predicted and observed rates track closely across bins (for example,
predicted 0.85 vs observed 0.87 in the 80–90% bin, and predicted 0.25 vs observed 0.22 in the 20–30% bin).
The model tends to be slightly over-confident for chases that end up at 30–70%, where predicted rates are 3–8 points above observed. Full bins are in `evaluation.json`.

**Feature importance (permutation, ROC-AUC drop on the test set):** `runs_needed` (0.214), `rrr_minus_crr` (0.150) and `rrr` (0.136) dominate, followed by `wickets_left` (0.041) and `target` (0.040). Current run rate (0.017) and balls left (0.008) contribute little beyond these. Several features are strongly correlated, so these values show how much the fitted model relies on each one, not independent effects.

## 6. Strategy engine

The Chase Strategy page combines three components, each labelled in the interface:

1. **Historical analogues (observed).** Past chase states with the same wickets left, balls left (±6),
   required run rate (±0.75) and current run rate (±1.0). The page shows the win rate and sample size.
2. **Model probability (learned).** The model estimate for the same state.
3. **Scenario simulator (assumption-based).** Monte Carlo simulation of the remaining balls. Each ball is
   drawn from second-innings outcomes for its phase (powerplay 1–6, middle 7–15, death 16–20) and
   wickets-fallen group (0–2, 3–5, 6–7, 8–9). Outcomes are scaled by a **pace factor** (the chase's current
   scoring rate relative to the historical average, capped at 0.5×–2.0×). The balanced scenario continues at
   that pace. Conservative (×0.85 runs, ×0.80 wicket chance) and aggressive (×1.20 runs, ×1.30 wicket chance) are
   stated assumptions in `config.py`, not estimates from data.

A recommendation names a best scenario only if it beats the next option by at least 2 percentage points
(`MATERIAL_DIFFERENCE_PP`). Otherwise the page says no scenario clearly wins.

**Simulator check (`scripts/validate_simulator.py`):** across 27 historical states with at least 100
comparable chases, the balanced scenario has a mean absolute error of 0.072 against realised win rates
(maximum 0.254) and a bias of −0.068. The first version, without pace scaling, had an error of 0.28, which is
why pace scaling was added. The model's error on the same states is 0.060, but those states are in its training
data, so that figure is not out-of-sample.

## 7. Player and team analytics

Definitions (also on the Methodology page):

* Batting: balls faced exclude wides. Runs are off the bat. Dismissals exclude retired hurt and retired not out.
* Bowling: legal balls only. Runs conceded exclude byes and leg-byes. Wickets are credited only for bowled,
  caught, caught and bowled, lbw, stumped and hit wicket.
* **Pressure Performance Index (custom, not an official statistic):** 100 × mean over pressure contexts of
  (strike rate in context ÷ career strike rate). Contexts: death overs; chases with required rate ≥ 10;
  four or more wickets down; innings in chases the side won. Each context counts only with at least 30 balls.
* Matchups under 12 balls are labelled "very small sample" and under 30 "small sample".

## 8. Architecture

See `docs/architecture.png`. Ingestion writes processed tables. Feature building and training write the model
bundle and the evaluation report. The server loads these into memory and serves JSON to the browser.

## 9. Reproducibility

* Fixed random seeds: `RANDOM_STATE = 42`, simulator `SIM_SEED = 7`, permutation importance seeded.
* One command rebuilds everything from the ZIP: `python scripts/rebuild_model.py`.
* Dependencies are pinned by minimum version in `requirements.txt`. Tested with Python 3.13, pandas 3.0.5,
  NumPy 2.5.3, scikit-learn 1.9.1 and joblib 1.6.0.
* Tests: 17 unit tests (`tests/test_core.py`, including two proving that adjusted-result matches are excluded from chase-state training) and 15 integration tests (`tests/test_app.py`, including the HTTP server
  and path-traversal protection). All 32 pass on the built data.
* A headless browser check loaded all eleven pages with no console or HTTP errors, and confirmed that the
  scenario inputs keep focus while typing.

## 10. Limitations

* The model uses score-state features only. Team strength, venue, pitch, toss, and batter and bowler quality are
  not modelled. Probabilities are historical frequencies, not forecasts for a particular match.
* The simulator's absolute probabilities are less reliable than the model. Use scenario differences, and see the
  calibration check above.
* Only the first dismissal kind on a ball is recorded (multi-wicket deliveries are rare).
* Player names are Cricsheet registry names. Spelling variants are not merged.
* Matches with revised targets but no `method` value cannot be detected by the exclusion rule (11 such matches have an innings target that differs from first-innings runs + 1).
* Results describe the uploaded archive. Newer Cricsheet releases will change the numbers.
* The app is not connected to live data and has no authentication; it binds to 127.0.0.1.
* Not built: a ball-by-ball match simulator for full innings play-out.

## 11. Data licence and attribution

Ball-by-ball data: [Cricsheet](https://cricsheet.org/). Cricsheet's licensing page says most of its data is under the Open Data Commons Attribution License (ODC-By) 1.0 (https://opendatacommons.org/licenses/by/1.0/), and that its XML match data alone is under CC BY 4.0. The page does not name the JSON archive individually; ChaseIQ treats the JSON archive as ODC-By. The JSON data is not labelled CC BY 4.0. ChaseIQ processed and transformed the data into its own tables, model and reports. See `DATA_ATTRIBUTION.md`.
