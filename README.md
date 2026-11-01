# ChaseIQ — Cricket Match Intelligence & Run-Chase Strategy

*Every ball. Every decision. A smarter way to win.*

ChaseIQ turns Cricsheet ball-by-ball T20 data into match analytics: a live-style chase
dashboard, a win-probability model with honest validation, a run-chase strategy engine,
player and matchup analytics, team analytics, a data explorer and a methodology page.

It runs as a single Python process using only the standard library for the web server and
plain HTML, CSS and JavaScript for the interface (charts are drawn as SVG, no chart library).
pandas, NumPy and scikit-learn do the data work and modelling.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8000 (it opens automatically; use `--no-browser` to skip).

**Data is not bundled.** Download the Cricsheet T20 JSON archive and either:

1. copy it into `data/raw/` (for example `data/raw/t20s_json.zip`) and run `python app.py`, which
   builds the data and model automatically the first time, or
2. start the app and use **Data & Upload** → choose the ZIP → **Upload and build**.

Source: https://cricsheet.org/downloads/ (file `t20s_json.zip`). Credit Cricsheet (https://cricsheet.org/), as the app does on its methodology page.
Cricsheet's licensing page says most of its data is under the Open Data Commons Attribution License (ODC-By) 1.0 (https://opendatacommons.org/licenses/by/1.0/), and that its XML match data alone is under CC BY 4.0. The page does not name the JSON archive individually; ChaseIQ treats the JSON archive as ODC-By, so check https://cricsheet.org/licensing/ before publishing.
See `DATA_ATTRIBUTION.md`.

### Commands

| Command | What it does |
|---|---|
| `python app.py` | Start the app (builds data and model first if they do not exist yet) |
| `python app.py --port 8080` | Use another port |
| `python app.py --build` | Re-ingest `data/raw/*.zip`, rebuild chase states and retrain |
| `python app.py --no-browser` | Start without opening a browser |
| `python -m unittest tests.test_core` | Unit tests (ingestion, features, simulator, analytics) |
| `python -m unittest tests.test_app` | Integration tests (real data, API, HTTP server); skipped until data is built |
| `python scripts/validate_simulator.py` | Check the strategy simulator against realised historical outcomes |
| `python scripts/rebuild_model.py` | Rebuild everything from `data/raw/*.zip` with a fixed seed and print the test metrics |

## Rebuilding the model from the original dataset

1. Download the Cricsheet T20 JSON archive from https://cricsheet.org/downloads/ (file `t20s_json.zip`).
2. Place it at `data/raw/t20s_json.zip` (the folder is created if missing).
3. Run `python scripts/rebuild_model.py` (about 2 minutes on a laptop).

This regenerates `data/processed/` (matches, deliveries, chase states, quality report), `models/chaseiq_win_model.joblib`,
`models/evaluation.json` and `models/simulator_validation.json`. Model training uses fixed random seeds, so
rebuilding from the same ZIP reproduces the same metrics. If your rebuilt test accuracy differs from the table below,
check that the ZIP is the same file as Cricsheet's current `t20s_json.zip`.

Cricsheet's data is updated over time, so newer archives will give slightly different numbers.

## Documentation

* `docs/architecture.png` (source `docs/architecture.mmd`): system architecture.
* `docs/TECHNICAL_REPORT.md`: method, validation, limitations.
* `docs/DEVPOST_SUBMISSION.md`: project description for the submission.
* `docs/screenshots/`: screenshots of the working dashboard, taken from the app on the real data.
* `DATA_ATTRIBUTION.md`: data source, terms and credit.
* `LICENSE`: MIT licence for the code. The Cricsheet data is licensed separately (see `DATA_ATTRIBUTION.md`).

## Pages

1. **Overview** — headline counts, matches per season, top teams, batters and bowlers (filterable).
2. **Match Intelligence** — scorecard summary, worm chart, manhattan charts, and the model's win-probability path through the chase.
3. **Win Probability** — enter any chase state; see the model probability and a runs-needed × balls-left heat map.
4. **Chase Strategy** — pressure level, CRR/RRR, historical analogues (with sample sizes), and conservative / balanced / aggressive scenarios.
5. **Player Analytics** — batting, bowling, pressure index (custom, labelled), strike rate by season.
6. **Player Comparison** — side-by-side metrics and a profile radar chart.
7. **Batter vs Bowler** — head-to-head record with sample-size warnings.
8. **Team Analytics** — win rates, chase vs defend, head-to-head heat map, season trends.
9. **Data Explorer** — matches and ball-by-ball deliveries; CSV download of matches.
10. **Methodology & Model** — model results, calibration, feature importance, formulas, limitations.
11. **Data & Upload** — upload a ZIP, rebuild, view the cleaning report.

Filters (format, season, tournament, team) apply to Overview, Match, Player, Comparison, Matchup and Explorer pages.

## Results on the uploaded data (from `models/evaluation.json`)

Data: 5,845 T20 matches parsed; 5,676 with a winner; 5,657 decided 20-over chases. 194 adjusted-result (D/L or awarded)
matches are excluded from training, leaving 5,463 chases and 552,963 chase states (one per legal ball).
Split: match-level and chronological.

Test set: the most recent 20% of matches (1,093 matches, 107,391 ball-states, 2025-10-09 → 2026-10-25), scored once.
All metrics are computed across the test ball-states pooled together, not averaged per match.

| Model | Accuracy | F1 | ROC-AUC | Brier ↓ | Log loss ↓ |
|---|---|---|---|---|---|
| **Logistic regression (selected)** | 0.849 | 0.824 | 0.935 | 0.103 | 0.321 |
| Random forest | 0.851 | 0.826 | 0.936 | 0.102 | 0.318 |
| Gradient boosting (HistGB) | 0.850 | 0.824 | 0.934 | 0.103 | 0.323 |
| Baseline: RRR only | 0.818 | 0.792 | 0.909 | 0.123 | 0.378 |
| Baseline: majority | 0.581 | 0.000 | 0.500 | 0.245 | 0.682 |

Selection used the validation partition (log loss: logistic regression 0.353, random forest 0.354), not the test set.
On the test set the random forest is marginally better on every headline metric, but it was not selected on that basis.
The selected model beats the RRR-only baseline by about 3 accuracy points and 0.03 ROC-AUC. These are historical hold-out numbers; they do
not guarantee accuracy for future matches or for any live match.

### Strategy simulator check

The simulator is used to compare scenarios with each other, not to predict absolute outcomes.
`scripts/validate_simulator.py` compares it with realised win rates in 27 matching historical states:
mean absolute error 0.072, bias −0.07. The model's error on the same states is 0.060, but those states were
in its training data, so that figure is not an out-of-sample result. The app shows this warning on the strategy page.

## Method summary

* **Ingestion** (`src/ingest.py`): reads Cricsheet JSON inside the ZIP; excludes super-over innings; keeps
  no-result and tie matches but flags them; writes a quality report (`data/processed/quality_report.json`).
* **Features** (`src/features.py`): one row per legal ball of a complete 20-over chase, using only the state
  before that ball. Features: runs needed, balls left, wickets left, current and required run rate,
  run-rate gap, target. Matches with a non-empty outcome `method` (D/L, awarded) are excluded from training; their counts are in
  `data/processed/quality_report.json` under `chase_training`.
* **Validation** (`src/model.py`): chronological, match-grouped split; model selection on validation log loss;
  single test evaluation; baselines; calibration bins; permutation importance.
* **Strategy** (`src/strategy.py`): historical analogues (matched on wickets, balls, RRR and CRR) plus a Monte
  Carlo simulator. Balanced scenario = continue at the current scoring pace with historical wicket rates.
  Conservative/aggressive multiply runs and wicket chances by assumed factors set in `config.py`.
* **Analytics** (`src/analytics.py`): definitions are on the Methodology page (strike rate, economy, dot %,
  dismissal rules, and the custom Pressure Performance Index with its formula).

## Project structure

```
app.py                 HTTP server (python standard library) + static files; `python app.py`
config.py              paths, scenario assumptions, thresholds, model settings
requirements.txt
src/
  ingest.py            Cricsheet ZIP parser and cleaning
  features.py          chase states, feature engineering, chronological split
  model.py             training, selection, evaluation, persistence
  strategy.py          historical analogues, simulator, scenario analysis
  analytics.py         player, matchup, team, pressure, match analytics
  store.py             in-memory data store and full build (ingest → states → model → check)
  api.py               JSON endpoint handlers
static/                index.html, styles.css, app.js (vanilla JavaScript, SVG charts)
scripts/rebuild_model.py     one-command rebuild from the ZIP
scripts/validate_simulator.py
tests/                 test_core.py (unit), test_app.py (integration)
docs/                  architecture, technical report, Devpost text, screenshots
LICENSE, DATA_ATTRIBUTION.md
data/, models/         generated outputs (git-ignored, except READMEs)
```

## Completed

* Cricsheet T20 ZIP ingestion with cleaning and a quality report (manual upload supported)
* Win-probability model with baselines, chronological validation, calibration and importance
* Run-chase strategy engine with analogues, simulator and scenario comparison
* Overview, Match Intelligence, Win Probability, Chase Strategy, Player Analytics, Player Comparison,
  Batter vs Bowler, Team Analytics, Data Explorer, Methodology & Model, Data & Upload pages
* Filters that work across pages; player/team/match selection; CSV export of matches
* Unit tests (17) and integration tests (15) passing (32 in total); headless browser check of every page with no console errors

## Not complete or known limitations

* **Match simulator (live "what if" over-by-over match play)**: not built. The Win Probability page covers user-entered states.
* **Simulator absolute probabilities** are not calibrated (see above); use the differences between scenarios.
* **Model features are score-only.** Team strength, venue, pitch, toss, and batter/bowler quality are not modelled.
  Test accuracy is for the uploaded data's period, not a forecast.
* **Multi-wicket deliveries**: only the first dismissal kind on a ball is stored (rare).
* **Adjusted targets (partial)**: 194 D/L or awarded decided 20-over chases are excluded from training because they carry a result method. Eleven further matches have no method but an innings-level target that differs from first-innings runs + 1 (Cricsheet's `innings[].target`, not parsed). Those eleven remain in the training set. Not every adjusted-target match is excluded.
* **Name matching**: players are matched by Cricsheet's registry name; spelling variants are not merged.
* **Live matches** are not connected. The app analyses historical files only and is not a live feed.
* **Pressure Performance Index** is a custom, documented index, not an official statistic.
* **No authentication**: the server binds to 127.0.0.1 and is meant for local use.
* Google Fonts (Inter) loads from the web; without it the page falls back to the system font.

## GitHub submission

```bash
git init
git add .
git status            # confirm data/raw, data/processed and models/*.joblib are NOT listed
git commit -m "ChaseIQ: cricket match intelligence and chase strategy"
gh repo create chaseiq --public --source=. --remote=origin --push
```

Before pushing, confirm the repo is public and contains no ZIP or large generated files (they are git-ignored).
The README above is the project description for judges to read on GitHub.

## Devpost submission (AQX Sports Analytics Data Bowl 4.0)

1. Open the hackathon page → **Create project** (or edit your existing project).
2. **Project name:** ChaseIQ
3. **Tagline:** Every Ball. Every Decision. A Smarter Way to Win.
4. **Built with:** Python, pandas, NumPy, scikit-learn, HTML/CSS/JavaScript (SVG charts)
5. **Links:** public GitHub repository URL; optional demo video.
6. **Description:** use the sections below.
   * **Problem:** coaches and analysts need more than scorecards to judge a run chase: how much pressure a side is under, and how the chances change with wickets and balls left.
   * **Solution:** ChaseIQ builds a validated win-probability model from ball-by-ball T20 data, shows historical analogues with sample sizes, and compares conservative, balanced and aggressive chase strategies.
   * **Data:** Cricsheet T20 JSON archive (5,845 matches; 1.3M+ deliveries).
   * **Analytical insight:** the selected logistic regression reaches 84.9% accuracy and 0.935 ROC-AUC on the most recent 20% of matches (ball-state metrics), versus 81.8% and 0.909 for a required-run-rate-only baseline. Player and matchup metrics include sample-size warnings.
   * **Practical application:** pre-match and mid-chase planning for analysts and coaches; a clear view of when the required rate is out of reach.
   * **Honest limits:** score-only model, simulator used only for relative scenario comparisons, no live feed.
7. **Demo:** add the screenshots or run notes; a short video is optional.
8. Submit well before the deadline (1 Nov 2026, 2:00 AM EST).

## Attribution and licence

Ball-by-ball data: [Cricsheet](https://cricsheet.org/). Cricsheet's licensing page says most of its data is under the Open Data Commons Attribution License (ODC-By) 1.0 (https://opendatacommons.org/licenses/by/1.0/), and that its XML match data alone is under CC BY 4.0. The page does not name the JSON archive individually; ChaseIQ treats the JSON archive as ODC-By, so check https://cricsheet.org/licensing/ before publishing. ChaseIQ processed and transformed the data. See `DATA_ATTRIBUTION.md`. ChaseIQ is an independent analysis project and is not
affiliated with Cricsheet, the ICC, or any league. Code is released under the MIT licence (`LICENSE`).
