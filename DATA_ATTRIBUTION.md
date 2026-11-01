# Data attribution and licence

## Source

Ball-by-ball T20 match data comes from **Cricsheet**: https://cricsheet.org/

* Download page: https://cricsheet.org/downloads/ (file used: `t20s_json.zip`, JSON format)
* JSON format reference: https://cricsheet.org/format/json/
* Withheld matches and policy: https://cricsheet.org/withheld-matches
* Cricsheet licensing: https://cricsheet.org/licensing/

## Licence of the Cricsheet data

* **ChaseIQ treats the JSON archive as licensed under the Open Data Commons Attribution License (ODC-By) 1.0**:
  https://opendatacommons.org/licenses/by/1.0/
* Cricsheet's licensing page states that most of its data is covered by ODC-By. It also says the
  XML match data alone is licensed under Creative Commons Attribution 4.0 (CC BY 4.0):
  https://creativecommons.org/licenses/by/4.0/legalcode.en. The JSON files are not named individually on that page,
  so the ODC-By basis above rests on "most of the data". Check https://cricsheet.org/licensing/ before publishing
  the project, and follow the licence Cricsheet states for the files you download.
* Do not describe the JSON data as CC BY 4.0. That licence applies only to the XML match data.

## How ChaseIQ uses the data

ChaseIQ **processed and transformed** the Cricsheet data. It parses the JSON files, applies cleaning rules
(for example, excluding super-over innings and flagging adjusted-result matches), and derives new tables and
statistics: match and delivery tables, a chase-state table, a trained win-probability model, and summary reports
(`models/evaluation.json`, `models/simulator_validation.json`, `data/processed/quality_report.json`).
Those derived outputs are not Cricsheet's files.

## Attribution and notices

When you share or publish results from ChaseIQ, credit **Cricsheet** (https://cricsheet.org/) as the source of the
ball-by-ball data, state that ChaseIQ processed and transformed it, and keep the licence reference
(https://opendatacommons.org/licenses/by/1.0/) with the data. The app footer and the Methodology page show a Cricsheet
credit and state that ChaseIQ is an independent analysis.

ChaseIQ is not affiliated with, endorsed by or sponsored by Cricsheet, the ICC or any league.

## What this repository contains

* This repository does **not** include the Cricsheet ZIP, the processed match tables, or the trained model binary.
  Those are generated on your machine from the ZIP you download.
* `models/evaluation.json` and `models/simulator_validation.json` are small reports computed from that data.
  They contain summary statistics only, not match records.

## Licence of the ChaseIQ code

The ChaseIQ source code is licensed under the MIT License (see `LICENSE`). That licence does **not** change the
terms that apply to the Cricsheet data above.

## Other sources

No other external datasets are used. Chart and interface code is original work. Fonts (Inter) are loaded from
Google Fonts when online; the app falls back to a system font otherwise.
