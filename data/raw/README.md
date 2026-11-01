# Raw data folder

Place the Cricsheet T20 JSON archive here:

    data/raw/t20s_json.zip

Download it from https://cricsheet.org/downloads/ (file `t20s_json.zip`). The archive is not included in
this project; it is git-ignored and must be downloaded separately (see DATA_ATTRIBUTION.md).

Then run `python scripts/rebuild_model.py`, or `python app.py` (which builds automatically on first start).
