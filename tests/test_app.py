"""Integration tests: real processed data (if built), API endpoints and the HTTP server.

These tests are skipped automatically when data/processed has not been built yet.
"""
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import app  # noqa: E402
from src import api  # noqa: E402
from src.api import ApiError  # noqa: E402
from src.store import DataStore  # noqa: E402

DATA_READY = config.PROCESSED_MATCHES.exists() and config.PROCESSED_DELIVERIES.exists() \
    and config.PROCESSED_CHASE_STATES.exists() and config.MODEL_FILE.exists()


@unittest.skipUnless(DATA_READY, "Build the data first: python app.py --build")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DataStore()
        assert cls.store.load()
        cls.match_id = str(cls.store.matches["match_id"].iloc[-1])

    def test_overview_has_counts(self):
        o = api.overview(self.store, {})
        self.assertGreater(o["matches"], 5000)
        self.assertIsNotNone(o["chase_win_rate"])

    def test_match_detail_and_winprob_timeline(self):
        m = api.match(self.store, {"id": self.match_id})
        self.assertTrue(len(m["innings"]) >= 1)
        self.assertIn("winprob", m)
        if m["winprob"]["available"]:
            self.assertIn("points", m["winprob"])

    def test_strategy_returns_ordered_scenarios(self):
        s = api.strategy(self.store, {"runs_needed": "45", "balls_left": "30",
                                      "wickets_left": "6", "target": "180"})
        self.assertTrue(s["model_available"])
        self.assertGreater(s["model_win_probability"], 0.0)
        sc = s["scenarios"]
        for key in sc:
            self.assertTrue(0.0 <= sc[key]["win_probability"] <= 1.0)
        self.assertAlmostEqual(sc["balanced"]["delta_vs_balanced_pp"], 0.0)
        self.assertIn("simulator_calibration", s)

    def test_strategy_rejects_bad_input(self):
        with self.assertRaises(ApiError) as ctx:
            api.strategy(self.store, {"runs_needed": "200", "balls_left": "10",
                                      "wickets_left": "2", "target": "100"})
        self.assertEqual(ctx.exception.status, 400)
        with self.assertRaises(ApiError):
            api.strategy(self.store, {"runs_needed": "5"})

    def test_winprob_grid_monotonic_in_runs_needed(self):
        g = api.winprob_grid(self.store, {"wickets_left": "8", "target": "170"})
        col = [row[0] for row in g["probability"]]          # balls_left = 120
        self.assertEqual(col, sorted(col, reverse=True))

    def test_player_profile_and_compare(self):
        name = api.overview(self.store, {})["top_batters"][0]["player"]
        p = api.player(self.store, {"name": name})
        self.assertTrue(p["found"])
        c = api.compare(self.store, {"a": name, "b": name})
        self.assertEqual(len(c["radar"]), 6)

    def test_matchup_options(self):
        names = api.filters(self.store, {})["players"]
        out = api.matchup(self.store, {"batter": names[0]})
        self.assertIn("bowlers", out)

    def test_teams_table_and_h2h(self):
        t = api.teams(self.store, {})
        self.assertGreater(len(t["table"]), 10)
        self.assertLessEqual(len(t["head_to_head"]["teams"]), 12)

    def test_explorer_limits(self):
        e = api.explorer(self.store, {"table": "matches", "limit": "2"})
        self.assertEqual(len(e["rows"]), 2)
        with self.assertRaises(ApiError):
            api.explorer(self.store, {"table": "deliveries"})

    def test_no_nan_in_json(self):
        payload = json.dumps(api.overview(self.store, {}), allow_nan=False)
        self.assertTrue(payload)


@unittest.skipUnless(DATA_READY, "Build the data first: python app.py --build")
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.STORE.load()
        cls.server = ThreadingHTTPServer((config.HOST, 0), app.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        url = f"http://{config.HOST}:{self.port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_index_and_static(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"ChaseIQ", body)
        status, _, _ = self.get("/app.js")
        self.assertEqual(status, 200)

    def test_path_traversal_blocked(self):
        status, _, _ = self.get("/..%2fconfig.py")
        self.assertIn(status, (403, 404))
        status, _, _ = self.get("/../config.py")
        self.assertIn(status, (403, 404))

    def test_api_status_and_unknown(self):
        status, ctype, body = self.get("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["data_loaded"])
        status, _, _ = self.get("/api/nope")
        self.assertEqual(status, 404)

    def test_api_bad_request_is_json_400(self):
        status, _, body = self.get("/api/strategy?runs_needed=abc")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_csv_export(self):
        status, ctype, body = self.get("/api/export/matches.csv")
        self.assertEqual(status, 200)
        self.assertIn("text/csv", ctype)
        self.assertTrue(body.startswith(b"match_id"))


if __name__ == "__main__":
    unittest.main()
