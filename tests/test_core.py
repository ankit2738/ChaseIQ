"""Unit tests for ingestion, features, analytics and the strategy simulator.

The small match dictionaries below are TEST FIXTURES written in the Cricsheet JSON
shape. They are not real matches and are never shown in the app as data.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import analytics as an  # noqa: E402
from src.features import FEATURES, chronological_split, state_frame  # noqa: E402
from src.ingest import parse_match  # noqa: E402
from src.strategy import analyse, phase_distributions, simulate_chase  # noqa: E402


def dlv(batter, bowler, runs_batter=0, extras=None, extras_total=0, wicket=None, non_striker="N"):
    d = {"batter": batter, "bowler": bowler, "non_striker": non_striker,
         "runs": {"batter": runs_batter, "extras": extras_total,
                  "total": runs_batter + extras_total}}
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"player_out": wicket[0], "kind": wicket[1]}]
    return d


def fixture_match():
    """Two innings, one super-over. Team A bats first, team B chases and wins."""
    first_over = [dlv("A1", "B1", 4), dlv("A1", "B1", 0), dlv("A1", "B1", 0,
                  extras={"wides": 1}, extras_total=1), dlv("A1", "B1", 1),
                  dlv("A2", "B1", 6, wicket=("A1", "caught")), dlv("A2", "B1", 0)]
    overs_a = [{"over": 0, "deliveries": first_over}]
    overs_b = [{"over": 0, "deliveries": [dlv("B1", "A1", 1), dlv("B1", "A1", 1, wicket=("B1", "bowled")),
                                          dlv("B2", "A1", 4, extras={"legbyes": 1}, extras_total=1),
                                          dlv("B2", "A1", 0, extras={"noballs": 1}, extras_total=1),
                                          dlv("B2", "A1", 0), dlv("B2", "A1", 0)]}]
    return {
        "info": {"dates": ["2024-05-01"], "season": "2024", "event": {"name": "Test League"},
                 "match_type": "T20", "gender": "male", "team_type": "club", "overs": 20,
                 "teams": ["TeamA", "TeamB"], "venue": "V", "city": "C",
                 "toss": {"winner": "TeamA", "decision": "bat"},
                 "outcome": {"winner": "TeamB", "by": {"wickets": 8}}},
        "innings": [
            {"team": "TeamA", "overs": overs_a},
            {"team": "TeamB", "overs": overs_b},
            {"team": "TeamA", "super_over": True, "overs": [{"over": 0, "deliveries": [dlv("Z", "Y", 6)]}]},
        ],
    }


class IngestTests(unittest.TestCase):
    def test_parse_counts_and_cleaning(self):
        report = Counter()
        row, balls = parse_match("999.json", fixture_match(), report)
        self.assertEqual(row["match_id"], "999")
        self.assertEqual(row["winner"], "TeamB")
        self.assertEqual(report["super_over_innings_excluded"], 1)
        # Super-over ball must not appear in the ball table.
        self.assertFalse(any(b[5] == "Z" for b in balls))
        self.assertEqual(row["inn1_runs"], 4 + 0 + 0 + 1 + 6 + 0 + 1)  # wide adds 1 extra run
        self.assertEqual(row["inn1_wickets"], 1)
        self.assertEqual(row["inn1_balls"], 5)  # the wide is not a legal ball
        self.assertEqual(row["inn2_team"], "TeamB")
        self.assertEqual(row["inn2_wickets"], 1)

    def test_legal_and_wicket_flags(self):
        report = Counter()
        _, balls = parse_match("999.json", fixture_match(), report)
        cols = ["match_id", "innings", "batting_team", "bowling_team", "over", "batter", "bowler",
                "non_striker", "legal", "runs_batter", "runs_extras", "runs_total", "byes_legbyes",
                "extra_type", "wicket_count", "player_out", "wicket_kind", "legal_before", "runs_before",
                "wickets_before"]
        df = pd.DataFrame(balls, columns=cols)
        wide = df[df["extra_type"] == "wides"]
        self.assertEqual(len(wide), 1)
        self.assertEqual(int(wide["legal"].iloc[0]), 0)
        noball = df[df["extra_type"] == "noballs"]
        self.assertEqual(int(noball["legal"].iloc[0]), 0)
        self.assertEqual(int(df[df["innings"] == 2]["byes_legbyes"].sum()), 1)


class FeatureTests(unittest.TestCase):
    def test_state_rates(self):
        f = state_frame(runs_needed=[60], balls_left=[30], wickets_left=[5], target=[160])
        self.assertAlmostEqual(f.loc[0, "rrr"], 12.0)
        self.assertAlmostEqual(f.loc[0, "crr"], 100 * 6 / 90)
        self.assertAlmostEqual(f.loc[0, "rrr_minus_crr"], 12.0 - 100 * 6 / 90)
        self.assertListEqual(list(f.columns), FEATURES)

    def test_no_balls_left_is_capped(self):
        f = state_frame(runs_needed=[5], balls_left=[0], wickets_left=[3], target=[100])
        self.assertLessEqual(f.loc[0, "rrr"], 99.0)

    def test_chronological_split_has_no_match_overlap(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(200):
            date = f"2020-01-{(i % 28) + 1:02d}" if i < 100 else f"2024-01-{(i % 28) + 1:02d}"
            for _ in range(6):
                rows.append({"match_id": f"m{i}", "date": date, "label": int(rng.integers(0, 2))})
        states = pd.DataFrame(rows)
        fit, val, test = chronological_split(states)
        self.assertFalse((fit & val).any())
        self.assertFalse((fit & test).any())
        self.assertFalse((val & test).any())
        fit_ids = set(states.loc[fit, "match_id"])
        test_ids = set(states.loc[test, "match_id"])
        self.assertFalse(fit_ids & test_ids)
        self.assertTrue(states.loc[test, "date"].min() >= states.loc[fit, "date"].max())


def _adjusted_chase_fixture():
    """Three decided 20-over chases: plain, D/L-adjusted and blank-method. Fixture data only."""
    base = {"date": "2024-01-01", "season": "2024", "event": "E", "match_type": "T20",
            "inn1_runs": 120, "inn1_team": "A", "inn2_team": "B", "winner": "B", "overs": 20}
    matches = pd.DataFrame([
        {**base, "match_id": "100", "method": np.nan},
        {**base, "match_id": "200", "method": "D/L", "date": "2024-01-02"},
        {**base, "match_id": "300", "method": "", "date": "2024-01-03"},
    ])
    deliveries = pd.DataFrame([
        {"match_id": mid, "innings": 2, "legal": 1, "legal_before": k,
         "runs_before": 2 * k, "wickets_before": 0}
        for mid in ["100", "200", "300"] for k in range(6)
    ])
    return deliveries, matches


class AdjustedResultExclusionTests(unittest.TestCase):
    def test_dl_matches_are_excluded_from_chase_states(self):
        from src.features import build_chase_states
        deliveries, matches = _adjusted_chase_fixture()
        states = build_chase_states(deliveries, matches)
        kept = set(states["match_id"].astype(str))
        self.assertIn("100", kept)
        self.assertIn("300", kept)  # blank method is not an adjustment
        self.assertNotIn("200", kept)  # D/L-adjusted result is excluded from training
        self.assertGreater(len(states), 0)

    def test_exclusion_counts_are_reported(self):
        from src.features import chase_training_exclusions
        _, matches = _adjusted_chase_fixture()
        counts = chase_training_exclusions(matches)
        self.assertEqual(counts["candidate_decided_20_over_chases"], 3)
        self.assertEqual(counts["excluded_adjusted_result_matches"], 1)
        self.assertEqual(counts["excluded_by_method"], {"D/L": 1})
        self.assertEqual(counts["included_matches"], 2)


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        rng = np.random.default_rng(1)
        for i in range(3000):
            runs = rng.choice([0, 1, 2, 4, 6], p=[0.4, 0.4, 0.07, 0.08, 0.05])
            wicket = int(rng.random() < 0.06)
            phase_over = rng.integers(0, 20)
            wk_before = rng.integers(0, 10)
            rows.append({"legal": 1, "innings": 2, "runs_total": runs, "wicket_count": wicket,
                         "phase": config.phase_of_over(int(phase_over)), "wickets_before": wk_before})
        cls.dists = phase_distributions(pd.DataFrame(rows))

    def test_already_won_and_no_wickets(self):
        s = config.SCENARIOS["balanced"]
        self.assertEqual(simulate_chase(0, 10, 5, 110, self.dists, s)["win_probability"], 1.0)
        self.assertEqual(simulate_chase(50, 10, 0, 110, self.dists, s)["win_probability"], 0.0)
        self.assertEqual(simulate_chase(50, 0, 5, 120, self.dists, s)["win_probability"], 0.0)

    def test_monotonic_in_runs_needed(self):
        s = config.SCENARIOS["balanced"]
        easy = simulate_chase(20, 30, 6, 90, self.dists, s)["win_probability"]
        hard = simulate_chase(60, 30, 6, 90, self.dists, s)["win_probability"]
        self.assertGreater(easy, hard)

    def test_aggressive_is_more_volatile_than_conservative(self):
        agg = simulate_chase(45, 30, 6, 90, self.dists, config.SCENARIOS["aggressive"])
        con = simulate_chase(45, 30, 6, 90, self.dists, config.SCENARIOS["conservative"])
        self.assertGreater(agg["avg_wickets_lost"], con["avg_wickets_lost"])
        self.assertGreater(agg["avg_runs_scored"], con["avg_runs_scored"])

    def test_same_seed_is_reproducible(self):
        s = config.SCENARIOS["balanced"]
        a = simulate_chase(40, 24, 5, 96, self.dists, s, seed=3)
        b = simulate_chase(40, 24, 5, 96, self.dists, s, seed=3)
        self.assertEqual(a, b)

    def test_analyse_returns_all_scenarios_and_relative_deltas(self):
        out = analyse({"runs_needed": 45, "balls_left": 30, "wickets_left": 6, "target": 180},
                      self.dists, None, 0.5)
        self.assertEqual(set(out["scenarios"]), {"conservative", "balanced", "aggressive"})
        self.assertAlmostEqual(out["scenarios"]["balanced"]["delta_vs_balanced_pp"], 0.0)
        self.assertEqual(out["model_win_probability"], 0.5)
        self.assertIn(out["pressure_level"], {"Low", "Medium", "High", "Very high"})


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        report = Counter()
        row, balls = parse_match("999.json", fixture_match(), report)
        cols = ["match_id", "innings", "batting_team", "bowling_team", "over", "batter", "bowler",
                "non_striker", "legal", "runs_batter", "runs_extras", "runs_total", "runs_byes_legbyes",
                "extra_type", "wicket_count", "player_out", "wicket_kind", "legal_before", "runs_before",
                "wickets_before"]
        self.d = pd.DataFrame(balls, columns=cols)
        self.d["extra_type"] = self.d["extra_type"].replace("", None)
        self.d["phase"] = self.d["over"].map(config.phase_of_over)
        self.d["won_innings"] = 0
        self.d["rrr_before"] = np.nan
        self.matches = pd.DataFrame([row])

    def test_batting_strike_rate_excludes_wides(self):
        bt = an.batting_table(self.d)
        # A1 faces: 4, 0, [wide - not a ball faced], 1, then is dismissed on A2's 6-ball
        # (the 6 is off A2's bat). So A1 faced 3 balls for 5 runs.
        a1 = bt.loc["A1"]
        self.assertEqual(int(a1["balls"]), 3)
        self.assertEqual(int(a1["runs"]), 5)
        self.assertEqual(int(a1["dismissals"]), 1)

    def test_bowling_economy_excludes_byes(self):
        bowl = an.bowling_table(self.d)
        b2 = bowl.loc["A1"]
        self.assertGreater(b2["legal_balls"], 0)
        self.assertGreaterEqual(b2["runs_conceded"], 0)

    def test_matchup_sample_warning(self):
        m = an.matchup(self.d, "B1", "A1")
        self.assertEqual(m["balls"], 2)
        self.assertIn("Very small sample", m["warning"])

    def test_team_table_win_counts(self):
        t = an.team_table(self.matches).set_index("team")
        self.assertEqual(int(t.loc["TeamB", "wins"]), 1)
        self.assertEqual(int(t.loc["TeamA", "wins"]), 0)
        self.assertEqual(int(t.loc["TeamB", "chase_wins"]), 1)

    def test_pressure_index_marks_small_samples_unused(self):
        out = an.pressure_index(self.d, "A1")
        self.assertIsNone(out["index"])
        self.assertIn("note", out)


if __name__ == "__main__":
    unittest.main()
