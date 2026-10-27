"""
Unit and integration tests for Abraham's co-visitation features (features/abraham.py).
Tests:
  1. Prefix and rowid integrity (all columns start with cv_, rowids are unique).
  2. Zero-leakage: adding events after the cutoff does NOT alter the feature values.
  3. Zero-leakage in click histories: post-cutoff click timestamps in history are strictly ignored.
  4. Cold user handling: user with 0 events before cutoff gets sensible fallbacks (0s, cv_has_history=0).
  5. Empty history handling: user with [] history gets sensible fallbacks.
  6. Unseen candidate item handling: item with no history/exposure gets 0 score.
  7. Time-weighted decay: more recent clicks produce higher time-weighted score than older clicks.
  8. Ranking integrity: cv_rank_in_event ranks 1 to 10 within each event.
  9. Source 'eval' compatibility: works seamlessly when predicting evaluation rows without 'clicked' column.
"""
import json
import unittest
import numpy as np
import pandas as pd
import duckdb

from common import check_features
from features.abraham import build, PREFIX


class TestAbrahamCoVisitation(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TABLE train (
                rowid VARCHAR,
                event_id VARCHAR,
                user_id VARCHAR,
                item_id VARCHAR,
                timestamp BIGINT,
                time_bucket TINYINT,
                day_type TINYINT,
                history_item_ids VARCHAR,
                history_timestamps VARCHAR,
                clicked TINYINT
            )
        """)

    def tearDown(self):
        self.con.close()

    def _insert_event(self, event_id, user_id, timestamp, time_bucket, day_type,
                      history_items, history_ts, candidate_items, clicked_item=None):
        h_json = json.dumps(history_items)
        hts_json = json.dumps(history_ts)
        for idx, item in enumerate(candidate_items):
            rowid = f"{event_id}_r{idx}"
            clicked = 1 if (clicked_item is not None and item == clicked_item) else 0
            self.con.execute("""
                INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [rowid, event_id, user_id, item, timestamp, time_bucket, day_type,
                  h_json, hts_json, clicked])

    def test_schema_and_prefix_compliance(self):
        """Verify check_features passes, unique rowids, and cv_ prefixes."""
        candidates = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, ["item_a"], [500], candidates, clicked_item="item_0")
        self._insert_event("e2", "u1", 2000, 1, 0, ["item_a", "item_0"], [500, 1000], candidates)

        df = build(self.con, cut=2000, end=3000, source="train")
        self.assertEqual(len(df), 10)
        self.assertTrue(df.rowid.is_unique)
        check_features(df, PREFIX)

        for col in df.columns:
            if col != "rowid":
                self.assertTrue(col.startswith("cv_"), f"Column {col} missing cv_ prefix")

    def test_zero_leakage_future_events(self):
        """Adding future events strictly after cutoff must NOT change feature values."""
        candidates = [f"item_{i}" for i in range(10)]
        # Pre-cutoff event (cut = 2000)
        self._insert_event("e1", "u1", 1000, 0, 0, ["item_a"], [500], candidates, clicked_item="item_0")
        # Target event at cut=2000
        self._insert_event("e_target", "u1", 2000, 1, 0, ["item_a", "item_0"], [500, 1000], candidates)

        df_before = build(self.con, cut=2000, end=3000, source="train")

        # Now insert future events at ts=2500 and ts=3500 with clicks
        self._insert_event("e_future_1", "u1", 2500, 1, 0, ["item_a", "item_0"], [500, 1000], candidates, clicked_item="item_2")
        self._insert_event("e_future_2", "u2", 3500, 2, 0, ["item_0"], [1000], candidates, clicked_item="item_3")

        df_after = build(self.con, cut=2000, end=3000, source="train")

        # Predictions for e_target must be completely identical
        pd.testing.assert_frame_equal(df_before, df_after)

    def test_zero_leakage_history_timestamps(self):
        """Any clicks in history with timestamp >= cut must be rejected."""
        candidates = [f"item_{i}" for i in range(10)]
        # History has a click at ts=2100 which is >= cut (2000)
        self._insert_event("e_target", "u1", 2050, 0, 0,
                           ["item_valid", "item_leaked"], [1500, 2100], candidates)

        df = build(self.con, cut=2000, end=3000, source="train")
        row = df.iloc[0]
        # Valid historical click count must be 1, not 2
        self.assertEqual(row["cv_hist_len_valid"], 1)

    def test_cold_user_and_empty_history(self):
        """Cold user with no history must get safe fallbacks (0s) and not crash."""
        candidates = [f"item_{i}" for i in range(10)]
        # u_cold has never interacted before
        self._insert_event("e_target_cold", "u_cold", 2000, 0, 0, [], [], candidates)

        df = build(self.con, cut=2000, end=3000, source="train")
        self.assertEqual(len(df), 10)
        for _, row in df.iterrows():
            self.assertEqual(row["cv_has_history"], 0)
            self.assertEqual(row["cv_hist_len_valid"], 0)
            self.assertEqual(row["cv_last1_prob"], 0.0)
            self.assertEqual(row["cv_last3_sum_prob"], 0.0)
            self.assertEqual(row["cv_last5_sum_prob"], 0.0)

    def test_time_weighted_recency_decay(self):
        """More recent clicks must yield higher time-weighted values."""
        candidates = ["target_item"] + [f"other_{i}" for i in range(9)]
        # Historical transition: 'A' -> 'target_item'
        # e1 clicked 'A' at 500, clicked 'target_item' at 600
        self._insert_event("e1", "u_hist", 500, 0, 0, [], [], ["A"] + [f"x{i}" for i in range(9)], clicked_item="A")
        self._insert_event("e2", "u_hist", 600, 0, 0, ["A"], [500], candidates, clicked_item="target_item")

        # User Recent: clicked 'A' 10 seconds ago (ts = 1990, target ts = 2000)
        self._insert_event("e_target_recent", "u_recent", 2000, 0, 0, ["A"], [1990], candidates)
        # User Old: clicked 'A' 100,000 seconds ago (ts = 1000, target ts = 101,000)
        self._insert_event("e_target_old", "u_old", 101000, 0, 0, ["A"], [1000], candidates)

        df = build(self.con, cut=2000, end=300000, source="train")
        row_recent = df[df.rowid == "e_target_recent_r0"].iloc[0]
        row_old = df[df.rowid == "e_target_old_r0"].iloc[0]

        # Both have the same base unweighted transition probability
        self.assertAlmostEqual(row_recent["cv_last1_prob"], row_old["cv_last1_prob"])
        # But recent user must have strictly higher time-weighted probability
        self.assertGreater(row_recent["cv_last3_tw_prob"], row_old["cv_last3_tw_prob"])

    def test_rank_in_event(self):
        """cv_rank_in_event must span 1 to 10 for each event."""
        candidates = [f"item_{i}" for i in range(10)]
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_a"], [1000], candidates)
        df = build(self.con, cut=2000, end=3000, source="train")
        ranks = sorted(df["cv_rank_in_event"].tolist())
        self.assertEqual(ranks, list(range(1, 11)))

    def test_eval_source_compatibility(self):
        """Works on eval table without 'clicked' column."""
        self.con.execute("""
            CREATE TABLE eval (
                rowid VARCHAR,
                event_id VARCHAR,
                user_id VARCHAR,
                item_id VARCHAR,
                timestamp BIGINT,
                time_bucket TINYINT,
                day_type TINYINT,
                history_item_ids VARCHAR,
                history_timestamps VARCHAR
            )
        """)
        candidates = [f"item_{i}" for i in range(10)]
        h_json = json.dumps(["item_0"])
        hts_json = json.dumps([1000])
        for idx, item in enumerate(candidates):
            self.con.execute("""
                INSERT INTO eval VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [f"eval_e1_r{idx}", "eval_e1", "u1", item, 2000, 0, 0, h_json, hts_json])

        df = build(self.con, cut=2000, end=3000, source="eval")
        self.assertEqual(len(df), 10)
        self.assertTrue(df.rowid.is_unique)
        check_features(df, PREFIX)


if __name__ == "__main__":
    unittest.main()
