"""
Comprehensive synthetic unit and integration tests for Ife's user features module.
File: tests/test_ife.py

Tests:
  1. Correct historical click-rate calculation (clicked events / total events, smoothed)
  2. No-click events counted correctly (distinct event zero-click ratio)
  3. Distinct event counting across 10 candidates
  4. Correct last-click and last-event recency
  5. Previous unclicked exposure counts
  6. Correct time-bucket and weekend statistics
  7. New-user fallback behaviour
  8. Empty-history behaviour
  9. No division-by-zero errors
  10. Unique rowids and required u_ prefixes
  11. Correct output row count and target alignment
  12. No target-label leakage
  13. No future-event leakage (identical output when adding future records)
  14. No double-counting of overlapping warm-up and training clicks
"""
import json
import unittest
import numpy as np
import pandas as pd
import duckdb

from common import check_features
from features.ife import build, PREFIX


class TestIfeUserFeatures(unittest.TestCase):
    def setUp(self):
        """Create an in-memory DuckDB database matching the competition schema."""
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
        """Helper to insert a full 10-candidate recommendation event."""
        h_json = json.dumps(history_items)
        hts_json = json.dumps(history_ts)
        for idx, item in enumerate(candidate_items):
            rowid = f"{event_id}_r{idx}"
            clicked = 1 if (clicked_item is not None and item == clicked_item) else 0
            self.con.execute("""
                INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [rowid, event_id, user_id, item, timestamp, time_bucket, day_type,
                  h_json, hts_json, clicked])

    def test_schema_rowids_and_prefixes(self):
        """Test 10 & 11: Unique rowids, required u_ prefixes, output row count and target alignment."""
        # User 1: 1 event before cutoff
        items1 = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, ["w0"], [500], items1, clicked_item="item_0")

        # Target event for u1 at cutoff
        items_target = [f"item_{i}" for i in range(10)]
        self._insert_event("e2", "u1", 2000, 1, 0, ["w0", "item_0"], [500, 1000], items_target, clicked_item="item_1")

        # Target event for u_cold (completely cold user)
        self._insert_event("e3", "u_cold", 2000, 2, 1, [], [], items_target, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")

        # Must have exactly 20 rows (2 target events * 10 candidates)
        self.assertEqual(len(df), 20)
        self.assertTrue(df.rowid.is_unique)

        # Validation helper assertion
        check_features(df, PREFIX)

        # Every feature column must start with u_
        feature_cols = [c for c in df.columns if c != "rowid"]
        for c in feature_cols:
            self.assertTrue(c.startswith("u_"), f"Column {c} does not start with u_")

    def test_distinct_events_click_rate_and_zero_clicks(self):
        """Test 1, 2 & 3: Distinct event counting across 10 candidates, click rate, zero-click events."""
        items = [f"item_{i}" for i in range(10)]

        # User 1: 3 distinct events before cut=2000:
        # e1: 10 candidates, 1 clicked (event timestamp 1000)
        # e2: 10 candidates, 0 clicked (event timestamp 1200)
        # e3: 10 candidates, 0 clicked (event timestamp 1400)
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items, clicked_item="item_0")
        self._insert_event("e2", "u1", 1200, 0, 0, ["item_0"], [1000], items, clicked_item=None)
        self._insert_event("e3", "u1", 1400, 0, 0, ["item_0"], [1000], items, clicked_item=None)

        # Target event at 2000
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_0"], [1000], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        u1_rows = df[df.rowid.str.startswith("e_target")]

        # u1 had 3 events, 1 clicked event, 2 zero-click events
        # Global stats: only u1 exists in fit, so:
        # g_click_rate = 1/3, g_zero_click_ratio = 2/3
        # u_click_rate smoothed: (1 + 20 * (1/3)) / (3 + 20) = (1 + 20/3) / 23 = (23/3) / 23 = 1/3
        # u_zero_click_event_ratio smoothed: (2 + 20 * (2/3)) / (3 + 20) = (2 + 40/3) / 23 = (46/3) / 23 = 2/3
        for val in u1_rows["u_click_rate"]:
            self.assertAlmostEqual(val, 1.0 / 3.0, places=5)

        for val in u1_rows["u_zero_click_event_ratio"]:
            self.assertAlmostEqual(val, 2.0 / 3.0, places=5)

        # u_events_per_day: 3 events over (2000 - 1000)/86400 days -> span < 1 day, so greatest(1.0, span) = 1.0 -> 3.0
        for val in u1_rows["u_events_per_day"]:
            self.assertAlmostEqual(val, 3.0, places=5)

    def test_recency_calculations(self):
        """Test 4: Correct last-click, last-event, and inter-event recency."""
        items = [f"item_{i}" for i in range(10)]

        # u1 events:
        # Event 1 at ts=1000 (clicked item_0)
        self._insert_event("e1", "u1", 1000, 0, 0, ["warmup_item"], [500], items, clicked_item="item_0")
        # Event 2 at ts=1600 (no click)
        self._insert_event("e2", "u1", 1600, 0, 0, ["warmup_item", "item_0"], [500, 1000], items, clicked_item=None)

        # Target event at ts=2000
        self._insert_event("e_target", "u1", 2000, 0, 0, ["warmup_item", "item_0"], [500, 1000], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        u1_row = df.iloc[0]

        # Last click was at ts=1000, target is at ts=2000 -> 2000 - 1000 = 1000
        self.assertEqual(u1_row["u_secs_since_last_click"], 1000.0)

        # Last event was at ts=1600, target is at ts=2000 -> 2000 - 1600 = 400
        self.assertEqual(u1_row["u_secs_since_last_event"], 400.0)

        # First seen: warmup click was at ts=500, event was at ts=1000 -> min is 500 -> 2000 - 500 = 1500
        self.assertEqual(u1_row["u_secs_since_first_seen"], 1500.0)

        # Inter-event interval: 2 events (ts 1000 and ts 1600), avg interval = (1600 - 1000) / 1 = 600
        # Recency = 400. Ratio = 400 / 600 = 2/3
        self.assertAlmostEqual(u1_row["u_recency_to_avg_interval_ratio"], 400.0 / 600.0, places=5)

    def test_unclicked_exposure_counts(self):
        """Test 5: Previous unclicked exposure counts and time since last unclicked."""
        # e1 has items [item_0, item_1, ..., item_9], item_0 is clicked
        items1 = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items1, clicked_item="item_0")

        # e2 has items [other_0, item_1, ..., other_8], item_1 is unclicked again, item_2 is not in e2
        # item_0 is NOT in e2 (replaced with other_0)
        items2 = ["other_0", "item_1", "item_2"] + [f"other_{i}" for i in range(3, 10)]
        self._insert_event("e2", "u1", 1500, 0, 0, ["item_0"], [1000], items2, clicked_item="item_2")

        # Target event at ts=2000 with candidates item_0, item_1, item_2, item_99
        items_target = ["item_0", "item_1", "item_2", "item_99"] + [f"t_item_{i}" for i in range(4, 10)]
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_0", "item_2"], [1000, 1500], items_target, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")

        # Candidate item_0: shown 1 time in e1 and clicked -> unclicked count = 0, secs = NaN, ratio = 0.0
        r_item0 = df[df.rowid == "e_target_r0"].iloc[0]
        self.assertEqual(r_item0["u_item_shown_unclicked_cnt"], 0)
        self.assertTrue(pd.isna(r_item0["u_secs_since_last_unclicked"]))
        self.assertEqual(r_item0["u_item_unclicked_ratio"], 0.0)

        # Candidate item_1: shown 2 times (e1 at 1000, e2 at 1500), 0 clicks -> unclicked count = 2
        # Last unclicked was at ts=1500 -> secs since = 2000 - 1500 = 500
        # Unclicked ratio = 2 / 2 = 1.0
        r_item1 = df[df.rowid == "e_target_r1"].iloc[0]
        self.assertEqual(r_item1["u_item_shown_unclicked_cnt"], 2)
        self.assertEqual(r_item1["u_secs_since_last_unclicked"], 500.0)
        self.assertEqual(r_item1["u_item_unclicked_ratio"], 1.0)

        # Candidate item_2: shown 1 time in e1 (unclicked at 1000), shown 1 time in e2 (clicked at 1500)
        # Total shown = 2, unclicked = 1, last unclicked = 1000 -> secs = 2000 - 1000 = 1000, ratio = 0.5
        r_item2 = df[df.rowid == "e_target_r2"].iloc[0]
        self.assertEqual(r_item2["u_item_shown_unclicked_cnt"], 1)
        self.assertEqual(r_item2["u_secs_since_last_unclicked"], 1000.0)
        self.assertEqual(r_item2["u_item_unclicked_ratio"], 0.5)

        # Candidate item_99: never shown in fit -> unclicked count = 0, secs = NaN, ratio = NaN
        r_item99 = df[df.rowid == "e_target_r3"].iloc[0]
        self.assertEqual(r_item99["u_item_shown_unclicked_cnt"], 0)
        self.assertTrue(pd.isna(r_item99["u_secs_since_last_unclicked"]))
        self.assertTrue(pd.isna(r_item99["u_item_unclicked_ratio"]))

    def test_time_bucket_and_weekend_affinity(self):
        """Test 6: Time-bucket and weekend statistics with clear affinity documentation."""
        items = [f"item_{i}" for i in range(10)]

        # u1 history: 3 events in bucket 2 (ts 100, 200, 300), 1 event in bucket 1 (ts 400)
        # 3 weekend events (day_type=1), 1 weekday event (day_type=0)
        self._insert_event("e1", "u1", 100, 2, 1, [], [], items, clicked_item=None)
        self._insert_event("e2", "u1", 200, 2, 1, [], [], items, clicked_item=None)
        self._insert_event("e3", "u1", 300, 2, 1, [], [], items, clicked_item=None)
        self._insert_event("e4", "u1", 400, 1, 0, [], [], items, clicked_item=None)

        # Target event 1: at bucket 2, day_type 1 (weekend)
        self._insert_event("e_target_sat", "u1", 2000, 2, 1, [], [], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        row = df.iloc[0]

        # Modal bucket is 2
        self.assertEqual(row["u_fav_time_bucket"], 2)
        self.assertEqual(row["u_is_fav_time_bucket"], 1)

        # Weekend affinity on weekend: user has 3/4 weekend share (smoothed).
        # On day_type = 1, affinity equals the weekend share.
        self.assertAlmostEqual(row["u_day_type_affinity"], row["u_weekend_event_share"], places=5)

    def test_cold_start_and_empty_history(self):
        """Test 7, 8 & 9: New user fallback, empty history, and no division by zero."""
        items = [f"item_{i}" for i in range(10)]

        # Fit has some existing activity from user_other
        self._insert_event("e0", "user_other", 1000, 1, 0, [], [], items, clicked_item="item_0")

        # Target event for completely cold user with empty history
        self._insert_event("e_target_cold", "user_cold", 2000, 3, 1, [], [], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        cold_row = df[df.rowid.str.startswith("e_target_cold")].iloc[0]

        # Cold start indicators
        self.assertEqual(cold_row["u_is_cold_user"], 1)
        self.assertEqual(cold_row["u_has_click_history"], 0)

        # Fallback values
        self.assertEqual(cold_row["u_total_clicks"], 0)
        self.assertEqual(cold_row["u_events_per_day"], 0.0)
        self.assertTrue(pd.isna(cold_row["u_secs_since_last_click"]))
        self.assertTrue(pd.isna(cold_row["u_secs_since_last_event"]))
        self.assertTrue(pd.isna(cold_row["u_secs_since_first_seen"]))
        self.assertTrue(pd.isna(cold_row["u_recency_to_avg_interval_ratio"]))

        # Rates should be smoothed and non-NaN
        self.assertFalse(np.isnan(cold_row["u_click_rate"]))
        self.assertFalse(np.isnan(cold_row["u_zero_click_event_ratio"]))
        self.assertFalse(np.isnan(cold_row["u_share_in_current_bucket"]))
        self.assertFalse(np.isnan(cold_row["u_weekend_event_share"]))
        self.assertFalse(np.isnan(cold_row["u_day_type_affinity"]))

    def test_no_target_label_leakage(self):
        """Test 12: Target clicked label must never influence features."""
        items = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items, clicked_item="item_0")

        # Scenario A: target event clicked label = 0 for item_0
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_0"], [1000], items, clicked_item=None)
        df_a = build(self.con, cut=2000, end=3000, source="train")

        # Scenario B: target event clicked label = 1 for item_0
        self.con.execute("DELETE FROM train WHERE event_id = 'e_target'")
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_0"], [1000], items, clicked_item="item_0")
        df_b = build(self.con, cut=2000, end=3000, source="train")

        pd.testing.assert_frame_equal(df_a, df_b)

    def test_no_future_event_leakage(self):
        """Test 13: Future events after cutoff must not alter features computed for cutoff."""
        items = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items, clicked_item="item_0")
        self._insert_event("e_target", "u1", 2000, 0, 0, ["item_0"], [1000], items, clicked_item=None)

        # Baseline build at cut=2000
        df_before = build(self.con, cut=2000, end=3000, source="train")

        # Add 5 future events for u1 after cut=2000
        for fut_idx in range(5):
            self._insert_event(f"e_fut_{fut_idx}", "u1", 2500 + fut_idx * 50, 1, 1,
                               ["item_0"], [1000], items, clicked_item="item_5")

        # Re-build at cut=2000: must be strictly identical
        df_after = build(self.con, cut=2000, end=3000, source="train")

        pd.testing.assert_frame_equal(df_before, df_after)

    def test_no_double_counting_clicks(self):
        """Test 14: Deduplication between warm-up history and fit clicks."""
        items = [f"item_{i}" for i in range(10)]

        # Warm-up click at ts=500
        # In fit: event e1 at ts=1000 clicks item_0
        self._insert_event("e1", "u1", 1000, 0, 0, ["warmup_item"], [500], items, clicked_item="item_0")

        # Target event at ts=2000: history array contains BOTH the warm-up click AND item_0 at ts=1000
        # Plus an illegal future click at ts=2500 that must be rejected
        self._insert_event("e_target", "u1", 2000, 0, 0,
                           ["warmup_item", "item_0", "illegal_future_item"],
                           [500, 1000, 2500],
                           items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        u1_row = df.iloc[0]

        # Total clicks must be exactly 2:
        # 1: 'warmup_item' at 500 (from history)
        # 2: 'item_0' at 1000 (present in both fit AND history, unified to 1)
        # The click at 2500 is >= cut, so it must be rejected!
        self.assertEqual(u1_row["u_total_clicks"], 2)


    def test_base_and_user_join(self):
        """Test integration: Mirrors train.make_table rowid join with features/base.py."""
        from features.base import build as build_base
        items = [f"item_{i}" for i in range(10)]
        self._insert_event("e1", "u1", 1000, 0, 0, ["w0"], [500], items, clicked_item="item_0")
        self._insert_event("e_target", "u1", 2000, 1, 0, ["w0", "item_0"], [500, 1000], items, clicked_item=None)

        base = build_base(self.con, 2000, 3000)
        user = build(self.con, 2000, 3000)

        combined = base.merge(user, on="rowid", how="left", validate="one_to_one")
        self.assertEqual(len(combined), 10)
        self.assertEqual(set(combined.rowid), set(user.rowid))
        self.assertTrue(any(c.startswith("base_") for c in combined.columns))
        self.assertTrue(any(c.startswith("u_") for c in combined.columns))
        # Ensure no overlapping column names between base_ and u_
        base_cols = {c for c in base.columns if c != "rowid"}
        user_cols = {c for c in user.columns if c != "rowid"}
        self.assertEqual(len(base_cols.intersection(user_cols)), 0)

    def test_repeated_clicks_and_same_timestamps(self):
        """Test edge cases: repeated clicks on same item, multiple clicks at identical timestamps."""
        items = [f"item_{i}" for i in range(10)]
        # In history:
        # - item_A clicked at 100 and again at 300 (2 legitimate separate clicks)
        # - item_B and item_C clicked at identical timestamp 500 (2 separate items)
        # - item_A clicked at 100 repeated in array (duplicate, should not double count)
        h = ["item_A", "item_A", "item_B", "item_C", "item_A"]
        hts = [100, 300, 500, 500, 100]
        self._insert_event("e_target", "u1", 2000, 0, 0, h, hts, items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        u1_row = df.iloc[0]

        # Total clicks should be exactly 4:
        # (item_A, 100), (item_A, 300), (item_B, 500), (item_C, 500)
        self.assertEqual(u1_row["u_total_clicks"], 4)
        # Max click ts = 500 -> secs since last click = 2000 - 500 = 1500
        self.assertEqual(u1_row["u_secs_since_last_click"], 1500.0)

    def test_cutoff_boundary_and_future_timestamps(self):
        """Test edge cases: timestamps exactly equal to cut or strictly in future must be excluded."""
        items = [f"item_{i}" for i in range(10)]
        # History for u1 has:
        # - click at 1500 (< cut=2000) -> valid
        # - click at 2000 (== cut=2000) -> MUST BE REJECTED
        # - click at 2500 (> cut=2000) -> MUST BE REJECTED
        h = ["item_valid", "item_boundary", "item_future"]
        hts = [1500, 2000, 2500]
        self._insert_event("e_target", "u1", 3000, 0, 0, h, hts, items, clicked_item=None)

        # In fit: an event at timestamp=2000 (== cut) for u_boundary must NOT be in fit_sql(2000)
        self._insert_event("e_fit_boundary", "u_boundary", 2000, 0, 0, [], [], items, clicked_item="item_0")

        df = build(self.con, cut=2000, end=4000, source="train")
        u1_row = df[df.rowid.str.startswith("e_target")].iloc[0]
        ub_row = df[df.rowid.str.startswith("e_fit_boundary")].iloc[0]

        # For u1: Only the click at 1500 is accepted (clicks at 2000 and 2500 rejected)
        self.assertEqual(u1_row["u_total_clicks"], 1)
        # Recency uses 1500: 3000 - 1500 = 1500
        self.assertEqual(u1_row["u_secs_since_last_click"], 1500.0)

        # For u_boundary: The event at timestamp 2000 was NOT before cut=2000, so user has 0 events in fit
        self.assertEqual(ub_row["u_is_cold_user"], 1)

    def test_malformed_and_mismatched_history_arrays(self):
        """Test edge cases: null, empty, or mismatched length history arrays."""
        items = [f"item_{i}" for i in range(10)]

        # u_null: h is NULL, hts is NULL
        for idx, item in enumerate(items):
            self.con.execute("""
                INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [f"e_null_r{idx}", "e_null", "u_null", item, 2000, 0, 0, None, None, 0])

        # u_mismatch: len(h) = 2, len(hts) = 1 (corrupted)
        h_json = json.dumps(["item_1", "item_2"])
        hts_json = json.dumps([500])
        for idx, item in enumerate(items):
            self.con.execute("""
                INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [f"e_mismatch_r{idx}", "e_mismatch", "u_mismatch", item, 2000, 0, 0, h_json, hts_json, 0])

        df = build(self.con, cut=2000, end=3000, source="train")
        self.assertEqual(len(df), 20)

        # Both should safely default without raising exceptions
        r_null = df[df.rowid == "e_null_r0"].iloc[0]
        self.assertEqual(r_null["u_total_clicks"], 0)
        self.assertEqual(r_null["u_has_click_history"], 0)

        r_mis = df[df.rowid == "e_mismatch_r0"].iloc[0]
        self.assertEqual(r_mis["u_total_clicks"], 0)
        self.assertEqual(r_mis["u_has_click_history"], 0)

    def test_warmup_only_users(self):
        """Test edge case: users with NO events in fit, but with historical warm-up clicks."""
        items = [f"item_{i}" for i in range(10)]
        # u_warmup has no events in fit, but target event has warm-up clicks from ts=400 and 800
        self._insert_event("e_target_warmup", "u_warmup", 2000, 1, 0,
                           ["warm_1", "warm_2"], [400, 800], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        row = df.iloc[0]

        # Cold user regarding recommendation events, but has click history
        self.assertEqual(row["u_is_cold_user"], 1)
        self.assertEqual(row["u_has_click_history"], 1)
        self.assertEqual(row["u_total_clicks"], 2)
        # Recency from clicks: 2000 - 800 = 1200
        self.assertEqual(row["u_secs_since_last_click"], 1200.0)
        # First seen from clicks: 2000 - 400 = 1600
        self.assertEqual(row["u_secs_since_first_seen"], 1600.0)
        # Event recency is NaN (never had a recommendation event before cut)
        self.assertTrue(pd.isna(row["u_secs_since_last_event"]))
        self.assertTrue(pd.isna(row["u_recency_to_avg_interval_ratio"]))

    def test_identical_event_timestamps_zero_interval(self):
        """Test edge case: multiple events with identical timestamps (interval = 0)."""
        items = [f"item_{i}" for i in range(10)]
        # 2 events in fit with the exact same timestamp 1000
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items, clicked_item=None)
        self._insert_event("e2", "u1", 1000, 0, 0, [], [], items, clicked_item=None)

        # Target event at 2000
        self._insert_event("e_target", "u1", 2000, 0, 0, [], [], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        row = df.iloc[0]

        # max_ts == min_ts == 1000 -> interval is 0 -> ratio must safely return NaN, not crash
        self.assertTrue(pd.isna(row["u_recency_to_avg_interval_ratio"]))
        self.assertEqual(row["u_secs_since_last_event"], 1000.0)

    def test_all_four_time_buckets(self):
        """Test edge case: correct handling of all 4 time buckets (0, 1, 2, 3)."""
        items = [f"item_{i}" for i in range(10)]
        # Insert events across all 4 time buckets in fit
        self._insert_event("e0", "u1", 100, 0, 0, [], [], items, clicked_item=None)
        self._insert_event("e1", "u1", 200, 1, 0, [], [], items, clicked_item=None)
        self._insert_event("e2", "u1", 300, 2, 0, [], [], items, clicked_item=None)
        self._insert_event("e3a", "u1", 400, 3, 0, [], [], items, clicked_item=None)
        self._insert_event("e3b", "u1", 500, 3, 0, [], [], items, clicked_item=None)  # bucket 3 has 2 events (modal)

        # Target in bucket 3
        self._insert_event("e_target_b3", "u1", 2000, 3, 0, [], [], items, clicked_item=None)

        df = build(self.con, cut=2000, end=3000, source="train")
        row = df.iloc[0]

        self.assertEqual(row["u_fav_time_bucket"], 3)
        self.assertEqual(row["u_is_fav_time_bucket"], 1)

    def test_source_eval(self):
        """Test edge case: source='eval' table where 'clicked' column is absent."""
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
        items = [f"item_{i}" for i in range(10)]
        # Fit activity in train
        self._insert_event("e1", "u1", 1000, 0, 0, [], [], items, clicked_item="item_0")

        # Target event in eval (note: NO clicked column!)
        h_json = json.dumps(["item_0"])
        hts_json = json.dumps([1000])
        for idx, item in enumerate(items):
            self.con.execute("""
                INSERT INTO eval VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [f"eval_e2_r{idx}", "eval_e2", "u1", item, 2000, 1, 0, h_json, hts_json])

        df = build(self.con, cut=2000, end=3000, source="eval")
        self.assertEqual(len(df), 10)
        self.assertTrue(df.rowid.is_unique)
        check_features(df, PREFIX)
        row = df.iloc[0]
        self.assertAlmostEqual(row["u_secs_since_last_click"], 1000.0)


if __name__ == "__main__":
    unittest.main()
