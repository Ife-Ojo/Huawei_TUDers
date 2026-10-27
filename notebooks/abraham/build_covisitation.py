#!/usr/bin/env python3
"""
Co-visitation matrix builder for Huawei TechArena CTR challenge.
Team: Huawei Tuders | Author: Abraham

Builds directed "B clicked after A" co-visitation tables using DuckDB:
  - Sequence window: <= W clicks (default W=5)
  - Time window: <= 24 hours (86,400s)
  - Memory limit enforced (8GB)
  - Strict leakage prevention: queries fit_sql(cut) where ts < cut
  - Writes results to Parquet in Data/

Usage:
  python notebooks/abraham/build_covisitation.py                     # sample DB, cut=C1
  python notebooks/abraham/build_covisitation.py --cut 1737936000     # specific cutoff
  python notebooks/abraham/build_covisitation.py full                # full DB
"""
import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import connect, fit_sql, C1, C2, C3, DAY


def build_covisitation_table(con, cut, window_size=5, max_time_diff=DAY, out_parquet=None):
    """
    Build directed co-visitation pairs (item_a -> item_b) strictly from events before cut.

    Args:
        con: DuckDB connection.
        cut: Cutoff timestamp (BIGINT epoch seconds).
        window_size: Maximum sequence steps between A and B (default: 5).
        max_time_diff: Maximum time difference in seconds between A and B (default: 86400s).
        out_parquet: Optional Path to save resulting Parquet file.

    Returns:
        pd.DataFrame containing co-visitation transition summary.
    """
    print(f"Building co-visitation table for cutoff {cut} (window={window_size}, max_dt={max_time_diff}s)...")
    t0 = time.time()

    query = f"""
    WITH fit AS {fit_sql(cut)},

    -- 1. All valid clicks strictly before cut
    raw_clicks AS (
        SELECT DISTINCT user_id, item_id, ts
        FROM (
            SELECT user_id,
                   unnest(from_json(history_item_ids, '["VARCHAR"]')) AS item_id,
                   unnest(from_json(history_timestamps, '["BIGINT"]')) AS ts
            FROM (SELECT DISTINCT event_id, user_id, history_item_ids, history_timestamps FROM fit)
            UNION ALL
            SELECT user_id, item_id, timestamp AS ts
            FROM fit WHERE clicked = 1
        )
        WHERE ts < {cut}
    ),

    -- 2. Chronological sequence per user
    ordered_clicks AS (
        SELECT user_id, item_id, ts,
               row_number() OVER (PARTITION BY user_id ORDER BY ts, item_id) AS pos
        FROM raw_clicks
    ),

    -- 3. Extract directed pairs "B after A"
    pairs AS (
        SELECT c1.item_id AS item_a,
               c2.item_id AS item_b,
               c2.pos - c1.pos AS step_diff,
               c2.ts - c1.ts AS time_diff
        FROM ordered_clicks c1
        JOIN ordered_clicks c2
          ON c1.user_id = c2.user_id
         AND c2.pos > c1.pos
         AND c2.pos <= c1.pos + {window_size}
         AND c2.ts > c1.ts
         AND c2.ts <= c1.ts + {max_time_diff}
    ),

    -- 4. Item marginal click counts for normalization
    item_counts AS (
        SELECT item_id, count(*)::BIGINT AS click_count
        FROM raw_clicks
        GROUP BY item_id
    )

    -- 5. Summary statistics per (item_a, item_b) pair
    SELECT p.item_a,
           p.item_b,
           count(*)::BIGINT AS pair_cnt,
           sum(1.0 / p.step_diff)::DOUBLE AS pos_weighted_cnt,
           sum(1.0 / (1.0 + p.time_diff / 3600.0))::DOUBLE AS time_weighted_cnt,
           count(*)::DOUBLE / nullif(coalesce(ca.click_count, 1), 0) AS prob_b_given_a,
           count(*)::DOUBLE / sqrt(nullif(ca.click_count * cb.click_count, 0)) AS cos_sim
    FROM pairs p
    LEFT JOIN item_counts ca ON ca.item_id = p.item_a
    LEFT JOIN item_counts cb ON cb.item_id = p.item_b
    GROUP BY p.item_a, p.item_b, ca.click_count, cb.click_count
    ORDER BY pair_cnt DESC
    """

    if out_parquet:
        out_parquet = Path(out_parquet)
        out_parquet.parent.mkdir(exist_ok=True)
        con.execute(f"COPY ({query}) TO '{out_parquet}' (FORMAT PARQUET)")
        print(f"Saved to {out_parquet} ({out_parquet.stat().st_size / 1024:.1f} KB)")

    df = con.sql(query).df()
    duration = time.time() - t0
    print(f"Generated {len(df):,} directed co-visitation pairs in {duration:.2f}s")
    return df


def main():
    ap = argparse.ArgumentParser(description="Build directed co-visitation tables")
    ap.add_argument("data", nargs="?", default="sample", choices=["sample", "full"])
    ap.add_argument("--cut", type=int, default=C1, help="Cutoff epoch seconds (default: C1)")
    ap.add_argument("--window", type=int, default=5, help="Sliding click window size (default: 5)")
    ap.add_argument("--out", type=str, default=None, help="Output Parquet path")
    args = ap.parse_args()

    con = connect(full=(args.data == "full"))
    out_path = args.out or (ROOT / "Data" / f"covis_cut_{args.cut}_{args.data}.parquet")
    df = build_covisitation_table(con, args.cut, window_size=args.window, out_parquet=out_path)

    print("\nTop 10 Co-visitation pairs:")
    print(df.head(10))


if __name__ == "__main__":
    main()
