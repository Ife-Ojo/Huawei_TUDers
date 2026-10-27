"""
Marcela's clues: how good is each candidate ITEM?   (prefix i_)

Learned only from `fit` (training rows before the cutoff). Things Eoin's base file
does NOT have: item recency/age, clicks in the last 6h, trend, the item's click rate
in the row's own time bucket, and comparisons with the other 9 candidates.
  i_ctr_bucket         item click rate in the row's own time_bucket (smoothed)
  i_bucket_lift        that rate minus the item's overall rate (is this a good time for it?)
  i_clicks_per_day     clicks per day (histories incl. warm-up + labelled clicks)
  i_clicks_6h          clicks in the 6 hours before the cutoff
  i_trend_24h          last-24h clicks vs the item's usual daily clicks
  i_age_days           days since the item's first click
  i_hours_since_click  hours since the item's latest click
  *_rel / i_recency_rank   the same clues compared with the other 9 candidates of the event
Test it:  python features/marcela.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_sql, targets_sql, check_features  # type: ignore

PREFIX = "i_"
K = 20   # smoothing weight: items with few impressions move toward the overall rate

# Unique clicks (user, item, time): histories of fit events + labelled clicks in fit
CLICKS = """
    SELECT DISTINCT user_id, item_id, ts FROM (
        SELECT user_id,
               unnest(from_json(history_item_ids, '["VARCHAR"]')) AS item_id,
               unnest(from_json(history_timestamps, '["BIGINT"]')) AS ts
        FROM (SELECT DISTINCT event_id, user_id, history_item_ids, history_timestamps FROM fit)
        UNION ALL
        SELECT user_id, item_id, timestamp AS ts FROM fit WHERE clicked = 1
    )"""


def build(con, cut, end=None, source="train"):
    return con.sql(f"""
    WITH fit AS {fit_sql(cut)},
    t AS ({targets_sql(cut, end, source)}),
    clk AS ({CLICKS}),
    span AS (SELECT ({cut} - MIN(ts)) / 86400.0 AS d FROM clk),
    ck AS (
        SELECT item_id,
               COUNT(*) AS n,
               SUM(CASE WHEN ts >= {cut} - 6 * 3600 THEN 1 ELSE 0 END) AS n6,
               SUM(CASE WHEN ts >= {cut} - 86400 THEN 1 ELSE 0 END) AS n24,
               MIN(ts) AS first_ts, MAX(ts) AS last_ts
        FROM clk GROUP BY item_id
    ),
    imp   AS (SELECT item_id, COUNT(*) AS n, SUM(clicked) AS c FROM fit GROUP BY item_id),
    imp_b AS (SELECT item_id, time_bucket, COUNT(*) AS n, SUM(clicked) AS c
              FROM fit GROUP BY item_id, time_bucket),
    g AS (SELECT SUM(c) * 1.0 / SUM(n) AS r FROM imp),
    item AS (
        SELECT p.item_id,
               (p.c + {K} * g.r) / (p.n + {K})                          AS ctr,
               COALESCE(k.n, 0) / s.d                                   AS clk_per_day,
               COALESCE(k.n6, 0)                                        AS clk_6h,
               COALESCE(k.n24, 0) / NULLIF(COALESCE(k.n, 0) / s.d, 0)   AS trend,
               ({cut} - k.first_ts) / 86400.0                           AS age_days,
               ({cut} - k.last_ts) / 3600.0                             AS hrs_since
        FROM imp p
        CROSS JOIN g CROSS JOIN span s
        LEFT JOIN ck k ON k.item_id = p.item_id
    ),
    f AS (
        SELECT t.event_id, t.rowid,
               it.ctr, it.clk_per_day, it.clk_6h, it.trend, it.age_days, it.hrs_since,
               (COALESCE(b.c, 0) + {K} * it.ctr) / (COALESCE(b.n, 0) + {K}) AS ctr_b
        FROM t
        LEFT JOIN item it ON it.item_id = t.item_id
        LEFT JOIN imp_b b ON b.item_id = t.item_id AND b.time_bucket = t.time_bucket
    )
    SELECT rowid,
           ctr_b                                   AS i_ctr_bucket,
           ctr_b - ctr                             AS i_bucket_lift,
           clk_per_day                             AS i_clicks_per_day,
           clk_6h                                  AS i_clicks_6h,
           trend                                   AS i_trend_24h,
           age_days                                AS i_age_days,
           hrs_since                               AS i_hours_since_click,
           -- compared with the other 9 candidates of the same event
           clk_per_day - avg(clk_per_day) OVER (PARTITION BY event_id) AS i_clicks_per_day_rel,
           clk_6h - avg(clk_6h) OVER (PARTITION BY event_id)           AS i_clicks_6h_rel,
           ctr_b - avg(ctr_b) OVER (PARTITION BY event_id)             AS i_ctr_bucket_rel,
           age_days - avg(age_days) OVER (PARTITION BY event_id)       AS i_age_rel,
           rank() OVER (PARTITION BY event_id ORDER BY hrs_since ASC NULLS LAST) AS i_recency_rank
    FROM f
    """).df()


if __name__ == "__main__":
    from common import connect, C1, C2  # type: ignore
    con = connect(full="full" in sys.argv)
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(df.describe().T[["count", "mean", "min", "max"]])