import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.parquet"
con = duckdb.connect(str(ROOT / "data" / "marcela.duckdb"))

FEATS = ["clicks_total", "users_total", "clicks_6h", "clicks_24h",
         "clicks_per_day", "trend_24h", "item_age_days",
         "hours_since_last_click", "share_b0", "share_b1", "share_b2",
         "share_b3", "share_weekend", "impressions", "ctr_smooth"]

def build_item_features(cutoff: int, name: str):
    """Item features using only data strictly before `cutoff` (Unix seconds)."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {name} AS
    WITH c AS (
        SELECT * FROM clicks WHERE click_ts < {cutoff}
    ),
    span AS (
        SELECT ({cutoff} - MIN(click_ts)) / 86400.0 AS days FROM c
    ),
    base AS (
        SELECT item_id,
               COUNT(*) AS clicks_total,
               COUNT(DISTINCT user_id) AS users_total,
               SUM(CASE WHEN click_ts >= {cutoff} - 6*3600 THEN 1 ELSE 0 END) AS clicks_6h,
               SUM(CASE WHEN click_ts >= {cutoff} - 86400  THEN 1 ELSE 0 END) AS clicks_24h,
               MIN(click_ts) AS first_seen,
               MAX(click_ts) AS last_click,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 0 THEN 1.0 ELSE 0 END) AS share_b0,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 1 THEN 1.0 ELSE 0 END) AS share_b1,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 2 THEN 1.0 ELSE 0 END) AS share_b2,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 3 THEN 1.0 ELSE 0 END) AS share_b3,
               AVG(CASE WHEN ((click_ts // 86400) + 3) % 7 >= 5 THEN 1.0 ELSE 0 END) AS share_weekend
        FROM c GROUP BY item_id
    ),
    imp AS (
        SELECT item_id, COUNT(*) AS impressions, SUM(clicked) AS imp_clicks
        FROM '{TRAIN}' WHERE timestamp < {cutoff} GROUP BY item_id
    ),
    overall AS (
        SELECT SUM(imp_clicks) * 1.0 / SUM(impressions) AS rate FROM imp
    ),
    items AS (
        SELECT item_id FROM base UNION SELECT item_id FROM imp
    )
    SELECT i.item_id,
           CAST(COALESCE(b.clicks_total, 0) AS BIGINT) AS clicks_total,
           CAST(COALESCE(b.users_total, 0) AS BIGINT)  AS users_total,
           CAST(COALESCE(b.clicks_6h, 0) AS BIGINT)    AS clicks_6h,
           CAST(COALESCE(b.clicks_24h, 0) AS BIGINT)   AS clicks_24h,
           COALESCE(b.clicks_total, 0) / s.days        AS clicks_per_day,
           COALESCE(b.clicks_24h, 0)
             / NULLIF(COALESCE(b.clicks_total, 0) / s.days, 0) AS trend_24h,
           ({cutoff} - b.first_seen) / 86400.0         AS item_age_days,
           ({cutoff} - b.last_click) / 3600.0          AS hours_since_last_click,
           b.share_b0, b.share_b1, b.share_b2, b.share_b3, b.share_weekend,
           CAST(COALESCE(p.impressions, 0) AS BIGINT)  AS impressions,
           (COALESCE(p.imp_clicks, 0) + 20 * g.rate)
             / (COALESCE(p.impressions, 0) + 20)       AS ctr_smooth
    FROM items i
    LEFT JOIN base b ON b.item_id = i.item_id
    LEFT JOIN imp p  ON p.item_id = i.item_id
    CROSS JOIN span s
    CROSS JOIN overall g
    """)
    print(name, con.execute(f"SELECT COUNT(*) FROM {name}").fetchone())

if __name__ == "__main__":
    build_item_features(1737936000, "item_feats_20250127")