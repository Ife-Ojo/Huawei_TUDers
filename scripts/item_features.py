import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data" / "marcela.duckdb"))

def build_item_features(cutoff: int, name: str):
    con.execute(f"""
    CREATE OR REPLACE TABLE {name} AS
    WITH c AS (
        SELECT * FROM clicks WHERE click_ts < {cutoff}
    ),
    span AS (
        SELECT (({cutoff} - MIN(click_ts)) / 86400.0) AS days FROM c
    ),
    base AS (
        SELECT item_id,
               COUNT(*)                                          AS clicks_total,
               COUNT(DISTINCT user_id)                           AS users_total,
               SUM(CASE WHEN click_ts >= {cutoff} - 6*3600  THEN 1 ELSE 0 END) AS clicks_6h,
               SUM(CASE WHEN click_ts >= {cutoff} - 86400   THEN 1 ELSE 0 END) AS clicks_24h,
               MIN(click_ts)                                     AS first_seen,
               MAX(click_ts)                                     AS last_click,
               -- share of this item's clicks per time bucket and weekend
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 0 THEN 1.0 ELSE 0 END) AS share_b0,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 1 THEN 1.0 ELSE 0 END) AS share_b1,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 2 THEN 1.0 ELSE 0 END) AS share_b2,
               AVG(CASE WHEN (click_ts % 86400) // 21600 = 3 THEN 1.0 ELSE 0 END) AS share_b3,
               AVG(CASE WHEN ((click_ts // 86400) + 3) % 7 >= 5 THEN 1.0 ELSE 0 END) AS share_weekend
        FROM c GROUP BY item_id
    )
    SELECT b.*,
           b.clicks_total / (SELECT days FROM span)              AS clicks_per_day,
           b.clicks_24h / NULLIF(b.clicks_total / (SELECT days FROM span), 0) AS trend_24h,
           ({cutoff} - b.first_seen) / 86400.0                   AS item_age_days,
           ({cutoff} - b.last_click) / 3600.0                    AS hours_since_last_click
    FROM base b
    """)
    print(name, con.execute(f"SELECT COUNT(*) FROM {name}").fetchone())

build_item_features(1737763200, "item_feats_jan25")
build_item_features(1737849600, "item_feats_jan26")
build_item_features(1737936000, "item_feats_eval")