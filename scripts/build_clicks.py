import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.parquet"
EVAL = ROOT / "data" / "eval.parquet"

con = duckdb.connect(str(ROOT / "data" / "marcela.duckdb"))  # saves your tables

# 1. One row per event, history parsed into real lists
con.execute(f"""
CREATE OR REPLACE TABLE events AS
SELECT DISTINCT
    event_id, user_id, timestamp,
    from_json(history_item_ids, '["VARCHAR"]')   AS hist_items,
    from_json(history_timestamps, '["BIGINT"]')  AS hist_ts
FROM '{TRAIN}'
""")
print(con.execute("SELECT COUNT(*) FROM events").fetchone())  # expect 686,824

# 2. Unique clicks: unnest items and timestamps together, then deduplicate
con.execute("""
CREATE OR REPLACE TABLE clicks AS
SELECT DISTINCT user_id, item AS item_id, click_ts
FROM (
    SELECT user_id,
           unnest(hist_items) AS item,
           unnest(hist_ts)    AS click_ts
    FROM events
)
""")
print(con.execute("SELECT COUNT(*) FROM clicks").fetchone())
print(con.execute("SELECT * FROM clicks LIMIT 5").df())