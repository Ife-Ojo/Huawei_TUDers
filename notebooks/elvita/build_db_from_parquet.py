"""
Build the DuckDB databases from the parquet files in ~/Downloads/.
Run from anywhere:  python notebooks/elvita/build_db_from_parquet.py

This creates:
  Data/techarena.duckdb   (full database - used by train.py full)
  Data/sample.duckdb      (5% sample  - used by train.py for quick tests)

These files stay on your laptop (they're in .gitignore).
"""
import os
import sys
from pathlib import Path

# Make sure we can import common.py
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import duckdb

DOWNLOADS = Path.home() / "Downloads"
TRAIN_PQ = DOWNLOADS / "train.parquet"
EVAL_PQ = DOWNLOADS / "eval.parquet"
DATA = PROJECT / "Data"
DATA.mkdir(exist_ok=True)

for f in (TRAIN_PQ, EVAL_PQ):
    if not f.exists():
        raise SystemExit(f"Can't find {f}")

# --- Full database ---
full_db = DATA / "techarena.duckdb"
print(f"Building full database: {full_db.name} ...")
if full_db.exists():
    full_db.unlink()

con = duckdb.connect(str(full_db))
con.execute("SET memory_limit='8GB'")

print("  Loading train.parquet ...")
con.execute(f"""
    CREATE TABLE train AS
    SELECT rowid::VARCHAR AS rowid, event_id::VARCHAR AS event_id,
           user_id::VARCHAR AS user_id, item_id::VARCHAR AS item_id,
           timestamp::BIGINT AS timestamp, time_bucket::TINYINT AS time_bucket,
           day_type::TINYINT AS day_type,
           history_item_ids::VARCHAR AS history_item_ids,
           history_timestamps::VARCHAR AS history_timestamps,
           clicked::TINYINT AS clicked
    FROM read_parquet('{TRAIN_PQ}')
""")
train_count = con.sql("SELECT count(*) FROM train").fetchall()[0][0]
print(f"  train: {train_count:,} rows")

print("  Loading eval.parquet ...")
con.execute(f"""
    CREATE TABLE eval AS
    SELECT rowid::VARCHAR AS rowid, event_id::VARCHAR AS event_id,
           user_id::VARCHAR AS user_id, item_id::VARCHAR AS item_id,
           timestamp::BIGINT AS timestamp, time_bucket::TINYINT AS time_bucket,
           day_type::TINYINT AS day_type,
           history_item_ids::VARCHAR AS history_item_ids,
           history_timestamps::VARCHAR AS history_timestamps
    FROM read_parquet('{EVAL_PQ}')
""")
eval_count = con.sql("SELECT count(*) FROM eval").fetchall()[0][0]
print(f"  eval:  {eval_count:,} rows")
con.close()
print(f"  Done: {full_db.name} ({full_db.stat().st_size / 1e9:.1f} GB)\n")

# --- Sample database (5% of training events) ---
sample_db = DATA / "sample.duckdb"
print(f"Building sample database: {sample_db.name} ...")
if sample_db.exists():
    sample_db.unlink()

con = duckdb.connect(str(sample_db))
con.execute("SET memory_limit='8GB'")
con.execute(f"""
    CREATE TABLE train AS
    SELECT * FROM read_parquet('{TRAIN_PQ}')
    WHERE hash(event_id) % 20 = 0
""")
con.execute(f"""
    CREATE TABLE eval AS
    SELECT * FROM read_parquet('{EVAL_PQ}')
""")
sample_count = con.sql("SELECT count(*) FROM train").fetchall()[0][0]
print(f"  train sample: {sample_count:,} rows (~5%)")
print(f"  eval: {eval_count:,} rows (all)")
con.close()
print(f"  Done: {sample_db.name}\n")

print("All done! You can now run:")
print("  python train.py          (quick test on 5% sample)")
print("  python train.py full     (real score on all data)")
