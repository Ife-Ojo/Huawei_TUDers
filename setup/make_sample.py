"""
Makes a small 5% copy of the training data so you can test quickly.
Run from the HuaweiTud folder:  python setup/make_sample.py
"""
import os
import duckdb

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the HuaweiTud folder
DATA = os.path.join(PROJECT, "Data")
TRAIN = os.path.join(DATA, "train.csv")
EVAL = os.path.join(DATA, "evaluation.csv")
DB = os.path.join(DATA, "sample.duckdb")

for f in (TRAIN, EVAL):
    if not os.path.exists(f):
        raise SystemExit(f"Can't find {f} - check the file name and folder.")

con = duckdb.connect(DB)
con.execute("SET memory_limit='8GB'")

types = """{'rowid':'VARCHAR','event_id':'VARCHAR','user_id':'VARCHAR','item_id':'VARCHAR',
 'timestamp':'BIGINT','time_bucket':'TINYINT','day_type':'TINYINT',
 'history_item_ids':'VARCHAR','history_timestamps':'VARCHAR','clicked':'TINYINT'}"""
ev_types = types.replace(",'clicked':'TINYINT'", "")

print("Reading train.csv and keeping ~1 in 20 events (takes a few minutes)...")
con.execute(f"""CREATE OR REPLACE TABLE train AS
    SELECT * FROM read_csv('{TRAIN}', header=true, columns={types})
    WHERE hash(event_id) % 20 = 0""")

print("Reading evaluation.csv (all of it)...")
con.execute(f"""CREATE OR REPLACE TABLE eval AS
    SELECT * FROM read_csv('{EVAL}', header=true, columns={ev_types})""")

print(con.sql("SELECT count(*) AS train_rows, count(DISTINCT event_id) AS events FROM train"))
print(con.sql("SELECT count(*) AS eval_rows FROM eval"))
print(f"Saved to {DB}")