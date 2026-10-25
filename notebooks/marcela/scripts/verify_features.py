import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect()
for f in sorted((ROOT / "data" / "features").glob("item_feats_*.parquet")):
    n, d = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT item_id) FROM '{f}'").fetchone()
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{f}'").fetchall()]
    bad = [c for c in cols if c != "item_id" and not c.startswith("itm_")]
    print(f.name, "rows", n, "| one row per item:", n == d, "| bad column names:", bad)