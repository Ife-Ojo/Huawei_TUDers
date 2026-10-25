import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval.parquet"
con = duckdb.connect(str(ROOT / "data" / "marcela.duckdb"))

# 1. Bucket shares should sum to about 1
print(con.execute("""
SELECT AVG(share_b0+share_b1+share_b2+share_b3) FROM item_feats_jan25
""").fetchone())

# 2. Popularity should be uneven
print(con.execute("""
SELECT MAX(clicks_total), MEDIAN(clicks_total), AVG(clicks_total)
FROM item_feats_jan25
""").fetchone())

# 3. Share of evaluation items with no clicks before the cutoff
print(con.execute(f"""
SELECT AVG(CASE WHEN f.item_id IS NULL THEN 1.0 ELSE 0 END) AS share_new
FROM (SELECT DISTINCT item_id FROM '{EVAL}') t
LEFT JOIN item_feats_eval f USING (item_id)
""").fetchone())