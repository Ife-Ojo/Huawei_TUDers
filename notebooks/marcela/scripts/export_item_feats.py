from datetime import datetime, timezone
from pathlib import Path
import item_features as itf

con = itf.con
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)

JAN20, DAY = 1737331200, 86400          # 2025-01-20 00:00 UTC

for i in range(8):                      # Jan 20 ... Jan 27
    cutoff = JAN20 + i * DAY
    date = datetime.fromtimestamp(cutoff, timezone.utc).strftime("%Y%m%d")
    table = f"item_feats_{date}"
    itf.build_item_features(cutoff, table)
    cols = ", ".join(f"{c} AS itm_{c}" for c in itf.FEATS)
    path = OUT / f"item_feats_{date}.parquet"
    con.execute(f"COPY (SELECT item_id, {cols} FROM {table}) "
                f"TO '{path}' (FORMAT PARQUET)")
    print("wrote", path.name)
