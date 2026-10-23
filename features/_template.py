"""
TEMPLATE - copy this file to features/<yourname>.py, change PREFIX, write your clues.

Rules:
  - build(con, cut, end=None, source="train") returns `rowid` + your feature columns
  - every column name starts with your PREFIX   (Ife u_, Marcela i_, Abraham cv_, Elvita emb_)
  - learn ONLY from `fit` (training rows before cut). Never use `clicked` from the target rows.

Test it:  python features/<yourname>.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_sql, targets_sql, check_features

PREFIX = "xx_"


def build(con, cut, end=None, source="train"):
    return con.sql(f"""
    WITH fit AS {fit_sql(cut)},                  -- rows you may learn from
    t AS ({targets_sql(cut, end, source)})       -- rows to make clues for (h = history items, hts = times)
    SELECT t.rowid,
           len(t.h) AS xx_example_hist_len       -- replace with your clues
    FROM t
    """).df()


if __name__ == "__main__":
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(df.head())
