"""Writes submission_dummy.csv (every row = the average click rate). Only for testing the checker."""
import duckdb
con = duckdb.connect()
from config import *
id_col, pred_col = SUBMISSION_COLUMNS
rate = con.execute(f"SELECT avg(clicked) FROM read_parquet('{TRAIN_PQ.as_posix()}')").fetchone()[0]
con.execute(f"""COPY (SELECT rowid AS {id_col}, {rate:.5f} AS {pred_col}
                     FROM read_parquet('{EVAL_PQ.as_posix()}'))
               TO 'submission_dummy.csv' (HEADER, DELIMITER ',')""")
print("Wrote submission_dummy.csv")
