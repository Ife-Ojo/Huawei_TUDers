"""Shared helper: an in-memory DuckDB with views over the shared Parquet files.
Read-only, so the whole team can use the same files at the same time."""
import duckdb
from config import *

def q(p): return p.as_posix()

def fold_name(cutoff): return "fold_" + cutoff.replace("-", "_")

def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=2; SET temp_directory='duckdb_tmp';")
    for name, pq in [("train", TRAIN_PQ), ("evalset", EVAL_PQ)]:
        con.execute(f"""CREATE VIEW {name} AS
            SELECT *, make_timestamp(timestamp * 1000000) AS ts_utc FROM read_parquet('{q(pq)}')""")
    for c in CUTOFFS:
        f = FOLDS / f"{fold_name(c)}.parquet"
        if f.exists():
            con.execute(f"CREATE VIEW {fold_name(c)} AS SELECT * FROM read_parquet('{q(f)}')")
    return con

def epoch_of(con, date_str):
    """Midnight UTC of a date, as epoch seconds (same unit as the 'timestamp' column)."""
    return int(con.execute(f"SELECT epoch(TIMESTAMP '{date_str}')::BIGINT").fetchone()[0])
