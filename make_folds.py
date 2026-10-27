"""Stage 2a: validation folds.
For each cutoff, every user's FIRST event at/after the cutoff (and before the end of the training
data). We keep all 10 candidate rows of that event, with the true 'clicked' label.
This mirrors the real evaluation: one event per user, scored on its 10 candidates.
No history truncation is needed: it is the user's first event after the cutoff, so the history
stored in the row can only contain clicks from before the cutoff."""
from config import *
from db import connect, epoch_of, fold_name, q

if __name__ == "__main__":
    FOLDS.mkdir(parents=True, exist_ok=True)
    con = connect()
    end = epoch_of(con, DATA_END)
    for c in CUTOFFS:
        lo, name = epoch_of(con, c), fold_name(c)
        con.execute(f"DROP VIEW IF EXISTS {name}")      # allows re-running when the fold file already exists
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            WITH firsts AS (
                SELECT user_id, arg_min(event_id, (timestamp, event_id)) AS event_id
                FROM (SELECT DISTINCT user_id, event_id, timestamp FROM train
                      WHERE timestamp >= {lo} AND timestamp < {end})
                GROUP BY user_id)
            SELECT t.* EXCLUDE (ts_utc) FROM train t JOIN firsts f ON t.event_id = f.event_id""")
        rows, events, users, clk, t0, t1 = con.execute(f"""
            SELECT count(*), count(DISTINCT event_id), count(DISTINCT user_id), sum(clicked),
                   make_timestamp(min(timestamp) * 1000000), make_timestamp(max(timestamp) * 1000000) FROM {name}""").fetchone()
        assert events == users, "must be exactly one event per user"
        assert rows == 10 * events, "every event must keep all 10 candidate rows"
        print(f"{name}: {users:,} users/events, {rows:,} rows, {clk / events:.1%} of events have a click, {t0} -> {t1}")
        con.execute(f"COPY {name} TO '{q(FOLDS / (name + '.parquet'))}' (FORMAT parquet, COMPRESSION zstd)")
    print("Saved in", FOLDS)
