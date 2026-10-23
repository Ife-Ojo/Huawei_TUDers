"""
Shared settings for the whole team. Everyone imports this file; nobody copies it.

  - where the databases are (works on any laptop, no /Users/... paths)
  - the cutoff dates
  - which rows we predict (each user's FIRST event after a cutoff, like the real evaluation)
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent
SAMPLE_DB = ROOT / "Data" / "sample.duckdb"      # 5% copy, for quick tests
FULL_DB = ROOT / "Data" / "techarena.duckdb"     # all the data, for real scores

DAY = 86400
C3 = 1737936000          # 2025-01-27 00:00 UTC  official cutoff: evaluation starts here
C2 = C3 - DAY            # 2025-01-26            validation day
C1 = C3 - 2 * DAY        # 2025-01-25            training day
C0 = C3 - 3 * DAY        # 2025-01-24            extra training day (second test fold)


def connect(full=False):
    """Open the sample database (default) or the full one (full=True), read-only."""
    db = FULL_DB if full else SAMPLE_DB
    if not db.exists():
        raise SystemExit(f"Can't find {db.name}. Build it first: python setup/make_sample.py  or  python setup/make_full.py")
    con = duckdb.connect(str(db), read_only=True)
    con.execute("SET memory_limit='8GB'")
    return con


def fit_sql(cut):
    """The rows a feature may LEARN from: training rows strictly before the cutoff."""
    return f"(SELECT * FROM train WHERE timestamp < {cut})"


def targets_sql(cut, end=None, source="train"):
    """The rows we PREDICT: all 10 candidates of each user's first event in [cut, end).
    Adds the parsed history as lists: h = clicked item ids, hts = their times."""
    end_filter = f"AND timestamp < {end}" if end is not None else ""
    return f"""
        SELECT *, from_json(history_item_ids, '["VARCHAR"]') AS h,
                  from_json(history_timestamps, '["BIGINT"]') AS hts
        FROM {source}
        WHERE event_id IN (
            SELECT event_id FROM {source}
            WHERE timestamp >= {cut} {end_filter}
            QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY timestamp, event_id) = 1)
    """


def labels(con, cut, end=None, source="train"):
    """rowid, event_id and clicked (empty for evaluation) for the rows we predict."""
    lab = "clicked" if source == "train" else "NULL::TINYINT AS clicked"
    return con.sql(f"SELECT rowid, event_id, {lab} FROM ({targets_sql(cut, end, source)})").df()


def check_features(df, prefix):
    """Checks a feature table follows the team rules. Run it before sharing your file."""
    assert "rowid" in df.columns, "missing rowid column"
    assert df.rowid.is_unique, "duplicate rowids"
    bad = [c for c in df.columns if c != "rowid" and not c.startswith(prefix)]
    assert not bad, f"these columns don't start with '{prefix}': {bad}"
    print(f"OK: {len(df):,} rows, {df.shape[1] - 1} features, all start with '{prefix}'")
