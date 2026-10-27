"""Stage 2b: submission checker.   Usage:  python check_submission.py my_submission.csv"""
import sys
import duckdb, pandas as pd
from config import *

def check(path):
    id_col, pred_col = SUBMISSION_COLUMNS
    header = list(pd.read_csv(path, nrows=0).columns)
    if header != SUBMISSION_COLUMNS:
        return [f"Header is {header}, expected {SUBMISSION_COLUMNS}"]      # stop here: columns unusable
    errors = []
    sub = pd.read_csv(path)

    if len(sub) != N_EVAL_ROWS:
        errors.append(f"Row count {len(sub):,} != expected {N_EVAL_ROWS:,}")
    dups = int(sub[id_col].duplicated().sum())
    if dups:
        errors.append(f"{dups:,} duplicate ids in '{id_col}'")

    v = pd.to_numeric(sub[pred_col], errors="coerce")
    bad = int((v.isna() | ~v.abs().lt(float("inf"))).sum())
    if bad:
        errors.append(f"{bad:,} values in '{pred_col}' are empty / non-numeric / NaN / infinite")
    oob = int(((v < 0) | (v > 1)).sum())
    if oob:
        errors.append(f"{oob:,} values in '{pred_col}' are outside [0, 1]")

    want = set(duckdb.connect().execute(f"SELECT rowid FROM read_parquet('{EVAL_PQ.as_posix()}')").df()["rowid"])
    got = set(sub[id_col].astype(str))
    if want - got: errors.append(f"{len(want - got):,} eval rowids are missing from the submission")
    if got - want: errors.append(f"{len(got - want):,} ids in the submission are not eval rowids")
    return errors

if __name__ == "__main__":
    errs = check(sys.argv[1])
    if errs:
        print("SUBMISSION INVALID:")
        for e in errs: print("  -", e)
        sys.exit(1)
    print("Submission looks valid.")
