"""
Eoin's training script: collects everyone's feature files, trains the model, prints the score.

  python train.py                        all feature files, 5% sample (quick check)
  python train.py full                   all feature files, all data (real score)
  python train.py full --groups base,user        only these feature files
  python train.py full --ablation        does each feature file help? (score with vs without it)
  python train.py full --submit          also make a submission file in outputs/
  python train.py full --note "added recency"    a note saved in the log

Every run is added to experiments.csv, so the team can see what helped.
"""
import argparse
import datetime as dt
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, connect, labels, C0, C1, C2, C3

LOG = ROOT / "experiments.csv"
OUT_DIR = ROOT / "outputs"

# Two tests, each copying the real setup: learn from one day, get tested on the next day.
FOLDS = [
    ("test 1: learn Jan 24, test Jan 25", C0, C1, C2),
    ("test 2: learn Jan 25, test Jan 26", C1, C2, C3),
]

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, seed=42, deterministic=True, force_row_wise=True, verbose=-1)


def all_groups():
    """Every feature file in features/ (files starting with _ are skipped)."""
    return sorted(p.stem for p in (ROOT / "features").glob("*.py") if not p.stem.startswith("_"))


def module(group):
    return importlib.import_module(f"features.{group}")


def make_table(con, groups, cut, end=None, source="train"):
    """Rows to predict + every group's clues, joined on rowid."""
    df = labels(con, cut, end, source)
    for g in groups:
        f = module(g).build(con, cut, end, source)
        assert f.rowid.is_unique, f"features/{g}.py returned duplicate rowids"
        n = len(df)
        df = df.merge(f, on="rowid", how="left")
        assert len(df) == n, f"features/{g}.py changed the number of rows"
    return df


def columns(df, groups):
    prefixes = tuple(module(g).PREFIX for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]


def score(tables, cols):
    """Train + test on both folds. Returns the AUC of each test, best rounds, and clue importance."""
    aucs, rounds, imps = [], [], []
    for (tr, va) in tables:
        m = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr.clicked), 2000,
                      valid_sets=[lgb.Dataset(va[cols], va.clicked)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(roc_auc_score(va.clicked, m.predict(va[cols], num_iteration=m.best_iteration)))
        rounds.append(m.best_iteration)
        imps.append(pd.Series(m.feature_importance("gain"), index=cols))
    return aucs, rounds, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def log(row):
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="sample", choices=["sample", "full"])
    ap.add_argument("--groups", help="comma-separated feature files, e.g. base,user (default: all)")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    groups = a.groups.split(",") if a.groups else all_groups()
    con = connect(full=(a.data == "full"))
    print(f"Data: {a.data}   Feature files: {', '.join(groups)}")

    # build each day's table once (test 2 learns from the day test 1 is tested on)
    t0, cache = time.time(), {}
    def table(cut, end):
        if (cut, end) not in cache:
            cache[(cut, end)] = make_table(con, groups, cut, end)
        return cache[(cut, end)]
    tables = [(table(a_, b_), table(b_, c_)) for _, a_, b_, c_ in FOLDS]
    cols = columns(tables[0][0], groups)
    print(f"Clues: {len(cols)}   built in {time.time() - t0:.0f}s")

    aucs, rounds, imp = score(tables, cols)
    mean_auc = float(np.mean(aucs))
    print()
    for (name, *_), auc, r in zip(FOLDS, aucs, rounds):
        print(f"  {name}:  AUC {auc:.5f}   ({r} rounds)")
    print(f"  AVERAGE AUC: {mean_auc:.5f}")
    print("\nTop 15 clues (how much the model uses each):")
    print(imp.head(15).round(0).to_string())

    log(dict(time=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), who="eoin", data=a.data,
             groups="+".join(groups), clues=len(cols), test1_auc=round(aucs[0], 5),
             test2_auc=round(aucs[1], 5), avg_auc=round(mean_auc, 5), note=a.note))
    print(f"\nSaved to {LOG.name}")

    if a.ablation:
        if len(groups) < 2:
            print("\n--ablation needs at least 2 feature files (only one so far).")
        else:
            print("\nDoes each feature file help?  (average AUC without it, and the change)")
            if a.data != "full":
                print("  (sample scores are noisy - confirm with 'full' before deciding)")
            for g in groups:
                rest = [x for x in groups if x != g]
                aucs_wo, _, _ = score(tables, columns(tables[0][0], rest))
                gain = mean_auc - float(np.mean(aucs_wo))
                verdict = "HELPS" if gain > 0.0005 else ("hurts" if gain < -0.0005 else "no clear effect")
                print(f"  {g:<10} without it: {np.mean(aucs_wo):.5f}   it adds {gain:+.5f}   -> {verdict}")

    if a.submit:
        print("\nTraining the final model on Jan 25 + Jan 26 and predicting evaluation...")
        final_tr = pd.concat([table(C1, C2), table(C2, C3)], ignore_index=True)
        model = lgb.train(PARAMS, lgb.Dataset(final_tr[cols], final_tr.clicked), max(int(np.mean(rounds) * 1.1), 50))
        ev = make_table(con, groups, C3, None, "eval")
        sub = pd.DataFrame({"rowid": ev.rowid, "predicted_probability": model.predict(ev[cols])})
        n_eval = con.sql("SELECT count(*) FROM eval").fetchall()[0][0]
        assert len(sub) == n_eval == 529_390, (len(sub), n_eval)
        assert sub.rowid.is_unique and sub.predicted_probability.between(0, 1).all()
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"sub_{dt.date.today()}_{'+'.join(groups)}_{a.data}_{mean_auc:.4f}.csv"
        sub.to_csv(out, index=False)
        print(f"Wrote outputs/{out.name}  ({len(sub):,} rows, checks passed)")
        if a.data != "full":
            print("NOTE: made from the 5% sample - don't upload it. Use 'full' for real submissions.")


if __name__ == "__main__":
    main()
