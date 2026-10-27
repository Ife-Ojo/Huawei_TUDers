"""
Eoin's tuning script: tries many model settings automatically and keeps the best.

Same clues, same fair tests as train.py - only the model's settings change:
  num_leaves         how complicated each tree may be
  learning_rate      how big a step each round takes
  min_data_in_leaf   how many examples a rule needs before the model trusts it
  feature_fraction   share of clues each tree looks at
  bagging_fraction   share of rows each tree looks at
  lambda_l2          a brake on extreme values (higher = more careful)

  python tuning.py                               quick check on the 5% sample
  python tuning.py full                          all data, all feature files, 30 tries
  python tuning.py full --groups base,elvita --trials 40 --minutes 30

How it works:
  1. The first tries are random; later tries make small changes to the best one so far.
     Each try stops when its test score stops improving (that also gives the rounds it needs).
  2. The best 3 are checked again FAIRLY: fixed rounds and 3 seeds, compared with the
     current setup (4 days, 15 leaves, lr 0.02). A setting only wins if it is better on
     BOTH test days - otherwise you keep the current setup.
  3. The winner is saved to best_params.json. Use it with:
        python train.py full --params best_params.json --seeds 3 --ablation
        python train.py full --params best_params.json --seeds 3 --submit

Run it AFTER the clue files are merged: the best settings change when clues change.
"""
import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd

import train
from common import ROOT, connect

# name: (kind, low, high)   log = searched on a log scale
SPACE = {
    "num_leaves":       ("int_log", 7, 127),
    "learning_rate":    ("log", 0.01, 0.1),
    "min_data_in_leaf": ("int_log", 20, 2000),
    "feature_fraction": ("lin", 0.5, 1.0),
    "bagging_fraction": ("lin", 0.5, 1.0),
    "lambda_l2":        ("log", 0.01, 20.0),
}
CURRENT = dict(num_leaves=15, learning_rate=0.02)   # best setup found by compare_setups.py
BEST_FILE = ROOT / "best_params.json"


def random_setting(rng):
    out = {}
    for k, (kind, lo, hi) in SPACE.items():
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind in ("log", "int_log") else rng.uniform(lo, hi)
        out[k] = int(round(v)) if kind == "int_log" else round(v, 4)
    return out


def nudge(best, rng, size=0.35):
    """A small change to the best setting so far."""
    out = {}
    for k, (kind, lo, hi) in SPACE.items():
        v = best[k]
        v = v * float(np.exp(rng.normal(0, size))) if kind in ("log", "int_log") else v + rng.normal(0, size * (hi - lo) / 2)
        v = min(max(v, lo), hi)
        out[k] = int(round(v)) if kind == "int_log" else round(v, 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="sample", choices=["sample", "full"])
    ap.add_argument("--groups", help="comma-separated feature files (default: all)")
    ap.add_argument("--trials", type=int, default=30, help="how many settings to try (default 30)")
    ap.add_argument("--minutes", type=float, default=60, help="stop starting new tries after this many minutes")
    ap.add_argument("--train-days", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1, help="random seed for the search itself")
    a = ap.parse_args(argv)

    groups = a.groups.split(",") if a.groups else train.all_groups()
    con = connect(full=(a.data == "full"))
    t_all = time.time()
    store = train.Tables(con, groups)
    tables = store.tests(a.train_days)
    cols = train.columns(tables[0][0], groups)
    print(f"Data: {a.data}   Feature files: {', '.join(groups)}   Clues: {len(cols)}   "
          f"Learning from {a.train_days} day(s)   Up to {a.trials} tries / {a.minutes:g} min\n")

    rng = np.random.default_rng(a.seed)
    n_random = max(5, int(a.trials * 0.4))
    rows = []

    def run(setting):
        auc, used, _ = train.score(tables, cols, None, 1, train.make_params(base={**train.PARAMS, **setting}))
        rows.append(dict(**setting, test1_auc=auc[0].mean(), test2_auc=auc[1].mean(), avg_auc=auc.mean(),
                         rounds_needed=int(round(np.mean(used)))))
        return rows[-1]

    current = {**{k: train.PARAMS[k] for k in SPACE if k in train.PARAMS}, "lambda_l2": 0.0, **CURRENT}
    print("Search (each try stops when its test score stops improving):")
    r = run(current)
    print(f"  [ 0] current setup                              avg AUC {r['avg_auc']:.5f}  ({r['rounds_needed']} rounds)")
    for i in range(1, a.trials + 1):
        if (time.time() - t_all) / 60 > a.minutes:
            print(f"  time limit reached after {i - 1} tries")
            break
        best_so_far = max(rows, key=lambda x: x["avg_auc"])
        setting = random_setting(rng) if i <= n_random else nudge({k: best_so_far[k] for k in SPACE}, rng)
        t0 = time.time()
        r = run(setting)
        mark = "  <- best so far" if r["avg_auc"] >= max(x["avg_auc"] for x in rows) else ""
        print(f"  [{i:>2}] leaves {setting['num_leaves']:>3}, lr {setting['learning_rate']:.3f}, "
              f"min_data {setting['min_data_in_leaf']:>4}   avg AUC {r['avg_auc']:.5f}  "
              f"({r['rounds_needed']} rounds, {time.time() - t0:.0f}s){mark}")

    table = pd.DataFrame(rows).sort_values("avg_auc", ascending=False).reset_index(drop=True)

    # --- fair check of the best 3 vs the current setup: fixed rounds + 3 seeds
    print("\nFair check of the best 3 (fixed rounds, 3 seeds) against the current setup:")
    def fair(row):
        setting = {k: (int(row[k]) if SPACE[k][0] == "int_log" else float(row[k])) for k in SPACE}
        auc, _, _ = train.score(tables, cols, int(row["rounds_needed"]), 3,
                                train.make_params(base={**train.PARAMS, **setting}))
        return setting, int(row["rounds_needed"]), auc
    cur_row = table[(table.num_leaves == current["num_leaves"]) & (table.learning_rate == current["learning_rate"])
                    & (table.lambda_l2 == 0.0)].iloc[0]
    _, cur_rounds, cur_auc = fair(cur_row)
    print(f"  current setup:  test1 {cur_auc[0].mean():.5f}  test2 {cur_auc[1].mean():.5f}  avg {cur_auc.mean():.5f}")
    winner = None
    for _, row in table.head(3).iterrows():
        if row.num_leaves == current["num_leaves"] and row.learning_rate == current["learning_rate"] and row.lambda_l2 == 0:
            continue
        setting, rounds, auc = fair(row)
        both = auc[0].mean() > cur_auc[0].mean() and auc[1].mean() > cur_auc[1].mean()
        print(f"  leaves {setting['num_leaves']:>3}, lr {setting['learning_rate']:.3f}, {rounds} rounds:  "
              f"test1 {auc[0].mean():.5f}  test2 {auc[1].mean():.5f}  avg {auc.mean():.5f}  "
              f"({auc.mean() - cur_auc.mean():+.5f})  {'better on both days' if both else 'not better on both days'}")
        if both and (winner is None or auc.mean() > winner[2].mean()):
            winner = (setting, rounds, auc)

    train.OUT_DIR.mkdir(exist_ok=True)
    out = train.OUT_DIR / f"tuning_{dt.date.today()}_{'+'.join(groups)}_{a.data}.csv"
    table.to_csv(out, index=False)
    print(f"\nAll tries saved to outputs/{out.name}  (finished in {(time.time() - t_all) / 60:.1f} min)")

    if winner is None:
        print("\nRESULT: nothing beat the current setup on both test days - keep using "
              "--train-days 4 --leaves 15 --lr 0.02. (best_params.json not changed)")
        return
    setting, rounds, auc = winner
    BEST_FILE.write_text(json.dumps(dict(
        params=setting, rounds=rounds, train_days=a.train_days,
        info=dict(made=str(dt.datetime.now())[:16], data=a.data, groups="+".join(groups),
                  test1_auc=round(float(auc[0].mean()), 5), test2_auc=round(float(auc[1].mean()), 5),
                  avg_auc=round(float(auc.mean()), 5), current_avg_auc=round(float(cur_auc.mean()), 5))), indent=2))
    print(f"\nRESULT: new best settings, avg AUC {auc.mean():.5f} vs current {cur_auc.mean():.5f} "
          f"({auc.mean() - cur_auc.mean():+.5f}), better on both test days.")
    print(f"Saved to {BEST_FILE.name}. Use them with:\n"
          f"  python train.py {a.data} --params best_params.json --seeds 3 --ablation\n"
          f"  python train.py {a.data} --params best_params.json --seeds 3 --submit")
    if a.data != "full":
        print("NOTE: tuned on the sample - run 'python tuning.py full' for settings to actually use.")


if __name__ == "__main__":
    main()
