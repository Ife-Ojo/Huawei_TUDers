"""
Which TRAINING SETUP works best? Same clues, different ways of training the model.

Tries every combination of:
  - days to learn from before each test day   (1 = today's setup, 4 = Jan 21-24 / Jan 22-25)
  - num_leaves (how complicated each tree may be)
  - learning rate (how big a step each round takes)

  python compare_setups.py                         quick check on the 5% sample
  python compare_setups.py full                    all data, base clues (takes a while)
  python compare_setups.py full --groups base,marcela --seeds 2

Each setup stops when its test score stops improving, so the table also shows how many
rounds each setup needs - use that number for --rounds afterwards.
The table is printed best first and saved in outputs/.
"""
import argparse
import datetime as dt
import itertools
import time

import numpy as np
import pandas as pd

import train
from common import connect

GRID = dict(train_days=[1, 4], leaves=[15, 31, 63], lr=[0.05, 0.02])
CURRENT = dict(train_days=1, leaves=63, lr=0.05)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="sample", choices=["sample", "full"])
    ap.add_argument("--groups", help="comma-separated feature files (default: base)")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args(argv)

    groups = a.groups.split(",") if a.groups else ["base"]
    con = connect(full=(a.data == "full"))
    store = train.Tables(con, groups)
    combos = list(itertools.product(*GRID.values()))
    print(f"Data: {a.data}   Feature files: {', '.join(groups)}   {len(combos)} setups x {a.seeds} seed(s)\n")

    rows, t_all = [], time.time()
    for i, (days, leaves, lr) in enumerate(combos, 1):
        t0 = time.time()
        tables = store.tests(days)
        cols = train.columns(tables[0][0], groups)
        auc, used, _ = train.score(tables, cols, None, a.seeds, train.make_params(leaves, lr))
        rows.append(dict(train_days=days, num_leaves=leaves, learning_rate=lr,
                         test1_auc=auc[0].mean(), test2_auc=auc[1].mean(), avg_auc=auc.mean(),
                         rounds_needed=int(np.mean(used))))
        print(f"  [{i:>2}/{len(combos)}] days {days}, leaves {leaves:>2}, lr {lr:<4}  ->  "
              f"avg AUC {auc.mean():.5f}  ({int(np.mean(used))} rounds, {time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("avg_auc", ascending=False).reset_index(drop=True)
    now = df[(df.train_days == CURRENT["train_days"]) & (df.num_leaves == CURRENT["leaves"])
             & (df.learning_rate == CURRENT["lr"])].iloc[0]
    df["vs_current"] = df.avg_auc - now.avg_auc
    print(f"\nAll setups, best first  (finished in {(time.time() - t_all) / 60:.1f} min):\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    best = df.iloc[0]
    print(f"\nCurrent setup (1 day, 63 leaves, lr 0.05): {now.avg_auc:.5f}")
    print(f"Best setup ({int(best.train_days)} days, {int(best.num_leaves)} leaves, lr {best.learning_rate}): "
          f"{best.avg_auc:.5f}  ({best.vs_current:+.5f})")
    if best.test1_auc > now.test1_auc and best.test2_auc > now.test2_auc:
        print("It is better on BOTH test days.")
    else:
        print("It is NOT better on both test days - treat the gain with care.")
    print(f"\nTry it with fixed rounds:\n  python train.py {a.data} --train-days {int(best.train_days)} "
          f"--leaves {int(best.num_leaves)} --lr {best.learning_rate} --rounds {int(best.rounds_needed)} --seeds 3")

    train.OUT_DIR.mkdir(exist_ok=True)
    out = train.OUT_DIR / f"compare_setups_{dt.date.today()}_{'+'.join(groups)}_{a.data}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved the table to outputs/{out.name}")


if __name__ == "__main__":
    main()
