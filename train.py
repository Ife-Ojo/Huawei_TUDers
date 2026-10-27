"""
Eoin's training script: collects everyone's feature files, trains the model, prints the score.

  python train.py                        all feature files, 5% sample (quick check)
  python train.py full                   all feature files, all data (real score)
  python train.py full --groups base,marcela     only these feature files
  python train.py full --rounds 40       train for a fixed number of rounds (fair comparisons)
  python train.py full --seeds 3         repeat with 3 random seeds and average (shows the wobble)
  python train.py full --train-days 4    learn from 4 days before each test day instead of 1
  python train.py full --leaves 31 --lr 0.02     simpler trees / slower learning
  python train.py full --params best_params.json  use the settings tuning.py found
  python train.py full --ablation        does each feature file help? (score with vs without it)
  python train.py full --submit          also make a submission file in outputs/
  python train.py full --note "..."      a note saved in the log

To check whether YOUR clues help (paste the output into your pull request):
  python train.py full --groups base,<yours> --rounds 40 --seeds 3 --ablation

Why --rounds and --seeds: without them the model picks its stopping point by looking at
the test day, which flatters the score and makes it jump around a little. Small gains
(like +0.0003) can then be luck. Fixed rounds + several seeds make comparisons fair and
show how big the random wobble is.

To find the best training setup, run compare_setups.py.

Every run is added to experiments.csv (your own copy - don't commit it).
"""
import argparse
import datetime as dt
import getpass
import importlib
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, DAY, connect, labels, C1, C2, C3

LOG = ROOT / "experiments.csv"
OUT_DIR = ROOT / "outputs"

# Two tests, each copying the real setup: learn from the day(s) before, get tested on the next day.
TESTS = [C1, C2]                 # test 1 = Jan 25, test 2 = Jan 26
FIRST_DAY = C3 - 6 * DAY         # Jan 21: the earliest day that has an earlier day to learn clues from

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, seed=42, deterministic=True, force_row_wise=True, verbose=-1)


def day_name(ts):
    d = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    return f"{d:%b} {d.day}"


def learn_days(test_cut, n):
    """The n days just before the test day (never earlier than Jan 21)."""
    return [d for d in (test_cut - k * DAY for k in range(n, 0, -1)) if d >= FIRST_DAY]


def days_label(days):
    if len(days) == 1:
        return day_name(days[0])
    last = dt.datetime.fromtimestamp(days[-1], dt.timezone.utc)
    return f"{day_name(days[0])}-{last.day}"


def make_params(leaves=None, lr=None, base=None):
    p = dict(base or PARAMS)
    if leaves:
        p["num_leaves"] = leaves
    if lr:
        p["learning_rate"] = lr
    return p


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


class Tables:
    """Builds each day's table once and reuses it."""
    def __init__(self, con, groups):
        self.con, self.groups, self.cache = con, groups, {}

    def day(self, cut, source="train"):
        if (cut, source) not in self.cache:
            end = cut + DAY if source == "train" else None
            self.cache[(cut, source)] = make_table(self.con, self.groups, cut, end, source)
        return self.cache[(cut, source)]

    def learn(self, days):
        return pd.concat([self.day(d) for d in days], ignore_index=True)

    def tests(self, n_days):
        """[(learning table, test table)] for test 1 and test 2."""
        return [(self.learn(learn_days(cut, n_days)), self.day(cut)) for cut in TESTS]


def columns(df, groups):
    prefixes = tuple(module(g).PREFIX for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]


def seed_params(s, params=PARAMS):
    return {**params, "seed": 42 + s}


def score(tables, cols, rounds=None, seeds=1, params=PARAMS):
    """Train + test on both tests, once per seed.
    Returns auc[test, seed], the rounds used per test, and how much the model uses each clue."""
    auc = np.zeros((len(tables), seeds))
    used = [[] for _ in tables]
    imps = []
    for s in range(seeds):
        for i, (tr, va) in enumerate(tables):
            data = lgb.Dataset(tr[cols], tr.clicked)
            if rounds:   # fixed length: the test day is never used to choose anything
                m = lgb.train(seed_params(s, params), data, rounds)
                pred = m.predict(va[cols])
                used[i].append(rounds)
            else:        # early stopping: picks the best round BY LOOKING AT the test day
                m = lgb.train(seed_params(s, params), data, 3000,
                              valid_sets=[lgb.Dataset(va[cols], va.clicked)],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                pred = m.predict(va[cols], num_iteration=m.best_iteration)
                used[i].append(m.best_iteration)
            auc[i, s] = roc_auc_score(va.clicked, pred)
            imps.append(pd.Series(m.feature_importance("gain"), index=cols))
    return auc, used, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def log(row):
    """Add one line to experiments.csv (keeps older lines even if the columns changed)."""
    df = pd.DataFrame([{k: str(v) for k, v in row.items()}])
    if LOG.exists():
        df = pd.concat([pd.read_csv(LOG, dtype=str, keep_default_na=False), df], ignore_index=True)
    df.to_csv(LOG, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="sample", choices=["sample", "full"])
    ap.add_argument("--groups", help="comma-separated feature files, e.g. base,marcela (default: all)")
    ap.add_argument("--rounds", type=int, help="train a fixed number of rounds (no early stopping)")
    ap.add_argument("--seeds", type=int, default=1, help="repeat with this many random seeds and average")
    ap.add_argument("--train-days", type=int, help="days to learn from before each test day (1-5, default 1)")
    ap.add_argument("--params", help="settings file made by tuning.py, e.g. best_params.json")
    ap.add_argument("--leaves", type=int, help=f"num_leaves (default {PARAMS['num_leaves']})")
    ap.add_argument("--lr", type=float, help=f"learning rate (default {PARAMS['learning_rate']})")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    tuned = {}
    if a.params:                              # settings found by tuning.py (options typed here still win)
        with open(a.params) as fh:
            tuned = json.load(fh)
        a.rounds = a.rounds or tuned.get("rounds")
        a.train_days = a.train_days or tuned.get("train_days")
    a.train_days = a.train_days or 1
    if a.rounds is not None and a.rounds < 1:
        ap.error("--rounds must be 1 or more")
    if a.seeds < 1:
        ap.error("--seeds must be 1 or more")
    if a.train_days < 1:
        ap.error("--train-days must be 1 or more")
    params = make_params(a.leaves, a.lr, base={**PARAMS, **tuned.get("params", {})})

    groups = a.groups.split(",") if a.groups else all_groups()
    con = connect(full=(a.data == "full"))
    how = f"{a.rounds} rounds" if a.rounds else "early stopping"
    print(f"Data: {a.data}   Feature files: {', '.join(groups)}   Training: {how}, {a.seeds} seed(s), "
          f"num_leaves {params['num_leaves']}, learning rate {params['learning_rate']}"
          + (f", tuned settings from {a.params}" if a.params else ""))

    t0, store = time.time(), Tables(con, groups)
    tables = store.tests(a.train_days)
    names = [f"test {i + 1}: learn {days_label(learn_days(cut, a.train_days))}, test {day_name(cut)}"
             for i, cut in enumerate(TESTS)]
    if len(learn_days(TESTS[0], a.train_days)) < a.train_days:
        print(f"NOTE: test 1 only has {len(learn_days(TESTS[0], a.train_days))} earlier days to learn from (data starts Jan 20)")
    cols = columns(tables[0][0], groups)
    print(f"Clues: {len(cols)}   learning rows: {len(tables[0][0]):,} / {len(tables[1][0]):,}   built in {time.time() - t0:.0f}s")

    auc, used, imp = score(tables, cols, a.rounds, a.seeds, params)
    wobble = auc.std(axis=1, ddof=1) if a.seeds > 1 else None
    print()
    for i, name in enumerate(names):
        spread = f" ± {wobble[i]:.5f}" if wobble is not None else ""
        r = f"{a.rounds} rounds" if a.rounds else f"stopped at ~{int(np.mean(used[i]))} rounds"
        print(f"  {name}:  AUC {auc[i].mean():.5f}{spread}   ({r})")
    print(f"  AVERAGE AUC: {auc.mean():.5f}")
    if wobble is not None:
        print(f"  (± = how much the score wobbles between the {a.seeds} seeds)")
    print("\nTop 15 clues (how much the model uses each):")
    print(imp.head(15).round(0).to_string())

    log(dict(time=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), who=getpass.getuser(), data=a.data,
             groups="+".join(groups), clues=len(cols), rounds=a.rounds or "early-stop", seeds=a.seeds,
             train_days=a.train_days, leaves=params["num_leaves"], lr=params["learning_rate"],
             params_file=a.params or "",
             test1_auc=round(auc[0].mean(), 5), test2_auc=round(auc[1].mean(), 5),
             avg_auc=round(auc.mean(), 5),
             wobble=round(float(wobble.mean()), 5) if wobble is not None else "", note=a.note))
    print(f"\nSaved to {LOG.name}")

    if a.ablation:
        if len(groups) < 2:
            print("\n--ablation needs at least 2 feature files.")
        else:
            print("\nDoes each feature file help?  (same rounds and seeds for every run)")
            if not a.rounds or a.seeds < 2:
                print("  TIP: for a fair answer add --rounds 40 --seeds 3")
            if a.data != "full":
                print("  (sample scores are noisy - confirm with 'full' before deciding)")
            for g in groups:
                rest = [x for x in groups if x != g]
                auc_wo, _, _ = score(tables, columns(tables[0][0], rest), a.rounds, a.seeds, params)
                diff = auc - auc_wo                    # same test day + same seed, with vs without
                per_test = diff.mean(axis=1)
                gain = float(diff.mean())
                if a.seeds > 1:                        # ~95% band from the seed wobble
                    noise = max(2 * diff.mean(axis=0).std(ddof=1) / np.sqrt(a.seeds), 0.0001)
                else:
                    noise = 0.0005
                if gain > noise and (per_test > 0).all():
                    verdict = "HELPS"
                elif gain < -noise and (per_test < 0).all():
                    verdict = "hurts"
                else:
                    verdict = "no clear effect"
                print(f"  {g:<10} adds {gain:+.5f}  (test 1 {per_test[0]:+.5f}, test 2 {per_test[1]:+.5f}, "
                      f"noise ±{noise:.5f})  -> {verdict}")
            print("  HELPS = better on both test days, by more than the random wobble")

    if a.submit:
        n_rounds = a.rounds or max(int(np.mean(used) * 1.1), 50)
        final_days = learn_days(C3, a.train_days + 1)
        print(f"\nTraining the final model on {days_label(final_days)} ({n_rounds} rounds, {a.seeds} seed(s)) "
              "and predicting evaluation...")
        final_tr = store.learn(final_days)
        ev = store.day(C3, "eval")
        pred = np.zeros(len(ev))
        for s in range(a.seeds):               # several seeds = average their predictions
            model = lgb.train(seed_params(s, params), lgb.Dataset(final_tr[cols], final_tr.clicked), n_rounds)
            pred += model.predict(ev[cols]) / a.seeds
        sub = pd.DataFrame({"rowid": ev.rowid, "predicted_probability": pred})
        n_eval = con.sql("SELECT count(*) FROM eval").fetchall()[0][0]
        assert len(sub) == n_eval == 529_390, (len(sub), n_eval)
        assert sub.rowid.is_unique and sub.predicted_probability.between(0, 1).all()
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / (f"sub_{dt.date.today()}_{'+'.join(groups)}_{a.data}_d{a.train_days}"
                         f"_l{params['num_leaves']}_lr{params['learning_rate']}{'_tuned' if a.params else ''}"
                         f"_r{n_rounds}_s{a.seeds}_{auc.mean():.4f}.csv")
        sub.to_csv(out, index=False)
        print(f"Wrote outputs/{out.name}  ({len(sub):,} rows, checks passed)")
        if a.data != "full":
            print("NOTE: made from the sample - don't upload it. Use 'full' for real submissions.")


if __name__ == "__main__":
    main()
