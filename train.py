"""
Eoin's training script: collects everyone's feature files, trains the model, prints the score.

  python train.py                        all feature files, 5% sample (quick check)
  python train.py full                   all feature files, all data (real score)
  python train.py full --groups base,marcela     only these feature files
  python train.py full --rounds 40       train for a fixed number of rounds (fair comparisons)
  python train.py full --seeds 3         repeat with 3 random seeds and average (shows the wobble)
  python train.py full --ablation        does each feature file help? (score with vs without it)
  python train.py full --submit          also make a submission file in outputs/
  python train.py full --note "..."      a note saved in the log

To check whether YOUR clues help (paste the output into your pull request):
  python train.py full --groups base,<yours> --rounds 40 --seeds 3 --ablation

Why --rounds and --seeds: without them the model picks its stopping point by looking at
the test day, which flatters the score and makes it jump around a little. Small gains
(like +0.0003) can then be luck. Fixed rounds + several seeds make comparisons fair and
show how big the random wobble is.

Every run is added to experiments.csv (your own copy - don't commit it).
"""
import argparse
import datetime as dt
import getpass
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


def seed_params(s):
    return {**PARAMS, "seed": 42 + s}


def score(tables, cols, rounds=None, seeds=1):
    """Train + test on both tests, once per seed.
    Returns auc[test, seed], the rounds used per test, and how much the model uses each clue."""
    auc = np.zeros((len(tables), seeds))
    used = [[] for _ in tables]
    imps = []
    for s in range(seeds):
        for i, (tr, va) in enumerate(tables):
            data = lgb.Dataset(tr[cols], tr.clicked)
            if rounds:   # fixed length: the test day is never used to choose anything
                m = lgb.train(seed_params(s), data, rounds)
                pred = m.predict(va[cols])
                used[i].append(rounds)
            else:        # early stopping: picks the best round BY LOOKING AT the test day
                m = lgb.train(seed_params(s), data, 2000,
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
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.rounds is not None and a.rounds < 1:
        ap.error("--rounds must be 1 or more")
    if a.seeds < 1:
        ap.error("--seeds must be 1 or more")

    groups = a.groups.split(",") if a.groups else all_groups()
    con = connect(full=(a.data == "full"))
    how = f"{a.rounds} rounds" if a.rounds else "early stopping"
    print(f"Data: {a.data}   Feature files: {', '.join(groups)}   Training: {how}, {a.seeds} seed(s)")

    # build each day's table once (test 2 learns from the day test 1 is tested on)
    t0, cache = time.time(), {}
    def table(cut, end):
        if (cut, end) not in cache:
            cache[(cut, end)] = make_table(con, groups, cut, end)
        return cache[(cut, end)]
    tables = [(table(a_, b_), table(b_, c_)) for _, a_, b_, c_ in FOLDS]
    cols = columns(tables[0][0], groups)
    print(f"Clues: {len(cols)}   built in {time.time() - t0:.0f}s")

    auc, used, imp = score(tables, cols, a.rounds, a.seeds)
    wobble = auc.std(axis=1, ddof=1) if a.seeds > 1 else None
    print()
    for i, (name, *_) in enumerate(FOLDS):
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
                auc_wo, _, _ = score(tables, columns(tables[0][0], rest), a.rounds, a.seeds)
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
        print(f"\nTraining the final model on Jan 25 + Jan 26 ({n_rounds} rounds, {a.seeds} seed(s)) "
              "and predicting evaluation...")
        final_tr = pd.concat([table(C1, C2), table(C2, C3)], ignore_index=True)
        ev = make_table(con, groups, C3, None, "eval")
        pred = np.zeros(len(ev))
        for s in range(a.seeds):               # several seeds = average their predictions
            model = lgb.train(seed_params(s), lgb.Dataset(final_tr[cols], final_tr.clicked), n_rounds)
            pred += model.predict(ev[cols]) / a.seeds
        sub = pd.DataFrame({"rowid": ev.rowid, "predicted_probability": pred})
        n_eval = con.sql("SELECT count(*) FROM eval").fetchall()[0][0]
        assert len(sub) == n_eval == 529_390, (len(sub), n_eval)
        assert sub.rowid.is_unique and sub.predicted_probability.between(0, 1).all()
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / (f"sub_{dt.date.today()}_{'+'.join(groups)}_{a.data}"
                         f"_r{n_rounds}_s{a.seeds}_{auc.mean():.4f}.csv")
        sub.to_csv(out, index=False)
        print(f"Wrote outputs/{out.name}  ({len(sub):,} rows, checks passed)")
        if a.data != "full":
            print("NOTE: made from the sample - don't upload it. Use 'full' for real submissions.")


if __name__ == "__main__":
    main()
