"""Stage 3b: do the validation folds look like the evaluation set?
Three checks, all at EVENT level (one row per user event):
  1. side-by-side summary numbers,
  2. for each feature: how well does that feature ALONE tell fold events from evaluation events?
     (AUC 0.50 = identical, >0.60 = noticeably different, >0.70 = clearly different),
  3. 'adversarial validation': a small model tries to tell fold from evaluation using all features.
Writes val_vs_eval_report.md"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import *
from db import connect, epoch_of, fold_name
from scaling import observation_days

pd.set_option("display.width", 200)
CALENDAR = ["hours_since_cutoff", "hour", "dow", "day_type", "time_bucket"]
OTHER = ["hist_len", "hours_since_last_click", "n_cand_in_history", "user_events_per_day"]
# user_events_per_day is left out of the adversarial model: dividing a whole number by 5, 6 or 7 days
# gives values on different "grids" (0.2, 0.4.. vs 0.143, 0.286..) that reveal the day count and fake a difference.
ADV_OTHER = ["hist_len", "hours_since_last_click", "n_cand_in_history"]

def event_table(con, src, cutoff):
    lo, days = epoch_of(con, cutoff), observation_days(con, cutoff)
    return con.execute(f"""
        WITH ev AS (SELECT event_id, any_value(user_id) AS user_id, any_value(timestamp) AS ts,
                           any_value(time_bucket) AS time_bucket, any_value(day_type) AS day_type,
                           any_value(history_timestamps) AS hts,
                           sum(contains(history_item_ids, item_id)::INT) AS n_cand_in_history
                    FROM {src} GROUP BY 1),
             ue AS (SELECT user_id, count(DISTINCT event_id) AS n FROM train WHERE timestamp < {lo} GROUP BY 1),
             x AS (SELECT *, CAST(hts AS BIGINT[]) AS h FROM ev)
        SELECT (ts - {lo}) / 3600.0 AS hours_since_cutoff,
               hour(make_timestamp(ts * 1000000)) AS hour,
               dayofweek(make_timestamp(ts * 1000000)) AS dow,
               time_bucket, day_type, len(h) AS hist_len,
               CASE WHEN len(h) > 0 THEN (ts - list_max(h)) / 3600.0 END AS hours_since_last_click,
               n_cand_in_history,
               coalesce(ue.n, 0) / {days} AS user_events_per_day
        FROM x LEFT JOIN ue USING (user_id)""").df()

def summary(df):
    s = {"events": len(df),
         "share weekend (day_type=1)": df.day_type.mean(),
         "share in first 24h after cutoff": (df.hours_since_cutoff < 24).mean(),
         "history length, median": df.hist_len.median(),
         "share empty history": (df.hist_len == 0).mean(),
         "hours since last click, median": df.hours_since_last_click.median(),
         "candidates in history, mean (of 10)": df.n_cand_in_history.mean(),
         "user events per day (scaled), mean": df.user_events_per_day.mean()}
    for b in range(4):
        s[f"share time_bucket {b} ({b*6:02d}-{b*6+5:02d}h UTC)"] = (df.time_bucket == b).mean()
    return pd.Series(s)

if __name__ == "__main__":
    con = connect()
    sets = {c: event_table(con, fold_name(c), c) for c in CUTOFFS}
    ev = event_table(con, "evalset", DATA_END)

    table = pd.DataFrame({**{f"fold {c}": summary(d) for c, d in sets.items()}, "EVAL": summary(ev)})
    out = ["# Validation folds vs evaluation set", "", "## 1. Side by side", "", table.round(3).to_string(), ""]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    for c, d in sets.items():
        both = pd.concat([d.assign(is_eval=0), ev.assign(is_eval=1)], ignore_index=True)
        y = both.pop("is_eval")
        single = pd.Series({f: (lambda a: max(a, 1 - a))(roc_auc_score(y, both[f].fillna(-1)))
                            for f in CALENDAR + OTHER}).sort_values(ascending=False)
        model = lambda: HistGradientBoostingClassifier(max_iter=100)
        adv_all = cross_val_score(model(), both[CALENDAR + ADV_OTHER], y, cv=cv, scoring="roc_auc").mean()
        adv_non = cross_val_score(model(), both[ADV_OTHER], y, cv=cv, scoring="roc_auc").mean()
        out += [f"## 2-3. Fold {c} vs EVAL", "",
                "Single-feature difference AUC (0.50 = same, higher = more different):", "",
                single.round(3).to_string(), "",
                f"Adversarial AUC, all features:                {adv_all:.3f}",
                f"Adversarial AUC, WITHOUT calendar features:   {adv_non:.3f}", ""]
    text = "\n".join(out)
    open("val_vs_eval_report.md", "w", encoding="utf-8").write(text)
    print(text)
