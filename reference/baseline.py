"""
TechArena 2026 Phase 1 - first end-to-end baseline.

Needs: pip install duckdb pandas lightgbm scikit-learn
Expects a DuckDB file with tables `train` and `eval` (made by make_sample.py).

Setup (mirrors the official one: "each user's first event after a cutoff"):
  train model : features from events < Jan 25, labels = first event per user in [Jan 25, Jan 26)
  validate    : features from events < Jan 26, labels = first event per user in [Jan 26, Jan 27)
  final       : retrain on both windows, features from all train (< Jan 27), predict evaluation.csv
"""
import duckdb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

# python baseline.py       -> uses the 5% sample (fast testing)
# python baseline.py full  -> uses the full database (real score)
import sys
DB_NAME = "techarena.duckdb" if "full" in sys.argv else "sample.duckdb"
from pathlib import Path
DB = str(Path(__file__).resolve().parents[1] / "Data" / DB_NAME)
print("Using", DB_NAME)
con = duckdb.connect(DB)
con.execute("SET memory_limit='8GB'")

DAY = 86400
C3 = 1737936000          # 2025-01-27 00:00 UTC (official cutoff)
C2 = C3 - DAY            # 2025-01-26
C1 = C3 - 2 * DAY        # 2025-01-25


def build(cut, end=None, source="train"):
    """Features use only train rows before `cut`. Rows returned = each user's first event in [cut, end)."""
    end_filter = f"AND timestamp < {end}" if end else ""
    label = "t.clicked" if source == "train" else "NULL::TINYINT AS clicked"
    return con.sql(f"""
    WITH fit AS (SELECT * FROM train WHERE timestamp < {cut}),
    gp AS (SELECT avg(clicked) AS p FROM fit),

    -- item popularity / click rate (smoothed towards the global rate)
    item AS (
        SELECT item_id, count(*) AS item_shown, sum(clicked) AS item_clicks,
               (sum(clicked) + 50 * (SELECT p FROM gp)) / (count(*) + 50) AS item_ctr
        FROM fit GROUP BY item_id),
    item_1d AS (
        SELECT item_id, count(*) AS item_shown_1d, sum(clicked) AS item_clicks_1d
        FROM fit WHERE timestamp >= {cut} - 86400 GROUP BY item_id),

    -- user activity: how often does this user click at all?
    usr AS (
        SELECT user_id, count(DISTINCT event_id) AS user_events,
               sum(clicked) / count(DISTINCT event_id) AS user_click_rate,
               max(timestamp) AS user_last_event_ts
        FROM fit GROUP BY user_id),

    -- user x item: shown before? clicked before?
    ui AS (
        SELECT user_id, item_id, count(*) AS ui_shown, sum(clicked) AS ui_clicks
        FROM fit GROUP BY user_id, item_id),

    firsts AS (
        SELECT event_id FROM {source}
        WHERE timestamp >= {cut} {end_filter}
        QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY timestamp, event_id) = 1),

    t AS (
        SELECT *, from_json(history_item_ids, '["VARCHAR"]') AS h,
                  from_json(history_timestamps, '["BIGINT"]') AS hts
        FROM {source} WHERE event_id IN (SELECT event_id FROM firsts))

    SELECT t.rowid, t.event_id, {label},
           t.time_bucket, t.day_type,
           -- history features (history is legitimately known at event time)
           len(t.h)                                        AS hist_len,
           len(list_filter(t.h, lambda x: x = t.item_id))       AS hist_item_cnt,
           list_position(list_reverse(t.h), t.item_id)     AS hist_item_clicks_ago,
           (list_last(t.h) = t.item_id)::INT               AS is_last_click,
           t.timestamp - list_last(t.hts)                  AS secs_since_last_click,
           t.timestamp - list_first(t.hts)                 AS secs_since_first_hist_click,
           -- aggregates from the fitting period only
           item.item_shown, item.item_clicks, item.item_ctr,
           item_1d.item_shown_1d, item_1d.item_clicks_1d,
           usr.user_events, usr.user_click_rate,
           t.timestamp - usr.user_last_event_ts            AS secs_since_last_event,
           ui.ui_shown, ui.ui_clicks
    FROM t
    LEFT JOIN item    ON item.item_id = t.item_id
    LEFT JOIN item_1d ON item_1d.item_id = t.item_id
    LEFT JOIN usr     ON usr.user_id = t.user_id
    LEFT JOIN ui      ON ui.user_id = t.user_id AND ui.item_id = t.item_id
    """).df()


def add_event_feats(df):
    """Compare each candidate with the other 9 in its event."""
    g = df.groupby("event_id")
    df["ctr_rank_in_event"] = g["item_ctr"].rank(ascending=False)
    df["ctr_minus_event_mean"] = df["item_ctr"] - g["item_ctr"].transform("mean")
    df["event_any_in_hist"] = g["hist_item_cnt"].transform("max")
    return df


print("Building features...")
tr = add_event_feats(build(C1, C2))
va = add_event_feats(build(C2, C3))
X = [c for c in tr.columns if c not in ("rowid", "event_id", "clicked")]
print(f"train rows {len(tr):,}  val rows {len(va):,}  features {len(X)}")

params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1)

model = lgb.train(params, lgb.Dataset(tr[X], tr.clicked), 2000,
                  valid_sets=[lgb.Dataset(va[X], va.clicked)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
print("VALIDATION AUC:", round(roc_auc_score(va.clicked, model.predict(va[X])), 5))

imp = pd.Series(model.feature_importance("gain"), index=X).sort_values(ascending=False)
print("\nFeature importance (gain):\n", imp.round(0))

# ---- final model: both windows, features from all of train, predict evaluation ----
print("\nTraining final model...")
full = pd.concat([tr, va], ignore_index=True)
final = lgb.train(params, lgb.Dataset(full[X], full.clicked), model.best_iteration)

ev = add_event_feats(build(C3, source="eval"))
sub = pd.DataFrame({"rowid": ev.rowid, "predicted_probability": final.predict(ev[X])})

# ---- submission checks ----
n_eval = con.sql("SELECT count(*) FROM eval").fetchall()[0][0]
assert len(sub) == n_eval == 529_390, (len(sub), n_eval)
assert sub.rowid.is_unique
assert sub.predicted_probability.between(0, 1).all()
sub.to_csv(Path(__file__).resolve().parents[1] / "outputs" / "submission_baseline.csv", index=False)
print("Wrote outputs/submission_baseline.csv", sub.shape)