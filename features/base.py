"""
Eoin's base clues (taken from baseline.py), in the shared feature format.

Team rules every feature file follows:
  build(con, cut, end=None, source="train") -> table with `rowid` + feature columns
  - every feature column starts with this file's prefix
  - it only LEARNS from training rows before `cut` (use fit_sql(cut))

Quick test:  python features/base.py        (5% sample)
             python features/base.py full   (all data)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_sql, targets_sql, check_features

PREFIX = "base_"


def build(con, cut, end=None, source="train"):
    return con.sql(f"""
    WITH fit AS {fit_sql(cut)},
    gp AS (SELECT avg(clicked) AS p FROM fit),

    -- item: how often shown / clicked (click rate smoothed towards the average)
    item AS (
        SELECT item_id, count(*) AS shown, sum(clicked) AS clicks,
               (sum(clicked) + 50 * (SELECT p FROM gp)) / (count(*) + 50) AS ctr
        FROM fit GROUP BY item_id),
    item_1d AS (
        SELECT item_id, count(*) AS shown_1d, sum(clicked) AS clicks_1d
        FROM fit WHERE timestamp >= {cut} - 86400 GROUP BY item_id),

    -- user: how active, how often they click
    usr AS (
        SELECT user_id, count(DISTINCT event_id) AS events,
               sum(clicked) / count(DISTINCT event_id) AS click_rate,
               max(timestamp) AS last_event_ts
        FROM fit GROUP BY user_id),

    -- user + item: shown this item before? clicked it?
    ui AS (
        SELECT user_id, item_id, count(*) AS shown, sum(clicked) AS clicks
        FROM fit GROUP BY user_id, item_id),

    t AS ({targets_sql(cut, end, source)}),

    f AS (
        SELECT t.rowid, t.event_id,
               t.time_bucket                                    AS base_time_bucket,
               t.day_type                                       AS base_day_type,
               -- history clues
               len(t.h)                                         AS base_hist_len,
               len(list_filter(t.h, lambda x: x = t.item_id))   AS base_hist_item_cnt,
               list_position(list_reverse(t.h), t.item_id)      AS base_hist_item_clicks_ago,
               (list_last(t.h) = t.item_id)::INT                AS base_is_last_click,
               t.timestamp - list_last(t.hts)                   AS base_secs_since_last_click,
               t.timestamp - list_first(t.hts)                  AS base_secs_since_first_hist_click,
               -- item clues
               item.shown AS base_item_shown, item.clicks AS base_item_clicks, item.ctr AS base_item_ctr,
               item_1d.shown_1d AS base_item_shown_1d, item_1d.clicks_1d AS base_item_clicks_1d,
               -- user clues
               usr.events AS base_user_events, usr.click_rate AS base_user_click_rate,
               t.timestamp - usr.last_event_ts                  AS base_secs_since_last_event,
               -- user + item clues
               ui.shown AS base_ui_shown, ui.clicks AS base_ui_clicks
        FROM t
        LEFT JOIN item    ON item.item_id = t.item_id
        LEFT JOIN item_1d ON item_1d.item_id = t.item_id
        LEFT JOIN usr     ON usr.user_id = t.user_id
        LEFT JOIN ui      ON ui.user_id = t.user_id AND ui.item_id = t.item_id)

    -- compare each candidate with the other 9 in its event
    SELECT * EXCLUDE (event_id),
           rank() OVER (PARTITION BY event_id ORDER BY base_item_ctr DESC NULLS LAST) AS base_ctr_rank_in_event,
           base_item_ctr - avg(base_item_ctr) OVER (PARTITION BY event_id)            AS base_ctr_minus_event_mean,
           max(base_hist_item_cnt) OVER (PARTITION BY event_id)                       AS base_event_any_in_hist
    FROM f
    """).df()


if __name__ == "__main__":
    import time
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    t0 = time.time()
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(f"built in {time.time() - t0:.1f}s")
    print(df.head())
