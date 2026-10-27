"""
Ife's user and user time clues for Huawei TechArena 2026 CTR Challenge.
Module: features/ife.py (prefix u_)

Assigned feature categories:
  1. User engagement:
     - u_click_rate: Historical clicked events / total events (Bayesian smoothed, K=20)
     - u_events_per_day: Historical recommendation events per day
     - u_zero_click_event_ratio: Ratio of events where user clicked nothing (smoothed, K=20)
     - u_total_clicks: Total eligible historical clicks (deduplicated across warm-up and training)
  2. User recency:
     - u_secs_since_last_click: Seconds since user's previous click before cut
     - u_secs_since_last_event: Seconds since user's previous recommendation event before cut
     - u_secs_since_first_seen: Seconds since user's earliest interaction before cut
     - u_recency_to_avg_interval_ratio: Ratio of current inactivity to mean inter-event arrival time
  3. User-item exposure:
     - u_item_shown_unclicked_cnt: Occasions candidate item was shown without clicking
     - u_secs_since_last_unclicked: Seconds since candidate item was last seen without clicking
     - u_item_unclicked_ratio: Unclicked impressions / total impressions for candidate item
  4. User time preferences:
     - u_share_in_current_bucket: Historical share of events in target event's time_bucket
     - u_fav_time_bucket: User's modal activity time_bucket (stable tie-breaking)
     - u_is_fav_time_bucket: Binary indicator if current event matches modal bucket
     - u_weekend_event_share: Proportion of past events occurring on weekends (day_type=1)
     - u_day_type_affinity: Proximity match between current day_type and user's weekend propensity
  5. Cold-start handling:
     - u_is_cold_user: 1 if user has 0 events before cut, else 0
     - u_has_click_history: 1 if user has any eligible clicks before cut, else 0

Strict leakage prevention:
  - All historical aggregations query `fit_sql(cut)` where timestamp < cut.
  - History arrays (h, hts) are filtered to click_ts < cut to reject post-cutoff clicks.
  - Target labels are never accessed or referenced.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_sql, targets_sql, check_features

PREFIX = "u_"
SMOOTH_K = 20.0


def build(con, cut, end=None, source="train"):
    """
    Build user and user-time features for the requested prediction window.

    Args:
        con: DuckDB connection.
        cut: Cutoff timestamp (epoch seconds).
        end: Optional end timestamp.
        source: 'train' or 'eval' table.

    Returns:
        pd.DataFrame containing 'rowid' and all features prefixed with 'u_'.
    """
    sql = f"""
    WITH fit AS {fit_sql(cut)},

    -- 1. Distinct events in fit (each recommendation event has 10 candidates)
    fit_events AS (
        SELECT
            event_id,
            user_id,
            min(timestamp) AS event_ts,
            min(time_bucket) AS time_bucket,
            min(day_type) AS day_type,
            max(clicked) AS any_clicked
        FROM fit
        GROUP BY event_id, user_id
    ),

    -- 2. Global priors from fit_events for empirical Bayes smoothing
    gp AS (
        SELECT
            count(*)::BIGINT AS total_events,
            coalesce(avg(any_clicked), 0.10)::DOUBLE AS g_click_rate,
            coalesce(avg(case when any_clicked = 0 then 1.0 else 0.0 end), 0.90)::DOUBLE AS g_zero_click_ratio,
            coalesce(avg(case when day_type = 1 then 1.0 else 0.0 end), 0.28)::DOUBLE AS g_weekend_share
        FROM fit_events
    ),

    gp_bucket AS (
        SELECT
            time_bucket,
            count(*)::DOUBLE / nullif((SELECT total_events FROM gp), 0) AS g_bucket_share
        FROM fit_events
        GROUP BY time_bucket
    ),

    gp_fav_bucket AS (
        SELECT time_bucket AS g_fav_bucket
        FROM gp_bucket
        ORDER BY g_bucket_share DESC, time_bucket ASC
        LIMIT 1
    ),

    -- 3. User-level event aggregations from fit
    usr_events AS (
        SELECT
            user_id,
            count(*)::BIGINT AS user_events,
            sum(any_clicked)::BIGINT AS user_clicked_events,
            sum(case when any_clicked = 0 then 1 else 0 end)::BIGINT AS user_zero_click_events,
            sum(case when day_type = 1 then 1 else 0 end)::BIGINT AS user_weekend_events,
            min(event_ts)::BIGINT AS user_min_event_ts,
            max(event_ts)::BIGINT AS user_max_event_ts
        FROM fit_events
        GROUP BY user_id
    ),

    -- 4. User time-bucket counts and modal bucket from fit
    usr_bucket AS (
        SELECT
            user_id,
            time_bucket,
            count(*)::BIGINT AS bucket_events
        FROM fit_events
        GROUP BY user_id, time_bucket
    ),

    usr_fav_bucket AS (
        SELECT
            user_id,
            time_bucket AS fav_time_bucket
        FROM (
            SELECT
                user_id,
                time_bucket,
                row_number() OVER (PARTITION BY user_id ORDER BY bucket_events DESC, time_bucket ASC) AS rn
            FROM usr_bucket
        )
        WHERE rn = 1
    ),

    -- 5. Prediction targets
    t AS ({targets_sql(cut, end, source)}),

    -- 6. Historical clicks strictly before cut
    -- Unnest target click history, filter click_ts < cut, and combine with training clicks.
    -- UNION deduplicates overlapping warm-up and training clicks on (user_id, item_id, click_ts).
    raw_hist_clicks AS (
        SELECT DISTINCT user_id, unnest(h) AS item_id, unnest(hts) AS click_ts
        FROM (
            SELECT DISTINCT user_id, h, hts
            FROM t
            WHERE h IS NOT NULL AND hts IS NOT NULL AND len(h) > 0 AND len(h) = len(hts)
        )
    ),
    all_clicks AS (
        SELECT user_id, item_id, timestamp AS click_ts
        FROM fit
        WHERE clicked = 1
        UNION
        SELECT user_id, item_id, click_ts
        FROM raw_hist_clicks
        WHERE click_ts < {cut}
    ),
    usr_clicks AS (
        SELECT
            user_id,
            count(*)::BIGINT AS total_clicks,
            min(click_ts)::BIGINT AS user_min_click_ts,
            max(click_ts)::BIGINT AS user_max_click_ts
        FROM all_clicks
        GROUP BY user_id
    ),

    -- 7. User-Item exposure history before cut
    ui_exposure AS (
        SELECT
            user_id,
            item_id,
            count(*)::BIGINT AS ui_shown_cnt,
            sum(case when clicked = 0 then 1 else 0 end)::BIGINT AS ui_shown_unclicked_cnt,
            max(case when clicked = 0 then timestamp else null end)::BIGINT AS ui_last_unclicked_ts
        FROM fit
        GROUP BY user_id, item_id
    )

    -- 8. Assemble all features for target candidate rows
    SELECT
        t.rowid,

        -- A. User engagement
        (coalesce(ue.user_clicked_events, 0) + {SMOOTH_K} * coalesce((SELECT g_click_rate FROM gp), 0.10)) /
            (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE AS u_click_rate,

        coalesce(ue.user_events, 0)::DOUBLE /
            greatest(1.0, ({cut} - coalesce(ue.user_min_event_ts, {cut})) / 86400.0) AS u_events_per_day,

        (coalesce(ue.user_zero_click_events, 0) + {SMOOTH_K} * coalesce((SELECT g_zero_click_ratio FROM gp), 0.90)) /
            (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE AS u_zero_click_event_ratio,

        coalesce(uc.total_clicks, 0)::BIGINT AS u_total_clicks,

        -- B. User recency (seconds in epoch format; NULL/NaN when undefined)
        case when uc.user_max_click_ts is not null
             then (t.timestamp - uc.user_max_click_ts)::DOUBLE
             else null end AS u_secs_since_last_click,

        case when ue.user_max_event_ts is not null
             then (t.timestamp - ue.user_max_event_ts)::DOUBLE
             else null end AS u_secs_since_last_event,

        case when ue.user_min_event_ts is not null or uc.user_min_click_ts is not null
             then (t.timestamp - least(coalesce(ue.user_min_event_ts, uc.user_min_click_ts),
                                       coalesce(uc.user_min_click_ts, ue.user_min_event_ts)))::DOUBLE
             else null end AS u_secs_since_first_seen,

        case when ue.user_events >= 2 and ue.user_max_event_ts > ue.user_min_event_ts
             then (t.timestamp - ue.user_max_event_ts)::DOUBLE /
                  ((ue.user_max_event_ts - ue.user_min_event_ts)::DOUBLE / (ue.user_events - 1.0))
             else null end AS u_recency_to_avg_interval_ratio,

        -- C. User-item exposure
        coalesce(ui.ui_shown_unclicked_cnt, 0)::BIGINT AS u_item_shown_unclicked_cnt,

        case when ui.ui_last_unclicked_ts is not null
             then (t.timestamp - ui.ui_last_unclicked_ts)::DOUBLE
             else null end AS u_secs_since_last_unclicked,

        case when ui.ui_shown_cnt > 0
             then ui.ui_shown_unclicked_cnt::DOUBLE / ui.ui_shown_cnt
             else null end AS u_item_unclicked_ratio,

        -- D. User time preferences
        (coalesce(ub.bucket_events, 0) + {SMOOTH_K} * coalesce(gb.g_bucket_share, 0.25)) /
            (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE AS u_share_in_current_bucket,

        coalesce(ufb.fav_time_bucket, (SELECT g_fav_bucket FROM gp_fav_bucket), 0)::TINYINT AS u_fav_time_bucket,

        (t.time_bucket = coalesce(ufb.fav_time_bucket, (SELECT g_fav_bucket FROM gp_fav_bucket), 0))::INT AS u_is_fav_time_bucket,

        (coalesce(ue.user_weekend_events, 0) + {SMOOTH_K} * coalesce((SELECT g_weekend_share FROM gp), 0.28)) /
            (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE AS u_weekend_event_share,

        case when t.day_type = 1
             then (coalesce(ue.user_weekend_events, 0) + {SMOOTH_K} * coalesce((SELECT g_weekend_share FROM gp), 0.28)) /
                  (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE
             else 1.0 - (coalesce(ue.user_weekend_events, 0) + {SMOOTH_K} * coalesce((SELECT g_weekend_share FROM gp), 0.28)) /
                  (coalesce(ue.user_events, 0) + {SMOOTH_K})::DOUBLE
             end AS u_day_type_affinity,

        -- E. Cold-start handling indicators
        (ue.user_events is null or ue.user_events = 0)::INT AS u_is_cold_user,
        (coalesce(uc.total_clicks, 0) > 0)::INT AS u_has_click_history

    FROM t
    LEFT JOIN usr_events ue ON ue.user_id = t.user_id
    LEFT JOIN usr_bucket ub ON ub.user_id = t.user_id AND ub.time_bucket = t.time_bucket
    LEFT JOIN usr_fav_bucket ufb ON ufb.user_id = t.user_id
    LEFT JOIN usr_clicks uc ON uc.user_id = t.user_id
    LEFT JOIN ui_exposure ui ON ui.user_id = t.user_id AND ui.item_id = t.item_id
    LEFT JOIN gp_bucket gb ON gb.time_bucket = t.time_bucket
    """
    df = con.sql(sql).df()
    return df


if __name__ == "__main__":
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().T)
