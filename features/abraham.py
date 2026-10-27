"""
Abraham's Co-visitation Clues (prefix cv_)
Huawei TechArena CTR Challenge - Team Huawei Tuders

Features based on item-item co-visitation patterns ("B clicked after A"):
  1. Historical co-visitation counts and transition probabilities:
     - Learned strictly from pre-cutoff training events (fit_sql(cut)).
     - Directed pairs: B clicked after A within a sliding sequence window (<= 5 clicks)
       and time window (<= 24 hours).
  2. Candidate scoring against the user's recent clicks:
     - cv_last1_pair_cnt: Raw co-visitation count from user's most recent click to candidate.
     - cv_last1_prob: Empirical transition probability P(candidate | last_click_1).
     - cv_last1_cos: Cosine-normalized similarity between last_click_1 and candidate.
     - cv_last3_sum_cnt, cv_last3_sum_prob, cv_last3_max_prob, cv_last3_sum_cos, cv_last3_max_cos:
       Aggregations over the user's last 3 clicks.
     - cv_last3_tw_prob: Time-weighted transition probability decaying with click age.
     - cv_last5_sum_cnt, cv_last5_sum_prob, cv_last5_max_prob, cv_last5_sum_cos:
       Aggregations over the user's last 5 clicks.
     - cv_last5_tw_prob: Time-weighted transition probability across last 5 clicks.
  3. Event-relative comparisons:
     - cv_rank_in_event: Rank of candidate within its 10-candidate event by co-visitation score.
     - cv_prob_minus_event_mean: Difference between candidate score and event mean score.
     - cv_event_max_last3_prob: Maximum co-visitation score present among the 10 candidates.
  4. Cold-start and history flags:
     - cv_has_history: 1 if user has at least 1 valid pre-cutoff click, else 0.
     - cv_hist_len_valid: Number of valid pre-cutoff clicks available (capped at 5).
     - cv_in_recent_5: 1 if candidate item itself was clicked in the user's recent 5 clicks.

Leakage Prevention Enforcement:
  - All co-visitation statistics are computed exclusively from training rows where timestamp < cut (fit_sql(cut)).
  - Candidate history clicks are strictly filtered so that click timestamp < cut (any click at or after cut is ignored).
  - Target labels ('clicked') from prediction rows are never accessed or referenced.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_sql, targets_sql, check_features

PREFIX = "cv_"


def build(con, cut, end=None, source="train"):
    """
    Build co-visitation features for the target events in [cut, end).

    Args:
        con: DuckDB connection.
        cut: Cutoff timestamp (BIGINT epoch seconds).
        end: Optional end timestamp (BIGINT epoch seconds).
        source: 'train' or 'eval' table.

    Returns:
        pd.DataFrame containing 'rowid' and all features prefixed with 'cv_'.
    """
    sql = f"""
    WITH fit AS {fit_sql(cut)},

    -- 1. All historical clicks strictly before cutoff
    raw_clicks AS (
        SELECT DISTINCT user_id, item_id, ts
        FROM (
            SELECT user_id,
                   unnest(from_json(history_item_ids, '["VARCHAR"]')) AS item_id,
                   unnest(from_json(history_timestamps, '["BIGINT"]')) AS ts
            FROM (SELECT DISTINCT event_id, user_id, history_item_ids, history_timestamps FROM fit)
            UNION ALL
            SELECT user_id, item_id, timestamp AS ts
            FROM fit WHERE clicked = 1
        )
        WHERE ts < {cut}
    ),

    -- 2. Chronological sequence of clicks per user before cutoff
    ordered_clicks AS (
        SELECT user_id, item_id, ts,
               row_number() OVER (PARTITION BY user_id ORDER BY ts, item_id) AS pos
        FROM raw_clicks
    ),

    -- 3. Extract directed pairs "B clicked after A" within 5 steps and 24 hours
    pairs AS (
        SELECT c1.item_id AS item_a,
               c2.item_id AS item_b,
               c2.pos - c1.pos AS step_diff,
               c2.ts - c1.ts AS time_diff
        FROM ordered_clicks c1
        JOIN ordered_clicks c2
          ON c1.user_id = c2.user_id
         AND c2.pos > c1.pos
         AND c2.pos <= c1.pos + 5
         AND c2.ts > c1.ts
         AND c2.ts <= c1.ts + 86400
    ),

    -- 4. Item click counts for marginal normalization
    item_counts AS (
        SELECT item_id, count(*)::BIGINT AS click_count
        FROM raw_clicks
        GROUP BY item_id
    ),

    -- 5. Co-visitation transition summary: count, P(B|A), and cosine similarity
    covis AS (
        SELECT p.item_a,
               p.item_b,
               count(*)::BIGINT AS pair_cnt,
               count(*)::DOUBLE / nullif(coalesce(ca.click_count, 1), 0) AS prob_b_given_a,
               count(*)::DOUBLE / sqrt(nullif(ca.click_count * cb.click_count, 0)) AS cos_sim
        FROM pairs p
        LEFT JOIN item_counts ca ON ca.item_id = p.item_a
        LEFT JOIN item_counts cb ON cb.item_id = p.item_b
        GROUP BY p.item_a, p.item_b, ca.click_count, cb.click_count
    ),

    -- 6. Target prediction rows
    t AS ({targets_sql(cut, end, source)}),

    -- 7. Extract the user's last 5 clicks strictly before cutoff
    t_clicks AS (
        SELECT t.rowid, t.event_id, t.item_id, t.timestamp,
               -- 1st last click (most recent)
               case when len(t.hts) >= 1 and list_extract(t.hts, -1) < {cut}
                    then list_extract(t.h, -1) else null end AS last_item_1,
               case when len(t.hts) >= 1 and list_extract(t.hts, -1) < {cut}
                    then list_extract(t.hts, -1) else null end AS last_ts_1,
               -- 2nd last click
               case when len(t.hts) >= 2 and list_extract(t.hts, -2) < {cut}
                    then list_extract(t.h, -2) else null end AS last_item_2,
               case when len(t.hts) >= 2 and list_extract(t.hts, -2) < {cut}
                    then list_extract(t.hts, -2) else null end AS last_ts_2,
               -- 3rd last click
               case when len(t.hts) >= 3 and list_extract(t.hts, -3) < {cut}
                    then list_extract(t.h, -3) else null end AS last_item_3,
               case when len(t.hts) >= 3 and list_extract(t.hts, -3) < {cut}
                    then list_extract(t.hts, -3) else null end AS last_ts_3,
               -- 4th last click
               case when len(t.hts) >= 4 and list_extract(t.hts, -4) < {cut}
                    then list_extract(t.h, -4) else null end AS last_item_4,
               case when len(t.hts) >= 4 and list_extract(t.hts, -4) < {cut}
                    then list_extract(t.hts, -4) else null end AS last_ts_4,
               -- 5th last click
               case when len(t.hts) >= 5 and list_extract(t.hts, -5) < {cut}
                    then list_extract(t.h, -5) else null end AS last_item_5,
               case when len(t.hts) >= 5 and list_extract(t.hts, -5) < {cut}
                    then list_extract(t.hts, -5) else null end AS last_ts_5
        FROM t
    ),

    -- 8. Score candidate item against recent clicks
    scored AS (
        SELECT tc.rowid, tc.event_id, tc.item_id,
               (tc.last_item_1 is not null)::INT AS cv_has_history,
               ((tc.last_item_1 is not null)::INT + (tc.last_item_2 is not null)::INT +
                (tc.last_item_3 is not null)::INT + (tc.last_item_4 is not null)::INT +
                (tc.last_item_5 is not null)::INT)::INT AS cv_hist_len_valid,
               coalesce((tc.item_id = tc.last_item_1 or tc.item_id = tc.last_item_2 or
                         tc.item_id = tc.last_item_3 or tc.item_id = tc.last_item_4 or
                         tc.item_id = tc.last_item_5)::INT, 0) AS cv_in_recent_5,

               -- Last 1 click scores
               coalesce(c1.pair_cnt, 0)::BIGINT AS cv_last1_pair_cnt,
               coalesce(c1.prob_b_given_a, 0.0)::DOUBLE AS cv_last1_prob,
               coalesce(c1.cos_sim, 0.0)::DOUBLE AS cv_last1_cos,

               -- Last 3 clicks scores
               (coalesce(c1.pair_cnt, 0) + coalesce(c2.pair_cnt, 0) + coalesce(c3.pair_cnt, 0))::BIGINT AS cv_last3_sum_cnt,
               (coalesce(c1.prob_b_given_a, 0.0) + coalesce(c2.prob_b_given_a, 0.0) + coalesce(c3.prob_b_given_a, 0.0))::DOUBLE AS cv_last3_sum_prob,
               greatest(coalesce(c1.prob_b_given_a, 0.0), coalesce(c2.prob_b_given_a, 0.0), coalesce(c3.prob_b_given_a, 0.0))::DOUBLE AS cv_last3_max_prob,
               (coalesce(c1.cos_sim, 0.0) + coalesce(c2.cos_sim, 0.0) + coalesce(c3.cos_sim, 0.0))::DOUBLE AS cv_last3_sum_cos,
               greatest(coalesce(c1.cos_sim, 0.0), coalesce(c2.cos_sim, 0.0), coalesce(c3.cos_sim, 0.0))::DOUBLE AS cv_last3_max_cos,

               -- Time-weighted last 3 clicks (hourly decay)
               (coalesce(c1.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_1) / 3600.0) +
                coalesce(c2.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_2) / 3600.0) +
                coalesce(c3.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_3) / 3600.0))::DOUBLE AS cv_last3_tw_prob,

               -- Last 5 clicks scores
               (coalesce(c1.pair_cnt, 0) + coalesce(c2.pair_cnt, 0) + coalesce(c3.pair_cnt, 0) +
                coalesce(c4.pair_cnt, 0) + coalesce(c5.pair_cnt, 0))::BIGINT AS cv_last5_sum_cnt,
               (coalesce(c1.prob_b_given_a, 0.0) + coalesce(c2.prob_b_given_a, 0.0) + coalesce(c3.prob_b_given_a, 0.0) +
                coalesce(c4.prob_b_given_a, 0.0) + coalesce(c5.prob_b_given_a, 0.0))::DOUBLE AS cv_last5_sum_prob,
               greatest(coalesce(c1.prob_b_given_a, 0.0), coalesce(c2.prob_b_given_a, 0.0), coalesce(c3.prob_b_given_a, 0.0),
                        coalesce(c4.prob_b_given_a, 0.0), coalesce(c5.prob_b_given_a, 0.0))::DOUBLE AS cv_last5_max_prob,
               (coalesce(c1.cos_sim, 0.0) + coalesce(c2.cos_sim, 0.0) + coalesce(c3.cos_sim, 0.0) +
                coalesce(c4.cos_sim, 0.0) + coalesce(c5.cos_sim, 0.0))::DOUBLE AS cv_last5_sum_cos,

               -- Time-weighted last 5 clicks
               (coalesce(c1.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_1) / 3600.0) +
                coalesce(c2.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_2) / 3600.0) +
                coalesce(c3.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_3) / 3600.0) +
                coalesce(c4.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_4) / 3600.0) +
                coalesce(c5.prob_b_given_a, 0.0) / (1.0 + greatest(0.0, tc.timestamp - tc.last_ts_5) / 3600.0))::DOUBLE AS cv_last5_tw_prob

        FROM t_clicks tc
        LEFT JOIN covis c1 ON c1.item_a = tc.last_item_1 AND c1.item_b = tc.item_id
        LEFT JOIN covis c2 ON c2.item_a = tc.last_item_2 AND c2.item_b = tc.item_id
        LEFT JOIN covis c3 ON c3.item_a = tc.last_item_3 AND c3.item_b = tc.item_id
        LEFT JOIN covis c4 ON c4.item_a = tc.last_item_4 AND c4.item_b = tc.item_id
        LEFT JOIN covis c5 ON c5.item_a = tc.last_item_5 AND c5.item_b = tc.item_id
    )

    -- 9. Candidate comparison relative to other 9 candidates in the same event
    SELECT rowid,
           cv_has_history,
           cv_hist_len_valid,
           cv_in_recent_5,
           cv_last1_pair_cnt,
           cv_last1_prob,
           cv_last1_cos,
           cv_last3_sum_cnt,
           cv_last3_sum_prob,
           cv_last3_max_prob,
           cv_last3_sum_cos,
           cv_last3_max_cos,
           cv_last3_tw_prob,
           cv_last5_sum_cnt,
           cv_last5_sum_prob,
           cv_last5_max_prob,
           cv_last5_sum_cos,
           cv_last5_tw_prob,
           rank() OVER (PARTITION BY event_id ORDER BY cv_last3_sum_prob DESC, cv_last1_prob DESC, rowid) AS cv_rank_in_event,
           cv_last3_sum_prob - avg(cv_last3_sum_prob) OVER (PARTITION BY event_id) AS cv_prob_minus_event_mean,
           max(cv_last3_sum_prob) OVER (PARTITION BY event_id) AS cv_event_max_last3_prob
    FROM scored
    """
    df = con.sql(sql).df()
    return df


if __name__ == "__main__":
    import time
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    t0 = time.time()
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(f"built in {time.time() - t0:.2f}s")
    print(df.head())
    print("\nFeature Summary:")
    print(df.describe().T[["count", "mean", "min", "max"]])
