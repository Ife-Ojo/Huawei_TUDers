"""
Generates a realistic synthetic sample database matching the competition schema
for local development and validation testing.
Creates: Data/sample.duckdb with tables `train` and `eval`.
"""
import json
import os
import random
import sys
from pathlib import Path

# Competition cutoffs
C3 = 1737936000          # 2025-01-27 00:00 UTC (evaluation starts)
DAY = 86400
C2 = C3 - DAY            # 2025-01-26
C1 = C3 - 2 * DAY        # 2025-01-25
C0 = C3 - 3 * DAY        # 2025-01-24
START_TIME = C0 - 2 * DAY # 2025-01-22

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
DB_PATH = DATA_DIR / "sample.duckdb"


def generate_synthetic_data(num_users=300, num_items=150, num_clusters=10, seed=42):
    import duckdb

    random.seed(seed)
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    # Assign items to clusters so co-visitation has genuine statistical signal
    items = [f"item_{i}" for i in range(num_items)]
    item_cluster = {item: i % num_clusters for i, item in enumerate(items)}
    cluster_items = {c: [it for it, cl in item_cluster.items() if cl == c] for c in range(num_clusters)}

    # User activity levels
    users = [f"u_{i}" for i in range(num_users)]
    
    # Store user click histories: user_id -> list of (item, timestamp)
    user_histories = {u: [] for u in users}

    train_rows = []
    eval_rows = []
    rowid_counter = 0

    # Generate events for each user over time
    for u in users:
        # Determine number of events for this user (some cold-start with 1 event, some active)
        n_events = random.choices([1, 2, 5, 10, 20], weights=[0.2, 0.2, 0.3, 0.2, 0.1])[0]
        
        # User's primary cluster interest
        fav_cluster = random.randrange(num_clusters)

        # Generate timestamps for events
        # Spread from START_TIME to C3 + DAY
        event_times = sorted([random.randint(START_TIME, C3 + DAY - 3600) for _ in range(n_events)])

        for event_idx, ts in enumerate(event_times):
            event_id = f"ev_{u}_{event_idx}"
            time_bucket = (ts // 21600) % 4
            day_type = 1 if ((ts // 86400) % 7) in (5, 6) else 0

            # Current history of user strictly before this event
            h_items = [h[0] for h in user_histories[u] if h[1] < ts]
            h_ts = [h[1] for h in user_histories[u] if h[1] < ts]

            # 10 candidate items for this event
            # Include items from same cluster as recent history to create co-visitation signal
            last_item = h_items[-1] if h_items else None
            active_cluster = item_cluster[last_item] if last_item else fav_cluster

            # Pick 10 unique candidates
            cluster_candidates = random.sample(cluster_items[active_cluster], min(4, len(cluster_items[active_cluster])))
            other_candidates = random.sample([it for it in items if it not in cluster_candidates], 10 - len(cluster_candidates))
            candidates = cluster_candidates + other_candidates
            random.shuffle(candidates)

            # Determine click probability for each candidate
            # Item in same cluster as recent click has high transition probability
            probs = []
            for it in candidates:
                base_p = 0.05
                if last_item and item_cluster[it] == item_cluster[last_item]:
                    base_p += 0.35  # strong co-visitation signal
                elif item_cluster[it] == fav_cluster:
                    base_p += 0.10
                probs.append(base_p)

            # Pick at most 1 clicked item
            total_p = sum(probs)
            norm_probs = [p / total_p for p in probs]
            clicked_idx = random.choices(range(10), weights=norm_probs)[0] if random.random() < 0.6 else None

            # If clicked, update user history for subsequent events
            if clicked_idx is not None and ts < C3:
                clicked_item = candidates[clicked_idx]
                user_histories[u].append((clicked_item, ts))

            h_items_json = json.dumps(h_items)
            h_ts_json = json.dumps(h_ts)

            # Assign to train or eval
            if ts < C3:
                for idx, it in enumerate(candidates):
                    rowid = f"r_{rowid_counter}"
                    rowid_counter += 1
                    clicked = 1 if idx == clicked_idx else 0
                    train_rows.append((
                        rowid, event_id, u, it, ts, time_bucket, day_type,
                        h_items_json, h_ts_json, clicked
                    ))
            else:
                for idx, it in enumerate(candidates):
                    rowid = f"r_eval_{rowid_counter}"
                    rowid_counter += 1
                    eval_rows.append((
                        rowid, event_id, u, it, ts, time_bucket, day_type,
                        h_items_json, h_ts_json
                    ))

    print(f"Generated {len(train_rows):,} train candidate rows and {len(eval_rows):,} eval candidate rows.")

    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE train (
            rowid VARCHAR,
            event_id VARCHAR,
            user_id VARCHAR,
            item_id VARCHAR,
            timestamp BIGINT,
            time_bucket TINYINT,
            day_type TINYINT,
            history_item_ids VARCHAR,
            history_timestamps VARCHAR,
            clicked TINYINT
        )
    """)

    con.execute("""
        CREATE TABLE eval (
            rowid VARCHAR,
            event_id VARCHAR,
            user_id VARCHAR,
            item_id VARCHAR,
            timestamp BIGINT,
            time_bucket TINYINT,
            day_type TINYINT,
            history_item_ids VARCHAR,
            history_timestamps VARCHAR
        )
    """)

    con.executemany("INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", train_rows)
    con.executemany("INSERT INTO eval VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", eval_rows)

    print(f"Successfully saved to {DB_PATH}")
    print(con.sql("SELECT count(*) as train_rows, count(DISTINCT event_id) as train_events FROM train"))
    print(con.sql("SELECT count(*) as eval_rows, count(DISTINCT event_id) as eval_events FROM eval"))
    con.close()


if __name__ == "__main__":
    generate_synthetic_data()
