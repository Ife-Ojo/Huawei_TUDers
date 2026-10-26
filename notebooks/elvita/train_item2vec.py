"""
Train Item2Vec (Word2Vec on click sequences) and save embeddings.
Run:  python notebooks/elvita/train_item2vec.py

This extracts every user's click history from the training data, treats each
history as a "sentence" of item_ids, and trains a Word2Vec model.

Saves:
  Data/item2vec.model    (gensim model, reloaded by features/elvita.py)
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
import duckdb
from common import FULL_DB, SAMPLE_DB, C3

# ---- Settings ----
VECTOR_DIM = 64
WINDOW = 5
MIN_COUNT = 3
EPOCHS = 10
WORKERS = 4
MODEL_PATH = PROJECT / "Data" / "item2vec.model"


def extract_sequences(db_path, cutoff):
    """Pull click sequences from all training histories before the cutoff."""
    print(f"Extracting sequences from {db_path.name} ...")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET memory_limit='8GB'")

    # Method 1: histories from all events (these include warm-up clicks)
    # Each user's history_item_ids is already a chronological click sequence
    rows = con.sql(f"""
        SELECT DISTINCT ON (user_id, history_item_ids)
            user_id, history_item_ids
        FROM train
        WHERE timestamp < {cutoff}
          AND history_item_ids != '[]'
        ORDER BY user_id, history_item_ids, timestamp DESC
    """).fetchall()

    # Method 2: also build sequences from clicked impressions
    click_rows = con.sql(f"""
        SELECT user_id, item_id, timestamp
        FROM train
        WHERE clicked = 1 AND timestamp < {cutoff}
        ORDER BY user_id, timestamp
    """).fetchall()
    con.close()

    # Parse history sequences
    sequences = []
    for user_id, hist_json in rows:
        items = json.loads(hist_json)
        if len(items) >= 2:
            sequences.append(items)

    # Build sequences from clicked impressions (grouped by user)
    user_clicks = defaultdict(list)
    for user_id, item_id, ts in click_rows:
        user_clicks[user_id].append(item_id)
    for uid, items in user_clicks.items():
        if len(items) >= 2:
            sequences.append(items)

    print(f"  {len(sequences):,} sequences, "
          f"{sum(len(s) for s in sequences):,} total tokens")
    return sequences


def train_with_gensim(sequences):
    """Train using gensim Word2Vec."""
    from gensim.models import Word2Vec
    print(f"Training Word2Vec (dim={VECTOR_DIM}, window={WINDOW}, epochs={EPOCHS}) ...")
    model = Word2Vec(
        sentences=sequences,
        vector_size=VECTOR_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,           # skip-gram
        workers=WORKERS,
        epochs=EPOCHS,
        seed=42,
    )
    print(f"  Vocabulary: {len(model.wv):,} items")
    model.save(str(MODEL_PATH))
    print(f"  Saved to {MODEL_PATH.name}")
    return model


def train_manual(sequences):
    """Fallback: build co-occurrence matrix and use SVD (no gensim needed)."""
    print("Gensim not available, using SVD fallback ...")
    from collections import Counter

    # Build vocabulary
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = [item for item, count in item_counts.items() if count >= MIN_COUNT]
    item_to_idx = {item: i for i, item in enumerate(vocab)}
    n = len(vocab)
    print(f"  Vocabulary: {n:,} items")

    # Build co-occurrence matrix (sparse-ish using dict)
    cooc = defaultdict(float)
    for seq in sequences:
        indices = [item_to_idx[item] for item in seq if item in item_to_idx]
        for i, idx_i in enumerate(indices):
            for j in range(max(0, i - WINDOW), min(len(indices), i + WINDOW + 1)):
                if i != j:
                    cooc[(idx_i, indices[j])] += 1.0

    # Build sparse matrix and do truncated SVD
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    row_idx, col_idx, vals = [], [], []
    for (r, c), v in cooc.items():
        row_idx.append(r)
        col_idx.append(c)
        vals.append(np.log1p(v))
    M = csr_matrix((vals, (row_idx, col_idx)), shape=(n, n))

    print(f"  Running SVD (dim={VECTOR_DIM}) ...")
    U, S, _ = svds(M, k=min(VECTOR_DIM, n - 1))
    embeddings = U * np.sqrt(S)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Save as numpy arrays
    np.save(str(MODEL_PATH) + ".vectors.npy", embeddings)
    np.save(str(MODEL_PATH) + ".vocab.npy", np.array(vocab))
    print(f"  Saved to {MODEL_PATH.name}.vectors.npy and .vocab.npy")
    return vocab, embeddings


if __name__ == "__main__":
    t0 = time.time()

    # Use full DB if available, otherwise sample
    if FULL_DB.exists():
        db = FULL_DB
    elif SAMPLE_DB.exists():
        db = SAMPLE_DB
        print("WARNING: Using sample DB. Train on full for better embeddings.")
    else:
        raise SystemExit("No database found. Run build_db_from_parquet.py first.")

    sequences = extract_sequences(db, C3)

    try:
        import gensim
        train_with_gensim(sequences)
    except ImportError:
        print("gensim not installed. Trying SVD fallback (pip install scipy) ...")
        train_manual(sequences)

    print(f"\nDone in {time.time() - t0:.0f}s")
