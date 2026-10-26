"""Fit cutoff-specific item embeddings from the latest known click sequences.

Sparse log co-occurrence + truncated SVD, using numpy/scipy and DuckDB.
No evaluation rows or events at/after the cutoff are used to fit vectors.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


class Sequences:
    """Restartable, batched corpus: one latest pre-cutoff snapshot per user.

    Latest histories contain up to 20 clicks, including warm-up clicks. Append
    the current clicked candidate, if any, because that event is also in fit.
    Avoid replaying overlapping histories once for each candidate/event.
    """
    def __init__(self, con, cutoff):
        self.con, self.cutoff = con, int(cutoff)

    def __iter__(self):
        import json
        cursor = self.con.cursor()
        try:
            cursor.execute('''
                SELECT history_item_ids, clicked_item FROM (
                    SELECT user_id, event_id, timestamp, history_item_ids,
                           max(CASE WHEN clicked = 1 THEN item_id END) AS clicked_item
                    FROM train WHERE timestamp < ?
                    GROUP BY user_id, event_id, timestamp, history_item_ids
                    QUALIFY row_number() OVER (
                        PARTITION BY user_id ORDER BY timestamp DESC, event_id DESC
                    ) = 1
                ) ORDER BY user_id
            ''', [self.cutoff])
            while True:
                rows = cursor.fetchmany(4096)
                if not rows:
                    break
                for history, clicked in rows:
                    sequence = [str(item) for item in json.loads(history)]
                    if clicked is not None:
                        sequence.append(str(clicked))
                    if len(sequence) >= 2:
                        yield sequence
        finally:
            cursor.close()


def fit_embeddings(con, cutoff, vector_dim=64, window=5, min_count=3):
    """Learn only from train.timestamp < cutoff; return normalized vectors."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    corpus = Sequences(con, cutoff)
    counts = Counter(item for sequence in corpus for item in sequence)
    vocab = sorted(item for item, count in counts.items() if count >= min_count)
    index = {item: i for i, item in enumerate(vocab)}
    if len(vocab) < 2:
        return {}
    pairs = defaultdict(float)
    for sequence in corpus:
        # Keep original positions: removing OOVs would change context distances.
        for i, item in enumerate(sequence):
            if item not in index:
                continue
            for j in range(max(0, i - window), min(len(sequence), i + window + 1)):
                other = sequence[j]
                if i != j and other in index:
                    pairs[index[item], index[other]] += 1.0
    if not pairs:
        return {}
    row, col = zip(*pairs)
    matrix = coo_matrix((np.log1p(list(pairs.values())), (row, col)),
                        shape=(len(vocab), len(vocab))).tocsr()
    u, singular, _ = svds(matrix, k=min(vector_dim, len(vocab) - 1), random_state=42)
    vectors = u * np.sqrt(singular)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1)
    return dict(zip(vocab, vectors.astype(np.float32)))


if __name__ == '__main__':
    import duckdb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, required=True, help='Exclusive Unix-second fitting cutoff')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    con = duckdb.connect(str(args.db), read_only=True)
    try:
        embeddings = fit_embeddings(con, args.cutoff)
    finally:
        con.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cutoff=args.cutoff,
                        vocab=np.asarray(list(embeddings), dtype=str),
                        vectors=np.asarray(list(embeddings.values()), dtype=np.float32))
    print(f'Saved {len(embeddings):,} embeddings to {args.out}')
