"""
Elvita's clues: item embedding similarity features.   (prefix emb_)

For each candidate row, computes how similar the candidate item is to the
user's recent click history using Item2Vec embeddings.

Features:
  emb_sim_last1        cosine similarity to user's last clicked item
  emb_sim_last3_avg    average similarity to last 3 clicks
  emb_sim_last3_max    max similarity to last 3 clicks
  emb_sim_last5_avg    average similarity to last 5 clicks
  emb_sim_hist_avg     average similarity to full history
  emb_sim_hist_max     max similarity to full history
  emb_has_embedding    1 if the candidate item has an embedding, 0 otherwise
  emb_hist_coverage    fraction of history items that have embeddings
  emb_sim_rank         rank of this candidate's similarity among 10 in the event

Test:  python features/elvita.py
       python features/elvita.py full
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from common import targets_sql, check_features

PREFIX = "emb_"
MODEL_PATH = PROJECT / "Data" / "item2vec.model"
VECTORS_NPY = PROJECT / "Data" / "item2vec.model.vectors.npy"
VOCAB_NPY = PROJECT / "Data" / "item2vec.model.vocab.npy"


def _load_embeddings():
    """Load item embeddings from gensim model or SVD fallback files."""
    if MODEL_PATH.exists():
        try:
            from gensim.models import Word2Vec
            model = Word2Vec.load(str(MODEL_PATH))
            vocab = list(model.wv.key_to_index.keys())
            vectors = np.array([model.wv[item] for item in vocab])
            # Normalize for cosine similarity via dot product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
            return {item: vectors[i] for i, item in enumerate(vocab)}
        except ImportError:
            pass

    if VECTORS_NPY.exists() and VOCAB_NPY.exists():
        vectors = np.load(str(VECTORS_NPY))
        vocab = np.load(str(VOCAB_NPY), allow_pickle=True)
        return {item: vectors[i] for i, item in enumerate(vocab)}

    raise SystemExit(
        "No embeddings found. Run:  python notebooks/elvita/train_item2vec.py"
    )


def _cosine_sims(candidate_vec, history_vecs):
    """Cosine similarity between one candidate and a list of history vectors.
    Vectors are already normalized, so dot product = cosine similarity."""
    if len(history_vecs) == 0:
        return np.array([])
    return np.dot(np.array(history_vecs), candidate_vec)


def build(con, cut, end=None, source="train"):
    embeddings = _load_embeddings()

    # Get the rows we need to make features for
    df = con.sql(f"""
        SELECT rowid, event_id, item_id, history_item_ids
        FROM ({targets_sql(cut, end, source)})
    """).df()

    results = []
    for _, row in df.iterrows():
        item_id = row["item_id"]
        hist_items = json.loads(row["history_item_ids"]) if row["history_item_ids"] else []

        # Get candidate embedding
        cand_vec = embeddings.get(item_id)
        has_emb = 1 if cand_vec is not None else 0

        # Get history embeddings
        hist_vecs = []
        for h_item in hist_items:
            v = embeddings.get(h_item)
            if v is not None:
                hist_vecs.append(v)

        hist_coverage = len(hist_vecs) / max(len(hist_items), 1)

        if cand_vec is not None and len(hist_vecs) > 0:
            sims = _cosine_sims(cand_vec, hist_vecs)

            # Last N clicks (history is oldest-to-newest, so last = most recent)
            sim_last1 = sims[-1] if len(sims) >= 1 else np.nan
            sim_last3_avg = np.mean(sims[-3:]) if len(sims) >= 1 else np.nan
            sim_last3_max = np.max(sims[-3:]) if len(sims) >= 1 else np.nan
            sim_last5_avg = np.mean(sims[-5:]) if len(sims) >= 1 else np.nan
            sim_hist_avg = np.mean(sims)
            sim_hist_max = np.max(sims)
        else:
            sim_last1 = np.nan
            sim_last3_avg = np.nan
            sim_last3_max = np.nan
            sim_last5_avg = np.nan
            sim_hist_avg = np.nan
            sim_hist_max = np.nan

        results.append({
            "rowid": row["rowid"],
            "event_id": row["event_id"],
            "emb_sim_last1": sim_last1,
            "emb_sim_last3_avg": sim_last3_avg,
            "emb_sim_last3_max": sim_last3_max,
            "emb_sim_last5_avg": sim_last5_avg,
            "emb_sim_hist_avg": sim_hist_avg,
            "emb_sim_hist_max": sim_hist_max,
            "emb_has_embedding": has_emb,
            "emb_hist_coverage": hist_coverage,
        })

    out = pd.DataFrame(results)

    # Within-event ranking: how does this candidate compare to the other 9?
    out["emb_sim_rank"] = out.groupby("event_id")["emb_sim_hist_avg"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(float)

    out = out.drop(columns=["event_id"])
    return out


if __name__ == "__main__":
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    t0 = time.time()
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(f"Built in {time.time() - t0:.1f}s")
    print(df.describe().T[["count", "mean", "min", "max"]])
