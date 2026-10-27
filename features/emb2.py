"""Item embeddings v2 + whole-history similarity clues (prefix emb2_).

Same idea as features/elvita.py (candidate similar to what the user clicked), with:
  - PPMI weighting instead of log counts before the SVD
  - clues that use the WHOLE history: similarity to the history average, recency-weighted,
    top-3 mean, and every clue compared with the other 9 candidates of the event
Learns only from train rows before `cut`.

Test it:  python features/emb2.py        (5% sample)
          python features/emb2.py full
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import targets_sql, check_features
from notebooks.elvita.train_item2vec import Sequences

PREFIX = "emb2_"
DIM = 64
WINDOW = 5
MIN_COUNT = 3
CTX_ALPHA = 0.75     # smooths the context counts in PPMI (word2vec's trick)
RECENCY_DECAY = 0.85 # weight of a click k steps back = DECAY ** k


def fit_ppmi_embeddings(con, cut, dim=DIM, window=WINDOW, min_count=MIN_COUNT):
    """Unit-length item vectors from a PPMI co-click matrix + truncated SVD."""
    from collections import Counter
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    seqs = list(Sequences(con, cut))
    counts = Counter(i for s in seqs for i in s)
    vocab = sorted(i for i, c in counts.items() if c >= min_count)
    if len(vocab) < 3:
        return {}
    index = {it: k for k, it in enumerate(vocab)}
    flat = np.array([index.get(i, -1) for s in seqs for i in s], dtype=np.int64)
    sid = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])

    rows, cols, w = [], [], []
    for d in range(1, window + 1):
        a, b = flat[:-d], flat[d:]
        ok = (sid[:-d] == sid[d:]) & (a >= 0) & (b >= 0)
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
        w += [np.full(ok.sum() * 2, 1.0 / d)]       # nearer clicks count more
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    n = len(vocab)
    m = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    m.sum_duplicates()
    m = m.tocoo()
    total = m.data.sum()
    row_sum = np.asarray(m.sum(axis=1)).ravel()
    ctx = np.asarray(m.sum(axis=0)).ravel() ** CTX_ALPHA
    ctx_total = ctx.sum()
    pmi = np.log((m.data / total) / ((row_sum[m.row] / total) * (ctx[m.col] / ctx_total)))
    keep = pmi > 0
    ppmi = coo_matrix((pmi[keep], (m.row[keep], m.col[keep])), shape=(n, n)).tocsr()
    u, s, _ = svds(ppmi, k=min(dim, n - 1), random_state=42)
    vec = u * np.sqrt(s)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = (vec / np.where(norm > 0, norm, 1)).astype(np.float32)
    return dict(zip(vocab, vec))


def event_features(cands, history, emb, dim):
    """Clue table (10 x k) for the candidates of ONE event."""
    n = len(cands)
    C = np.array([emb.get(c, np.full(dim, np.nan, np.float32)) for c in cands])         # n x d
    H = np.array([emb.get(h, np.full(dim, np.nan, np.float32)) for h in history]).reshape(-1, dim)
    out = np.full((n, 10), np.nan)
    has_c = ~np.isnan(C[:, 0])
    known = ~np.isnan(H[:, 0]) if len(H) else np.zeros(0, bool)
    if known.any():
        S = C @ H.T                                       # n x len(history), NaN where unknown
        S[:, ~known] = np.nan
        L = S.shape[1]
        age = np.arange(L - 1, -1, -1)                    # 0 = most recent click
        w = np.where(known, RECENCY_DECAY ** age, 0.0)
        Sz = np.nan_to_num(S)
        with np.errstate(all="ignore"):
            out[:, 0] = np.nanmean(S, axis=1)                                   # mean sim
            out[:, 1] = np.nanmax(S, axis=1)                                    # best single match
            top = -np.sort(-np.where(np.isnan(S), -9, S), axis=1)[:, :3]
            out[:, 2] = np.where(top > -9, top, np.nan).mean(axis=1) if L else np.nan  # top-3 mean
            out[:, 3] = (Sz * w).sum(axis=1) / w.sum()                          # recency-weighted
            out[:, 4] = S[:, -1] if known[-1] else np.nan                       # last click
            cen = np.nansum(H * known[:, None], axis=0)
            nc = np.linalg.norm(cen)
            out[:, 5] = C @ (cen / nc) if nc > 0 else np.nan                    # sim to centroid
        out[~has_c, :6] = np.nan
    out[:, 6] = has_c
    out[:, 7] = known.mean() if len(H) else 0.0           # share of history we have vectors for
    # compare with the other candidates of the event (z-score + rank of mean sim / centroid sim)
    for col, src in ((8, 0), (9, 5)):
        v = out[:, src]
        if np.isfinite(v).sum() > 1:
            out[:, col] = pd.Series(v).rank(ascending=False, method="min", na_option="bottom").to_numpy()
    return out


NAMES = ["mean", "max", "top3", "recency", "last1", "centroid", "has_vec", "hist_coverage",
         "rank_mean", "rank_centroid"]


def build(con, cut, end=None, source="train"):
    emb = fit_ppmi_embeddings(con, cut)
    dim = len(next(iter(emb.values()))) if emb else DIM
    ev = con.sql(f"SELECT rowid, event_id, item_id, history_item_ids "
                 f"FROM ({targets_sql(cut, end, source)}) ORDER BY event_id").df()
    parts = []
    for _, g in ev.groupby("event_id", sort=False):
        feats = event_features(g.item_id.tolist(), json.loads(g.history_item_ids.iloc[0]), emb, dim)
        parts.append(pd.DataFrame(feats, index=g.index, columns=[f"{PREFIX}{n}" for n in NAMES]))
    out = pd.concat(parts).loc[ev.index]
    out.insert(0, "rowid", ev.rowid.values)
    out["event_id"] = ev.event_id.values
    # z-scores within the event (best clue relative to the other 9)
    for c in ("mean", "centroid", "recency"):
        col = f"{PREFIX}{c}"
        g = out.groupby("event_id")[col]
        out[f"{col}_z"] = (out[col] - g.transform("mean")) / (g.transform("std") + 1e-6)
        out[f"{col}_gap"] = out[col] - g.transform("max")
    return out.drop(columns="event_id")


if __name__ == "__main__":
    import time
    from common import connect, C1, C2
    con = connect(full="full" in sys.argv)
    t0 = time.time()
    df = build(con, C1, C2)
    check_features(df, PREFIX)
    print(f"built in {time.time() - t0:.1f}s")
    print(df.describe().T[["count", "mean", "min", "max"]])
