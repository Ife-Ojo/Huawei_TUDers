"""Cutoff-safe item embedding similarity features (emb_).

build learns vectors from strictly pre-cutoff training events on every call.
It never loads the old globally fitted Data/item2vec.model artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from notebooks.elvita.train_item2vec import fit_embeddings

PREFIX = 'emb_'
SIM_COLUMNS = ['emb_sim_last1', 'emb_sim_last3_avg', 'emb_sim_last3_max',
               'emb_sim_last5_avg', 'emb_sim_hist_avg', 'emb_sim_hist_max']
COLUMNS = ['rowid', *SIM_COLUMNS, 'emb_has_embedding', 'emb_hist_coverage', 'emb_sim_rank']


def similarity_features(item_id, history, embeddings):
    """Keep unknown items in their original positions and preserve repeats."""
    candidate = embeddings.get(str(item_id))
    sims = np.full(len(history), np.nan)
    known = 0
    for i, item in enumerate(history):
        vector = embeddings.get(str(item))
        if vector is not None:
            known += 1
            if candidate is not None:
                sims[i] = np.clip(np.dot(candidate, vector), -1.0, 1.0)

    def reduce(values, operation):
        values = values[np.isfinite(values)]
        return float(operation(values)) if len(values) else np.nan

    values = [float(sims[-1]) if len(sims) else np.nan,
              reduce(sims[-3:], np.mean), reduce(sims[-3:], np.max),
              reduce(sims[-5:], np.mean), reduce(sims, np.mean), reduce(sims, np.max)]
    return dict(zip(SIM_COLUMNS, values),
                emb_has_embedding=int(candidate is not None),
                emb_hist_coverage=known / len(history) if history else 0.0)


def build(con, cut, end=None, source='train'):
    from common import targets_sql
    embeddings = fit_embeddings(con, cut)
    cursor = con.cursor()
    results = []
    try:
        cursor.execute(f'''SELECT rowid, event_id, item_id, history_item_ids
                           FROM ({targets_sql(cut, end, source)})''')
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for rowid, event, item, history in rows:
                results.append(dict(rowid=rowid, event_id=event,
                                    **similarity_features(item, json.loads(history), embeddings)))
    finally:
        cursor.close()
    if not results:
        return pd.DataFrame({c: pd.Series(dtype='object' if c == 'rowid' else 'float64')
                             for c in COLUMNS})
    out = pd.DataFrame(results)
    out['emb_sim_rank'] = out.groupby('event_id')['emb_sim_hist_avg'].rank(
        ascending=False, method='min', na_option='bottom')
    return out[COLUMNS]


if __name__ == '__main__':
    from common import connect, check_features, C1, C2
    con = connect(full='full' in sys.argv)
    try:
        df = build(con, C1, C2)
        check_features(df, PREFIX)
        print(df.describe().T)
    finally:
        con.close()
