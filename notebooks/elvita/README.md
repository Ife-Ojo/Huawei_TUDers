# Elvita: item embedding features

Changes target `feat/elvita-embeddings`, based on commit
`bf5bd3e0c78dc689d879112f13b2a6a513d88edd`.
Only Elvita's feature module, trainer, documentation and tests are changed.
The earlier starter-solution patch is separate and is not needed for this work.

## Setup and checks

From the repository root, in the team's virtual environment:

```bash
pip install -r requirements-elvita.txt
python -m unittest discover -s tests -p 'test_elvita.py' -v
python features/elvita.py
python features/elvita.py full
python train.py full --groups base,elvita --ablation
```

The last command compares pooled row-level ROC AUC with and without embeddings
using the team's existing chronological folds. Record both fold scores and the
mean before deciding whether to include embeddings. No AUC improvement has yet
been measured for this change.

## Fitting and feature contract

`build(con, cut, end=None, source="train")` returns one row per target rowid and
nine `emb_` feature columns. Existing feature names and missing-value conventions
are preserved. The team's target selection keeps every candidate in each user's
first event in the requested window. Identifiers remain text and are never
parsed as numbers. NaNs represent unavailable features, not submission values;
the downstream model must produce finite probabilities.

Each call fits a new embedding space using only `train.timestamp < cut`.
Evaluation rows and target-window histories never train the embedding space.
The old `Data/item2vec.model` and NumPy files are deliberately ignored: these
were fitted through the official cutoff and leak future clicks into earlier
validation folds. No shared disk cache is used, preventing sample/full or cutoff
mixups; repeated builds will repeat the fitting cost.

The corpus takes each user's latest pre-cutoff event history and appends that
event's clicked candidate, if any. History order and repeated clicks are retained,
and warm-up clicks are included. This uses at most 21 tokens per user and avoids
counting overlapping history snapshots repeatedly. Older clicks beyond that
snapshot are omitted: this is a deliberate bounded-corpus baseline to validate,
not a complete reconstruction of each user's lifetime click sequence.

Embeddings use log co-occurrence counts in a five-click context window and sparse
truncated SVD, with 64 dimensions at most and minimum token count 3. Vectors are
normalized for cosine similarity. This replaces automatic Word2Vec/fallback
selection with one explicit algorithm; it requires SciPy and does not require
gensim. Sparse pair counts can still become large with a large vocabulary. Input
sequences are read in batches; the final feature DataFrame remains in memory, as
required by the team's interface. Performance on the full dataset is unmeasured.

Recent-window features refer to the actual last 1/3/5 clicks. Unknown embeddings
are excluded from each window's average but do not shift its boundary. If the
actual last click is unknown, `emb_sim_last1` is missing. Empty histories and
unknown candidates are supported; an empty corpus returns no known embeddings.

## Optional standalone artifact

```bash
python notebooks/elvita/train_item2vec.py \
  --db Data/techarena.duckdb --cutoff 1737936000 \
  --out Data/item2vec_1737936000.npz
```

This exports vectors, vocabulary and cutoff for inspection only. `build` fits
its own cutoff-specific vectors and does not load this file. Keep generated
artifacts and competition data out of Git.

## Commit

After all tests (including DuckDB integration tests) pass locally and you have
reviewed the feature ablation:

```bash
git diff --check
git add features/elvita.py notebooks/elvita/train_item2vec.py notebooks/elvita/README.md tests/test_elvita.py
git commit -m "Make Elvita embedding features cutoff-safe and preserve history positions"
```

## Combining with the team's other work

Elvita's branch already includes commit c77d465 from feat/eoin-model. Merging
that branch again adds no model code. Marcela's complementary item features are
on marcela/item-features (PR #1). Integrate them on a temporary branch first:

```bash
git fetch origin
git switch feat/elvita-embeddings
git switch -c integration/elvita-marcela
git merge origin/marcela/item-features
python -m pip install -r requirements-elvita.txt lightgbm scikit-learn
python -m unittest discover -s tests -p 'test_elvita.py' -v
python train.py full --groups base --note "baseline comparison"
python train.py full --groups base,marcela --note "item features comparison"
python train.py full --groups base,elvita --note "embedding comparison"
python train.py full --groups base,marcela,elvita --ablation --note "combined feature comparison"
```

Compare both folds on the same full database. In particular, compare the full
model with its `without elvita` ablation to isolate the embedding contribution.
Keep this integration separate from main until results support it. No submission
is needed for these local experiments, so they do not consume the daily limit.

The GitHub Actions workflow installs DuckDB and runs synthetic integration tests
on branch pushes and relevant pull requests. Synthetic checks verify wiring and
cutoff isolation; they do not establish an AUC improvement. The workflow does not
have the competition dataset.
