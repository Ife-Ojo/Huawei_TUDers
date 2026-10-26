import json
import unittest

try:
    import duckdb
except ImportError:
    duckdb = None
import numpy as np

from features.elvita import build, similarity_features, COLUMNS
from notebooks.elvita.train_item2vec import Sequences, fit_embeddings


@unittest.skipIf(duckdb is None, "DuckDB is not installed")
class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute('''CREATE TABLE train (
            rowid VARCHAR, event_id VARCHAR, user_id VARCHAR, item_id VARCHAR,
            timestamp BIGINT, history_item_ids VARCHAR, history_timestamps VARCHAR,
            clicked INTEGER)''')
        self.event('early', 'u', 10, ['a', 'b', 'a'], 'b')
        self.event('target', 'u', 20, ['a', 'b', 'a', 'b'], None)
        self.event('cold', 'new', 20, [], None)

    def tearDown(self):
        self.con.close()

    def event(self, event, user, timestamp, history, clicked):
        for i, item in enumerate(['a', 'b'] + [f'oov{i}' for i in range(8)]):
            self.con.execute('INSERT INTO train VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                             [f'{event}_{i}', event, user, item, timestamp,
                              json.dumps(history), json.dumps(list(range(len(history)))),
                              int(item == clicked)])

    def test_positions_repeats_and_cold_start(self):
        vectors = {'a': np.array([1., 0.]), 'b': np.array([0., 1.])}
        result = similarity_features('a', ['a', 'a', 'b', 'unknown'], vectors)
        self.assertTrue(np.isnan(result['emb_sim_last1']))
        self.assertEqual(result['emb_sim_last3_avg'], 0.5)
        self.assertAlmostEqual(result['emb_sim_hist_avg'], 2 / 3)
        self.assertEqual(result['emb_hist_coverage'], .75)
        self.assertEqual(similarity_features('unknown', [], vectors)['emb_has_embedding'], 0)

    def test_cutoff_isolation_and_sequences(self):
        self.assertEqual(list(Sequences(self.con, 20)), [['a', 'b', 'a', 'b']])
        before = fit_embeddings(self.con, 20, min_count=1)
        self.event('future', 'u', 30, ['future', 'a', 'future'], 'a')
        after = fit_embeddings(self.con, 20, min_count=1)
        self.assertNotIn('future', after)
        self.assertEqual(set(before), set(after))
        for item in before:
            np.testing.assert_allclose(before[item], after[item])
            self.assertAlmostEqual(float(np.linalg.norm(before[item])), 1., places=5)

    def test_build_contract_and_empty_targets(self):
        output = build(self.con, 20, 30)
        self.assertEqual(list(output.columns), COLUMNS)
        self.assertEqual(len(output), 20)
        self.assertTrue(output.rowid.is_unique)
        self.assertTrue((output.filter(like='emb_').columns.str.startswith('emb_')).all())
        self.assertFalse(np.isinf(output.select_dtypes('number').to_numpy()).any())
        self.assertEqual(list(build(self.con, 100, 200).columns), COLUMNS)
        self.assertTrue(build(self.con, 100, 200).empty)

    def test_base_and_embedding_join(self):
        # Mirrors train.make_table's rowid join, without requiring LightGBM.
        from features.base import build as build_base
        self.con.execute('ALTER TABLE train ADD COLUMN time_bucket INTEGER DEFAULT 0')
        self.con.execute('ALTER TABLE train ADD COLUMN day_type INTEGER DEFAULT 0')
        base = build_base(self.con, 20, 30)
        embedding = build(self.con, 20, 30)
        combined = base.merge(embedding, on='rowid', how='left', validate='one_to_one')
        self.assertEqual(len(combined), 20)
        self.assertEqual(set(combined.rowid), set(embedding.rowid))
        self.assertTrue(any(c.startswith('base_') for c in combined.columns))
        self.assertTrue(any(c.startswith('emb_') for c in combined.columns))

    def test_evaluation_needs_no_labels(self):
        self.con.execute('CREATE TABLE eval AS SELECT * EXCLUDE (clicked) FROM train WHERE timestamp = 20')
        output = build(self.con, 20, source='eval')
        self.assertEqual(len(output), 20)
        self.assertEqual(set(output.rowid),
                         {r[0] for r in self.con.execute('SELECT rowid FROM eval').fetchall()})

    def test_tiny_corpus(self):
        self.assertEqual(fit_embeddings(self.con, 0), {})


class PureEmbeddingTests(unittest.TestCase):
    test_positions_repeats_and_cold_start = EmbeddingTests.test_positions_repeats_and_cold_start

    def test_sparse_svd_and_tiny_corpus(self):
        from unittest.mock import patch
        with patch('notebooks.elvita.train_item2vec.Sequences',
                   return_value=[['a', 'b', 'a', 'b'], ['a', 'c', 'b']]):
            vectors = fit_embeddings(None, 20, min_count=1)
        self.assertEqual(set(vectors), {'a', 'b', 'c'})
        for vector in vectors.values():
            self.assertTrue(np.isfinite(vector).all())
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1., places=5)
        with patch('notebooks.elvita.train_item2vec.Sequences', return_value=[]):
            self.assertEqual(fit_embeddings(None, 20), {})


if __name__ == '__main__':
    unittest.main()
