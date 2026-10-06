"""Reference equivalence, candle mapping, and bootstrap validation."""
import unittest
import numpy as np
import pandas as pd
from HelperFunctions.sci_functions import (
    indicator_matrix, average_uniqueness, sequential_bootstrap,
)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.bars = pd.DatetimeIndex(['2026-10-02 09:30', '2026-10-02 10:30',
                                     '2026-10-02 11:30', '2026-10-05 09:30'])
        self.events = pd.DataFrame({
            'entry_time': self.bars.take([0, 1, 3]),
            't1': self.bars.take([1, 2, 3]),
        }, index=['a', 'b', 'c'])
        self.events.attrs['bar_index'] = tuple(self.bars)

    def test_matrix_and_uniqueness(self):
        matrix = indicator_matrix(self.events)
        np.testing.assert_array_equal(matrix.sparse.to_dense(),
                                      [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(average_uniqueness(matrix), [.75, .75, 1])
        np.testing.assert_allclose(average_uniqueness(matrix[['a', 'a']]), [.5, .5])

    def test_matches_literal_reference_with_repeated_draws(self):
        matrix = indicator_matrix(self.events).sparse.to_dense()
        rng = np.random.default_rng(23)
        expected = []
        for _ in range(30):
            scores = []
            for candidate in matrix.columns:
                subset = matrix[expected + [candidate]]
                u = subset.div(subset.sum(axis=1), axis=0)
                scores.append(u.where(u > 0).mean().iloc[-1])
            p = np.array(scores) / np.sum(scores)
            expected.append(rng.choice(matrix.columns, p=p))
        actual, info = sequential_bootstrap(self.events, 30, 23, diagnostics=True)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, sequential_bootstrap(self.events, 30, 23))
        np.testing.assert_array_equal(info['sample_concurrency'], matrix[actual].sum(axis=1))
        self.assertEqual(len(sequential_bootstrap(self.events, seed=1)), 3)
        self.assertEqual(sequential_bootstrap(self.events, 0), [])

    def test_candle_mapping_and_union(self):
        events = self.events.loc[['a', 'c']].copy()
        events.loc['a', 'entry_time'] += pd.Timedelta(minutes=15)
        matrix = indicator_matrix(events)
        self.assertEqual(list(matrix.index), list(self.bars.take([0, 1, 3])))

    def test_invalid_events_raise(self):
        for end in [pd.NaT, self.bars[0] - pd.Timedelta(hours=1),
                    self.bars[2] + pd.Timedelta(hours=2)]:
            events = self.events.copy()
            events.loc['a', 't1'] = end
            with self.assertRaises(ValueError):
                sequential_bootstrap(events)
        for length in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                sequential_bootstrap(self.events, length)


if __name__ == '__main__':
    unittest.main()
