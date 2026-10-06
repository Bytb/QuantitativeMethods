import unittest
import numpy as np
import pandas as pd
from HelperFunctions.sci_functions import sample_weights


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.bars = pd.date_range('2026-10-01 09:30', periods=5, freq='h')
        self.close = pd.Series(np.exp([0., .1, .3, .2, .6]), index=self.bars)
        self.events = pd.DataFrame({
            'entry_time': self.bars.take([0, 1, 3]),
            't1': self.bars.take([2, 3, 4]),
        }, index=['a', 'b', 'c'])
        self.labels = pd.Series([0, 0, 1], index=self.events.index)

    def test_hand_calculation_and_alignment(self):
        result = sample_weights(self.events, self.close, self.labels[::-1])
        # Concurrency [1,2,2,2,1]; entry returns excluded.
        np.testing.assert_allclose(result.attributed_return, [.15, .05, .4])
        np.testing.assert_allclose(result.average_uniqueness, [2/3, 1/2, 3/4])
        np.testing.assert_allclose(result.class_factor, [.75, .75, 1.5])
        expected_time = .5 + .5 * np.cumsum([2/3, 1/2, 3/4]) / (23/12)
        np.testing.assert_allclose(result.time_factor, expected_time)
        self.assertAlmostEqual(result.return_weight.sum(), 3)
        np.testing.assert_allclose(result.final_weight,
                                  result.return_weight * result.time_factor * result.class_factor)
        normalized = sample_weights(self.events, self.close, self.labels, normalize_final=True)
        self.assertAlmostEqual(normalized.final_weight.mean(), 1)
        np.testing.assert_allclose(normalized.final_weight / result.final_weight,
                                  3 / result.final_weight.sum())

    def test_cancel_returns_before_absolute_and_zero_event(self):
        close = pd.Series(np.exp([0., .1, -.1, 0., .4]), index=self.bars)
        result = sample_weights(self.events, close, self.labels, decay=1, class_weight=None)
        self.assertAlmostEqual(result.attributed_return.iloc[0], -.05)
        self.assertAlmostEqual(result.absolute_attributed_return.iloc[0], .05)
        events = self.events.copy()
        events.loc['c', 't1'] = events.loc['c', 'entry_time']
        result = sample_weights(events, close, self.labels)
        self.assertEqual(result.final_weight.loc['c'], 0)

    def test_chronology_negative_decay_and_errors(self):
        shuffled = self.events.iloc[[2, 0, 1]]
        result = sample_weights(shuffled, self.close, self.labels, decay=-.5)
        self.assertEqual(result.time_factor.loc['a'], 0)
        self.assertEqual(result.time_factor.loc['c'], 1)
        self.assertEqual(list(result.index), list(shuffled.index))
        for kwargs in [dict(decay=-1), dict(decay=2), dict(normalize_final=1),
                       dict(class_weight='wrong')]:
            with self.assertRaises(ValueError):
                sample_weights(self.events, self.close, self.labels, **kwargs)
        with self.assertRaises(ValueError):
            sample_weights(self.events, self.close * 0 + 1, self.labels)
        with self.assertRaises(ValueError):
            sample_weights(self.events, self.close, self.labels.iloc[:2])


if __name__ == '__main__':
    unittest.main()
