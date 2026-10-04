"""Deterministic tests of triple-barrier outcomes, timing, and overlap."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from HelperFunctions.labeling import (
    estimate_volatility, get_vertical_barriers, create_barriers,
    create_events, get_labels,
)
from HelperFunctions.signals import plot_cumulative_returns


class LabelingTests(unittest.TestCase):
    def fixture(self, values, signals=None, freq='h'):
        index = pd.date_range('2026-01-01', periods=len(values), freq=freq, tz='UTC')
        prices = pd.DataFrame({'MSFT': values}, index=index)
        sides = pd.DataFrame({'MSFT': signals or [1] + [0] * (len(values) - 1)}, index=index)
        vol = pd.DataFrame({'MSFT': [.01] * len(values)}, index=index)
        return prices, sides, vol

    def test_long_and_short_profit_and_stop(self):
        for side, end, expected in [(1, 102, 1), (1, 98, 0), (-1, 98, 1), (-1, 102, 0)]:
            with self.subTest(side=side, end=end):
                p, s, v = self.fixture([100, end], [side, 0])
                events = create_events(p, s, v)
                labels = get_labels(events)
                self.assertEqual(labels.loc[0, 'bin'], expected)
                self.assertEqual(events.loc[0, 't1'], p.index[1])
                self.assertEqual(events.loc[0, 'exit_price'], end)
                self.assertEqual(events.loc[0, 'exit_type'], 'profit_taking' if expected else 'stop_loss')

    def test_time_exit_profitable_losing_flat(self):
        for side, end, expected in [(1, 100.5, 1), (1, 99.5, 0), (1, 100, 0), (-1, 99.5, 1)]:
            p, s, v = self.fixture([100, end], [side, 0], freq='D')
            events = create_events(p, s, v, holding_days=1)
            self.assertEqual(events.loc[0, 'exit_type'], 'time')
            self.assertEqual(get_labels(events).loc[0, 'bin'], expected)

    def test_independent_multipliers_and_frozen_target(self):
        p, s, v = self.fixture([100, 101.5, 102.5, 99])
        v.iloc[1:] = .9
        barriers = create_barriers(p, s, v, pt_mult=2, sl_mult=1)
        self.assertAlmostEqual(barriers.loc[0, 'pt_price'], 102)
        self.assertAlmostEqual(barriers.loc[0, 'sl_price'], 99)
        events = create_events(p, s, v, pt_mult=2, sl_mult=1)
        self.assertEqual(events.loc[0, 't1'], p.index[2])
        self.assertEqual(events.loc[0, 'trgt'], .01)

    def test_overlapping_trades_are_retained(self):
        p, s, v = self.fixture([100, 100.1, 100.2, 102], [1, 1, 1, 0])
        events = create_events(p, s, v)
        self.assertEqual(len(events), 3)
        self.assertTrue(events['t1'].eq(p.index[3]).all())
        self.assertEqual(len(get_labels(events)), 3)
        self.assertTrue(events.index.is_unique)

    def test_deadline_gap_incomplete_and_early_exit(self):
        index = pd.DatetimeIndex(['2026-01-02 16:00', '2026-01-05 10:00'], tz='UTC')
        vertical = get_vertical_barriers(index, index, holding_days=1)
        self.assertEqual(vertical.iloc[0], index[1])
        self.assertTrue(pd.isna(vertical.iloc[1]))
        p, s, v = self.fixture([100, 100.1], [1, 1])
        events = create_events(p, s, v, holding_days=7)
        self.assertTrue(events['status'].eq('incomplete').all())
        self.assertTrue(events['t1'].isna().all())
        self.assertTrue(get_labels(events).empty)
        p.iloc[1] = 102
        events = create_events(p, s, v, holding_days=7)
        self.assertEqual(events.loc[0, 'status'], 'complete')
        self.assertEqual(events.loc[1, 'status'], 'incomplete')

    def test_volatility_is_causal_and_frequency_agnostic(self):
        p, _, _ = self.fixture([100, 101, 99, 102, 100, 103])
        vol = estimate_volatility(p, span=4, min_periods=2)
        expected = p.pct_change(fill_method=None).ewm(span=4, min_periods=2).std()
        pd.testing.assert_frame_equal(vol, expected)
        changed = p.copy()
        changed.iloc[4:] *= 5
        pd.testing.assert_frame_equal(vol.iloc[:4], estimate_volatility(changed, 4).iloc[:4])
        daily = p.copy()
        daily.index = pd.date_range('2026-01-01', periods=len(p), freq='D', tz='UTC')
        np.testing.assert_allclose(vol, estimate_volatility(daily, 4), equal_nan=True)

    def test_warmup_zero_targets_empty_signals_and_multiple_tickers(self):
        p, s, v = self.fixture([100, 101, 102], [1, 1, 0])
        v.iloc[0] = np.nan
        v.iloc[1] = 0
        events = create_events(p, s, v)
        self.assertTrue(events['status'].eq('excluded').all())
        self.assertTrue(get_labels(events).empty)
        events = create_events(p, s * 0, v)
        self.assertTrue(events.empty)
        self.assertTrue(get_labels(events).empty)
        p['AAPL'] = p['MSFT']
        s['AAPL'] = -s['MSFT']
        v['AAPL'] = .01
        events = create_events(p, s, v)
        self.assertEqual(len(events), 4)
        self.assertEqual(events['entry_time'].nunique(), 2)

    def test_subsequent_close_only_and_ties(self):
        p, s, v = self.fixture([100, 102, 99], [1, 0, 0], freq='D')
        events = create_events(p, s, v, holding_days=1)
        self.assertEqual(events.loc[0, 'exit_type'], 'profit_taking')
        self.assertEqual(events.loc[0, 't1'], p.index[1])
        self.assertTrue(pd.isna(events.loc[0, 'sl_time']))

    def test_threaded_matches_serial_with_multiple_tickers(self):
        p, s, v = self.fixture([100, 100.1, 102, 99], [1, -1, 1, 0])
        p['AAPL'], s['AAPL'], v['AAPL'] = p['MSFT'], -s['MSFT'], v['MSFT']
        serial = create_events(p, s, v, max_workers=1)
        threaded = create_events(p, s, v, max_workers=4)
        pd.testing.assert_frame_equal(serial, threaded)
        pd.testing.assert_frame_equal(get_labels(serial, max_workers=1),
                                      get_labels(threaded, max_workers=4))
        pd.testing.assert_frame_equal(estimate_volatility(p, max_workers=1),
                                      estimate_volatility(p, max_workers=4))

    def test_cumulative_curve_starts_at_one_and_compounds_by_exit(self):
        index = pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC')
        events = pd.DataFrame({
            'ticker': ['MSFT', 'MSFT', 'MSFT', 'AAPL', 'MSFT'],
            'entry_time': [index[0]] * 5, 'entry_price': [100.] * 5,
            'side': [1] * 5,
            't1': [index[2], index[1], index[2], index[3], pd.NaT],
            'exit_price': [110., 90., 120., 105., np.nan],
            'exit_type': ['time'] * 5,
            'status': ['complete'] * 4 + ['incomplete'],
        })
        with patch.object(plt, 'show'):
            curve = plot_cumulative_returns(events, max_workers=2)
        self.assertTrue(curve.iloc[0].eq(1).all())
        self.assertAlmostEqual(curve.at[index[1], 'MSFT'], .9)
        self.assertAlmostEqual(curve.at[index[2], 'MSFT'], .9 * 1.1 * 1.2)
        self.assertAlmostEqual(curve['AAPL'].iloc[-1], 1.05)
        self.assertEqual(len(plt.get_fignums()), 2)
        for figure in plt.get_fignums():
            self.assertIsNone(plt.figure(figure).axes[0].get_legend())
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
