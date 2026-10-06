"""Verify event alignment and exclusion of future daily observations."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from HelperFunctions.features import event_features


class FeatureTests(unittest.TestCase):
    def hourly_fixture(self):
        days = pd.bdate_range("2026-01-05", periods=30)
        times = pd.DatetimeIndex([
            day.tz_localize("America/New_York") + pd.Timedelta(hours=hour, minutes=30)
            for day in days for hour in range(9, 16)
        ])
        i = np.arange(len(times))
        bars = pd.DataFrame({"Close": 100 * np.exp(.001 * np.sin(i / 3) + .0001 * i),
                             "Open": 100., "Volume": 1000. + 10 * i}, index=times)
        daily_days = pd.bdate_range("2025-08-01", end=days[-1])
        daily = pd.DataFrame({"Close": 100. + np.arange(len(daily_days))}, index=daily_days)
        events = pd.DataFrame({"ticker": "MSFT", "signal_time": times},
                              index=pd.Index(i, name="event_id"))
        return dict(hourly={"MSFT": bars}, daily={"MSFT": daily}, spy_daily=daily,
                    events=events, volatility=pd.DataFrame(.01, index=times, columns=["MSFT"]),
                    rsi=pd.DataFrame(25., index=times, columns=["MSFT"]))

    def test_volatility_warmup_and_exact_values(self):
        args = self.hourly_fixture()
        features, _ = event_features(**args)
        ratio = features["volatility_ratio20_100"]
        # 100 valid returns need 101 closing prices: position 100 is first valid.
        self.assertTrue(ratio.iloc[:100].isna().all())
        self.assertTrue(ratio.iloc[100:].notna().all())
        returns = args["hourly"]["MSFT"].Close.pct_change(fill_method=None)
        short = returns.ewm(span=20, min_periods=20, adjust=True).std(bias=False)
        long = returns.ewm(span=100, min_periods=100, adjust=True).std(bias=False)
        np.testing.assert_allclose(ratio.to_numpy(), (short / long).to_numpy(), equal_nan=True)

    def test_relative_volume_warmup_and_same_slot_baseline(self):
        args = self.hourly_fixture()
        features, _ = event_features(**args)
        ratio = features["relative_volume20_sessions"]
        # Seven bars per session: the 21st session is the first with 20 predecessors.
        self.assertTrue(ratio.iloc[:140].isna().all())
        self.assertTrue(ratio.iloc[140:].notna().all())
        volume = args["hourly"]["MSFT"].Volume.to_numpy()
        for position in (140, 143, 146, 147, 209):
            prior_positions = position - 7 * np.arange(1, 21)
            self.assertAlmostEqual(ratio.iloc[position],
                                   volume[position] / volume[prior_positions].mean())

    def test_hourly_features_exclude_future_observations(self):
        args = self.hourly_fixture()
        before, _ = event_features(**args)
        changed = args["hourly"]["MSFT"].copy()
        changed.iloc[151:, changed.columns.get_loc("Close")] *= 10
        changed.iloc[151:, changed.columns.get_loc("Volume")] *= 100
        args["hourly"] = {"MSFT": changed}
        after, _ = event_features(**args)
        pd.testing.assert_frame_equal(before.iloc[:151], after.iloc[:151])

    def test_zero_denominators_remain_missing(self):
        args = self.hourly_fixture()
        args["hourly"]["MSFT"]["Close"] = 100.
        args["hourly"]["MSFT"]["Volume"] = 0.
        features, _ = event_features(**args)
        self.assertTrue(features["volatility_ratio20_100"].isna().all())
        self.assertTrue(features["relative_volume20_sessions"].isna().all())
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_previous_daily_values_and_event_ids(self):
        days = pd.bdate_range("2026-01-01", periods=80)
        daily = pd.DataFrame({"Close": np.arange(100., 180.)}, index=days)
        times = pd.DatetimeIndex([day.tz_localize("America/New_York") +
                                  pd.Timedelta(hours=9, minutes=30) for day in days])
        bars = pd.DataFrame({"Close": np.arange(100., 180.),
                             "Open": np.arange(99., 179.),
                             "Volume": np.arange(1000., 1080.)}, index=times)
        events = pd.DataFrame({"ticker": ["MSFT", "MSFT", "AAPL"],
                               "signal_time": [times[60], times[70], times[60]]},
                              index=pd.Index([9, 2, 15], name="event_id"))
        vol = pd.DataFrame(.01, index=times, columns=["MSFT", "AAPL"])
        rsi = pd.DataFrame(25., index=times, columns=vol.columns)
        args = dict(hourly={"MSFT": bars, "AAPL": bars},
                    daily={"MSFT": daily, "AAPL": daily}, spy_daily=daily,
                    events=events, volatility=vol, rsi=rsi)
        features, metadata = event_features(**args)
        self.assertTrue(features.index.equals(events.index))
        self.assertTrue(metadata.index.equals(events.index))
        self.assertEqual(features.shape, (3, 17))
        self.assertAlmostEqual(features.loc[9, "spy_return_3_daily_sessions_pct"],
                               (159 / 156 - 1) * 100)
        self.assertAlmostEqual(features.loc[9, "distance_daily_sma50_pct"],
                               (160 / daily.Close.iloc[10:60].mean() - 1) * 100)
        self.assertAlmostEqual(features.loc[9, "relative_volume20_sessions"],
                               1060 / np.arange(1040., 1060.).mean())
        self.assertNotIn("rsi14_threshold_distance", features.columns)
        changed = daily.copy()
        changed.iloc[60:, 0] = 9999
        args.update(daily={"MSFT": changed, "AAPL": changed}, spy_daily=changed)
        after, _ = event_features(**args)
        pd.testing.assert_series_equal(features.loc[9], after.loc[9])
        self.assertNotIn("side", features.columns)


if __name__ == "__main__":
    unittest.main()
