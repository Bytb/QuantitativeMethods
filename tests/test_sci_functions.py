"""Check per-ticker normality diagnostics and optional plotting."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from HelperFunctions.sci_functions import return_normality


class NormalityTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_multiple_tickers_and_conversion(self):
        rng = np.random.default_rng(42)
        prices = pd.DataFrame(100 * np.cumprod(
            1 + rng.normal(0, .01, (200, 2)), axis=0), columns=["MSFT", "AAPL"])
        returns = prices.pct_change(fill_method=None) * 100
        returns.loc[10, "AAPL"] = np.nan
        summary = return_normality(returns, "pct_change", False, False)
        self.assertEqual(summary.loc["MSFT", "n_obs"], 199)
        self.assertEqual(summary.loc["AAPL", "n_obs"], 198)
        expected = stats.anderson(returns["MSFT"].dropna(), dist="norm")
        self.assertAlmostEqual(summary.loc["MSFT", "statistic"], expected.statistic)
        self.assertEqual(summary.loc["MSFT", "reject_normality"],
                         expected.statistic > expected.critical_values[2])
        converted = return_normality(prices, plot_qq=False, plot_distribution=False)
        self.assertAlmostEqual(converted.loc["MSFT", "statistic"], expected.statistic)
        self.assertEqual(plt.get_fignums(), [])

    def test_invalid_samples_do_not_block_valid_tickers(self):
        values = pd.DataFrame({"valid": np.arange(20), "constant": np.ones(20),
                               "missing": [np.nan] * 20})
        result = return_normality(values, "pct_change", False, False)
        self.assertEqual(result.loc["valid", "status"], "ok")
        self.assertEqual(result.loc["constant", "status"], "constant returns")
        self.assertTrue(pd.isna(result.loc["missing", "reject_normality"]))

    def test_independent_plot_switches(self):
        values = pd.DataFrame({"MSFT": np.linspace(-2, 2, 30)})
        for qq, histogram, count in [(True, False, 1), (False, True, 1), (True, True, 2)]:
            with patch("matplotlib.pyplot.show") as show:
                return_normality(values, "pct_change", qq, histogram)
                self.assertEqual(show.call_count, count)
            plt.close("all")

    def test_no_forward_fill_across_missing_prices(self):
        prices = pd.DataFrame({"MSFT": [100, 101, np.nan, 102, 103, 104, 105]})
        result = return_normality(prices, plot_qq=False, plot_distribution=False)
        self.assertEqual(result.loc["MSFT", "n_obs"], 4)


if __name__ == "__main__":
    unittest.main()
