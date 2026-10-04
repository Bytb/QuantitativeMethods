"""Technical signals calculated from DataFrames of closing prices."""
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
import os

import numpy as np
import pandas as pd


def _rsi_column(task):
    values, periods = task
    result = np.full(len(values), np.nan)
    if len(values) <= periods:
        return result
    changes = np.diff(values)
    gains = np.maximum(changes, 0)
    losses = np.maximum(-changes, 0)
    gain = gains[:periods].mean()
    loss = losses[:periods].mean()
    for i in range(periods, len(values)):
        if i > periods:
            gain = ((periods - 1) * gain + gains[i - 1]) / periods
            loss = ((periods - 1) * loss + losses[i - 1]) / periods
        result[i] = (50.0 if gain == 0 else 100.0) if loss == 0 else (
            100.0 - 100.0 / (1.0 + gain / loss))
    return result


def _plot_rsi(rsi, periods):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection

    is_dates = isinstance(rsi.index, pd.DatetimeIndex)
    x = mdates.date2num(rsi.index.to_pydatetime()) if is_dates else np.arange(len(rsi))
    for ticker in rsi.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        y = rsi[ticker].to_numpy()
        segments, colors = [], []
        for i in range(1, len(y)):
            if not np.isfinite(y[i - 1:i + 1]).all():
                continue
            x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
            # Split at thresholds so colors change exactly at 30 and 70.
            cuts = [0.0, 1.0]
            if y1 != y0:
                cuts += [(level - y0) / (y1 - y0) for level in (30, 70)
                         if 0 < (level - y0) / (y1 - y0) < 1]
            cuts.sort()
            for a, b in zip(cuts[:-1], cuts[1:]):
                segments.append([(x0 + a * (x1 - x0), y0 + a * (y1 - y0)),
                                 (x0 + b * (x1 - x0), y0 + b * (y1 - y0))])
                midpoint = y0 + (a + b) / 2 * (y1 - y0)
                colors.append('red' if midpoint > 70 else
                              'royalblue' if midpoint < 30 else 'dimgray')
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.3))
        ax.axhspan(70, 100, color='red', alpha=0.06)
        ax.axhspan(0, 30, color='royalblue', alpha=0.06)
        ax.axhline(70, color='red', linestyle='--', linewidth=1)
        ax.axhline(30, color='royalblue', linestyle='--', linewidth=1)
        ax.axhline(50, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.autoscale_view(scalex=True, scaley=False)
        if is_dates:
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 30, 50, 70, 100])
        ax.set_title(f"{ticker} — Wilder's {periods}-Period RSI")
        ax.set_xlabel('Date / time' if is_dates else 'Bar')
        ax.set_ylabel('RSI')
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
    plt.show()


def wilder_rsi(prices, periods=14, plot=False, max_workers=None):
    """Return RSI for every closing-price column, preserving index and names.

    Accept a DataFrame (or a Series for one ticker). Seed Wilder smoothing
    with the mean of the first `periods` price changes. The first `periods`
    rows remain NaN. Flat prices give RSI 50; gains only give 100, losses
    only give 0. Input prices must be finite and in chronological order.

    Multiple columns use worker processes; one column runs locally.
    plot=True draws one chart per ticker, without legends, using red above
    70, blue below 30, and gray between thresholds. In standalone scripts,
    call this under `if __name__ == '__main__':` for Windows multiprocessing.
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=prices.name or 'Price')
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] == 0:
        raise ValueError('prices must contain at least one closing-price column')
    if not prices.columns.is_unique:
        raise ValueError('Ticker columns must be unique')
    if not isinstance(periods, Integral) or isinstance(periods, bool) or periods < 1:
        raise ValueError('periods must be a positive integer')
    if max_workers is not None and (
        not isinstance(max_workers, Integral) or isinstance(max_workers, bool)
        or max_workers < 1
    ):
        raise ValueError('max_workers must be a positive integer')
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('RSI requires finite closing prices for every bar')
    tasks = [(values[:, i], periods) for i in range(values.shape[1])]
    if len(tasks) > 1:
        workers = min(len(tasks), max_workers or (os.cpu_count() or 1), 61)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            columns = list(executor.map(_rsi_column, tasks))
    else:
        columns = [_rsi_column(tasks[0])]
    result = pd.DataFrame(np.column_stack(columns), index=prices.index,
                          columns=prices.columns)
    if plot:
        _plot_rsi(result, periods)
    return result
