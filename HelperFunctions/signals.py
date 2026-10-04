"""Technical signals calculated from DataFrames of closing prices."""
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter


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
    alpha = 1.0 / periods
    beta = 1.0 - alpha
    # Compiled recursive filters preserve Wilder's initial simple-average seed.
    smooth_gain = np.r_[gain, lfilter([alpha], [1, -beta], gains[periods:], zi=[beta * gain])[0]]
    smooth_loss = np.r_[loss, lfilter([alpha], [1, -beta], losses[periods:], zi=[beta * loss])[0]]
    total = smooth_gain + smooth_loss
    result[periods:] = np.divide(100 * smooth_gain, total,
                                out=np.full_like(total, 50.0), where=total != 0)
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

    Multiple columns use worker threads; one column runs locally.
    plot=True draws one chart per ticker, without legends, using red above
    70, blue below 30, and gray between thresholds. Plotting runs on the
    calling thread after all calculations finish.
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            columns = list(executor.map(_rsi_column, tasks))
    else:
        columns = [_rsi_column(tasks[0])]
    result = pd.DataFrame(np.column_stack(columns), index=prices.index,
                          columns=prices.columns)
    if plot:
        _plot_rsi(result, periods)
    return result

def plot_cumulative_returns(events, max_workers=None):
    """Plot compounded completed-trade wealth per ticker, starting at 1.

    Returns a wide DataFrame indexed by exit timestamps. Trades exiting at
    the same timestamp are combined multiplicatively; pre-first-exit values
    are one. Incomplete/excluded trades are omitted. This event-return
    summary is not a capital-weighted portfolio equity curve: overlapping
    trades have no capital allocation specified. No legend is displayed.
    """
    import matplotlib.pyplot as plt
    from .labeling import get_labels
    from .parallel import parallel_map
    labels = get_labels(events, max_workers=max_workers)
    if labels.empty:
        raise ValueError('No completed events are available to plot')
    if not np.isfinite(labels['ret']).all() or (labels['ret'] < -1).any():
        raise ValueError('Compounding requires finite trade returns of at least -100%')
    def calculate(group):
        gross = (1 + group.set_index('t1')['ret']).groupby(level=0).prod().sort_index()
        curve = gross.cumprod()
        curve.name = group['ticker'].iloc[0]
        return curve
    groups = [group for _, group in labels.groupby('ticker', sort=False)]
    curves = pd.concat(parallel_map(calculate, groups, max_workers), axis=1).sort_index()
    curves = curves.ffill().fillna(1.0)
    # Anchor at one before the first realized exit.
    start = labels['entry_time'].min()
    curves.loc[start] = 1.0
    curves = curves.sort_index()
    curves.index.name = 'exit_time'
    for ticker in curves.columns:
        plt.figure(figsize=(14, 5))
        ax = curves[ticker].plot(
            figsize=(14, 5), legend=False, drawstyle='steps-post',
            title=f'{ticker} — Compounded Event Returns (Start = 1)')
        ax.axhline(1, color='gray', linewidth=0.8)
        ax.set_xlabel('Exit date / time')
        ax.set_ylabel('Cumulative value (initial value = 1)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    plt.show()
    return curves
