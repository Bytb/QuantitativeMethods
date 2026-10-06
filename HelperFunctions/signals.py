"""Technical signals calculated from DataFrames of closing prices."""
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def plot_event_barriers(events, prices, *, allow_short=True, seed=42):
    """Plot four random completed events in a 2x2 grid, without replacement.

    Select two longs/two shorts if allow_short, otherwise four longs.
    Show observed closes before entry and beyond the vertical barrier, with
    entry approximately 40% across the time window, plus price barriers,
    and actual closing-price exit. Return (figure, axes, selected_event_ids).
    If an early exit has no observed vertical bar, show the calendar deadline
    and mark that future price coverage is unavailable. Insufficient events
    raise ValueError. prices may be a Series or ticker-column DataFrame.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    required = {'ticker', 'side', 'status', 'entry_time', 'entry_price',
                'pt_price', 'sl_price', 'vertical_time', 'deadline',
                't1', 'exit_price', 'exit_type'}
    if not isinstance(events, pd.DataFrame) or not required.issubset(events):
        raise ValueError('events must be create_events output')
    if not events.index.is_unique:
        raise ValueError('Event IDs must be unique')
    if not isinstance(allow_short, (bool, np.bool_)):
        raise ValueError('allow_short must be boolean')
    if isinstance(prices, pd.Series):
        tickers = events['ticker'].unique()
        if len(tickers) != 1:
            raise ValueError('A price Series requires single-ticker events')
        prices = prices.to_frame(name=tickers[0])
    if (not isinstance(prices, pd.DataFrame)
            or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.hasnans or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing
            or not prices.columns.is_unique):
        raise ValueError('prices must have unique columns and increasing candle timestamps')
    complete = events.loc[events['status'].eq('complete')]
    rng = np.random.default_rng(seed)
    selected = []
    for side, count in ([(1, 2), (-1, 2)] if allow_short else [(1, 4)]):
        candidates = complete.loc[complete['side'].eq(side)]
        if len(candidates) < count:
            direction = 'long' if side == 1 else 'short'
            raise ValueError(f'Need {count} completed {direction} events; found {len(candidates)}')
        positions = rng.choice(len(candidates), size=count, replace=False)
        selected.extend(candidates.index.take(positions).tolist())
    paths = []
    for event_id in selected:
        event = events.loc[event_id]
        if event['ticker'] not in prices:
            raise ValueError(f"Missing prices for {event['ticker']}")
        if not np.isfinite(event[['entry_price', 'pt_price', 'sl_price', 'exit_price']].to_numpy(dtype=float)).all():
            raise ValueError(f'Event {event_id} has invalid price fields')
        if pd.isna(event['entry_time']) or pd.isna(event['t1']):
            raise ValueError(f'Event {event_id} has missing entry/exit timestamps')
        if not pd.DatetimeIndex([event['entry_time'], event['t1']]).isin(prices.index).all():
            raise ValueError(f'Event {event_id} entry/exit candle missing from prices')
        barrier = event['vertical_time'] if pd.notna(event['vertical_time']) else event['deadline']
        if pd.isna(barrier):
            raise ValueError(f'Event {event_id} has no time barrier')
        entry = event['entry_time']
        entry_position = prices.index.get_loc(entry)
        barrier_position = prices.index.searchsorted(barrier, side='right') - 1
        # At least ten observed candles after the barrier, or a quarter of
        # its calendar horizon, whichever extends farther (when available).
        horizon = max(barrier - entry, pd.Timedelta(hours=1))
        after_position = min(len(prices) - 1, barrier_position + 10)
        right = max(barrier + horizon / 4, prices.index[after_position])
        right_span = right - entry
        # A 2:3 left/right span puts entry 40% across the calendar-time axis.
        # Expand both sides if needed to include ten preceding candles.
        before_position = max(0, entry_position - 10)
        left_span = max(right_span * (2 / 3), entry - prices.index[before_position])
        right_span = max(right_span, left_span * 1.5)
        left, right = entry - left_span, entry + right_span
        path = prices.loc[left:right, event['ticker']]
        if path.empty or not np.isfinite(path.to_numpy(dtype=float)).all():
            raise ValueError(f'Event {event_id} has missing/nonfinite closing prices')
        paths.append((event, path, barrier, left, right))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    for ax, event_id, (event, path, barrier, left, right) in zip(axes.flat, selected, paths):
        ax.plot(path.index, path, color='steelblue', linewidth=1.3, label='Candle close')
        lower, upper = sorted([event['pt_price'], event['sl_price']])
        ax.hlines(event['pt_price'], event['entry_time'], barrier,
                  color='seagreen', linestyle='--', linewidth=1.1, alpha=.8,
                  label='Profit-taking barrier')
        ax.hlines(event['sl_price'], event['entry_time'], barrier,
                  color='firebrick', linestyle='--', linewidth=1.1, alpha=.8,
                  label='Stop-loss barrier')
        ax.hlines(event['entry_price'], event['entry_time'], barrier,
                  color='gray', linestyle='--', linewidth=1.0, alpha=.65,
                  label='Entry price')
        ax.vlines(event['entry_time'], lower, upper, color='gray',
                  linestyle='--', linewidth=1.0, alpha=.65)
        time_label = ('Time barrier' if pd.notna(event['vertical_time'])
                      else 'Deadline (future candles unavailable)')
        ax.vlines(barrier, lower, upper, color='darkorange', linestyle=':',
                  linewidth=1.1, alpha=.8, label=time_label)
        is_long = event['side'] == 1
        entry_color = 'seagreen' if is_long else 'red'
        # Circles mark the actual entry/exit close. Exits reverse the
        # entry transaction: sell a long, buy back a short.
        for timestamp, price, color in [
            (event['entry_time'], event['entry_price'], entry_color),
            (event['t1'], event['exit_price'],
             'red' if is_long else 'seagreen'),
        ]:
            ax.scatter([timestamp], [price], marker='o', s=90,
                       facecolors='none', edgecolors=color, linewidths=1.6,
                       zorder=6)
        ax.text(event['t1'], event['exit_price'],
                f"  Exit: {event['exit_type']}", fontsize=8, va='center')
        ax.set_xlim(left, right)
        if left < prices.index[0] or right > prices.index[-1]:
            ax.text(.02, .98, 'Context limited by available price history',
                    transform=ax.transAxes, va='top', fontsize=8, color='dimgray')
        direction = 'Long' if event['side'] == 1 else 'Short'
        ax.set(title=f"{event['ticker']} — {direction} event {event_id}",
               xlabel='Candle timestamp', ylabel='Price ($)')
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=prices.index.tz))
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Include exit reasons that may differ between panels.
    legend_items = dict(zip(labels, handles))
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend_items.update(zip(labels, handles))
    from matplotlib.lines import Line2D
    legend_items['Buy / long entry / short exit'] = Line2D(
        [], [], color='seagreen', marker='o', markerfacecolor='none',
        markeredgewidth=1.6, markersize=9, linestyle='None')
    legend_items['Sell / short entry / long exit'] = Line2D(
        [], [], color='red', marker='o', markerfacecolor='none',
        markeredgewidth=1.6, markersize=9, linestyle='None')
    fig.legend(legend_items.values(), legend_items.keys(), loc='lower center',
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle('Random completed events — close-only triple barriers', fontsize=15)
    fig.tight_layout(rect=(0, .08, 1, .96))
    plt.show()
    return fig, axes, selected


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
