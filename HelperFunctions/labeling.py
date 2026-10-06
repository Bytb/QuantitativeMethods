"""Close-only, volatility-scaled triple-barrier meta-labeling.

Each nonzero side is an independent trade, including overlapping signals.
Entry is the signal bar's close; monitoring starts at the following bar.
Returns exclude costs. All timestamps retain the input index's timezone.
"""
from numbers import Integral

import numpy as np
import pandas as pd
from .parallel import parallel_map


def _prices_frame(prices):
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=prices.name or 'Price')
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a nonempty Series or DataFrame')
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('prices must have a DatetimeIndex')
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Price timestamps must be unique, increasing, and nonmissing')
    if not prices.columns.is_unique:
        raise ValueError('Ticker columns must be unique')
    prices = prices.astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Closing prices must be finite and strictly positive')
    return prices


def estimate_volatility(prices, span=100, min_periods=2, max_workers=None):
    """EW standard deviation of consecutive-bar simple returns, per ticker.

    Frequency follows input bars. No annualization or horizon scaling.
    Uses pandas ewm(span=span, adjust=True).std(bias=False), as in AFML.
    A target at time t uses only observations up to and including t.
    min_periods controls warm-up independently of the smoothing span.
    """
    prices = _prices_frame(prices)
    for name, value in [('span', span), ('min_periods', min_periods)]:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 2:
            raise ValueError(f'{name} must be an integer of at least 2')
    def calculate(ticker):
        return prices[ticker].pct_change(fill_method=None).ewm(
            span=span, min_periods=min_periods, adjust=True
        ).std(bias=False)
    return pd.concat(parallel_map(calculate, prices.columns, max_workers), axis=1)


def get_vertical_barriers(event_times, price_index, holding_days=7, max_workers=None):
    """First observed close at/after each elapsed-calendar-time deadline.

    One day is 24 elapsed hours, including weekends. NaT means the data
    does not extend to the deadline. No final-bar substitute is used.
    """
    if not isinstance(price_index, pd.DatetimeIndex) or not price_index.is_monotonic_increasing:
        raise ValueError('price_index must be an increasing DatetimeIndex')
    holding_days = float(holding_days)
    if not np.isfinite(holding_days) or holding_days <= 0:
        raise ValueError('holding_days must be positive and finite')
    starts = pd.DatetimeIndex(event_times)
    if starts.tz != price_index.tz:
        raise ValueError('Event and price timestamps must use the same timezone')
    deadlines = starts + pd.Timedelta(days=holding_days)
    chunks = np.array_split(np.arange(len(starts)), min(32, len(starts) or 1))
    parts = parallel_map(lambda chunk: price_index.searchsorted(deadlines.take(chunk), side='left'),
                         chunks, max_workers)
    positions = np.concatenate(parts)
    dtype = pd.DatetimeTZDtype(tz=price_index.tz) if price_index.tz is not None else 'datetime64[ns]'
    times = [price_index[p] if p < len(price_index) else pd.NaT for p in positions]
    return pd.Series(times, index=starts, dtype=dtype, name='vertical_time')


def create_barriers(prices, sides, volatility=None, *, pt_mult=1.0,
                    sl_mult=1.0, holding_days=7, min_ret=0.0,
                    volatility_span=100, volatility_min_periods=2, max_workers=None):
    """Build one row per nonzero signal, with frozen volatility and barriers.

    sides must match the price index and columns; values are -1, 0, +1.
    Optional volatility must also match, or is estimated from price bars.
    pt_mult/sl_mult are independent positive volatility multipliers.
    Targets must exceed min_ret. Invalid or warming-up targets remain in
    the table with status='excluded' and an explanatory exclusion_reason.
    Rows are indexed by event_id; repeated entry times across tickers and
    overlapping holding windows are preserved.
    """
    prices = _prices_frame(prices)
    if isinstance(sides, pd.Series) and prices.shape[1] == 1:
        sides = sides.to_frame(name=prices.columns[0])
    if not isinstance(sides, pd.DataFrame) or not sides.index.equals(prices.index) or not sides.columns.equals(prices.columns):
        raise ValueError('sides must match the price index and ticker columns exactly')
    if not sides.isin([-1, 0, 1]).all().all():
        raise ValueError('sides must contain only -1, 0, or +1')
    pt_mult, sl_mult, min_ret = float(pt_mult), float(sl_mult), float(min_ret)
    if not np.isfinite([pt_mult, sl_mult, min_ret]).all() or pt_mult <= 0 or sl_mult <= 0 or min_ret < 0:
        raise ValueError('Barrier multipliers must be positive; min_ret must be nonnegative')
    # Validate the calendar horizon even when there are no signals.
    get_vertical_barriers(prices.index[:0], prices.index, holding_days, max_workers)
    if volatility is None:
        volatility = estimate_volatility(prices, volatility_span, volatility_min_periods, max_workers)
    if isinstance(volatility, pd.Series) and prices.shape[1] == 1:
        volatility = volatility.to_frame(name=prices.columns[0])
    if not isinstance(volatility, pd.DataFrame) or not volatility.index.equals(prices.index) or not volatility.columns.equals(prices.columns):
        raise ValueError('volatility must match the price index and ticker columns exactly')
    positions = np.argwhere(sides.to_numpy() != 0)
    starts = prices.index.take(positions[:, 0])
    vertical = get_vertical_barriers(starts, prices.index, holding_days, max_workers)
    def build_row(item):
        i, (row, col) = item
        timestamp, ticker = prices.index[row], prices.columns[col]
        side, entry, target = int(sides.iloc[row, col]), prices.iloc[row, col], float(volatility.iloc[row, col])
        reason = ('missing_volatility' if not np.isfinite(target) else
                  'nonpositive_volatility' if target <= 0 else
                  'below_min_ret' if target <= min_ret else None)
        eligible = reason is None
        pt_return = pt_mult * target if eligible else np.nan
        sl_return = -sl_mult * target if eligible else np.nan
        return dict(
            ticker=ticker, signal_time=timestamp, entry_time=timestamp,
            entry_price=entry, side=side, trgt=target,
            pt_mult=pt_mult, sl_mult=sl_mult, pt_return=pt_return,
            sl_return=sl_return, pt_price=entry * (1 + side * pt_return),
            sl_price=entry * (1 + side * sl_return),
            deadline=timestamp + pd.Timedelta(days=holding_days),
            vertical_time=vertical.iloc[i],
            status='pending' if eligible else 'excluded', exclusion_reason=reason,
        )
    rows = parallel_map(build_row, enumerate(positions), max_workers)
    columns = ['ticker', 'signal_time', 'entry_time', 'entry_price', 'side', 'trgt',
               'pt_mult', 'sl_mult', 'pt_return', 'sl_return', 'pt_price', 'sl_price',
               'deadline', 'vertical_time', 'status', 'exclusion_reason']
    result = pd.DataFrame(rows, columns=columns)
    result.index.name = 'event_id'
    return result


def create_events(prices, sides, volatility=None, max_workers=None, **barrier_parameters):
    """Create barriers and find the first observed close touching a barrier.

    Horizontal touches use >= PT and <= SL, and can occur at the vertical
    close (horizontal reason takes precedence on that same timestamp).
    Missing future data leaves events incomplete unless a horizontal
    barrier has already been touched. Every signal is evaluated separately.
    pt_time/sl_time record the actual first-touch exit, not later touches.
    """
    prices = _prices_frame(prices)
    events = create_barriers(prices, sides, volatility, max_workers=max_workers, **barrier_parameters)
    for field in ['pt_time', 'sl_time', 't1']:
        dtype = pd.DatetimeTZDtype(tz=prices.index.tz) if prices.index.tz is not None else 'datetime64[ns]'
        events[field] = pd.Series(pd.NaT, index=events.index, dtype=dtype)
    events['exit_type'] = pd.Series(None, index=events.index, dtype=object)
    events['exit_price'] = np.nan
    def evaluate(item):
        event_id, event = item
        start = prices.index.get_loc(event['entry_time']) + 1
        stop = (prices.index.get_loc(event['vertical_time']) + 1
                if pd.notna(event['vertical_time']) else len(prices))
        path = prices[event['ticker']].iloc[start:stop]
        returns = event['side'] * (path.to_numpy() / event['entry_price'] - 1)
        pt_hits = returns >= event['pt_return']
        sl_hits = returns <= event['sl_return']
        touches = np.flatnonzero(pt_hits | sl_hits)
        if len(touches):
            touch = int(touches[0])
            exit_time, exit_price = path.index[touch], path.iloc[touch]
            exit_type = 'profit_taking' if pt_hits[touch] else 'stop_loss'
            touch_field = 'pt_time' if pt_hits[touch] else 'sl_time'
        elif pd.notna(event['vertical_time']):
            exit_time = event['vertical_time']
            exit_price = prices.at[exit_time, event['ticker']]
            exit_type = 'time'
            touch_field = None
        else:
            return event_id, {'status': 'incomplete'}
        updates = dict(t1=exit_time, exit_price=exit_price,
                       exit_type=exit_type, status='complete')
        if touch_field:
            updates[touch_field] = exit_time
        return event_id, updates
    pending = events.loc[events['status'].eq('pending')]
    # Workers only read price/event data; apply updates on the calling thread.
    for event_id, updates in parallel_map(evaluate, pending.iterrows(), max_workers):
        for field, value in updates.items():
            events.at[event_id, field] = value
    # Preserve observed candles for overlap calculations (including session gaps).
    # Tuple equality is scalar, so pandas can safely compare attrs on concat.
    events.attrs['bar_index'] = tuple(prices.index)
    return events


def get_labels(events, max_workers=None):
    """Binary meta-labels for completed events: side-adjusted ret > 0.

    No costs. Profitable trades get 1; losing or flat trades get 0.
    Excluded/incomplete events receive no label. Preserve event_id so
    labels can be joined to events for overlap and sample-weight studies.
    """
    required = {'ticker', 'entry_time', 'entry_price', 'side', 't1',
                'exit_price', 'exit_type', 'status'}
    if not isinstance(events, pd.DataFrame) or not required.issubset(events.columns):
        raise ValueError('events must be the DataFrame returned by create_events')
    complete = events.loc[events['status'].eq('complete')]
    def calculate(group):
        labels = group[['ticker', 'entry_time', 't1', 'side', 'exit_type']].copy()
        labels['raw_return'] = group['exit_price'] / group['entry_price'] - 1
        labels['ret'] = labels['side'] * labels['raw_return']
        labels['bin'] = labels['ret'].gt(0).astype('int64')
        return labels
    if complete.empty:
        return calculate(complete)
    groups = [group for _, group in complete.groupby('ticker', sort=False)]
    return pd.concat(parallel_map(calculate, groups, max_workers)).reindex(complete.index)
