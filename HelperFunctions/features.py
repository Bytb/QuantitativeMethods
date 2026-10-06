"""Features known at the close of each hourly signal bar."""
import numpy as np
import pandas as pd


def event_features(hourly, daily, spy_daily, events, volatility, rsi):
    """Build one feature row per event, preserving event IDs.

    hourly/daily are dictionaries mapping ticker to adjusted OHLCV DataFrames.
    spy_daily is an adjusted SPY daily OHLCV DataFrame. All hourly indexes must
    be timezone-aware; daily indexes identify trading dates. Daily features
    always use dates strictly before the signal's New York trading date.
    volatility and rsi are wide DataFrames aligned to hourly price timestamps.
    Returns (features, metadata); metadata contains ticker and signal_time.
    Missing lookbacks remain NaN. No labels, exits, or direction enter features.
    """
    def daily_close(frame):
        close = frame["Close"].astype(float).copy()
        index = pd.DatetimeIndex(close.index)
        if index.tz is not None:
            index = index.tz_convert("America/New_York").tz_localize(None)
        close.index = index.normalize()
        if not close.index.is_unique or not close.index.is_monotonic_increasing:
            raise ValueError("Daily trading dates must be unique and increasing.")
        return close

    def previous_daily(series, dates):
        # Strictly previous trading date; never read the current day's close.
        positions = series.index.searchsorted(dates, side="left") - 1
        result = np.full(len(dates), np.nan)
        valid = positions >= 0
        result[valid] = series.to_numpy(dtype=float)[positions[valid]]
        return result

    spy_return = daily_close(spy_daily).pct_change(3, fill_method=None) * 100
    frames = {}
    for ticker, bars in hourly.items():
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
            raise ValueError("Hourly bars require a timezone-aware DatetimeIndex.")
        if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
            raise ValueError("Hourly timestamps must be unique and increasing.")
        local = bars.index.tz_convert("America/New_York")
        dates = local.tz_localize(None).normalize()
        close = bars["Close"].astype(float)
        volume = bars["Volume"].astype(float)
        returns = close.pct_change(fill_method=None)
        daily_prices = daily_close(daily[ticker])
        sma50 = daily_prices.rolling(50, min_periods=50).mean()
        prior_sma = previous_daily(sma50, dates)
        slots = local.hour * 60 + local.minute
        volume_baseline = volume.groupby(slots).transform(
            lambda group: group.shift(1).rolling(20, min_periods=20).mean())
        f = pd.DataFrame(index=bars.index)
        f["volatility_ewma"] = volatility[ticker].reindex(bars.index)
        short_vol = returns.ewm(span=20, min_periods=20, adjust=True).std(bias=False)
        long_vol = returns.ewm(span=100, min_periods=100, adjust=True).std(bias=False)
        f["volatility_ratio20_100"] = short_vol / long_vol.replace(0, np.nan)
        f["volume_shares"] = volume
        f["relative_volume20_sessions"] = volume / volume_baseline.replace(0, np.nan)
        f["above_daily_sma50"] = np.where(
            np.isfinite(prior_sma), (close.to_numpy() > prior_sma).astype(float), np.nan)
        f["distance_daily_sma50_pct"] = (close.to_numpy() / prior_sma - 1) * 100
        f["daily_sma50_slope5_pct"] = previous_daily(
            sma50.pct_change(5, fill_method=None) * 100, dates)
        for lag in (1, 3, 6):
            f[f"return_{lag}_bars_pct"] = close.pct_change(lag, fill_method=None) * 100
        f["return_5_daily_sessions_pct"] = previous_daily(
            daily_prices.pct_change(5, fill_method=None) * 100, dates)
        current_rsi = rsi[ticker].reindex(bars.index)
        f["rsi14"] = current_rsi
        f["rsi14_change1_bar"] = current_rsi.diff()
        session_open = bars["Open"].groupby(dates).transform("first")
        prior_close = previous_daily(daily_prices, dates)
        f["overnight_gap_pct"] = (session_open.to_numpy() / prior_close - 1) * 100
        f["bar_start_minutes_since0930"] = slots - (9 * 60 + 30)
        f["weekday"] = local.dayofweek
        f["spy_return_3_daily_sessions_pct"] = previous_daily(spy_return, dates)
        frames[ticker] = f.replace([np.inf, -np.inf], np.nan)

    columns = next(iter(frames.values())).columns
    features = pd.DataFrame(index=events.index, columns=columns, dtype=float)
    for ticker, group in events.groupby("ticker", sort=False):
        if ticker not in frames:
            raise ValueError(f"No hourly data for event ticker {ticker}.")
        times = pd.DatetimeIndex(group["signal_time"])
        if not times.isin(frames[ticker].index).all():
            raise ValueError(f"Signal timestamps are missing from {ticker} hourly data.")
        features.loc[group.index] = frames[ticker].loc[times].to_numpy()
    metadata = events[["ticker", "signal_time"]].copy()
    return features, metadata
