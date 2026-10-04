"""S&P 500 closing-price downloads and relative market-cap grouping."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

# Existing notebook constituent snapshot; Yahoo Finance symbol format.
# Source: https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv
SP500_TICKERS = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A',
    'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL',
    'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK',
    'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT',
    'APP', 'APTV', 'ACGL', 'ADM', 'ARES', 'ANET', 'AJG', 'AIZ', 'T', 'ATO',
    'ADSK', 'ADP', 'AZO', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX',
    'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BE', 'BNY', 'BA',
    'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BG', 'BXP', 'CHRW',
    'CDNS', 'CPT', 'COF', 'CAH', 'CCL', 'CARR', 'CVNA', 'CASY', 'CAT', 'CBOE',
    'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX',
    'CMG', 'CB', 'CHD', 'CIEN', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
    'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COHR', 'COIN', 'CL', 'CMCSA', 'FIX',
    'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP',
    'COST', 'CRH', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG',
    'DVA', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG',
    'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD',
    'ETN', 'EBAY', 'ECHO', 'ECL', 'EIX', 'EW', 'ELV', 'EME', 'EMR', 'ETR',
    'EOG', 'EQT', 'EFX', 'EQIX', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'P',
    'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO',
    'FAST', 'FRT', 'FDX', 'FDXF', 'FERG', 'FIS', 'FITB', 'FSLR', 'FE', 'FISV',
    'FLEX', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT',
    'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD',
    'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC',
    'HSY', 'HPE', 'HLT', 'HD', 'HONA', 'HON', 'HRL', 'HST', 'HWM', 'HPQ',
    'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
    'IR', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP', 'INTU', 'ISRG', 'IVZ',
    'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM',
    'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC',
    'KR', 'LHX', 'LH', 'LRCX', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN',
    'LYV', 'LMT', 'L', 'LOW', 'LULU', 'LITE', 'LYB', 'MTB', 'MPC', 'MAR',
    'MRSH', 'MLM', 'MRVL', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
    'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MDLZ',
    'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX',
    'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC',
    'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC',
    'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PSKY', 'PH',
    'PAYX', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC',
    'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA',
    'PHM', 'PWR', 'QCOM', 'DGX', 'Q', 'RL', 'RJF', 'RDDT', 'RTX', 'O',
    'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'HOOD', 'ROK', 'ROL', 'ROP',
    'ROST', 'RCL', 'SPGI', 'CRM', 'SNDK', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW',
    'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK',
    'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS',
    'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN',
    'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB',
    'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS',
    'URI', 'UNH', 'UHS', 'VLO', 'VEEV', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ',
    'VRTX', 'VRT', 'VTRS', 'VICI', 'V', 'VST', 'VMRK', 'VMC', 'WRB', 'GWW',
    'WAB', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST',
    'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM',
    'ZBRA', 'ZBH', 'ZTS',
]


def _normalize_interval(interval):
    if not isinstance(interval, str):
        raise ValueError('interval must be a string such as "1H" or "1D"')
    interval = interval.strip().lower()
    interval = {'1hr': '1h', '1hour': '1h', '1dy': '1d',
                '1day': '1d'}.get(interval, interval)
    valid = {'1m', '2m', '5m', '15m', '30m', '60m', '90m',
             '1h', '1d', '5d', '1wk', '1mo', '3mo'}
    if interval not in valid:
        raise ValueError(f'Unsupported interval {interval!r}; choose from {sorted(valid)}')
    return interval


def _fetch_close(ticker, period='1y', start=None, end=None, interval='1d'):
    interval = _normalize_interval(interval)
    if start is not None or end is not None:
        history = yf.Ticker(ticker).history(
            start=start, end=end, interval=interval, auto_adjust=True)
    else:
        history = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True)
    close = history['Close']
    if close.empty:
        raise ValueError(f'No price history returned for {ticker}')
    if interval in {'1d', '5d', '1wk', '1mo', '3mo'}:
        close.index = close.index.tz_localize(None).normalize()
    elif close.index.tz is not None:
        # Preserve each bar's time and align intraday timestamps across exchanges.
        close.index = close.index.tz_convert('UTC')
    return close.rename(ticker)


def get_SP500(period='1y', max_workers=32, *, start=None, end=None, interval='1D'):
    """Return closing prices for the stored S&P 500 ticker snapshot.

    Rows are bar timestamps and columns are ticker symbols. Downloads run
    concurrently, once per ticker, preserving the stored ticker order.
    This function does not plot or refresh index membership.

    Supply start/end as 'YYYY-MM-DD' strings or datetime objects to use
    a date range instead of period. Start is inclusive; end is exclusive
    and defaults to now. Example: get_SP500(start='2020-01-01').
    interval accepts '1H', '1D', or other yfinance interval strings, ignoring
    case. Intraday timestamps retain their time in UTC; daily dates are
    timezone-naive. Yahoo limits available history for intraday intervals.
    """
    interval = _normalize_interval(interval)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prices = list(executor.map(
            lambda ticker: _fetch_close(ticker, period, start, end, interval), SP500_TICKERS
        ))
    return pd.concat(prices, axis=1)


def get_stock_returns(ticker, start, end=None, interval='1D', max_workers=None):
    """Return fractional adjusted returns per bar for a ticker or ticker list.

    Start/end accept yfinance dates ('YYYY-MM-DD' or datetime objects).
    Start is inclusive; end is exclusive and defaults to now. The column
    is the normalized ticker, and 0.01 means a 1% return. The first price
    timestamp is omitted because it has no prior observation in the range.
    Missing prices are not forward-filled. interval accepts '1H', '1D',
    or other yfinance interval strings, ignoring case. Returns follow the
    selected interval. Intraday timestamps retain their time in UTC.
    A list of tickers is downloaded concurrently using worker threads.
    """
    if not isinstance(ticker, str):
        tickers = list(ticker)
        if not tickers:
            raise ValueError('At least one ticker is required')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            frames = list(executor.map(
                lambda symbol: get_stock_returns(symbol, start, end, interval), tickers))
        return pd.concat(frames, axis=1)
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError('ticker must be a nonempty string')
    if start is None:
        raise ValueError('A start date is required')
    ticker = ticker.strip().upper().replace('.', '-')
    close = _fetch_close(ticker, start=start, end=end, interval=interval)
    return close.pct_change(fill_method=None).iloc[1:].to_frame(name=ticker)


def get_stock_volume(ticker, start=None, end=None, interval='1D', period='1y', max_workers=None):
    """Return traded share volume per bar for a ticker or ticker list.

    Supports yfinance start/end dates and intervals such as '1H' or '1D'.
    Explicit dates override period; end is exclusive. Use period='max'
    for the longest available history at the selected interval. Intraday
    timestamps are retained in UTC; daily dates are timezone-naive.
    Zero-volume bars are retained rather than treated as missing values.
    A list of tickers is downloaded concurrently using worker threads.
    """
    if not isinstance(ticker, str):
        tickers = list(ticker)
        if not tickers:
            raise ValueError('At least one ticker is required')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            frames = list(executor.map(
                lambda symbol: get_stock_volume(symbol, start, end, interval, period), tickers))
        return pd.concat(frames, axis=1)
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError('ticker must be a nonempty string')
    ticker = ticker.strip().upper().replace('.', '-')
    interval = _normalize_interval(interval)
    if start is not None or end is not None:
        history = yf.Ticker(ticker).history(
            start=start, end=end, interval=interval, auto_adjust=True)
    else:
        history = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True)
    if history.empty or 'Volume' not in history:
        raise ValueError(f'No volume history returned for {ticker}')
    volume = history['Volume'].copy()
    if interval in {'1d', '5d', '1wk', '1mo', '3mo'}:
        volume.index = volume.index.tz_localize(None).normalize()
    elif volume.index.tz is not None:
        volume.index = volume.index.tz_convert('UTC')
    return volume.sort_index().to_frame(name=ticker)


def _fetch_market_cap(ticker):
    stock = yf.Ticker(ticker)
    for lookup in (lambda: stock.fast_info['market_cap'],
                   lambda: stock.info.get('marketCap')):
        try:
            cap = float(lookup())
            if np.isfinite(cap) and cap > 0:
                return ticker, cap
        except Exception:
            continue
    return ticker, np.nan


def split_by_market_cap(sp500_df, max_workers=8):
    """Return (small_cap_df, medium_cap_df, large_cap_df), in that order.

    Fetch current market caps for the input columns, rank ascending with
    alphabetical tie-breaking, and split into nearly equal thirds. Labels
    are relative ranks within the input universe, not dollar thresholds.
    Preserve dates and prices; never download price history again or
    mutate the input. Missing market caps raise ValueError.
    """
    if not isinstance(sp500_df, pd.DataFrame):
        raise TypeError('sp500_df must be a pandas DataFrame')
    if len(sp500_df.columns) < 3:
        raise ValueError('At least three ticker columns are required')
    if not sp500_df.columns.is_unique:
        raise ValueError('Ticker columns must be unique')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_fetch_market_cap, sp500_df.columns))
    caps = pd.Series(dict(results), name='market_cap')
    missing = caps[caps.isna()].index.tolist()
    if missing:
        raise ValueError('Missing market caps; retry: ' + ', '.join(missing))
    ranked = (
        caps.rename_axis('ticker').reset_index()
        .sort_values(['market_cap', 'ticker'], kind='stable')['ticker']
        .to_numpy()
    )
    groups = np.array_split(ranked, 3)
    return tuple(sp500_df.loc[:, group].copy() for group in groups)
