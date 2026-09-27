import pandas as pd

MARKET_TZ = "America/New_York"  # NYSE and TSX both trade on Eastern Time


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Download intraday OHLCV data for a ticker, indexed in market (Eastern) time.

    yfinance limits intraday history depth depending on interval
    (e.g. 5m/15m/30m bars are only available for the trailing ~60 days; 1m for
    ~7 days). yfinance is imported lazily so the rest of the package - including
    the clipboard prompt flow - works without the data library installed.
    """
    import yfinance as yf  # lazy: only the live fetch needs it

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval})")

    # yfinance may return columns as a MultiIndex (field, ticker) or (ticker, field).
    # Pick whichever level actually holds the OHLC field names.
    if isinstance(df.columns, pd.MultiIndex):
        ohlc = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        level = 0 if ohlc.intersection(df.columns.get_level_values(0)) else 1
        df.columns = df.columns.get_level_values(level)

    df = _to_market_time(df, interval)
    df.index.name = "datetime"
    return df


def _to_market_time(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Normalise the index to Eastern time so 'local market time' hour labels
    downstream (seasonality, hold windows) are actually correct. Daily bars are
    left date-indexed; only intraday bars carry a meaningful time-of-day.
    """
    if interval.endswith("d") or interval.endswith("wk") or interval.endswith("mo"):
        return df

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(MARKET_TZ)
    return df


def completed_bars(df: pd.DataFrame, interval: str, now=None) -> pd.DataFrame:
    """Validate OHLCV, then omit unclosed bars (5s publication grace).

    Daily inputs use only dates before today, even after close; this deliberately
    waits until the following date for daily confirmation. Intraday timestamps
    are bar-start timestamps. Regular sessions only; exchange holidays are not
    supplied by this helper.
    """
    import numpy as np
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('OHLCV timestamps must be ordered, unique datetimes')
    # Today's daily bar is not used by daily signals; validate only completed
    # daily rows so a malformed partial daily update cannot poison old history.
    if interval == '1d':
        cutoff = pd.Timestamp.now(tz=MARKET_TZ) if now is None else pd.Timestamp(now)
        cutoff = cutoff.tz_localize(MARKET_TZ) if cutoff.tzinfo is None else cutoff.tz_convert(MARKET_TZ)
        df = df.loc[df.index.date < cutoff.date()]
    values = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('OHLCV contains missing or non-finite values')
    invalid = ((values[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1) | 
        (values.Volume < 0) | 
        (values.High < values[['Open', 'Close', 'Low']].max(axis=1)) | 
        (values.Low > values[['Open', 'Close', 'High']].min(axis=1)))
    if invalid.any():
        raise ValueError('OHLCV contains invalid prices, ranges or volumes: '+str(values.loc[invalid].head(2).to_dict('index')))
    now = pd.Timestamp.now(tz=MARKET_TZ) if now is None else pd.Timestamp(now)
    now = now.tz_localize(MARKET_TZ) if now.tzinfo is None else now.tz_convert(MARKET_TZ)
    idx = df.index.tz_localize(MARKET_TZ) if df.index.tz is None else df.index.tz_convert(MARKET_TZ)
    if interval == '1d':
        return df.loc[idx.date < now.date()].copy()
    durations = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '90m': 90, '1h': 60}
    if interval not in durations:
        raise ValueError(f'Unsupported signal interval: {interval}')
    mins = idx.hour*60 + idx.minute
    end = idx + pd.Timedelta(minutes=durations[interval], seconds=5)
    return df.loc[(end <= now) & (mins >= 570) & (mins < 960)].copy()
