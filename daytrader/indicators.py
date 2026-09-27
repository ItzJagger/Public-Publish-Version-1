import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA) - the standard basis for RSI/ATR."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (the conventional
    definition used by most charting tools, so values line up with what a
    broker shows). Handles the no-loss / flat edge cases instead of returning
    NaN there.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _wilder(gain, window)
    avg_loss = _wilder(loss, window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    # No losses in the window -> maximally strong -> RSI 100.
    out = out.where(avg_loss != 0, 100.0)
    # Perfectly flat (no gains and no losses) -> neutral 50.
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    # Re-mask the warmup period (first `window` bars have no real reading).
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, reset at the start of each trading day."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df.groupby(df.index.date)["Volume"].cumsum()
    cum_vol_price = (typical_price * df["Volume"]).groupby(df.index.date).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing) - the typical bar-to-bar price
    swing, in dollars.
    """
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder(true_range, window)
