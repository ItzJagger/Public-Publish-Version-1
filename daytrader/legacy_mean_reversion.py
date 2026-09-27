import pandas as pd

from . import indicators


def generate_mean_reversion_signals(df: pd.DataFrame, lookback: int = 20, num_std: float = 1.5) -> pd.DataFrame:
    """Buy near the bottom of a stock's recent trading range, sell near the top.

    Suited to range-bound, low-dispersion stocks rather than trending ones —
    targets small, frequent gains instead of riding a trend. Shares the
    rsi/vwap/buy/sell column shape of signals.generate_signals so it's a
    drop-in for backtest.run_backtest / recommend.current_recommendation.
    """
    out = df.copy()
    sma = out["Close"].rolling(lookback).mean()
    std = out["Close"].rolling(lookback).std()
    out["lower_band"] = sma - num_std * std
    out["upper_band"] = sma + num_std * std
    out["rsi"] = indicators.rsi(out["Close"], 14)
    out["vwap"] = indicators.vwap(out)

    out["buy"] = out["Close"] <= out["lower_band"]
    out["sell"] = out["Close"] >= out["upper_band"]
    return out
