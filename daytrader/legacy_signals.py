import pandas as pd

from . import indicators


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = indicators.ema(out["Close"], 9)
    out["ema_slow"] = indicators.ema(out["Close"], 21)
    out["rsi"] = indicators.rsi(out["Close"], 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = indicators.macd(out["Close"])
    out["vwap"] = indicators.vwap(out)
    out["avg_volume"] = out["Volume"].rolling(window=20).mean()
    return out


def generate_signals(df: pd.DataFrame, volume_filter: bool = True) -> pd.DataFrame:
    """Flag long entry/exit bars using an EMA-crossover + RSI + VWAP filter.

    Entry: fast EMA crosses above slow EMA, RSI not overbought, price above VWAP,
    and (if volume_filter) current volume above its 20-bar average — this avoids
    entering on thin, low-conviction bars (e.g. midday chop).
    Exit: fast EMA crosses below slow EMA, or RSI overbought.
    This is a simple long-only intraday rule set, not a recommendation.
    """
    out = build_indicators(df)

    ema_cross_up = (out["ema_fast"] > out["ema_slow"]) & (
        out["ema_fast"].shift(1) <= out["ema_slow"].shift(1)
    )
    ema_cross_down = (out["ema_fast"] < out["ema_slow"]) & (
        out["ema_fast"].shift(1) >= out["ema_slow"].shift(1)
    )

    out["buy"] = ema_cross_up & (out["rsi"] < 70) & (out["Close"] > out["vwap"])
    if volume_filter:
        out["buy"] = out["buy"] & (out["Volume"] > out["avg_volume"])
    out["sell"] = ema_cross_down | (out["rsi"] > 80)

    return out


def daily_trade_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse buy/sell signal bars into a per-signal timestamp table."""
    signals = generate_signals(df)
    events = signals[signals["buy"] | signals["sell"]].copy()
    events["action"] = events.apply(lambda r: "BUY" if r["buy"] else "SELL", axis=1)
    return events[["Close", "rsi", "vwap", "action"]]
