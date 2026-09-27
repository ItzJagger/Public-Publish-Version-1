import pandas as pd

from . import indicators
from .policy import ENTRY_FILTERS, SIGNAL_PROFILE
from .filters import relative_volume, intraday_liquidity


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

    # Buy a recovery into the band, never a fresh break below it.
    out["atr"] = indicators.atr(out)
    out["relative_volume"] = relative_volume(out)
    mid = indicators.ema(out["Close"], 50)
    same_session = pd.Series(out.index.date, index=out.index).eq(
        pd.Series(out.index.date, index=out.index).shift(1))
    out["recovery_confirmed"] = ((out["Close"].shift(1) < out["lower_band"].shift(1))
                                  & (out["Close"] >= out["lower_band"]) & same_session)
    if SIGNAL_PROFILE == 'active':
        out['recovery_confirmed'] |= ((out['Low'].shift(1) <= out['lower_band'].shift(1))
                                     & (out['Close'] > out['lower_band']) & same_session)
    checks = {
        'trigger': out['recovery_confirmed'],
        'recovery': (out['Close'] > out['Close'].shift(1)) & (out['Close'] > out['Open']),
        'rsi': out['rsi'].between(30, ENTRY_FILTERS['meanrev_rsi_max']) & (out['rsi'] > out['rsi'].shift(1)),
        'trend': (mid >= mid.shift(3)-.25*out['atr']) & (out['Close'] >= mid-out['atr']),
        'volume': out['relative_volume'] >= ENTRY_FILTERS['meanrev_rvol'],
        'liquidity': intraday_liquidity(out),
    }
    for name, passed in checks.items():
        out['check_' + name] = passed.fillna(False)
    out["buy"] = (out["recovery_confirmed"] & (out["Close"] > out["Close"].shift(1))
                  & (out["Close"] > out["Open"]) & out["rsi"].between(30, ENTRY_FILTERS['meanrev_rsi_max'])
                  & (out["rsi"] > out["rsi"].shift(1))
                  & (mid >= mid.shift(3) - .25*out["atr"])
                  & (out["Close"] >= mid - out["atr"])
                  & (out["relative_volume"] >= ENTRY_FILTERS['meanrev_rvol']) & intraday_liquidity(out))
    out["sell"] = out["Close"] >= out["upper_band"]
    return out
