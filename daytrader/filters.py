"""Causal market features shared by entries and research checks."""
import numpy as np
import pandas as pd

from . import indicators
from .policy import SIGNAL_PROFILE, ENTRY_FILTERS, MIN_DAILY_DOLLARS, MIN_LIQUIDITY_DAYS


def relative_volume(df, sessions=10, min_sessions=5):
    """Compare the same clock slot on earlier sessions; excludes current volume.

    Avoids comparing busy opening bars against quiet midday bars. Missing slots
    reduce the sample; no fallback claims liquidity from inadequate history.
    """
    slots = df.index.hour * 60 + df.index.minute
    return df['Volume'] / df['Volume'].groupby(slots).transform(
        lambda s: s.shift(1).rolling(sessions, min_periods=min_sessions).mean()).replace(0, np.nan)


def liquid_daily(daily):
    shares = daily['Volume'].tail(20)
    dollars = (daily['Volume'] * daily['Close']).tail(20)
    return bool(len(shares) >= MIN_LIQUIDITY_DAYS and
                shares.mean() >= ENTRY_FILTERS['daily_shares'] and dollars.mean() >= MIN_DAILY_DOLLARS)


def intraday_liquidity(df):
    daily = df.groupby(df.index.date).agg(Volume=('Volume', 'sum'), Close=('Close', 'last'))
    avg_v = daily['Volume'].shift(1).rolling(20, min_periods=MIN_LIQUIDITY_DAYS).mean()
    avg_d = (daily['Volume']*daily['Close']).shift(1).rolling(20, min_periods=MIN_LIQUIDITY_DAYS).mean()
    good = (avg_v >= ENTRY_FILTERS['daily_shares']) & (avg_d >= MIN_DAILY_DOLLARS)
    return pd.Series([bool(good.loc[d]) for d in df.index.date], index=df.index)


def daily_confirmation(daily, long=False):
    close = daily['Close']
    a = indicators.atr(daily, 14)
    mid = indicators.sma(close, 50)
    slow = indicators.sma(close, 200 if long else 50)
    fast = indicators.sma(close, 50 if long else 20)
    rv = daily['Volume'] / daily['Volume'].shift(1).rolling(20).mean()
    span = (daily['High'] - daily['Low']).replace(0, np.nan)
    strong_close = (close - daily['Low']) / span >= .6
    support = mid if long else fast
    if long and SIGNAL_PROFILE == 'active':
        ma20 = indicators.sma(close, 20)
        support = mid.where((close-mid).abs() <= (close-ma20).abs(), ma20)
    return {
        'trend_confirmed': bool(close.iloc[-1] > slow.iloc[-1] and
                                fast.iloc[-1] > slow.iloc[-1] and
                                slow.iloc[-1] > slow.iloc[-6]),
        'recovery_confirmed': bool(close.iloc[-1] > close.iloc[-2] and
                                   close.iloc[-1] > daily['Open'].iloc[-1] and strong_close.iloc[-1]),
        'volume_confirmed': bool(rv.iloc[-1] >= .8),
        'liquidity_confirmed': liquid_daily(daily.iloc[:-1]),
        'near_support': bool(abs(close.iloc[-1] - support.iloc[-1]) <= 2*a.iloc[-1]),
    }
