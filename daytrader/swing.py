"""Swing screen: find stocks sitting near a multi-day low that have started to
turn back up, for a hold of a few days (not intraday).

This is a daily-timeframe mean-reversion idea: buy a pullback near a recent low
*once it shows a first sign of stabilizing*, target a bounce, and hold a few
days with a stop and a take-profit.

Honest limits, same as everywhere else in this tool:
- "at a low" is not "about to go up". In a real sell-off oversold names keep
  falling, so this requires a small up-tick first and still misses plenty.
- A multi-day hold carries overnight gap risk every single night that a stop
  cannot protect - earnings and headlines especially. Check the calendar before
  holding. This is a weak, noisy edge, not a profit machine.
"""
import pandas as pd

from .data import MARKET_TZ, fetch_intraday, completed_bars
from .filters import daily_confirmation
from .policy import round_trip_cost, net_reward_risk, MIN_NET_RR
from .indicators import atr, rsi

SWING_PERIOD = "6mo"
MIN_BARS = 60
LOOKBACK_LOW = 20       # window (trading days) that defines "a recent low"
NEAR_LOW_PCT = 3.0      # within this % of that low counts as "at the low"
OVERSOLD_RSI = 40.0     # daily RSI at/below this is a meaningful pullback
STOP_ATR_MULT = 1.5     # stop = entry - 1.5 x daily ATR
RR = 2.5                # reward:risk for the take-profit
MAX_HOLD_DAYS = 5       # backstop exit if the target isn't reached first
MIN_CLEAN_RR = 1.5      # after capping the target at the recent high, need >= this


def _weekday_tilt(daily: pd.DataFrame):
    """Best historical up-weekday by average close-to-close daily return."""
    ret = daily["Close"].pct_change() * 100
    by_day = ret.groupby(daily.index.day_name()).mean()
    by_day = by_day[by_day > 0]
    if by_day.empty:
        return None, None
    return by_day.idxmax(), float(by_day.max())


def swing_candidate(ticker: str, period: str = SWING_PERIOD, asof=None) -> dict | None:
    """Daily-timeframe dip stats for one ticker. None if not enough history.

    `candidate` is True when the name is near a recent low (or oversold) AND has
    ticked up off the prior day AND a bounce target keeps a sane reward:risk.
    """
    try:
        daily = fetch_intraday(ticker, period=period, interval="1d")
    except Exception:
        return None
    daily = completed_bars(daily, '1d', asof)
    return swing_from_frame(ticker, daily, asof)


def swing_from_frame(ticker, daily, asof=None, cost=None):
    """Pure causal screen: caller supplies only completed, historical bars."""
    if len(daily) < MIN_BARS:
        return None

    close = daily["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    rsi_val = rsi(close, 14).iloc[-1]
    atr_val = atr(daily, 14).iloc[-1]
    if not (rsi_val == rsi_val) or not (atr_val == atr_val) or atr_val <= 0:
        return None
    day_rsi = float(rsi_val)
    day_atr = float(atr_val)

    low_n = float(close.tail(LOOKBACK_LOW).min())
    high_n = float(close.tail(LOOKBACK_LOW).max())
    dist_from_low = (price - low_n) / low_n * 100 if low_n else 0.0

    near_low = dist_from_low <= NEAR_LOW_PCT
    oversold = day_rsi <= OVERSOLD_RSI
    turning_up = price > prev  # first sign it has stopped falling

    floor = price - STOP_ATR_MULT * day_atr
    risk = price - floor
    ceiling = price + RR * risk
    if high_n > price:  # don't promise a target above the recent high
        ceiling = min(ceiling, high_n)
    clean_rr = (ceiling - price) / risk if risk > 0 else 0.0

    confirmations = daily_confirmation(daily)
    net_rr = net_reward_risk(price, floor, ceiling, round_trip_cost(ticker) if cost is None else cost)
    candidate = bool((near_low or oversold or confirmations['near_support']) and
                     35 <= day_rsi <= 60 and turning_up and all(confirmations.values())
                     and clean_rr >= MIN_CLEAN_RR and net_rr >= MIN_NET_RR)

    asof = asof or pd.Timestamp.now(tz=MARKET_TZ)
    exit_by = (asof.normalize() + pd.offsets.BDay(MAX_HOLD_DAYS)).strftime("%a %Y-%m-%d")
    best_day, best_day_ret = _weekday_tilt(daily)

    return {
        "ticker": ticker,
        "price": price,
        "rsi": day_rsi,
        "atr": day_atr,
        "low_20": low_n,
        "high_20": high_n,
        "dist_from_low_pct": dist_from_low,
        "near_low": near_low,
        "oversold": oversold,
        "turning_up": turning_up,
        "still_falling": price <= prev,
        "floor": floor,
        "ceiling": ceiling,
        "rr": clean_rr,
        "hold_days": MAX_HOLD_DAYS,
        "exit_by": exit_by,
        "best_weekday": best_day,
        "best_weekday_ret": best_day_ret,
        "candidate": bool(candidate),
        "net_rr": net_rr,
        "confirmation": confirmations,
        "as_of": str(daily.index[-1]),
    }
