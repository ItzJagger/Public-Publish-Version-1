"""Long-hold screen: find stocks in a longer-term uptrend that have pulled back
toward support, for a position held weeks (not intraday, not a few days).

The idea is different from the swing screen. Swing catches a bounce off an
absolute recent low. This looks for a name whose longer trend is still UP
(price above its 200-day average) but which has dipped back toward its 50-day
average and started to turn up again - a "buy the dip in an uptrend" position,
where the thesis is trend continuation to new highs over weeks.

Honest limits (same as everywhere in this tool):
- A pullback in an uptrend and the start of a breakdown look identical in real
  time. The 200-day trend filter and the stop are the only things separating
  them; plenty of these will still roll over. This is a discretionary, weak
  edge, not a screen that "knows" a stock will go up.
- A weeks-long hold carries an overnight gap every single night, through
  earnings and headlines, that no stop can protect. Check the calendar. US
  names also carry the ~3% round-trip FX; over a multi-week move that is
  survivable but not free.
- This is not financial advice.
"""
import pandas as pd

from .data import MARKET_TZ, fetch_intraday, completed_bars
from .filters import daily_confirmation, liquid_daily
from .patterns import head_shoulders
from .policy import round_trip_cost, net_reward_risk, MIN_NET_RR, SIGNAL_PROFILE
from .indicators import atr, rsi, sma

LONG_PERIOD = "1y"          # ~252 daily bars: enough for a 200-day average
MIN_BARS = 210
SMA_MID_WINDOW = 50         # the pullback reference
SMA_LONG_WINDOW = 200       # the long-term trend filter
RSI_MIN = 40.0              # below this the pullback is looking more like a breakdown
RSI_MAX = 62.0              # above this it isn't a pullback, it's extended
NEAR_MID_PCT = 5.0          # price must be within +5% of the 50-day (near/below it)
STOP_ATR_MULT = 3.0         # wide stop - the hold is long
RR = 2.0                    # target reward:risk for a position trade
REVIEW_BDAYS = 20 if SIGNAL_PROFILE == "active" else 25           # ~5 weeks: a review checkpoint, not a hard exit
MIN_CLEAN_RR = 1.5


def position_candidate(ticker: str, period: str = LONG_PERIOD, asof=None) -> dict | None:
    """Long-term-uptrend pullback stats for one ticker. None if not enough history.

    `candidate` is True when the long trend is up (price > 200-day average),
    price has pulled back to/near the 50-day average (not extended above it),
    momentum is in a moderate band (RSI 40-62, i.e. dipped but not crashing or
    overbought), and it has ticked up off the prior day.
    """
    try:
        daily = fetch_intraday(ticker, period=period, interval="1d")
    except Exception:
        return None
    daily = completed_bars(daily, '1d', asof)
    return position_from_frame(ticker, daily, asof)


def position_from_frame(ticker, daily, asof=None, cost=None):
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

    sma_mid = float(sma(close, SMA_MID_WINDOW).iloc[-1])
    long_window = SMA_LONG_WINDOW
    sma_long = float(sma(close, long_window).iloc[-1])
    if not (sma_mid == sma_mid) or not (sma_long == sma_long):
        return None

    dist_mid_pct = (price - sma_mid) / sma_mid * 100 if sma_mid else 0.0
    mom_3m_pct = None
    if len(close) > 64:
        past = float(close.iloc[-64])
        if past:
            mom_3m_pct = (price / past - 1) * 100

    uptrend = price > sma_long                       # long-term trend still up
    support = min([sma_mid, float(sma(close,20).iloc[-1])], key=lambda x: abs(price-x)) if SIGNAL_PROFILE == "active" else sma_mid
    pulled_back = abs(price-support) <= 2*day_atr if SIGNAL_PROFILE == "active" else price <= sma_mid * (1 + NEAR_MID_PCT / 100)  # near/below the 50-day
    not_overbought = day_rsi <= (68 if SIGNAL_PROFILE == "active" else RSI_MAX)
    not_crashing = day_rsi >= RSI_MIN
    turning_up = price > prev

    floor = price - STOP_ATR_MULT * day_atr
    risk = price - floor
    ceiling = price + RR * risk                      # trend continuation -> target new highs
    clean_rr = (ceiling - price) / risk if risk > 0 else 0.0

    confirmations = daily_confirmation(daily, long=True)
    patterns=head_shoulders(daily) if SIGNAL_PROFILE == 'active' else {}
    bull=patterns.get('bullish',{})
    pattern_entry=False
    if bull.get('entry'):
        rv=float(daily.Volume.iloc[-1]/daily.Volume.iloc[:-1].tail(20).mean())
        pattern_entry=bool(rv >= (0.8 if bull['state']=='retest' else 1.1)
                           and price>float(daily.Open.iloc[-1]) and liquid_daily(daily.iloc[:-1]))
        bull['entry_confirmed']=pattern_entry
    if pattern_entry:
        floor=min(price-STOP_ATR_MULT*day_atr, float(bull['right_shoulder'])-.1*day_atr)
        risk=price-floor
        ceiling=float(bull['target'])
        clean_rr=(ceiling-price)/risk if risk>0 else 0.0
    net_rr = net_reward_risk(price, floor, ceiling, round_trip_cost(ticker) if cost is None else cost)
    candidate = (
        ((uptrend and pulled_back and not_overbought and not_crashing
          and turning_up and all(confirmations.values())) or pattern_entry)
        and clean_rr >= MIN_CLEAN_RR and net_rr >= MIN_NET_RR
        and not patterns.get("bearish",{}).get("bearish")
    )

    asof = asof or pd.Timestamp.now(tz=MARKET_TZ)
    review_by = (asof.normalize() + pd.offsets.BDay(REVIEW_BDAYS)).strftime("%a %Y-%m-%d")

    return {
        "ticker": ticker,
        "patterns": patterns,
        "strategy": "inverse-hs-"+bull["state"] if pattern_entry else "uptrend-pullback",
        "price": price,
        "rsi": day_rsi,
        "atr": day_atr,
        "sma_mid": sma_mid,
        "sma_long": sma_long,
        "long_window": long_window,
        "dist_sma50_pct": dist_mid_pct,
        "mom_3m_pct": mom_3m_pct,
        "uptrend": uptrend,
        "pulled_back": pulled_back,
        "not_overbought": not_overbought,
        "not_crashing": not_crashing,
        "turning_up": turning_up,
        "floor": floor,
        "ceiling": ceiling,
        "rr": clean_rr,
        "hold_weeks": max(1, round(REVIEW_BDAYS / 5)),
        "review_by": review_by,
        "candidate": bool(candidate),
        "net_rr": net_rr,
        "confirmation": confirmations,
        "as_of": str(daily.index[-1]),
    }
