"""Best-effort next-earnings-date lookup, so holds can be checked against the
one gap a stop can never protect: an earnings report inside the hold window.

Honest limits: earnings dates come from Yahoo Finance and are sometimes
missing, tentative, or later rescheduled. Treat every date here as
"approximately" and confirm in your broker before relying on it. When the
lookup fails, the tool says UNKNOWN rather than guessing - an unknown earnings
date on a multi-day hold is itself a risk worth flagging.
"""
import pandas as pd
import time

from .data import MARKET_TZ

_cache: dict = {}
_cached_at: dict = {}
CACHE_TTL_SECONDS = 3600

# Broad/sector/bond ETFs and funds: no single-company earnings, so the
# earnings-gap risk that matters for a stock hold does not apply. Recognising
# them avoids a misleading "earnings UNKNOWN" flag on a core long-term holding.
KNOWN_ETFS = {
    # Canadian-listed (FX-free)
    "VFV.TO", "XEQT.TO", "XIU.TO", "VDY.TO", "XGRO.TO", "XBAL.TO", "ZAG.TO",
    "VCN.TO", "XIC.TO", "ZCN.TO", "VBAL.TO", "HXQ.TO", "XQQ.TO", "ZEB.TO",
    # US-listed
    "SPY", "VOO", "VTI", "QQQ", "IVV", "SCHD", "DIA", "IWM", "ARKX", "ARKK",
    "XLK", "XLF", "XLE", "XLV", "SMH", "VIG", "BND", "AGG",
}


def is_etf(ticker: str) -> bool:
    return ticker.upper() in KNOWN_ETFS


def _fetch_next_earnings(ticker: str):
    """Raw lookup (network). Returns a tz-naive date-like Timestamp or None."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    dates = []
    try:
        cal = t.calendar  # dict on modern yfinance
        raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if raw:
            dates += list(raw) if isinstance(raw, (list, tuple)) else [raw]
    except Exception:
        pass
    if not dates:
        try:
            df = t.get_earnings_dates(limit=8)
            if df is not None and len(df):
                dates += [d for d in df.index]
        except Exception:
            pass
    now = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None)
    future = []
    for d in dates:
        try:
            ts = pd.Timestamp(d)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(MARKET_TZ).tz_localize(None)
            ts = ts.normalize()
            if ts >= now:
                future.append(ts)
        except Exception:
            continue
    return min(future) if future else None


def next_earnings(ticker: str):
    """Next earnings date (approx) for `ticker`, cached; None if unknown."""
    key = ticker.upper()
    if is_etf(key):
        return None
    if key not in _cache or time.monotonic() - _cached_at.get(key, -1e20) >= CACHE_TTL_SECONDS:
        try:
            _cache[key] = _fetch_next_earnings(key)
        except Exception:
            _cache[key] = None
        _cached_at[key] = time.monotonic()
    return _cache[key]


def earnings_note(ticker: str, exit_by) -> str:
    """One compact clause describing earnings risk for a hold ending `exit_by`.

    Examples: 'earnings ~Jul 30 INSIDE hold - exit before', 'earnings ~Aug 27
    (clear of window)', 'earnings date UNKNOWN - check before holding'.
    """
    date = next_earnings(ticker)
    if is_etf(ticker):
        return "ETF - no single-company earnings gap"
    if date is None:
        return "earnings date UNKNOWN - check before holding"
    label = date.strftime("%b %d")
    try:
        exit_ts = pd.Timestamp(exit_by)
        if exit_ts.tzinfo is not None:
            exit_ts = exit_ts.tz_localize(None)
    except Exception:
        return f"earnings ~{label} - check vs your exit date"
    if date <= exit_ts.normalize():
        return f"earnings ~{label} INSIDE hold - exit before"
    return f"earnings ~{label} (clear of window)"
