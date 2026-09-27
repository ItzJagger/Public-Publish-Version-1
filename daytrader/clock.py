"""Current date/time and market-session awareness.

Everything is reported in US/Eastern (NYSE and TSX both trade on ET), so the
header the scanner prints matches the timezone the price bars are in. Uses
pandas for the timezone conversion (already a hard dependency) so there's no
separate tzdata/zoneinfo requirement.
"""
import pandas as pd

from .data import MARKET_TZ

# Phase boundaries in ET, as decimal hours (9.5 == 9:30am).
_OPEN = 9.5
_MORNING_END = 11.0
_MIDDAY_END = 14.5
_CLOSE = 16.0


def market_now() -> pd.Timestamp:
    """Current wall-clock time in market (Eastern) time."""
    return pd.Timestamp.now(tz=MARKET_TZ)


def session_phase(now: pd.Timestamp | None = None) -> str:
    """Coarse label for where we are in the trading day: weekend, pre-market,
    morning, midday, late-day, or after-hours.
    """
    now = now or market_now()
    if now.weekday() >= 5:
        return "weekend"
    t = now.hour + now.minute / 60.0
    if t < _OPEN:
        return "pre-market"
    if t < _MORNING_END:
        return "morning"
    if t < _MIDDAY_END:
        return "midday"
    if t < _CLOSE:
        return "late-day"
    return "after-hours"


def market_is_open(now: pd.Timestamp | None = None) -> bool:
    now = now or market_now()
    from .trade_plan import session_bounds
    return any(bounds is not None and bounds[0] <= now < bounds[1]
               for bounds in (session_bounds('SPY', now), session_bounds('XIU.TO', now)))


def describe_now(now: pd.Timestamp | None = None) -> str:
    """One-line human header, e.g. 'Tuesday, 2026-06-23, 09:47 ET (morning)'."""
    now = now or market_now()
    return f"{now.day_name()}, {now.strftime('%Y-%m-%d, %H:%M')} ET ({session_phase(now)})"


def minutes_since(ts, now: pd.Timestamp | None = None) -> float | None:
    """Minutes between a (tz-aware) bar timestamp and now. None if not usable."""
    if ts is None:
        return None
    now = now or market_now()
    try:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(MARKET_TZ)
        return (now - ts.tz_convert(MARKET_TZ)).total_seconds() / 60.0
    except Exception:
        return None


def recommended_extra_views(phase: str) -> set[str]:
    """Which time-specific sections fit the current phase. The buy-now section
    always shows; this only decides the two scheduled extras.

    - morning / pre-market  -> the weekday-riser ("hold through today") view
    - late-day / after-hours -> the overnight ("hold to next morning") view
    - midday                 -> neither (just buy-now)
    """
    if phase in ("pre-market", "morning"):
        return {"morning"}
    if phase in ("late-day", "after-hours"):
        return {"overnight"}
    return set()
