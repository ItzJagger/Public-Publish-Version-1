"""Overnight edge: does a stock tend to gap UP from today's close to tomorrow's
open? This backs the "buy late, hold to next morning" view.

Big honesty caveat lives with the data it produces: overnight returns are
driven by news, earnings, and macro that land while the market is closed, the
sample here is small (~3 months of daily bars), and an intraday stop cannot
protect an overnight hold - the worst historical gap is the real downside, not
any floor. Treat the output as a weak, noisy tendency, never a sure thing.
"""
from .data import fetch_intraday, completed_bars

OVERNIGHT_PERIOD = "3mo"
MIN_SAMPLES = 20


def overnight_edge(ticker: str, period: str = OVERNIGHT_PERIOD) -> dict | None:
    """Stats on the close -> next-open move, in percent. None if there isn't
    enough daily history to say anything.

    Returns: samples, mean/median gap %, hit_rate (share of nights that gapped
    up), average up-gap and down-gap, and the best/worst single gaps.
    """
    try:
        daily = fetch_intraday(ticker, period=period, interval="1d")
    except Exception:
        return None

    daily = completed_bars(daily, '1d')
    if len(daily) < MIN_SAMPLES + 1:
        return None

    prev_close = daily["Close"].shift(1)
    gap_pct = ((daily["Open"] - prev_close) / prev_close * 100).dropna()

    if len(gap_pct) < MIN_SAMPLES:
        return None

    up = gap_pct[gap_pct > 0]
    down = gap_pct[gap_pct < 0]

    return {
        "gap_returns": gap_pct.tolist(),
        "samples": int(len(gap_pct)),
        "mean_gap_pct": float(gap_pct.mean()),
        "median_gap_pct": float(gap_pct.median()),
        "hit_rate": float((gap_pct > 0).mean()),
        "avg_up_pct": float(up.mean()) if len(up) else 0.0,
        "avg_down_pct": float(down.mean()) if len(down) else 0.0,
        "best_gap_pct": float(gap_pct.max()),
        "worst_gap_pct": float(gap_pct.min()),
    }
