"""Data-quality guard: catch the two ways a scan silently lies.

1. STALE data - the latest bar is old (feed hiccup, halted name, pre-open),
   so every 'current price' downstream is actually minutes-to-hours stale.
2. IMPLAUSIBLE PRINT - the last price is far from the prior daily close. That
   is *either* real news *or* a bad tick from the feed (this tool saw Walmart
   print a phantom -6% on 2026-07-01 with no news). One feed cannot tell which,
   so the honest output is a VERIFY flag: check the price in your broker before
   acting on any signal for that name.

These checks never block a scan; they annotate it. A flagged name's numbers
should be confirmed against the broker before any order.
"""
import pandas as pd

STALE_MINUTES = 45           # older than this during a session is suspicious
BIG_MOVE_PCT = 8.0           # |last vs prior daily close| beyond this -> VERIFY


def check_quality(df: pd.DataFrame, now=None) -> dict:
    """Quality flags for one ticker's intraday frame.

    Returns {stale_minutes, is_stale, move_pct, verify_move, note}.
    Never raises; on any trouble returns a 'could not assess' note.
    """
    out = {"stale_minutes": None, "is_stale": False, "move_pct": None,
           "verify_move": False, "note": ""}
    try:
        if df is None or len(df) < 2:
            out["note"] = "too little data to quality-check"
            return out
        last_ts = df.index[-1]
        if now is None:
            now = pd.Timestamp.now(tz=getattr(last_ts, "tz", None))
        age = (now - last_ts).total_seconds() / 60.0
        out["stale_minutes"] = age
        out["is_stale"] = age > STALE_MINUTES

        last_close = float(df["Close"].iloc[-1])
        # prior *daily* close: last bar of the previous calendar day if present,
        # else just the previous bar (short frames)
        days = df.index.normalize()
        prev_mask = days < days[-1]
        prev_close = float(df["Close"][prev_mask].iloc[-1]) if prev_mask.any() else float(df["Close"].iloc[-2])
        if prev_close:
            move = (last_close / prev_close - 1) * 100
            out["move_pct"] = move
            out["verify_move"] = abs(move) >= BIG_MOVE_PCT

        notes = []
        if out["is_stale"]:
            notes.append(f"data is {age:.0f} min old")
        if out["verify_move"]:
            notes.append(f"{out['move_pct']:+.1f}% vs prior close - VERIFY price in broker (real news or bad tick)")
        out["note"] = "; ".join(notes)
        return out
    except Exception:
        out["note"] = "could not assess data quality"
        return out


def quality_warnings(results: list[dict], include_stale: bool = True) -> list[str]:
    """Compact warning lines for flagged names. Staleness is only meaningful
    while the market is open (after hours every bar is 'old'), so pass
    include_stale=False when the market is closed."""
    lines = []
    for r in results:
        q = r.get("quality")
        if not q:
            continue
        flags = []
        if include_stale and q.get("is_stale") and q.get("stale_minutes") is not None:
            flags.append(f"data is {q['stale_minutes']:.0f} min old")
        if q.get("verify_move") and q.get("move_pct") is not None:
            flags.append(f"{q['move_pct']:+.1f}% vs prior close - VERIFY price in broker (real news or bad tick)")
        if flags:
            lines.append(f"  {r['ticker']:<9} {'; '.join(flags)}")
    return lines
