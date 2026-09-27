"""Holdings tracker: the tool's record of what YOU actually own.

Why this exists: the scanner's IN_POSITION statuses are *replays* of its own
signals, not your account - which caused real confusion. This file is the
source of truth for your open positions, so every scan and every copied prompt
states plainly what is held, at what entry, with what exit plan - for both
day-trades and longer-term holds.

Format of holdings.txt - one position per line, comments with '#':

    TICKER  ENTRY_PRICE  BUY_DATE     MODE   [FLOOR]  [CEILING]
    DEMO    100.00       2026-01-05   long   95.00    110.00
    EXAMPLE.TO 50.00     2026-01-05   swing  48.00    54.00

MODE is one of: intraday, swing, long. FLOOR/CEILING are optional; if given
they are shown as your stop/target. Edit this file when you buy or sell - the
tool never writes it. Bad lines are reported (see validate_holdings), never
silently dropped.
"""
from pathlib import Path

import pandas as pd

VALID_MODES = ("intraday", "swing", "long")


def validate_holdings(path: str = "holdings.txt") -> tuple[list[dict], list[str]]:
    """Parse holdings.txt into (positions, issues).

    `positions` is the list of valid position dicts. `issues` is a list of
    human-readable problems (bad lines, duplicate tickers, illogical
    floor/ceiling) so nothing fails silently. Missing file -> ([], []).
    """
    p = Path(path)
    if not p.exists():
        return [], []

    rows: list[dict] = []
    issues: list[str] = []
    seen: dict[str, int] = {}

    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            issues.append(f"line {lineno}: need at least TICKER ENTRY DATE MODE - skipped ('{line}')")
            continue

        ticker = parts[0].upper()
        try:
            entry = float(parts[1])
        except ValueError:
            issues.append(f"line {lineno}: entry price '{parts[1]}' is not a number - skipped")
            continue
        if entry <= 0:
            issues.append(f"line {lineno}: entry price must be positive - skipped")
            continue
        try:
            date = str(pd.Timestamp(parts[2]).date())
        except (ValueError, TypeError):
            issues.append(f"line {lineno}: date '{parts[2]}' is not YYYY-MM-DD - skipped")
            continue
        mode = parts[3].lower()
        if mode not in VALID_MODES:
            issues.append(f"line {lineno}: mode '{parts[3]}' must be one of {VALID_MODES} - skipped")
            continue

        floor = ceiling = None
        if len(parts) > 4:
            try:
                floor = float(parts[4])
            except ValueError:
                issues.append(f"line {lineno} ({ticker}): floor '{parts[4]}' is not a number - ignored")
        if len(parts) > 5:
            try:
                ceiling = float(parts[5])
            except ValueError:
                issues.append(f"line {lineno} ({ticker}): ceiling '{parts[5]}' is not a number - ignored")

        # logical checks (warn, keep the row)
        if floor is not None and floor >= entry:
            issues.append(f"{ticker}: floor {floor:.2f} is at/above entry {entry:.2f} - a stop should sit below")
        if ceiling is not None and ceiling <= entry:
            issues.append(f"{ticker}: ceiling {ceiling:.2f} is at/below entry {entry:.2f} - a target should sit above")
        if floor is not None and ceiling is not None and floor >= ceiling:
            issues.append(f"{ticker}: floor {floor:.2f} is not below ceiling {ceiling:.2f}")
        if ticker in seen:
            issues.append(f"{ticker}: listed more than once (lines {seen[ticker]} and {lineno}) - using the first")
            continue
        seen[ticker] = lineno

        rows.append({"ticker": ticker, "entry": entry, "date": date,
                     "mode": mode, "floor": floor, "ceiling": ceiling})
    return rows, issues


def load_holdings(path: str = "holdings.txt") -> list[dict]:
    """Just the valid positions (see validate_holdings for the issue list)."""
    rows, _ = validate_holdings(path)
    return rows


def attach_current(holdings: list[dict], results: list[dict]) -> list[dict]:
    """Add current price / unrealized % to each holding from this scan's results.

    Prefers the recommendation's current price, falls back to the ticker's last
    bar. If a held name wasn't in the scan at all, current stays None and a flag
    says so - so a missing quote is visible, never a silently wrong P&L.
    """
    price_by_ticker: dict[str, float] = {}
    scanned = set()
    for r in results:
        tk = r["ticker"].upper()
        scanned.add(tk)
        rec = r.get("rec") or {}
        px = rec.get("current_price")
        if px is None or px != px:
            px = r.get("last_price")
        if px is not None and px == px:
            price_by_ticker[tk] = float(px)

    out = []
    for h in holdings:
        cur = price_by_ticker.get(h["ticker"])
        pnl = (cur / h["entry"] - 1) * 100 if (cur and h["entry"]) else None
        out.append({**h, "current": cur, "pnl_pct": pnl,
                    "quote_missing": h["ticker"] not in scanned or cur is None})
    return out


def price_status(h: dict) -> list[str]:
    """Level-based flags for a holding (ceiling hit / floor breached).

    Pure price-vs-levels; horizon and earnings live in the scan renderer so this
    stays dependency-free. Returns e.g. ['AT/ABOVE CEILING - take profit'].
    """
    flags = []
    cur, fl, ce = h.get("current"), h.get("floor"), h.get("ceiling")
    if cur is None:
        return flags
    if ce is not None and cur >= ce:
        flags.append("AT/ABOVE CEILING - take profit")
    elif fl is not None and cur <= fl:
        flags.append("AT/BELOW FLOOR - stop hit, exit")
    elif fl is not None and ce is not None:
        span = ce - fl
        if span > 0 and (cur - fl) / span < 0.15:
            flags.append("near floor - watch the stop")
    return flags
