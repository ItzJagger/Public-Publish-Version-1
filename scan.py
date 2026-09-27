import argparse
import math
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
from daytrader.policy import DEFAULT_SIGNAL_INTERVAL

from daytrader.clock import (
    describe_now,
    market_is_open,
    market_now,
    minutes_since,
    recommended_extra_views,
    session_phase,
)
from daytrader.earnings import earnings_note
from daytrader.holdings import attach_current, load_holdings, price_status, validate_holdings
from daytrader.playbook import buy_now_candidates, is_us_listed, overnight_gappers, position_picks, swing_picks, weekday_risers
from daytrader.quality import quality_warnings
from daytrader.diagnostics import diagnostic_lines
from daytrader.seasonality import format_clock, format_hour
from daytrader.watchlist import DEFAULT_HOLD_HOURS, scan_watchlist

WIDTH = 64
BARS_PER_TRADING_DAY_30M = 13  # ~6.5h / 30m, for a rough per-bar noise estimate
STALE_MINUTES = 45


def load_watchlist(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the OS clipboard, cross-platform. Returns True on success."""
    if sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        candidates = [["clip"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]

    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return True
        except Exception:
            continue
    return False


def _num(value) -> str:
    return f"{value:.2f}" if value is not None and value == value else "n/a"


def latest_bar_time(results: list[dict]):
    """Most recent bar timestamp seen across results (from the live rec)."""
    stamps = [r["rec"]["as_of"] for r in results if r.get("rec") and r["rec"].get("as_of") is not None]
    return max(stamps) if stamps else None


def print_atr_block(atr_stop: dict) -> None:
    classification = atr_stop.get("classification")
    hold_bars = atr_stop.get("hold_bars")

    if classification:
        buy_hour = atr_stop.get("buy_hour")
        sell_hour = atr_stop.get("sell_hour")
        sell_minute = atr_stop.get("sell_minute", 0)
        est = " (entry hour estimated - thin timing history)" if atr_stop.get("hours_estimated") else ""
        print(
            f"  Plan hold:      ~{format_clock(buy_hour, atr_stop.get('buy_minute', 0))} -> ~{format_clock(sell_hour, sell_minute)} ET  "
            f"({classification}, {hold_bars} x 15m bars){est}"
        )

    raw_high = atr_stop.get("raw_best_sell_hour")
    if raw_high is not None:
        note = "  (an overnight pattern - not traded that way here)" if atr_stop.get("raw_overnight") else ""
        print(f"  Hist. high hr:  ~{format_hour(raw_high)} ET{note}")

    print(f"  Single-bar ATR: {_num(atr_stop.get('single_bar_atr'))}   Horizon adverse move: {_num(atr_stop.get('horizon_atr'))}")
    if atr_stop.get("risk_model") == "empirical-horizon-v1":
        print(f"  Risk sample:    {atr_stop.get('samples', 0)} prior same-time sessions; independent stop/target")
    if atr_stop.get('risk_model') == 'structure-v6-unvalidated':
        print("  Risk model:     structure + ATR; planned target, not a forecast")
    print(f"  MAE floor dist: {_num(atr_stop.get('mae_distance'))}")
    if classification == "OVERNIGHT":
        print(f"  Typical gap:    {_num(atr_stop.get('typical_gap'))}   Worst gap: {_num(atr_stop.get('worst_gap'))}")
    if atr_stop.get("consistency_bug"):
        print("  [BUG] hours were labeled INTRADAY but fall outside one session - reclassified OVERNIGHT")
    if "floor" in atr_stop:
        anchor = " (anchored to your entry)" if atr_stop.get("anchored_to") == "entry" else ""
        print(f"  Floor:          {atr_stop['floor']:.2f}   Ceiling: {atr_stop['ceiling']:.2f}   R:R {atr_stop['rr']:.2f}:1{anchor}")


def print_ticker_block(result: dict, show_ai: bool) -> None:
    ticker = result["ticker"]
    print("=" * WIDTH)

    if "error" in result:
        print(f"  {ticker}")
        print("-" * WIDTH)
        print(f"  Error: {result['error']}")
        print()
        return

    season = result["season"]
    is_meanrev = result.get("atr_stop") is not None

    if result.get("no_trade"):
        atr_stop = result["atr_stop"]
        print(f"  {ticker:<10}${result['current_price']:,.2f}")
        print("-" * WIDTH)
        print(f"  Status:         {atr_stop['status']}")
        print(
            f"  Strategy:       {result['strategy']}  "
            f"(risk: {result['risk']}, avg daily range {result['dispersion']['avg_daily_range_pct']:.2f}%)"
        )
        print()
        print_atr_block(atr_stop)
        for line in textwrap.wrap(f"Reason: {atr_stop['reason']}", width=WIDTH - 4):
            print(f"  {line}")
        print()
    else:
        rec = result["rec"]

        print(f"  {ticker:<10}${rec['current_price']:,.2f}")
        print("-" * WIDTH)
        print(f"  Status:         {rec['status']}")
        if rec['status'] == 'BLOCKED':
            print('  Reasons:        ' + '; '.join(result['eligibility']['buy-now']['reasons']))
        print(
            f"  Strategy:       {result['strategy']}  "
            f"(risk: {result['risk']}, avg daily range {result['dispersion']['avg_daily_range_pct']:.2f}%)"
        )
        print()

        if is_meanrev:
            print_atr_block(result["atr_stop"])
            print()

        if rec["status"] in ("IN_POSITION", "BUY_NOW"):
            print(f"  Entry price:    {rec['entry_price']:.2f}  (at {rec['entry_time']})")
            print(f"  Unrealized:     {rec['unrealized_pct']:+.2f}%")
            if is_meanrev:
                print("  Set the Floor/Ceiling above as your stop / limit orders.")
            else:
                print(f"  Ceiling (sell): {rec['ceiling_price']:.2f}   (+{result['take_profit_pct']:.2f}%)")
                print(f"  Floor (sell):   {rec['floor_price']:.2f}   (-{result['stop_loss_pct']:.2f}%)")
        else:
            print(f"  RSI:            {rec['rsi']:.1f}")
            print("  Action:         wait for next buy signal")
        print()

    if season.get("insufficient_data"):
        print(f"  Timing data:    not enough history yet ({season['num_days']} days)")
    else:
        print(
            f"  Best buy time:  ~{format_hour(season['best_buy_hour'])} local market time "
            f"(historically near the daily low)"
        )
        print(
            f"  Best sell time: ~{format_hour(season['best_sell_hour'])} local market time "
            f"(historically near the daily high)"
        )
        if season.get("best_day"):
            print(f"  Strongest day:  {season['best_day']}  (avg {season['best_day_return_pct']:+.2f}%)")
        if season.get("worst_day"):
            print(f"  Weakest day:    {season['worst_day']}  (avg {season['worst_day_return_pct']:+.2f}%)")
    print()

    if show_ai:
        from daytrader.ai_judge import get_ai_recommendation

        print("  AI take:")
        try:
            take = get_ai_recommendation(ticker, result)
            for line in textwrap.wrap(take, width=WIDTH - 4):
                print(f"    {line}")
        except Exception as exc:
            print(f"    Unavailable ({exc})")
        print()


MODE_HORIZON_DAYS = {"intraday": 0, "swing": 7, "long": 35}
MODE_LABEL = {"intraday": "day-trade", "swing": "swing (days)", "long": "long hold (weeks)"}


def _holding_exit_by(h: dict) -> str:
    days = MODE_HORIZON_DAYS.get(h["mode"], 7)
    return str((pd.Timestamp(h["date"]) + pd.Timedelta(days=days)).date())


def _horizon_flag(h: dict, now) -> str:
    """Text flag if a position is at/past the horizon for its mode."""
    exit_by = pd.Timestamp(_holding_exit_by(h))
    today = pd.Timestamp(now).normalize().tz_localize(None)
    if h["mode"] == "intraday" and today > pd.Timestamp(h["date"]).normalize():
        return "STALE DAY-TRADE - an intraday buy should be closed the same day"
    if today > exit_by:
        return f"PAST EXIT HORIZON (was ~{exit_by.date()}) - close it"
    if today == exit_by:
        return "EXIT HORIZON IS TODAY"
    return ""


def holdings_lines(holds: list[dict], now=None, issues=None) -> list[str]:
    """Section 0: what the user ACTUALLY owns (from holdings.txt), with live
    price, P&L, level/horizon/earnings status for both day-trades and holds."""
    now = now if now is not None else market_now()
    lines = ["=" * WIDTH, "  0) YOUR OPEN POSITIONS  -  from holdings.txt (user-held, real)", "=" * WIDTH]
    if issues:
        lines.append("  HOLDINGS FILE NEEDS ATTENTION:")
        for msg in issues:
            lines.append(f"    - {msg}")
        lines.append("")
    if not holds:
        lines.append("  (none listed - edit holdings.txt when you buy or sell)")
        return lines
    for h in holds:
        label = MODE_LABEL.get(h["mode"], h["mode"])
        if h.get("quote_missing"):
            cur, pnl = "no quote", "n/a"
        else:
            cur = f"{h['current']:.2f}"
            pnl = f"{h['pnl_pct']:+.2f}%" if h.get("pnl_pct") is not None else "n/a"
        fl = f"{h['floor']:.2f}" if h.get("floor") else "-"
        ce = f"{h['ceiling']:.2f}" if h.get("ceiling") else "-"
        us = "  | US (FX on exit)" if is_us_listed(h["ticker"]) else ""
        lines.append(
            f"  {h['ticker']:<9} {label:<18} entry {h['entry']:.2f} ({h['date']}) | "
            f"now {cur} ({pnl}) | floor {fl} | ceiling {ce}{us}"
        )
        status = price_status(h)
        hz = _horizon_flag(h, now)
        if hz:
            status.append(hz)
        if h["mode"] != "intraday":
            status.append(earnings_note(h["ticker"], _holding_exit_by(h)))
        if h.get("quote_missing"):
            status.append("no live quote this scan - verify price in your broker")
        detail = " | ".join(s for s in status if s)
        lines.append(f"    {detail}" if detail else f"    holding OK - horizon ends ~{_holding_exit_by(h)}")
    lines.append("  Follow your broker stop/target, sell-by deadline and earnings plan. Stop-limit fills are not guaranteed.")
    return lines


def summary_lines(results: list[dict], holds: list[dict], market_open: bool = True) -> list[str]:
    """A 3-6 line decision summary so the pasted-to AI sees the point first."""
    lines = ["-" * WIDTH, "  DECISION SUMMARY"]
    lines.extend("  " + line for line in diagnostic_lines(results))
    buys = buy_now_candidates(results)
    lines.append(f"  Fresh BUY signals this scan: {len(buys)}"
                 + (f" ({', '.join(b['ticker'] for b in buys)})" if buys else " (none - waiting is fine)"))
    if holds:
        parts = [f"{h['ticker']} ({h['mode']})" for h in holds]
        lines.append(f"  User-held positions needing exit calls: {', '.join(parts)}")
    else:
        lines.append("  User-held positions: none listed")
    for r in results:
        for mode, gate in r.get('eligibility', {}).items():
            if gate['reasons'] and 'no confirmed setup' not in gate['reasons']:
                lines.append(f"  {r['ticker']} {mode} BLOCKED: {'; '.join(gate['reasons'])}")
    qw = quality_warnings(results, include_stale=market_open)
    if qw:
        lines.append("  DATA-QUALITY FLAGS (verify these in the broker before acting):")
        lines.extend(qw)
    lines.append("-" * WIDTH)
    return lines


def _buy_now_levels(result: dict) -> str:
    if result.get("atr_stop") and "floor" in result["atr_stop"]:
        a = result["atr_stop"]
        return f"Floor {a['floor']:.2f} | Ceiling {a['ceiling']:.2f}"
    rec = result["rec"]
    return f"Floor {rec['floor_price']:.2f} | Ceiling {rec['ceiling_price']:.2f}"


def buy_now_lines(results: list[dict]) -> list[str]:
    lines = ["=" * WIDTH, "  1) BUY NOW  -  confirmed setup; intended hold ~1-2h", "=" * WIDTH]
    cands = buy_now_candidates(results)
    if not cands:
        lines.append("  No fresh buy signals right now. (This is common; waiting is fine.)")
        return lines
    for r in cands:
        rec = r["rec"]
        lines.append(f"  {r['ticker']:<9} {r['strategy']:<8} buy ~{r.get('eligibility', {}).get('buy-now', {}).get('entry_quote', rec['current_price']):.2f}  |  {_buy_now_levels(r)}")
        if is_us_listed(r["ticker"]):
            lines.append("    - US-listed: ~1.5% FX each way on a CAD account can swamp the target")
        if r["strategy"] == "trend" and not r.get("atr_stop"):
            adr = r["dispersion"]["avg_daily_range_pct"]
            per_bar_noise = adr / math.sqrt(BARS_PER_TRADING_DAY_30M)
            if r["stop_loss_pct"] < per_bar_noise:
                lines.append(
                    f"    - stop ({r['stop_loss_pct']:.2f}%) sits inside one 30m bar of typical "
                    f"noise (~{per_bar_noise:.2f}%): likely to be wiggled out"
                )
    return lines


def weekday_lines(results: list[dict], today_weekday: str) -> list[str]:
    lines = [
        "=" * WIDTH,
        f"  2) MORNING HOLD  -  strong on {today_weekday}s (buy early, hold the day)",
        "=" * WIDTH,
    ]
    rows = weekday_risers(results, today_weekday)
    if not rows:
        lines.append(f"  No names in your watchlist are historically strongest on {today_weekday}.")
        return lines
    lines.append("  (ranked: intraday timing-agreement first, then tilt size)")
    for x in rows:
        timing = "timing AGREES" if x["timing_agrees"] else "timing mismatch"
        hrs = ""
        if x["best_buy_hour"] is not None and x["best_sell_hour"] is not None:
            hrs = f" (low ~{format_hour(x['best_buy_hour'])} -> high ~{format_hour(x['best_sell_hour'])})"
        us = " | US (FX cost)" if x["is_us"] else ""
        lines.append(f"  {x['ticker']:<9} tilt {x['tilt_pct']:+.2f}% | {timing}{hrs}{us}")
    lines.append("  Weekday tilts are ~60-day noise; a buy needs its own signal, not just the day.")
    return lines


def overnight_lines(results: list[dict]) -> list[str]:
    lines = ["=" * WIDTH, "  3) OVERNIGHT  -  buy late, hold to next morning", "=" * WIDTH]
    rows = overnight_gappers(results)
    if not rows:
        lines.append("  No positive overnight gap-up tendencies. Nothing to hold overnight.")
        return lines
    lines.append("  (filtered for stable net evidence; ranked by penalized net mean)")
    for x in rows:
        us = " | US (FX cost)" if x["is_us"] else ""
        lines.append(
            f"  {x['ticker']:<9} up {x['hit_rate']*100:.0f}% of nights | avg {x['mean_gap_pct']:+.2f}% | "
            f"worst night {x['worst_gap_pct']:+.2f}% | {x['samples']} nights{us}"
        )
    lines.append("  WARNING: a stop CANNOT protect an overnight hold - if the stock gaps")
    lines.append("  down at the open you're filled below your floor. The 'worst night'")
    lines.append("  column is historical only: future losses can be larger.")
    return lines


def swing_lines(results: list[dict]) -> list[str]:
    lines = ["=" * WIDTH, "  4) SWING  -  buy a dip, hold a few days", "=" * WIDTH]
    rows = swing_picks(results)
    if not rows:
        lines.append("  No names are sitting near a recent low and turning up right now.")
        return lines
    lines.append("  (daily: confirmed uptrend pullback, liquidity and net R:R gates)")
    for x in rows:
        us = " | US (FX cost)" if x["is_us"] else ""
        lines.append(
            f"  {x['ticker']:<9} buy ~{x['price']:.2f} | floor {x['floor']:.2f} | "
            f"ceiling {x['ceiling']:.2f}  (RSI {x['rsi']:.0f}, {x['dist_from_low_pct']:+.1f}% off 20d low){us}"
        )
        sell = (f"    sell: target {x['ceiling']:.2f}, or by the close on {x['exit_by']} "
                f"(~{x['hold_days']} trading days); stop {x['floor']:.2f}")
        if x.get("best_weekday"):
            sell += f"; if near target, historically firmest on {x['best_weekday']}"
        lines.append(sell)
        lines.append(f"    {earnings_note(x['ticker'], x['exit_by'].split(None, 1)[-1] if isinstance(x['exit_by'], str) else x['exit_by'])}")
    lines.append("  Multi-day = nightly gap risk no stop covers; CHECK earnings. Weak edge, size small.")
    return lines


def position_lines(results: list[dict]) -> list[str]:
    lines = ["=" * WIDTH, "  5) LONG HOLD  -  buy an uptrend pullback, hold weeks", "=" * WIDTH]
    rows = position_picks(results)
    if not rows:
        lines.append("  No uptrend pullbacks lining up right now.")
        return lines
    lines.append("  (daily; long-term uptrend, pulled back to the 50-day; ranked by trend strength)")
    for x in rows:
        us = " | US (FX cost)" if x["is_us"] else ""
        mom = x.get("mom_3m_pct")
        mom_str = f", 3mo {mom:+.0f}%" if mom is not None else ""
        lines.append(
            f"  {x['ticker']:<9} buy ~{x['price']:.2f} | floor {x['floor']:.2f} | "
            f"ceiling {x['ceiling']:.2f}  (RSI {x['rsi']:.0f}, {x['dist_sma50_pct']:+.1f}% vs 50-day{mom_str}){us}"
        )
        lines.append(
            f"    plan: target {x['ceiling']:.2f}, or review by {x['review_by']} "
            f"(~{x['hold_weeks']} weeks); stop {x['floor']:.2f} (below the trend)"
        )
        lines.append(f"    {earnings_note(x['ticker'], x['review_by'].split(None, 1)[-1] if isinstance(x['review_by'], str) else x['review_by'])}")
    lines.append("  Weeks = nightly gaps + earnings risk; the stop is your line. CHECK earnings, size small.")
    return lines


def build_sections(results: list[dict], extra_views: set, today_weekday: str) -> list[list[str]]:
    """The summary sections to show, in order. Buy-now always; the others by view."""
    sections = [buy_now_lines(results)]
    if "morning" in extra_views:
        sections.append(weekday_lines(results, today_weekday))
    if "overnight" in extra_views:
        sections.append(overnight_lines(results))
    if "swing" in extra_views:
        sections.append(swing_lines(results))
    if "position" in extra_views:
        sections.append(position_lines(results))
    return sections


def resolve_views(view: str, phase: str) -> set:
    """Which extra sections to include, from an explicit view or the time of day."""
    if view == "auto":
        return recommended_extra_views(phase)
    if view == "all":
        return {"morning", "overnight", "swing", "position"}
    if view == "day":
        return {"morning", "overnight", "swing"}
    if view in ("morning", "overnight", "swing", "position"):
        return {view}
    return set()  # intraday / buy-now only


def run_scan(view: str = "auto", tickers=None, watchlist: str = "watchlist.txt",
             interval: str = DEFAULT_SIGNAL_INTERVAL, period: str = "60d", take_profit: float = 0.75,
             stop_loss: float = 0.4, hold_hours: float = DEFAULT_HOLD_HOURS,
             holdings_path: str = "holdings.txt") -> dict:
    """Run the watchlist scan and return everything needed to render it (used by
    both the command line and the phone server).
    """
    now = market_now()
    phase = session_phase(now)
    extra_views = resolve_views(view, phase)
    tickers = [t.upper() for t in tickers] if tickers else load_watchlist(watchlist)
    holds, hold_issues = validate_holdings(holdings_path)
    held_tickers = [h["ticker"] for h in holds if h["ticker"] not in tickers]
    results = scan_watchlist(
        tickers + held_tickers,          # always scan held names too
        interval=interval,
        period=period,
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        hold_hours=hold_hours,
        include_overnight="overnight" in extra_views,
        include_swing="swing" in extra_views,
        include_position="position" in extra_views,
    )
    holds = attach_current(holds, results)
    return {"results": results, "now": now, "phase": phase,
            "extra_views": extra_views, "holdings": holds, "holdings_issues": hold_issues}


def scan_holdings_only(holdings_path: str = "holdings.txt", interval: str = DEFAULT_SIGNAL_INTERVAL,
                       period: str = "60d") -> dict:
    """Scan ONLY the names you hold (fast) - for a 'check my positions' view."""
    now = market_now()
    holds, issues = validate_holdings(holdings_path)
    tickers = [h["ticker"] for h in holds]
    results = scan_watchlist(tickers, interval=interval, period=period,
                             include_position=True) if tickers else []
    holds = attach_current(holds, results)
    return {"results": results, "now": now, "holdings": holds, "holdings_issues": issues}


def build_holdings_text(now, holdings, holdings_issues=None) -> str:
    """Positions-only paste text: what you hold + explicit hold/sell instruction."""
    header = f"Positions check: {describe_now(now)}\n\n"
    body = "\n".join(holdings_lines(holdings, now=now, issues=holdings_issues))
    if not holdings:
        return header + body
    ask = (
        "\n\nFor EACH position above give a clear HOLD or SELL call with a one-line reason, "
        "restate its floor/ceiling, and name the exit deadline (its earnings date or horizon "
        "end, whichever is sooner) as a concrete date. Verify each earnings date with web "
        "search - the dates here are approximate. These are the user's REAL holdings."
    )
    return header + body + ask


def build_prompt_text(results: list[dict], now, extra_views: set, holdings=None,
                      holdings_issues=None) -> str:
    """The exact paste-into-Claude text: dated header + decision summary +
    holdings + summary sections + per-ticker prompt."""
    from daytrader.ai_judge import build_batch_prompt

    holdings = holdings if holdings is not None else []
    bar_ts = latest_bar_time(results)
    mins = minutes_since(bar_ts, now)
    ago = f"{mins:.0f} min ago" if mins is not None else "unknown"
    bar_line = bar_ts.strftime("%Y-%m-%d %H:%M ET") if bar_ts is not None else "n/a"
    dated_header = f"Scan time: {describe_now(now)}\nLatest data bar: {bar_line} ({ago})\n\n"
    top = "\n".join(summary_lines(results, holdings, market_open=market_is_open(now))) + "\n\n"
    hold_block = "\n".join(holdings_lines(holdings, now=now, issues=holdings_issues)) + "\n\n" if (holdings or holdings_issues) else ""
    sections = build_sections(results, extra_views, now.day_name())
    playbook_block = "\n\n".join("\n".join(sec) for sec in sections)
    return dated_header + top + hold_block + playbook_block + "\n\n" + build_batch_prompt(results, holdings=holdings)


def main():
    parser = argparse.ArgumentParser(description="Scan a watchlist for buy/sell/hold recommendations")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols to scan. If omitted, reads watchlist.txt")
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist file")
    parser.add_argument("--holdings", default="holdings.txt", help="Path to your open-positions file")
    parser.add_argument("--interval", default=DEFAULT_SIGNAL_INTERVAL)
    parser.add_argument("--period", default="60d", help="History window to pull, e.g. 60d, 1mo, 5d")
    parser.add_argument("--take-profit", type=float, default=0.75)
    parser.add_argument("--stop-loss", type=float, default=0.4)
    parser.add_argument(
        "--hold-hours", type=float, default=DEFAULT_HOLD_HOURS,
        help=(
            "Planned intraday hold length for mean-reversion names (the main dial). "
            f"Default {DEFAULT_HOLD_HOURS:g}."
        ),
    )
    parser.add_argument(
        "--view", choices=["auto", "intraday", "morning", "overnight", "swing", "position", "all"], default="auto",
        help=(
            "Which sections to show. 'auto' (default) picks by time of day: the morning "
            "hold list before ~11am, the overnight list after ~2:30pm. The buy-now list "
            "always shows. 'swing' adds the buy-a-dip/hold-a-few-days list; 'position' adds "
            "the uptrend-pullback/hold-weeks list. 'all' shows everything."
        ),
    )
    parser.add_argument(
        "--ai", action="store_true",
        help="Ask Claude for a narrative take on each ticker (requires ANTHROPIC_API_KEY; small cost per call)",
    )
    parser.add_argument(
        "--prompt", action="store_true",
        help="Print (and copy) a dated prompt with today's scan data, ready to paste into claude.ai - free",
    )
    args = parser.parse_args()

    scan = run_scan(
        view=args.view,
        tickers=args.tickers or None,
        watchlist=args.watchlist,
        interval=args.interval,
        period=args.period,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        hold_hours=args.hold_hours,
        holdings_path=args.holdings,
    )
    results, now, extra_views = scan["results"], scan["now"], scan["extra_views"]
    holds = scan["holdings"]
    hold_issues = scan.get("holdings_issues")

    bar_ts = latest_bar_time(results)
    mins = minutes_since(bar_ts, now)

    print()
    print("  WATCHLIST SCAN")
    print(f"  {describe_now(now)}")
    print(f"  {args.period} of {args.interval} bars | hold ~{args.hold_hours:g}h | {len(results)} tickers")
    if bar_ts is not None:
        ago = f"{mins:.0f} min ago" if mins is not None else "time unknown"
        print(f"  Latest data bar: {bar_ts.strftime('%Y-%m-%d %H:%M')} ET ({ago})")
        if mins is not None and market_is_open(now) and mins > STALE_MINUTES:
            print(f"  NOTE: latest bar is over {STALE_MINUTES} min old - data may be lagging or the feed is stale.")
    if not market_is_open(now):
        print("  NOTE: market is closed right now - prices/signals are from the last session.")
    print()

    for result in results:
        print_ticker_block(result, show_ai=args.ai)

    print("\n".join(summary_lines(results, holds, market_open=market_is_open(now))))
    print()
    if holds or hold_issues:
        print("\n".join(holdings_lines(holds, now=now, issues=hold_issues)))
        print()
    sections = build_sections(results, extra_views, now.day_name())
    for sec in sections:
        print("\n".join(sec))
        print()

    print("=" * WIDTH)
    print()
    print("  Notes:")
    print("  - Review scan coverage and risk rejection reasons when alerts are absent.")
    print("  - This screens for setups. It does NOT guarantee a profit - no tool")
    print("    can. The weekday/overnight edges are weak and mostly noise.")
    print("  - Signals only, not instructions. Execute manually in Wealthsimple")
    print("    if you agree with the call.")
    print()

    if args.prompt:
        prompt_text = build_prompt_text(results, now, extra_views, holdings=holds, holdings_issues=hold_issues)

        print("=" * WIDTH)
        print("  PASTE-INTO-CLAUDE.AI PROMPT")
        print("=" * WIDTH)
        print()
        print(prompt_text)
        print()

        if copy_to_clipboard(prompt_text):
            print("  (Copied to clipboard - paste into a claude.ai chat.)")
        else:
            print("  (Could not copy to clipboard automatically - copy the text above.)")
        print()


if __name__ == "__main__":
    main()
