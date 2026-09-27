"""Backtest report: how each mode WOULD have done on real history, after costs.

Run it occasionally (not daily) to get a straight read on whether any mode has
actually paid. It downloads price history (on your machine) and replays trades
for each mode - buy-now, swing, overnight, weekday - subtracting realistic costs
(the ~3% round-trip FX on US names from a CAD account, plus a little slippage).

    python backtest_report.py

Decisions use prior information, but these are research proxies, not a replay
of all live gates. Buy-now uses fixed percentage exits rather than live ATR
levels. Long hold uses the shared daily entry screen and a 25-session exit proxy. Earnings, manual reaction delay and portfolio
constraints are not simulated. These limitations prevent treating the report as live performance evidence.

THIS IS HYPOTHETICAL. Past results don't predict the future, the intraday
(buy-now) sample is short because Yahoo only serves ~60 days of 30-minute bars,
and the model ignores taxes, partial fills, and exact intrabar order. It places
no trades. Not financial advice.
"""
import argparse

import numpy as np
import pandas as pd

from daytrader import data as _data  # call _data.fetch_intraday so tests can patch it
from daytrader.swing import swing_from_frame
from daytrader.position import position_from_frame
from daytrader.evidence import tendency
from daytrader.policy import SIGNAL_PROFILE, FX_ROUND_TRIP_PCT, SLIPPAGE_ROUND_TRIP_PCT, net_reward_risk, MIN_NET_RR
from daytrader.backtest import long_exit_on_bar, run_backtest
from daytrader.dispersion import classify_risk, compute_dispersion
from daytrader.indicators import atr, rsi
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.signals import generate_signals
from daytrader.swing import (
    LOOKBACK_LOW, MAX_HOLD_DAYS, MIN_BARS, MIN_CLEAN_RR, NEAR_LOW_PCT,
    OVERSOLD_RSI, RR, STOP_ATR_MULT,
)

WIDTH = 74
FX_US_ROUND_TRIP = FX_ROUND_TRIP_PCT       # ~1.5% each way converting CAD<->USD (Wealthsimple)
SLIPPAGE_ROUND_TRIP = SLIPPAGE_ROUND_TRIP_PCT    # small allowance for imperfect fills
OVERNIGHT_TRAIL = 40         # trailing days to judge an overnight tendency
OVERNIGHT_MIN_HIT = 0.5
WEEKDAY_MIN_HISTORY = 40     # min trailing days before trading a weekday


def is_us(ticker: str) -> bool:
    return not ticker.upper().endswith(".TO")


def cost_pct(ticker: str, fx: float, slippage: float) -> float:
    return (fx if is_us(ticker) else 0.0) + slippage


def _daily(ticker: str, period: str) -> pd.DataFrame:
    return _data.completed_bars(_data.fetch_intraday(ticker, period=period, interval="1d"), "1d")


# ---- per-mode trade generators (each returns a list of net % per trade) ----

def walk_forward_signals(df: pd.DataFrame, min_history_days: int = 5) -> pd.DataFrame:
    """Freeze each session's regime using only earlier sessions.

    Both indicator implementations are causal. Future prices never choose the
    strategy for earlier trades. The first five sessions are a warmup only.
    """
    trend = generate_signals(df)
    meanrev = generate_mean_reversion_signals(df)
    out = trend.copy()
    out["buy"] = False
    out["sell"] = False
    days = pd.Index(df.index.date)
    for day in days.unique():
        past = df.loc[days < day]
        if len(pd.Index(past.index.date).unique()) < min_history_days:
            continue
        risk = classify_risk(compute_dispersion(past))
        source = meanrev if risk == "LOW" else trend
        mask = days == day
        out.loc[mask, ["buy", "sell"]] = source.loc[mask, ["buy", "sell"]]
    return out


def buy_now_trades(ticker: str, intraday_period: str, fx: float, slip: float) -> list[float]:
    try:
        df = _data.completed_bars(_data.fetch_intraday(ticker, period=intraday_period, interval="30m"), "30m")
        df = df.loc[df.index.date < pd.Timestamp.now(tz="America/New_York").date()]
        res = run_backtest(df, signal_fn=walk_forward_signals)
    except Exception:
        return []
    if not res["num_trades"]:
        return []
    c = cost_pct(ticker, fx, slip)
    return [p - c for p in res["trades"]["pnl_pct"].tolist()]


def daily_mode_trades(ticker, daily_period, fx, slip, mode='swing'):
    """Use the same pure daily entry rules as live; next-open fills, fixed levels.

    Historical earnings dates and human execution latency are unavailable here.
    """
    try:
        d = _daily(ticker, daily_period)
    except Exception:
        return []
    evaluator = swing_from_frame if mode == 'swing' else position_from_frame
    min_bars, hold = (60, 5) if mode == 'swing' else (210, 25)
    c = cost_pct(ticker, fx, slip)
    trades, i = [], min_bars - 1
    while i < len(d)-1:
        setup = evaluator(ticker, d.iloc[:i+1], asof=d.index[i], cost=c)
        if not setup or not setup['candidate']:
            i += 1
            continue
        entry = float(d['Open'].iloc[i+1])
        # Preserve the planned absolute support/resistance levels across gaps.
        floor, ceiling = setup['floor'], setup['ceiling']
        if net_reward_risk(entry, floor, ceiling, c) < MIN_NET_RR:
            i += 1
            continue
        last = min(i+hold, len(d)-1)
        exit_price = None
        for j in range(i+1, last+1):
            exit_price, _ = long_exit_on_bar(d.iloc[j], floor, ceiling)
            if exit_price is not None:
                break
        if exit_price is None:
            exit_price = float(d['Close'].iloc[last])
        trades.append((exit_price/entry-1)*100-c)
        i = j
    return trades


def swing_trades(ticker: str, daily_period: str, fx: float, slip: float) -> list[float]:
    return daily_mode_trades(ticker, daily_period, fx, slip, 'swing')


def position_trades(ticker: str, daily_period: str, fx: float, slip: float) -> list[float]:
    return daily_mode_trades(ticker, daily_period, fx, slip, 'position')


def overnight_trades(ticker: str, daily_period: str, fx: float, slip: float) -> list[float]:
    try:
        d = _daily(ticker, daily_period)
    except Exception:
        return []
    if len(d) < OVERNIGHT_TRAIL + 5:
        return []
    close, open_ = d["Close"], d["Open"]
    gap = (open_ - close.shift(1)) / close.shift(1) * 100
    trail_mean = gap.rolling(OVERNIGHT_TRAIL).mean()
    trail_hit = (gap > 0).rolling(OVERNIGHT_TRAIL).mean()
    future = (open_.shift(-1) - close) / close * 100        # buy close d, sell open d+1
    c = cost_pct(ticker, fx, slip)

    trades = []
    for i in range(len(d) - 1):
        m, h, f = trail_mean.iloc[i], trail_hit.iloc[i], future.iloc[i]
        if f == f and tendency(gap.iloc[max(0, i-OVERNIGHT_TRAIL+1):i+1], c, 30)["eligible"]:
            trades.append(f - c)
    return trades


def weekday_trades(ticker: str, daily_period: str, fx: float, slip: float) -> list[float]:
    try:
        d = _daily(ticker, daily_period)
    except Exception:
        return []
    if len(d) < WEEKDAY_MIN_HISTORY + 5:
        return []
    day_ret = ((d["Close"] - d["Open"]) / d["Open"] * 100).to_numpy()
    names = np.array(d.index.day_name())
    c = cost_pct(ticker, fx, slip)

    trades = []
    for i in range(WEEKDAY_MIN_HISTORY, len(d)):
        hist = pd.Series(day_ret[:i], index=names[:i])
        means = hist.groupby(level=0).mean()
        means = means[means > 0]
        if means.empty:
            continue
        if (names[i] == means.idxmax() and day_ret[i] == day_ret[i]
            and tendency(hist.loc[hist.index == names[i]], c, 8)["eligible"]):
            trades.append(day_ret[i] - c)
    return trades


# ---- aggregation + reporting ----

def summarize(net: list[float]) -> dict | None:
    if not net:
        return None
    s = pd.Series(net, dtype=float)
    cum = pd.concat([pd.Series([0.0]), s.cumsum()], ignore_index=True)
    drawdown = (cum.cummax() - cum).max()
    return {
        "n": int(len(s)),
        "win": float((s > 0).mean() * 100),
        "avg": float(s.mean()),
        "median": float(s.median()),
        "worst": float(s.min()),
        "best": float(s.max()),
        "total": float(s.sum()),
        "max_dd": float(drawdown),
    }


def verdict(m: dict | None) -> str:
    if m is None or m["n"] < 10:
        return "too few trades to judge"
    if m["avg"] <= 0:
        return "NO edge after costs (lost money on average)"
    if m["avg"] < 0.1:
        return "marginal - within noise"
    return "positive in this sample (still no guarantee)"


def buy_hold_benchmark(tickers: list[str], daily_period: str, fx: float, slip: float) -> float | None:
    rets = []
    for t in tickers:
        try:
            d = _daily(t, daily_period)
            if len(d) > 2:
                rets.append((d["Close"].iloc[-1] / d["Close"].iloc[0] - 1) * 100 - cost_pct(t, fx, slip))
        except Exception:
            continue
    return float(np.mean(rets)) if rets else None


def run_report(tickers: list[str], daily_period: str = "1y", intraday_period: str = "60d",
               fx: float = FX_US_ROUND_TRIP, slip: float = SLIPPAGE_ROUND_TRIP) -> dict:
    modes = {"buy-now": [], "swing": [], "overnight": [], "weekday": [], "position": []}
    for t in tickers:
        modes["buy-now"] += buy_now_trades(t, intraday_period, fx, slip)
        modes["swing"] += swing_trades(t, daily_period, fx, slip)
        modes["position"] += position_trades(t, daily_period, fx, slip)
        modes["overnight"] += overnight_trades(t, daily_period, fx, slip)
        modes["weekday"] += weekday_trades(t, daily_period, fx, slip)
    report = {name: summarize(net) for name, net in modes.items()}
    # Trades are pooled ticker-by-ticker, not a dated portfolio equity curve.
    for stats in report.values():
        if stats is not None:
            stats["max_dd"] = None
    report["_buy_hold"] = buy_hold_benchmark(tickers, daily_period, fx, slip)
    return report


def format_report(report: dict, tickers: list[str], daily_period: str, intraday_period: str,
                  fx: float, slip: float) -> str:
    out = []
    out.append("=" * WIDTH)
    out.append("  BACKTEST REPORT - research proxies, after assumed costs")
    out.append("=" * WIDTH)
    out.append(f"  Signal profile: {SIGNAL_PROFILE}")
    out.append(f"  Names: {len(tickers)} | daily history: {daily_period} | intraday: {intraday_period}")
    out.append(f"  Costs applied per round trip: {fx:.1f}% FX on US names, {slip:.1f}% slippage")
    out.append("")
    header = f"  {'mode':<10}{'trades':>7}{'win%':>7}{'avg%':>8}{'med%':>7}{'worst%':>8}{'DD':>8}"
    out.append(header)
    out.append("  " + "-" * (WIDTH - 4))
    order = ["buy-now", "swing", "overnight", "weekday", "position"]
    for name in order:
        m = report.get(name)
        if m is None:
            out.append(f"  {name:<10}{'0':>7}   (no trades in this sample)")
            continue
        out.append(
            f"  {name:<10}{m['n']:>7}{m['win']:>6.0f}%{m['avg']:>8.2f}{m['median']:>7.2f}"
            f"{m['worst']:>8.2f}{'N/A':>8}"
        )
    out.append("")
    out.append("  DD unavailable: pooled trades are not a portfolio equity curve.")
    out.append("  Long hold: shared entry rules, 25-session exit proxy.")
    out.append("  Buy-now: fixed exits, not live ATR gates. Entry rules are updated.")
    out.append("  No historical earnings gate, manual latency, or portfolio sizing.")
    out.append("  Data failures and no signals can both produce zero trades.")
    out.append("  Read (avg% is the average NET result per trade after costs):")
    for name in order:
        out.append(f"   - {name:<10}: {verdict(report.get(name))}")
    bh = report.get("_buy_hold")
    if bh is not None:
        out.append("")
        out.append(f"  For context, buy & hold (avg name over {daily_period}, after one FX round trip): "
                   f"{bh:+.1f}%")
    out.append("")
    out.append("  Hypothetical only - past results don't predict the future. The buy-now")
    out.append("  sample is short (~60 days of intraday data is all Yahoo serves). No")
    out.append("  trades are placed. Not financial advice.")
    out.append("=" * WIDTH)
    return "\n".join(out)


def main():
    from scan import load_watchlist

    parser = argparse.ArgumentParser(description="Backtest each mode on real history, after costs")
    parser.add_argument("tickers", nargs="*", help="Tickers to test (default: watchlist.txt)")
    parser.add_argument("--watchlist", default="watchlist.txt")
    parser.add_argument("--period", default="1y", help="Daily history window (e.g. 1y, 2y, 6mo)")
    parser.add_argument("--intraday-period", default="60d", help="Intraday window for buy-now (max ~60d)")
    parser.add_argument("--fx", type=float, default=FX_US_ROUND_TRIP, help="US round-trip FX cost %%")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE_ROUND_TRIP, help="Round-trip slippage %%")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else load_watchlist(args.watchlist)
    print(f"\n  Backtesting {len(tickers)} names... (downloads history and replays rules; may take several minutes)\n")
    report = run_report(tickers, args.period, args.intraday_period, args.fx, args.slippage)
    print(format_report(report, tickers, args.period, args.intraday_period, args.fx, args.slippage))


if __name__ == "__main__":
    main()
