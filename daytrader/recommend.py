import pandas as pd

from .sessions import last_bar_of_day
from .signals import generate_signals


def current_recommendation(
    df: pd.DataFrame,
    take_profit_pct: float = 0.75,
    stop_loss_pct: float = 0.4,
    signal_fn=generate_signals,
) -> dict:
    """Replay signals up to the latest bar and report what to do right now.

    This replays hypothetical signal state at bar closes; IN_POSITION is not
    evidence of an actual fill. The research backtest executes at next-bar
    opens, so its position state can differ. Actual fills belong in holdings.txt.
    It does not place trades - you act manually in Wealthsimple. `signal_fn` swaps the strategy (trend-following
    EMA crossover vs. mean-reversion) while reusing this same replay logic.

    For mean-reversion tickers the displayed floor/ceiling come from the ATR
    risk engine (see watchlist/risk_levels), not from these take_profit_pct /
    stop_loss_pct - those are only used here to replay the position state.
    """
    signals = signal_fn(df)
    eod = last_bar_of_day(signals.index)
    final_ts = signals.index[-1]

    position = None
    entry_ts = None

    for ts, row in signals.iterrows():
        # A completed day's closing bar forces a flat-overnight reset - but the
        # final bar of the data is "now" (a live mid-session reading), not an
        # overnight boundary, so a fresh signal there is a real BUY_NOW.
        overnight_flat = (ts in eod) and (ts != final_ts)

        if position is None and row["buy"]:
            if overnight_flat:  # no overnight holds on a completed day's last bar
                continue
            position = row["Close"]
            entry_ts = ts
            continue

        if position is None:
            continue

        floor_price = position * (1 - stop_loss_pct / 100)
        ceiling_price = position * (1 + take_profit_pct / 100)

        if row["Low"] <= floor_price or row["High"] >= ceiling_price or row["sell"] or overnight_flat:
            position = None
            entry_ts = None

    last_ts = signals.index[-1]
    last_row = signals.loc[last_ts]

    if position is not None:
        floor_price = position * (1 - stop_loss_pct / 100)
        ceiling_price = position * (1 + take_profit_pct / 100)
        unrealized_pct = (last_row["Close"] / position - 1) * 100
        status = "BUY_NOW" if entry_ts == last_ts else "IN_POSITION"
        return {
            "status": status,
            "as_of": last_ts,
            "entry_time": entry_ts,
            "entry_price": position,
            "current_price": last_row["Close"],
            "unrealized_pct": unrealized_pct,
            "floor_price": floor_price,
            "ceiling_price": ceiling_price,
        }

    return {
        "status": "FLAT",
        "as_of": last_ts,
        "current_price": last_row["Close"],
        "rsi": last_row["rsi"],
        "vwap": last_row["vwap"],
    }


def format_recommendation(rec: dict, take_profit_pct: float, stop_loss_pct: float) -> str:
    if rec["status"] in ("IN_POSITION", "BUY_NOW"):
        headline = (
            "FRESH BUY SIGNAL - consider buying now"
            if rec["status"] == "BUY_NOW"
            else f"IN POSITION - entered {rec['entry_time']} @ {rec['entry_price']:.2f}"
        )
        return (
            f"\n=== RECOMMENDATION (as of {rec['as_of']}) ===\n"
            f"{headline}\n"
            f"Current price: {rec['current_price']:.2f}  "
            f"(unrealized {rec['unrealized_pct']:+.2f}%)\n"
            f"  CEILING (sell, take profit): {rec['ceiling_price']:.2f}  "
            f"(+{take_profit_pct:.2f}%)\n"
            f"  FLOOR   (sell, stop loss):   {rec['floor_price']:.2f}  "
            f"(-{stop_loss_pct:.2f}%)\n"
            f"Action: hold and manually sell in Wealthsimple if price touches "
            f"either level, or sooner if a SELL signal fires."
        )

    return (
        f"\n=== RECOMMENDATION (as of {rec['as_of']}) ===\n"
        f"FLAT - no open position.\n"
        f"Current price: {rec['current_price']:.2f}  RSI: {rec['rsi']:.1f}\n"
        f"Action: wait for the next BUY signal before entering."
    )
