"""Bar-based, one-share research simulation; not a manual execution model."""
import math

import pandas as pd

from .sessions import last_bar_of_day
from .signals import generate_signals


def long_exit_on_bar(row, floor_price: float, ceiling_price: float):
    """Model resting exits: opening gaps first, then stop-first ambiguity.

    Downside gaps fill at the open, not at an unavailable stop price. Upside
    gaps receive only the target (no assumed price improvement). Manual exits
    can be later and worse; this helper cannot model human reaction latency.
    """
    if row["Open"] <= floor_price:
        return float(row["Open"]), "stop_loss"
    if row["Open"] >= ceiling_price:
        return float(ceiling_price), "take_profit"
    if row["Low"] <= floor_price:
        return float(floor_price), "stop_loss"
    if row["High"] >= ceiling_price:
        return float(ceiling_price), "take_profit"
    return None, None


def run_backtest(
    df: pd.DataFrame,
    starting_cash: float = 10_000.0,
    take_profit_pct: float = 0.75,
    stop_loss_pct: float = 0.4,
    signal_fn=generate_signals,
) -> dict:
    """Use completed-bar signals and execute at the next same-session open.

    Exits are checked on the entry bar too. Signals at session end are discarded;
    held positions exit at the final supplied bar's close. Input must contain
    complete sessions: that final close is a scheduled flattening assumption.
    Custom signal functions must themselves be causal. One share per trade,
    before costs; the report separately subtracts configured FX and slippage.
    """
    if not math.isfinite(take_profit_pct) or take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive and finite")
    if not math.isfinite(stop_loss_pct) or not 0 < stop_loss_pct < 100:
        raise ValueError("stop_loss_pct must be finite and between 0 and 100")
    signals = signal_fn(df)
    eod = last_bar_of_day(signals.index)
    position = entry_ts = pending = None
    trades = []
    for ts, row in signals.iterrows():
        is_last = ts in eod
        exit_price = exit_reason = None
        if pending is not None:
            action, signal_ts = pending
            if ts.date() == signal_ts.date():
                if action == "buy" and position is None:
                    position, entry_ts = float(row["Open"]), ts
                elif action == "sell" and position is not None:
                    exit_price, exit_reason = float(row["Open"]), "signal"
            pending = None

        if position is not None:
            if exit_price is None:
                exit_price, exit_reason = long_exit_on_bar(
                    row, position * (1 - stop_loss_pct / 100),
                    position * (1 + take_profit_pct / 100))
            if exit_price is None and is_last:
                exit_price, exit_reason = float(row["Close"]), "eod_flat"
            if exit_price is not None:
                trades.append({
                    "entry_time": entry_ts, "exit_time": ts,
                    "entry_price": position, "exit_price": exit_price,
                    "pnl": exit_price - position,
                    "pnl_pct": (exit_price / position - 1) * 100,
                    "exit_reason": exit_reason,
                })
                position = entry_ts = None

        if not is_last:
            if position is None and row["buy"]:
                pending = ("buy", ts)
            elif position is not None and row["sell"]:
                pending = ("sell", ts)

    trades_df = pd.DataFrame(trades, columns=["entry_time", "exit_time", "entry_price",
                                            "exit_price", "pnl", "pnl_pct", "exit_reason"])
    total_pnl = float(trades_df["pnl"].sum())
    return {"trades": trades_df, "total_pnl": total_pnl,
            "win_rate": float((trades_df["pnl"] > 0).mean()) if len(trades_df) else 0.0,
            "final_cash": starting_cash + total_pnl, "num_trades": len(trades_df)}
