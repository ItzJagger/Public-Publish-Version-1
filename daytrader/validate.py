import pandas as pd

from .backtest import run_backtest
from .data import fetch_intraday


def run_validation(
    tickers: list[str],
    period: str = "60d",
    interval: str = "15m",
    take_profit_pct: float = 0.75,
    stop_loss_pct: float = 0.4,
) -> dict:
    """Backtest the strategy across multiple tickers and summarize aggregate edge."""
    per_ticker = []
    all_trades = []

    for ticker in tickers:
        try:
            df = fetch_intraday(ticker, period=period, interval=interval)
        except ValueError as exc:
            per_ticker.append({"ticker": ticker, "error": str(exc)})
            continue

        result = run_backtest(df, take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct)
        trades = result["trades"]

        if not trades.empty:
            trades = trades.copy()
            trades["ticker"] = ticker
            all_trades.append(trades)

        per_ticker.append(
            {
                "ticker": ticker,
                "num_trades": result["num_trades"],
                "win_rate": result["win_rate"],
                "total_pnl": result["total_pnl"],
                "avg_pnl_pct": trades["pnl_pct"].mean() if not trades.empty else 0.0,
            }
        )

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary = {
        "per_ticker": pd.DataFrame(per_ticker),
        "trades": combined,
        "num_trades": len(combined),
        "win_rate": (combined["pnl_pct"] > 0).mean() if not combined.empty else 0.0,
        "avg_pnl_pct": combined["pnl_pct"].mean() if not combined.empty else 0.0,
        "avg_win_pct": combined.loc[combined["pnl_pct"] > 0, "pnl_pct"].mean() if not combined.empty else 0.0,
        "avg_loss_pct": combined.loc[combined["pnl_pct"] <= 0, "pnl_pct"].mean() if not combined.empty else 0.0,
        "exit_reason_counts": combined["exit_reason"].value_counts() if not combined.empty else pd.Series(dtype=int),
    }
    return summary
