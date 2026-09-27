import argparse

from daytrader.backtest import run_backtest
from daytrader.data import fetch_intraday, completed_bars
from daytrader.recommend import current_recommendation, format_recommendation
from daytrader.signals import daily_trade_windows


def main():
    parser = argparse.ArgumentParser(description="Intraday buy/sell signal scanner")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--interval", default="5m", help="Bar interval (1m, 5m, 15m, ...)")
    parser.add_argument("--period", default="60d", help="History window to pull, e.g. 5d, 1mo")
    parser.add_argument("--backtest", action="store_true", help="Run the historical backtest")
    parser.add_argument(
        "--take-profit", type=float, default=0.75,
        help="Ceiling: auto-sell when price is up this %% from entry (default 0.75)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=0.4,
        help="Floor: auto-sell when price is down this %% from entry (default 0.4)",
    )
    args = parser.parse_args()

    df = completed_bars(fetch_intraday(args.ticker, period=args.period, interval=args.interval), args.interval)

    print(f"\nHistorical signal events (before live eligibility gates) for {args.ticker} ({args.interval} bars, last {args.period}):\n")
    windows = daily_trade_windows(df)
    if windows.empty:
        print("No buy/sell signals triggered in this window.")
    else:
        print(windows.to_string())

    if args.backtest:
        result = run_backtest(
            df, take_profit_pct=args.take_profit, stop_loss_pct=args.stop_loss
        )
        print(f"\nBacktest (take-profit {args.take_profit}%, stop-loss {args.stop_loss}%): "
              f"{result['num_trades']} trades, "
              f"win rate {result['win_rate']:.1%}, "
              f"total P&L ${result['total_pnl']:.2f}, "
              f"final cash ${result['final_cash']:.2f}")
        if not result["trades"].empty:
            print(result["trades"].to_string(index=False))

    from daytrader.watchlist import scan_ticker
    from scan import print_ticker_block
    result = scan_ticker(args.ticker, interval=args.interval, period=args.period,
                         take_profit_pct=args.take_profit, stop_loss_pct=args.stop_loss)
    print_ticker_block(result, show_ai=False)



if __name__ == "__main__":
    main()
