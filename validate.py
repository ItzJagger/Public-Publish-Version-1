import argparse

from daytrader.validate import run_validation


def main():
    parser = argparse.ArgumentParser(description="Multi-ticker strategy validation")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT TSLA")
    parser.add_argument("--interval", default="15m", help="Bar interval (60d history max for 15m)")
    parser.add_argument("--period", default="60d", help="History window, e.g. 60d")
    parser.add_argument("--take-profit", type=float, default=0.75)
    parser.add_argument("--stop-loss", type=float, default=0.4)
    args = parser.parse_args()

    summary = run_validation(
        args.tickers,
        period=args.period,
        interval=args.interval,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
    )

    print("\nPer-ticker results:")
    print(summary["per_ticker"].to_string(index=False))

    print(f"\nOverall across {len(args.tickers)} tickers, {args.period} of {args.interval} bars:")
    print(f"  Trades: {summary['num_trades']}")
    print(f"  Win rate: {summary['win_rate']:.1%}")
    print(f"  Avg P&L per trade: {summary['avg_pnl_pct']:.3f}%")
    print(f"  Avg winner: {summary['avg_win_pct']:.3f}%   Avg loser: {summary['avg_loss_pct']:.3f}%")
    print(f"\nExit reason breakdown:\n{summary['exit_reason_counts'].to_string()}")


if __name__ == "__main__":
    main()
