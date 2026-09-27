"""Compare original/revised intraday entries on a chronological holdout.

No parameter search. Both versions receive the same prices, fixed exits and
costs. This isolates entry changes; it does NOT replay live ATR/earnings gates.
Run with downloaded Yahoo history, or an offline OHLCV CSV with datetime index.
"""
import argparse
import pandas as pd
from daytrader.data import fetch_intraday, completed_bars
from daytrader.backtest import run_backtest
from daytrader.policy import round_trip_cost, SIGNAL_PROFILE
from daytrader.signals import generate_signals
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.legacy_signals import generate_signals as old_trend
from daytrader.legacy_mean_reversion import generate_mean_reversion_signals as old_meanrev


def compare(df, ticker, holdout_fraction=.3, take_profit=1.0, stop_loss=.5, cost=None):
    if not 0 < holdout_fraction < 1:
        raise ValueError('Holdout fraction must be between 0 and 1')
    days = pd.Index(df.index.date).unique()
    if len(days) < 20:
        raise ValueError('Need at least 20 sessions; even that is a small research sample')
    cut = days[max(1, int(len(days)*(1-holdout_fraction)))]
    c = round_trip_cost(ticker) if cost is None else cost
    rows = []
    for mode, old, new in [('trend', old_trend, generate_signals),
                            ('meanrev', old_meanrev, generate_mean_reversion_signals)]:
        for label, fn in [('original', old), ('revised', new)]:
            def heldout(x):
                signals = fn(x)
                signals.loc[signals.index.date < cut, 'buy'] = False
                return signals
            trades = run_backtest(df, signal_fn=heldout, take_profit_pct=take_profit,
                                  stop_loss_pct=stop_loss)['trades']
            net = trades['pnl_pct'].astype(float)-c
            losses = -net[net < 0].sum()
            rows.append({'mode':mode, 'version':label, 'trades':len(net),
                         'win_pct': float((net>0).mean()*100) if len(net) else None,
                         'avg_net_pct': float(net.mean()) if len(net) else None,
                         'worst_net_pct': float(net.min()) if len(net) else None,
                         'profit_factor': float(net[net>0].sum()/losses) if losses > 0 else None,
                         'holdout_from':str(cut)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ticker')
    p.add_argument('--csv', help='Offline OHLCV CSV; first column is timestamp')
    p.add_argument('--interval', default='30m')
    p.add_argument('--period', default='60d')
    p.add_argument('--holdout', type=float, default=.3)
    p.add_argument('--take-profit', type=float, default=1.0)
    p.add_argument('--stop-loss', type=float, default=.5)
    a = p.parse_args()
    try:
        df = (pd.read_csv(a.csv, index_col=0, parse_dates=True) if a.csv else
              fetch_intraday(a.ticker, period=a.period, interval=a.interval))
        df = completed_bars(df, a.interval)
        # Drop the current date to avoid treating a partial session as a close.
        now = pd.Timestamp.now(tz='America/New_York')
        df = df.loc[df.index.date < now.date()]
        print(f"Revised signal profile: {SIGNAL_PROFILE}")
        print(compare(df, a.ticker, a.holdout, a.take_profit, a.stop_loss).to_string(index=False))
        print('\nResearch only: identical fixed exits, after assumed costs. No live ATR/earnings gates.')
        print('No trades = no evidence. Sparse/dependent samples cannot establish an improvement.')
        print('Do not tune on the holdout then keep calling it unseen data; forward-paper-test next.')
    except Exception as exc:
        p.exit(1, f'Comparison unavailable: {exc}\n')


if __name__ == '__main__':
    main()
