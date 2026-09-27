"""Offline regressions for execution timing and research-report integrity."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest_report as br
from daytrader.backtest import long_exit_on_bar, run_backtest


def bars(opens, buys=(), sells=()):
    d = pd.DataFrame({'Open': opens}, index=pd.date_range('2026-06-01 09:30', periods=len(opens), freq='30min'))
    d['Close'] = d['Open']
    d['High'] = d['Open'] + .1
    d['Low'] = d['Open'] - .1
    d['Volume'] = 1000
    d['buy'] = [i in buys for i in range(len(d))]
    d['sell'] = [i in sells for i in range(len(d))]
    return d


def replay(d, **kwargs):
    return run_backtest(d, signal_fn=lambda x: x, take_profit_pct=10, stop_loss_pct=5, **kwargs)


class BacktestIntegrity(unittest.TestCase):
    def test_entry_uses_next_open(self):
        d = bars([100, 102, 103], buys=[0])
        t = replay(d)['trades'].iloc[0]
        self.assertEqual(t.entry_price, 102)
        self.assertEqual(t.entry_time, d.index[1])

    def test_entry_bar_stop_is_checked(self):
        d = bars([100, 100, 100], buys=[0]); d.iloc[1, d.columns.get_loc('Low')] = 94
        t = replay(d)['trades'].iloc[0]
        self.assertEqual(t.exit_price, 95)
        self.assertEqual(t.entry_time, t.exit_time)

    def test_downside_gap_fills_at_open(self):
        t = replay(bars([100, 100, 90, 90], buys=[0]))['trades'].iloc[0]
        self.assertEqual(t.exit_price, 90)
        self.assertAlmostEqual(t.pnl_pct, -10)

    def test_both_touched_stop_first(self):
        self.assertEqual(long_exit_on_bar({'Open': 100, 'Low': 90, 'High': 120}, 95, 110), (95, 'stop_loss'))

    def test_open_above_target_precedes_later_low(self):
        self.assertEqual(long_exit_on_bar({'Open': 115, 'Low': 90, 'High': 120}, 95, 110), (110, 'take_profit'))

    def test_signal_exit_uses_next_open(self):
        d = bars([100, 100, 101, 102, 103], buys=[0], sells=[2])
        t = replay(d)['trades'].iloc[0]
        self.assertEqual((t.exit_price, t.exit_reason), (102, 'signal'))
        self.assertEqual(t.exit_time, d.index[3])

    def test_final_signal_does_not_enter(self):
        self.assertEqual(replay(bars([100, 100], buys=[1]))['num_trades'], 0)

    def test_no_entry_from_previous_session(self):
        d = bars([100, 100, 100], buys=[0])
        d.index = pd.to_datetime(['2026-06-01 15:30', '2026-06-02 09:30', '2026-06-02 10:00'])
        self.assertEqual(replay(d)['num_trades'], 0)

    def test_empty_frame(self):
        self.assertEqual(replay(bars([]))['num_trades'], 0)

    def test_invalid_risk_is_rejected(self):
        with self.assertRaises(ValueError):
            run_backtest(bars([100]), stop_loss_pct=float('nan'))

    def test_initial_loss_counts_in_sequence_drawdown(self):
        self.assertEqual(br.summarize([-5, 2, -1])['max_dd'], 5)

    def test_regime_and_signals_are_prefix_invariant(self):
        idx = pd.DatetimeIndex([day + pd.Timedelta(hours=9, minutes=30 + 30*b)
                               for day in pd.bdate_range('2026-01-05', periods=14) for b in range(13)])
        rng = np.random.default_rng(31)
        price = 100 + np.cumsum(rng.normal(0, .25, len(idx)))
        d = pd.DataFrame({'Open': price, 'Close': price, 'High': price + .3,
                          'Low': price - .3, 'Volume': rng.integers(1000, 5000, len(idx))}, index=idx)
        cut = 10*13
        first = br.walk_forward_signals(d.iloc[:cut])[['buy', 'sell']]
        d.iloc[cut:, d.columns.get_loc('High')] *= 10  # extreme future regime change
        full = br.walk_forward_signals(d)[['buy', 'sell']]
        pd.testing.assert_frame_equal(first, full.iloc[:cut])
        self.assertFalse(full.iloc[:5*13].to_numpy().any())

    def test_report_suppresses_pooled_drawdown(self):
        with patch.object(br, 'buy_now_trades', return_value=[-5, 2]), \
             patch.object(br, 'swing_trades', return_value=[]), \
             patch.object(br, 'overnight_trades', return_value=[]), \
             patch.object(br, 'weekday_trades', return_value=[]), \
             patch.object(br, 'position_trades', return_value=[]), \
             patch.object(br, 'buy_hold_benchmark', return_value=None):
            report = br.run_report(['TEST.TO'])
        self.assertIsNone(report['buy-now']['max_dd'])
        rendered = br.format_report(report, ['TEST.TO'], '1y', '60d', 3, .1)
        self.assertIn('N/A', rendered)
        self.assertIn('25-session exit proxy', rendered)

    def test_swing_gap_loss_and_reentry_after_early_exit(self):
        d = bars([100]*66)
        d.index = pd.bdate_range('2026-01-05', periods=len(d))
        d.loc[d.index[60], 'Low'] = 97
        d.loc[d.index[63], ['Open', 'Low']] = [90, 89]
        setup = {'candidate':True, 'floor':98., 'ceiling':104.}
        with patch.object(br, '_daily', return_value=d), \
             patch.object(br, 'swing_from_frame', return_value=setup):
            trades = br.swing_trades('TEST.TO', '1y', 0, 0)
        self.assertGreaterEqual(len(trades), 2)
        self.assertTrue(any(x < -5 for x in trades))


if __name__ == '__main__':
    unittest.main()
