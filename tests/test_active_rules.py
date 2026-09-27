import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from daytrader.trade_plan import sell_by, size_example, previous_session_date
from daytrader.active_risk import active_risk
from daytrader.signals import generate_signals
from daytrader.alerts import format_categorized_alert
from daytrader.data import completed_bars
from daytrader.diagnostics import diagnostic_lines
from watch import closed_wait_seconds


def frame():
    idx = pd.DatetimeIndex([t for d in pd.bdate_range('2026-07-01',periods=25)
        for t in pd.date_range(str(d.date())+' 09:30',periods=26,freq='15min',tz='America/New_York')])
    x = 100+np.sin(np.arange(len(idx))*.15)*1.3
    return pd.DataFrame({'Open':x-.05,'High':x+.2,'Low':x-.2,'Close':x,'Volume':100000},index=idx)

class ActiveRules(unittest.TestCase):
    def test_active_entry_gate_and_dollar_floor(self):
        from test_signal_upgrade import candidate, NOW
        from daytrader.eligibility import attach_eligibility, permitted
        with patch('daytrader.eligibility.SIGNAL_PROFILE','active'), patch('daytrader.eligibility.SHORT_MIN_NET_RR',1.25):
            r=candidate();r['atr_stop']['ceiling']=101.6
            out=attach_eligibility(r,NOW)
            self.assertTrue(permitted(out,'buy-now'))
            self.assertIn('sell_by',out['eligibility']['buy-now'])
            r=candidate();r['last_price']=1000;r['rec']['current_price']=1000
            r['atr_stop'].update(floor=990,ceiling=1020)
            out=attach_eligibility(r,NOW)
            self.assertFalse(permitted(out,'buy-now'))
            self.assertTrue(any('position sizing' in x for x in out['eligibility']['buy-now']['reasons']))
    def test_deadline_and_late_entry(self):
        self.assertEqual(sell_by('RY.TO','2026-09-14 10:05',minutes=60).hour,11)
        self.assertIsNone(sell_by('RY.TO','2026-09-14 15:05',minutes=60))
    def test_holiday_and_early_close(self):
        self.assertIsNone(sell_by('SPY','2026-11-26 10:00',minutes=60))
        self.assertIsNone(sell_by('SPY','2026-11-27 12:05',minutes=60))
        self.assertIsNotNone(sell_by('RY.TO','2026-11-26 10:00',minutes=60))
    def test_long_deadline_counts_sessions(self):
        x=sell_by('RY.TO','2026-09-14 10:00',sessions=20)
        self.assertEqual(str(x.date()),'2026-10-13') # Canadian Thanksgiving skipped
        self.assertEqual((x.hour,x.minute),(15,55))
    def test_prior_session_holiday(self):
        self.assertEqual(str(previous_session_date('RY.TO','2026-09-08 10:00')),'2026-09-04')
    def test_size_respects_both_caps(self):
        x=size_example('RY.TO',100,99,102,budget=500,risk_budget=3,min_profit=1.01)
        self.assertTrue(x['eligible']);self.assertEqual(x['shares'],2)
        self.assertLessEqual(x['planned_stop_loss'],3)
        self.assertFalse(size_example('RY.TO',100,99,100.11,budget=500,risk_budget=3)['eligible'])
        self.assertFalse(size_example('AAPL',100,99,102)['eligible'])
    def test_risk_does_not_shrink_to_fit(self):
        d=frame();now=d.index[-1]+pd.Timedelta(minutes=15,seconds=6)
        now=now.normalize()+pd.Timedelta(hours=12)
        d=d.loc[d.index < now]
        a=active_risk(d,'RY.TO',float(d.Close.iloc[-1]),now)
        if 'floor_distance' in a:
            self.assertGreaterEqual(a['floor_distance'],1.5*a['single_bar_atr'])
            self.assertLessEqual(a['reward_distance'],a['range_cap'])
    def test_signal_causality(self):
        d=frame()
        with patch('daytrader.signals.SIGNAL_PROFILE','active'):
            full=generate_signals(d)
            part=generate_signals(d.iloc[:400])
        pd.testing.assert_series_equal(full.buy.iloc[:400],part.buy)
        checks=part.filter(like='check_').all(axis=1)
        pd.testing.assert_series_equal(checks,part.buy,check_names=False)
    def test_invalid_row_is_identified(self):
        d=frame();d.iloc[-1,d.columns.get_loc('High')]=1
        with self.assertRaisesRegex(ValueError,'2026'):
            completed_bars(d,'15m')
    def test_alert_has_date_time_and_bounded_profit(self):
        r={'ticker':'RY.TO','rec':{'current_price':100},'atr_stop':{'floor':99,'ceiling':102},
           'eligibility':{'buy-now':{'sizing':size_example('RY.TO',100,99,102)}}}
        _,msg=format_categorized_alert([r],[],pd.Timestamp('2026-09-14 10:00',tz='America/New_York'))
        self.assertIn('2026-09-14 11:00 ET',msg);self.assertIn('Stop-limit SELL: Stop $99.00 | Limit $99.00',msg);self.assertIn('Amount: 4 whole shares',msg)
    def test_incomplete_daily_bar_cannot_poison_completed_history(self):
        d=frame().groupby(frame().index.date).last()
        d.index=pd.DatetimeIndex(d.index)
        today=d.index[-1]
        d.loc[today,'High']=1
        out=completed_bars(d,'1d',today+pd.Timedelta(hours=12))
        self.assertEqual(len(out),len(d)-1)
        with self.assertRaises(ValueError):
            completed_bars(d,'1d',today+pd.Timedelta(days=1))
    def test_long_alert_has_deadline(self):
        r={'ticker':'RY.TO','price':100,'floor':97,'ceiling':106,'hold_weeks':4}
        _,msg=format_categorized_alert([],[r],pd.Timestamp('2026-09-14 10:00',tz='America/New_York'))
        self.assertIn('2026-10-13 15:55 ET',msg)
    def test_wakes_at_open(self):
        now=pd.Timestamp('2026-09-14 09:19',tz='America/New_York')
        self.assertEqual(closed_wait_seconds(now),11*60)
    def test_names_actual_blocked_setup(self):
        r={'ticker':'RY.TO','raw_buy_signal':True,'no_trade':True,'atr_stop':{'reason':'test'},'entry_checks':{'trigger':True,'volume':False}}
        text='\n'.join(diagnostic_lines([r]))
        self.assertIn('Actual setup RY.TO',text);self.assertIn('volume=1',text)

if __name__=='__main__':unittest.main()
