import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from daytrader.tracker import SignalTracker, evaluate


def row():
    return dict(ticker='VFV.TO',alert_at='2026-09-28T10:00:01-04:00',deadline='2026-09-28T10:20:00-04:00',
                entry_limit=100.,stop=99.,target=102.,shares=5,cost_pct=.1)


def bars():
    return pd.DataFrame({'Open':[100.,100.5,101.],'High':[100.8,101.5,102.1],
                         'Low':[99.5,100.,100.5],'Close':[100.5,101.,102.],'Volume':[1000]*3},
                        index=pd.date_range('2026-09-28 10:05',periods=3,freq='5min',tz='America/New_York'))

class TrackerTests(unittest.TestCase):
    def test_deadline_and_costs(self):
        result=evaluate(row(),bars(),'2026-09-28 10:21')
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['first_touch'],'TARGET')
        self.assertAlmostEqual(result['net_proxy'],9.5)
        self.assertAlmostEqual(result['max_drop_pct'],-.5)

    def test_no_pre_alert_bar_or_after_deadline_leakage(self):
        b=bars();extra=b.iloc[[0]].copy();extra.index=[b.index[0]-pd.Timedelta(minutes=5)];extra.High=200
        late=b.iloc[[0]].copy();late.index=[b.index[-1]+pd.Timedelta(minutes=5)];late.High=300
        r=evaluate(row(),pd.concat([extra,b,late]),'2026-09-28 11:00')
        self.assertAlmostEqual(r['max_rise_pct'],2.1)

    def test_same_bar_ambiguous_and_stop_nonfill(self):
        b=bars();b.iloc[0,b.columns.get_loc('High')]=103;b.iloc[0,b.columns.get_loc('Low')]=98
        self.assertEqual(evaluate(row(),b,'2026-09-28 10:21')['first_touch'],'AMBIGUOUS_SAME_BAR')
        b.High=[101,101.5,102.1]
        self.assertEqual(evaluate(row(),b,'2026-09-28 10:21')['first_touch'],'STOP (fill not assumed)')

    def test_missing_bar_not_success(self):
        r=evaluate(row(),bars().iloc[[0,2]],'2026-09-28 10:21')
        self.assertEqual(r['status'],'AWAITING_DATA')
        self.assertIsNone(r['net_proxy'])
        self.assertEqual(r['missing_bars'],1)

    def test_intrabar_touch_not_invented_fill(self):
        b=bars();b.Open=101.;b.High=103.;b.Close=102.;b.Low=99.5
        r=evaluate(row(),b,'2026-09-28 10:21')
        self.assertEqual(r['entry_status'],'UNCONFIRMED_INTRABAR_TOUCH')
        self.assertIsNone(r['net_proxy'])

    def test_no_entry_and_invalid_gap(self):
        b=bars()+5
        self.assertEqual(evaluate(row(),b,'2026-09-28 10:21')['entry_status'],'NO_ENTRY')
        b=bars();b.Open=98.
        self.assertEqual(evaluate(row(),b,'2026-09-28 10:21')['entry_status'],'INVALID_AT_ENTRY')

    def test_weekend_not_missing_and_early_close(self):
        from daytrader.tracker import expected_bars, stamp
        x=expected_bars('SPY',stamp('2026-11-27 12:55'),stamp('2026-11-30 09:35'))
        self.assertEqual(len(x),2)

    def test_persistence_and_dedup_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            tracker=SignalTracker(d)
            plan=({'entry':100.,'stop_price':99.,'limit_price':99.,'target_price':102.},
                  {'eligible':True,'shares':5},'2026-09-28 10:20')
            with patch('daytrader.alerts.alert_plan',return_value=plan):
                tracker.record({'ticker':'VFV.TO'},'2026-09-28 10:00:01',True,'message')
                tracker.record({'ticker':'VFV.TO'},'2026-09-28 10:00:01',True,'message')
            reloaded=SignalTracker(d)
            self.assertEqual(len(reloaded.records()),1)
            reloaded.update('2026-09-28 10:21',fetch=lambda *a,**kw:bars())
            self.assertEqual(reloaded.records()[0]['status'],'COMPLETE')
            report=(reloaded.directory/'reports/2026-09-28.txt').read_text()
            self.assertIn('VFV.TO',report);self.assertIn('SIMULATED',report)

    def test_fetch_failure_not_erased(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d)
            plan=({'entry':100.,'stop_price':99.,'limit_price':99.,'target_price':102.},
                  {'eligible':True,'shares':5},'2026-09-28 10:20')
            with patch('daytrader.alerts.alert_plan',return_value=plan):t.record({'ticker':'VFV.TO'},'2026-09-28 10:00:01',True,'m')
            def fail(*a,**kw):raise ValueError('unavailable')
            t.update('2026-09-28 10:21',fetch=fail)
            self.assertEqual(t.records()[0]['data_error'],'unavailable')
            self.assertNotEqual(t.records()[0]['status'],'COMPLETE')
            t.update('2026-09-28 10:30',fetch=lambda *a,**kw:bars())
            self.assertEqual(t.records()[0]['status'],'COMPLETE')

if __name__=='__main__':unittest.main()

class WatcherTracking(unittest.TestCase):
    def test_only_successful_delivery_recorded(self):
        import contextlib
        import io
        import scan
        import watch
        r={'ticker':'VFV.TO','raw_buy_signal':True,'rec':{'status':'BUY_NOW','current_price':100},
           'atr_stop':{'floor':99,'ceiling':102},'eligibility':{'buy-now':{'eligible':True}}}
        for delivered in [False,True]:
            with patch.object(scan,'run_scan',return_value={'results':[r]}), \
                 patch.object(watch,'market_is_open',return_value=True), \
                 patch.object(watch,'market_now',return_value=pd.Timestamp('2026-09-28 10:00',tz='America/New_York')), \
                 patch.object(watch,'notification_config',return_value={'kind':'ntfy','target':'unused'}), \
                 patch.object(watch,'send_notification',return_value=delivered), \
                 patch.object(watch,'SignalTracker') as tracker, \
                 contextlib.redirect_stdout(io.StringIO()):
                watch.run_watch(alerts='short',once=True)
                self.assertEqual(tracker.return_value.record.call_count,int(delivered))
                self.assertEqual(tracker.return_value.update.call_count,1)
