import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from daytrader.tracker import SignalTracker, evaluate, stamp, expected_bars
from daytrader.daily_review import (DailyReview, report_rows, render_report, next_report_seconds,
                                    trading_day, summary, chart)
from daytrader.learning import (FEATURES, atomic_json, training_rows, train_reports,
                                shadow_prediction, snapshot)


def record(mode='hourly'):
    return dict(id='one', ticker='VFV.TO', mode=mode, strategy='trend',
                alert_at='2026-09-28T10:00:01-04:00', deadline='2026-09-28T11:00:00-04:00',
                signal_price=100.,entry_limit=100.,stop=99.,stop_limit=99.,target=102.,
                shares=5,cost_pct=.1,currency='CAD',status='PENDING',features={k:1. for k in FEATURES})


def frame(day='2026-09-28', price=100.):
    ix=expected_bars('VFV.TO',stamp(day+' 09:30'),stamp(day+' 16:00'))
    return pd.DataFrame(dict(Open=price,High=price+1.,Low=price-.5,Close=price+.5,Volume=1000.),index=ix)


def insert(tracker,r,b):
    with tracker.connect() as db:
        db.execute('INSERT OR REPLACE INTO signals VALUES (?,?)',(r['id'],json.dumps(r)))
        db.executemany('INSERT OR REPLACE INTO bars VALUES (?,?,?)',
                       [(r['ticker'],t.isoformat(),json.dumps(v.to_dict())) for t,v in b.iterrows()])


class DailyTests(unittest.TestCase):
    def test_metrics_and_day_separate_from_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d);r=record();b=frame();b.loc[b.index.hour>=12,'High']=150.
            b.loc[b.index.hour>=12,'Close']=110.;insert(t,r,b)
            result=report_rows(t,datetime.date(2026,9,28))[0]
            self.assertEqual(result['peak_price'],101.)
            self.assertEqual(result['floor_price'],99.5)
            self.assertEqual(result['day_peak'],150.)
            self.assertEqual(result['last_price'],100.5)
            self.assertEqual(result['day_close'],110.)
            self.assertAlmostEqual(result['net_proxy'],2.)
            self.assertAlmostEqual(result['estimated_cost'],.5)

    def test_long_snapshot_does_not_leak_future(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d);r=record('long');r['deadline']='2026-09-29T15:55:00-04:00'
            r.update(status='COMPLETE',net_proxy=999,peak_price=999)
            insert(t,r,pd.concat([frame(),frame('2026-09-29',200)]))
            result=report_rows(t,datetime.date(2026,9,28))[0]
            self.assertEqual(result['status'],'OPEN')
            self.assertEqual(result['peak_price'],101.)
            self.assertEqual(result['last_price'],100.5)
            self.assertNotEqual(result['net_proxy'],999)

    def test_missing_and_no_entry_not_profitable(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d);b=frame().drop(stamp('2026-09-28 10:10'));insert(t,record(),b)
            result=report_rows(t,datetime.date(2026,9,28))[0]
            self.assertIsNone(result['net_proxy']);self.assertIsNone(result['day_peak'])
            self.assertEqual(result['day_missing_bars'],1)
            insert(t,record(),frame(price=105))
            result=report_rows(t,datetime.date(2026,9,28))[0]
            self.assertEqual(result['entry_status'],'NO_ENTRY')
            self.assertIsNone(result['net_proxy'])

    def test_currency_totals_never_mixed(self):
        rows=[dict(currency='CAD',net_proxy=5,gross_proxy=6),dict(currency='USD',net_proxy=3,gross_proxy=4)]
        totals=summary(rows)
        self.assertEqual(totals['CAD']['net_proxy'],5)
        self.assertEqual(totals['USD']['net_proxy'],3)

    def test_17_hour_schedule_restart_and_empty_day(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d);scheduler=DailyReview(t)
            self.assertEqual(scheduler.run_due(stamp('2026-09-28 16:59')),[])
            written=scheduler.run_due(stamp('2026-09-28 17:00'))
            self.assertEqual(len(written),1)
            text=(written[0]/'report.txt').read_text()
            self.assertIn('No BUY alerts',text)
            self.assertEqual(DailyReview(SignalTracker(d)).run_due(stamp('2026-09-28 18:00')),[])
            self.assertEqual(next_report_seconds(stamp('2026-09-28 16:59')),60)

    def test_catchup_skips_weekend_and_retries_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d)
            DailyReview(t).run_due(stamp('2026-09-25 16:59'))
            paths=DailyReview(t).run_due(stamp('2026-09-28 17:00'))
            self.assertEqual([p.name for p in paths],['2026-09-25','2026-09-28'])
            marker=paths[-1]/'complete.json';saved=json.loads(marker.read_text());saved['needs_retry']=True
            atomic_json(marker,saved)
            self.assertEqual(len(DailyReview(t).run_due(stamp('2026-09-28 17:31'))),1)

    def test_exchange_holidays_and_dst(self):
        self.assertFalse(trading_day(datetime.date(2026,12,25)))
        self.assertTrue(trading_day(datetime.date(2026,10,12))) # US open, Canadian holiday
        seconds=next_report_seconds(stamp('2026-10-30 17:01'))
        self.assertEqual(stamp('2026-10-30 17:01')+pd.Timedelta(seconds=seconds),stamp('2026-11-02 17:00'))

    def test_html_csv_json_chart_and_existing_db(self):
        with tempfile.TemporaryDirectory() as d:
            t=SignalTracker(d);insert(t,record(),frame())
            out=render_report(SignalTracker(d),datetime.date(2026,9,28),stamp('2026-09-28 17:00'))
            for name in ['report.html','report.txt','report.json','signals.csv','complete.json']:
                self.assertTrue((out/name).exists())
            self.assertEqual(len(list(out.glob('*.png'))),1)
            self.assertIn('SIMULATED', (out/'report.txt').read_text())
            self.assertEqual(json.loads((out/'report.json').read_text())['signals'][0]['net_proxy'],2.)


def learning_fixture(root, long=False):
    rows=[]
    for n,day in enumerate(pd.bdate_range('2026-01-05',periods=30)):
        date=day.date()
        for j in range(5):
            features={k:0. for k in FEATURES};features['stop_pct']=j+.1*n
            at=stamp(str(date)+' 10:00')
            deadline=at+pd.Timedelta(days=90) if long else at+pd.Timedelta(hours=1)
            r=dict(id=f'{n}-{j}',ticker=f'TEST{j}.TO',mode='long' if long else 'hourly',
                   alert_at=at.isoformat(),deadline=deadline.isoformat(),status='COMPLETE',entry_status='OPEN_PRICE_PROXY',
                   missing_bars=0,features=features,net_return_pct=features['stop_pct']*.5-1)
            rows.append(r)
    atomic_json(Path(root)/'daily'/'2026-07-01'/'report.json',
                dict(schema=1,as_of=stamp('2026-07-01 17:00').isoformat(),signals=rows))
    return rows


class LearningTests(unittest.TestCase):
    def test_insufficient_history_collects_not_fabricates_model(self):
        with tempfile.TemporaryDirectory() as d:
            result=train_reports(d,stamp('2026-09-28 17:00'))
            self.assertEqual(result['modes']['hourly']['status'],'COLLECTING')
            self.assertNotIn('model',result['modes']['hourly'])

    def test_trains_on_reports_and_saves_prospective_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            rows=learning_fixture(d)
            result=train_reports(d,stamp('2026-07-02 17:00'))
            model=result['modes']['hourly']
            self.assertEqual(model['status'],'SHADOW_TRAINED')
            self.assertLess(model['holdout_mse'],model['baseline_mse'])
            by_id={r['id']:r for r in rows}
            self.assertLess(max(stamp(by_id[k]['deadline']) for k in model['train_ids']),
                            min(stamp(by_id[k]['alert_at']) for k in model['test_ids']))
            self.assertIsNone(shadow_prediction(d,'hourly',rows[0]['features'],stamp('2026-07-01')))
            self.assertEqual(shadow_prediction(d,'hourly',rows[0]['features'],stamp('2026-07-03'))['operation'],'SHADOW_ONLY')

    def test_dedup_future_incomplete_and_missing_features(self):
        with tempfile.TemporaryDirectory() as d:
            rows=learning_fixture(d)
            self.assertEqual(training_rows(Path(d)/'daily',stamp('2026-06-01')),[])
            extra=dict(rows[0],id='repeat'); bad=dict(rows[1],id='bad',missing_bars=1)
            legacy=dict(rows[2],id='legacy',features=None)
            atomic_json(Path(d)/'daily'/'2026-07-02'/'report.json',dict(schema=1,as_of=stamp('2026-07-02').isoformat(),signals=rows+[extra,bad,legacy]))
            self.assertEqual(len(training_rows(Path(d)/'daily',stamp('2026-07-03'))),150)

    def test_long_overlapping_labels_purged(self):
        with tempfile.TemporaryDirectory() as d:
            learning_fixture(d,long=True)
            result=train_reports(d,stamp('2026-07-02 17:00'))
            self.assertEqual(result['modes']['long']['status'],'COLLECTING_AFTER_PURGE')
            self.assertEqual(result['modes']['long']['train_n'],0)

    def test_features_ignore_future_outcomes(self):
        order=dict(entry=100,stop_price=99,target_price=102)
        row=dict(strategy='trend',rec={'rsi':55,'relative_volume':1.2},peak_price=900,net_proxy=500)
        x=snapshot(row,order,stamp('2026-09-28 10:00'),.1)
        row.update(peak_price=1,net_proxy=-1000)
        self.assertEqual(x,snapshot(row,order,stamp('2026-09-28 10:00'),.1))
        self.assertEqual(x['rvol'],1.2)

if __name__=='__main__':unittest.main()
