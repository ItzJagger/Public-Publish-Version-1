"""Persistent, observational BUY-alert tracking. No orders or model training.

Uses completed 5m OHLCV; entry is a bar-open proxy during a fixed 15-minute
window after delivery. Bars overlapping notification time are excluded.
Net P&L is a hold-to-latest/deadline scenario, NOT a stop-limit execution model.
Missing paths suppress net/extrema claims; level touches never prove fills.
History is retried for one day after deadline, then labelled incomplete.
Only future successfully sent BUY alerts are recorded (not WATCH or console).
"""
import csv
import json
import sqlite3
from pathlib import Path
import pandas as pd
from .trade_plan import eastern, exchange
from .data import fetch_intraday, completed_bars
from .policy import round_trip_cost


def stamp(value):
    return eastern(value)


class SignalTracker:
    def __init__(self, directory='tracking'):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'signals.sqlite3'
        self.last_update=None
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS signals (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS bars (ticker TEXT, time TEXT, payload TEXT, PRIMARY KEY(ticker,time))')

    def connect(self):
        return sqlite3.connect(self.path,timeout=30)

    def records(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM signals ORDER BY id')]

    def record(self, row, now, short, message):
        from .alerts import alert_plan
        order,size,deadline=alert_plan(row,now,short)
        at=stamp(now)
        mode='hourly' if short else 'long'
        # Re-recording the exact same delivery is idempotent; later deliveries get new records.
        key=f'{at.isoformat()}:{row["ticker"]}:{mode}'
        record={'id':key,'ticker':row['ticker'],'mode':mode,'alert_at':at.isoformat(),
                'deadline':stamp(deadline).isoformat(),'entry_limit':order['entry'],
                'stop':order['stop_price'],'stop_limit':order['limit_price'],
                'target':order['target_price'],'shares':size.get('shares') if size.get('eligible') else None,
                'currency':'CAD' if row['ticker'].endswith('.TO') else 'USD',
                'cost_pct':round_trip_cost(row['ticker']), 'strategy':row.get('strategy'),
                'message':message,'status':'PENDING','data_error':None}
        if stamp(deadline)<=at:raise ValueError('Tracker deadline precedes notification')
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO signals VALUES (?,?)',(key,json.dumps(record)))
        return key

    def update(self, now, fetch=fetch_intraday, force=False):
        now=stamp(now)
        if not force and self.last_update is not None and (now-self.last_update).total_seconds()<300:return
        self.last_update=now
        records=self.records()
        active=[r for r in records if r['status'] not in ('COMPLETE','INCOMPLETE')]
        for ticker in sorted({r['ticker'] for r in active}):
            try:
                data=completed_bars(fetch(ticker,period='60d',interval='5m'),'5m',now)
                # Completed data is validated before merging into durable history.
                with self.connect() as db:
                    db.executemany('INSERT OR REPLACE INTO bars VALUES (?,?,?)',
                       [(ticker,stamp(t).isoformat(),json.dumps({k:float(b[k]) for k in ('Open','High','Low','Close','Volume')}))
                        for t,b in data.iterrows()])
                error=None
            except Exception as exc:
                error=str(exc)
            with self.connect() as db:
                cached=list(db.execute('SELECT time,payload FROM bars WHERE ticker=? ORDER BY time',(ticker,)))
            frame=pd.DataFrame([json.loads(x[1]) for x in cached],index=pd.DatetimeIndex([stamp(x[0]) for x in cached]))
            for r in [x for x in active if x['ticker']==ticker]:
                r.update(evaluate(r,frame,now));r['data_error']=error
                # Leave incomplete records retryable for one day after deadline.
                if now>=stamp(r['deadline'])+pd.Timedelta(days=1) and r['status']!='COMPLETE':r['status']='INCOMPLETE'
                with self.connect() as db:db.execute('UPDATE signals SET payload=? WHERE id=?',(json.dumps(r),r['id']))
        self.write_reports()

    def write_reports(self):
        groups={}
        for r in self.records():groups.setdefault(str(stamp(r['alert_at']).date()),[]).append(r)
        out=self.directory/'reports';out.mkdir(exist_ok=True)
        fields=['ticker','mode','strategy','alert_at','deadline','status','currency','entry_limit','shares',
                'entry_status','entry_proxy','entry_at','last_price','last_at','alert_change_pct','change_pct','net_proxy',
                'max_rise_pct','max_drop_pct','first_touch','target_touch','stop_touch','missing_bars','data_error']
        for day,rows in groups.items():
            with (out/f'{day}.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
            lines=[f'Signal report — {day} (Eastern)',
                   'SIMULATED observations, not actual trades. One row per BUY alert plan.',
                   'Entry proxy: first FULL post-alert 5m bar OPEN <= buy limit, within 15 minutes.',
                   'Net proxy: hold to latest/deadline bar close, costs deducted; NOT a stop/target exit simulation.',
                   'Stop/target touches do not prove fills. Missing bars and same-bar ordering remain uncertain.',
                   'Long signals remain open across days; deadline close may be up to 5 minutes early.', '']
            for r in rows:
                def fmt(key):
                    v=r.get(key);return '-' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
                lines.append(f"{r['ticker']} | {r['mode']} | {r['status']} | entry {fmt('entry_status')} "
                             f"| from alert {fmt('alert_change_pct')}% | from entry {fmt('change_pct')}% | net proxy {fmt('net_proxy')} {r['currency']} "
                             f"| first touch {fmt('first_touch')} | missing bars {fmt('missing_bars')}")
                if r.get('data_error'):lines.append('  Data error: '+r['data_error'])
            (out/f'{day}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def expected_bars(ticker, start, end):
    """Exchange sessions account for weekends, holidays and early closes."""
    cal=exchange(ticker);pieces=[]
    for session in cal.sessions_in_range(str(start.date()),str(end.date())):
        op=cal.session_open(session).tz_convert(start.tz)
        cl=cal.session_close(session).tz_convert(start.tz)
        pieces.extend(pd.date_range(op,cl-pd.Timedelta(minutes=5),freq='5min'))
    return pd.DatetimeIndex([t for t in pieces if t>=start and t+pd.Timedelta(minutes=5)<=end])


def evaluate(r, bars, now):
    at=stamp(r['alert_at']);deadline=stamp(r['deadline']);now=stamp(now)
    start=at.ceil('5min')
    # Publication grace: never treat a not-yet-completed bar as observed.
    end=min(deadline,now-pd.Timedelta(seconds=5))
    if end<start:return {'status':'PENDING','missing_bars':0,'entry_status':'WAITING'}
    expected=expected_bars(r['ticker'],start,end)
    frame=bars.loc[bars.index.isin(expected)].sort_index() if len(bars) else bars
    missing=len(expected.difference(frame.index))
    complete=now>=deadline+pd.Timedelta(seconds=5)
    out={'status':('COMPLETE' if not missing and len(expected) else 'AWAITING_DATA') if complete else 'OPEN',
         'missing_bars':missing,'entry_status':'WAITING','entry_proxy':None,'entry_at':None,
         'net_proxy':None,'first_touch':None,'target_touch':None,'stop_touch':None,
         'max_rise_pct':None,'max_drop_pct':None,'change_pct':None,'alert_change_pct':None,'last_price':None,'last_at':None}
    if frame.empty:return out
    out.update(last_price=float(frame.Close.iloc[-1]),last_at=(frame.index[-1]+pd.Timedelta(minutes=5)).isoformat())
    out['alert_change_pct']=(out['last_price']/r['entry_limit']-1)*100
    window_end=min(deadline,at+pd.Timedelta(minutes=15))
    entry_expected=expected_bars(r['ticker'],start,min(end,window_end+pd.Timedelta(minutes=5)))
    entry_expected=entry_expected[entry_expected<window_end]
    if len(entry_expected.difference(frame.index)):
        out['entry_status']='UNCERTAIN_MISSING_DATA';return out
    window=frame.loc[frame.index<window_end]
    # Avoid inventing an intrabar fill whose timing relative to extrema is unknown.
    fills=window.loc[window.Open<=r['entry_limit']]
    if fills.empty:
        out['entry_status']='UNCONFIRMED_INTRABAR_TOUCH' if len(window) and (window.Low<=r['entry_limit']).any() else ('NO_ENTRY' if now>=window_end+pd.Timedelta(minutes=5) else 'WAITING')
        return out
    entry_at=fills.index[0];entry=float(fills.Open.iloc[0]);path=frame.loc[entry_at:]
    out.update(entry_status='OPEN_PRICE_PROXY',entry_proxy=entry,entry_at=entry_at.isoformat())
    if entry<=r['stop'] or entry>=r['target']:
        out['entry_status']='INVALID_AT_ENTRY';return out
    target=path.index[path.High>=r['target']];stop=path.index[path.Low<=r['stop']]
    out['target_touch']=target[0].isoformat() if len(target) else None
    out['stop_touch']=stop[0].isoformat() if len(stop) else None
    if missing:out['first_touch']='UNCERTAIN_MISSING_DATA'
    elif len(target) and len(stop) and target[0]==stop[0]:out['first_touch']='AMBIGUOUS_SAME_BAR'
    elif len(target) and (not len(stop) or target[0]<stop[0]):out['first_touch']='TARGET'
    elif len(stop):out['first_touch']='STOP (fill not assumed)'
    else:out['first_touch']='NEITHER'
    out['change_pct']=(out['last_price']/entry-1)*100
    if not missing:
        out['max_rise_pct']=max(0.,(float(path.High.max())/entry-1)*100)
        out['max_drop_pct']=min(0.,(float(path.Low.min())/entry-1)*100)
        if r.get('shares'):
            out['net_proxy']=(out['last_price']-entry-entry*r['cost_pct']/100)*r['shares']
    return out
