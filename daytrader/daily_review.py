"""17:00 Eastern session reports and automatic local shadow-model training."""
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import pandas as pd
from .tracker import stamp, expected_bars, evaluate
from .trade_plan import exchange, session_bounds
from .learning import atomic_json, train_reports, number


def trading_day(day):
    return any(exchange(t).is_session(str(day)) for t in ('SPY', 'XIU.TO'))


def next_report_seconds(now):
    now = stamp(now)
    for offset in range(10):
        day = (now+pd.Timedelta(days=offset)).date()
        at = stamp(str(day)+' 17:00')
        if at > now and trading_day(day):
            return (at-now).total_seconds()
    return 1800.


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def clean(value):
    """Keep structured reports portable strict JSON; no NaN or Infinity."""
    if isinstance(value, dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [clean(v) for v in value]
    if isinstance(value, float) and number(value) is None:return None
    return value


def fmt(value):
    if value is None:return 'unavailable'
    return f'{value:,.2f}' if isinstance(value, (float,int)) else str(value)


def findings(r):
    notes = []
    if r.get('missing_bars'):
        notes.append('Incomplete price path: do not use this outcome for training or full-period extrema.')
    if r.get('entry_status') != 'OPEN_PRICE_PROXY':
        notes.append('No confirmed simulated entry: no trade profit/loss is assigned.')
    if r.get('net_return_pct') is not None:
        if r['net_return_pct'] < 0:
            if r.get('change_pct',0) > 0:
                notes.append('The price rose, but the estimated round-trip costs exceeded that gain.')
            else:
                notes.append('The observed holding-period finish was below the entry after costs.')
            if (r.get('max_rise_pct') or 0) > r['cost_pct']:
                notes.append('There was an earlier favourable excursion before the losing finish; this does not prove an earlier exit was predictable.')
        else:
            notes.append('Positive estimated hold-to-deadline/latest result after costs.')
    if r.get('stop_touch'):
        notes.append('The stop level was touched. A stop-limit fill is not assumed.')
    if r.get('first_touch') == 'AMBIGUOUS_SAME_BAR':
        notes.append('Stop and target touched in the same candle: their order is unknown.')
    if r.get('status') != 'COMPLETE':notes.append('The intended holding period is unfinished or its data is incomplete.')
    return notes or ['No diagnostic conclusion yet.']


def chart(r, bars, path, cutoff):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    start = stamp(r['alert_at']).ceil('5min')
    expected = expected_bars(r['ticker'], start, cutoff)
    if not len(expected):return False
    # Reindex to expected trading bars so missing data breaks the plotted line.
    frame = bars.reindex(expected)
    if frame.empty or 'Close' not in frame or frame.Close.notna().sum() == 0:return False
    fig, ax = plt.subplots(figsize=(10,3.6), layout='constrained')
    candle_ends = frame.index + pd.Timedelta(minutes=5)
    ax.plot(candle_ends, frame.Close, color='#1d4ed8', lw=1.4, label='5-minute close')
    ax.fill_between(candle_ends, frame.Low, frame.High, color='#93c5fd', alpha=.28, label='Observed high–low')
    for key, label, colour in [('entry_limit','Buy limit','#64748b'),('stop','Stop','#dc2626'),('target','Target','#15803d')]:
        ax.axhline(r[key], color=colour, ls='--', lw=.9, label=label)
    deadline = stamp(r['deadline'])
    if start <= deadline <= cutoff:
        ax.axvline(deadline, color='#7c3aed', ls=':', label='Sell-by deadline')
    if r.get('entry_at') and r.get('entry_proxy'):
        ax.scatter([stamp(r['entry_at'])],[r['entry_proxy']], color='#111827', s=30, zorder=4, label='Simulated entry')
    ax.set_title(f"{r['ticker']} · {r['mode']} · observed path, not actual fills")
    ax.set_ylabel(r['currency']);ax.set_xlabel('Eastern time (candle end); gaps include non-trading hours')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=start.tzinfo))
    ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.18)
    ax.legend(fontsize=7, loc='best', ncol=3)
    fig.savefig(path, dpi=130);plt.close(fig)
    return True


def report_rows(tracker, day):
    cutoff = stamp(str(day)+' 17:00')
    rows = []
    for original in tracker.records():
        if stamp(original['alert_at']) > cutoff or not stamp(original['alert_at']).date() <= day <= stamp(original['deadline']).date():
            continue
        bars = tracker.cached_bars(original['ticker'])
        r = dict(original)
        # Re-evaluate at this date, never copy a future completed long-hold outcome.
        r.update(evaluate(r, bars, cutoff))
        r['data_error'] = None  # Current fetch errors may belong to a later day.
        r['notes'] = findings(r)
        bounds = session_bounds(r['ticker'], cutoff)
        r.update(day_close=None, day_peak=None, day_floor=None, day_missing_bars=None,
                 day_start=None, day_close_at=None, day_change_from_signal_pct=None)
        if bounds:
            start = max(bounds[0], stamp(r['alert_at']).ceil('5min'))
            expected = expected_bars(r['ticker'], start, bounds[1])
            frame = bars.reindex(expected)
            missing = len(expected) if 'Close' not in frame else int(frame.Close.isna().sum())
            r['day_missing_bars'] = missing
            r['day_start'] = start.isoformat()
            if len(expected) and not missing:
                r.update(day_close=float(frame.Close.iloc[-1]), day_peak=float(frame.High.max()),
                         day_floor=float(frame.Low.min()), day_close_at=bounds[1].isoformat())
                price = r.get('signal_price', r['entry_limit'])
                r['day_change_from_signal_pct'] = (r['day_close']/price-1)*100
        r['chart_file'] = hashlib.sha256(r['id'].encode()).hexdigest()[:16]+'.png'
        rows.append(clean(r))
    return rows


def summary(rows):
    totals = {}
    for r in rows:
        bucket = totals.setdefault(r['currency'], {'signals':0,'priced_scenarios':0,'net_proxy':0.,'gross_proxy':0.})
        bucket['signals'] += 1
        if r.get('net_proxy') is not None:
            bucket['priced_scenarios'] += 1
            bucket['net_proxy'] += r['net_proxy'];bucket['gross_proxy'] += r['gross_proxy']
    return totals


def render_report(tracker, day, generated_at):
    out = tracker.directory/'daily'/str(day)
    out.mkdir(parents=True,exist_ok=True)
    rows = report_rows(tracker,day)
    for r in rows:
        if not chart(r,tracker.cached_bars(r['ticker']),out/r['chart_file'],stamp(str(day)+' 17:00')):
            r['chart_file'] = None
    totals = summary(rows)
    doc = {'schema':1,'session_date':str(day),'as_of':stamp(str(day)+' 17:00').isoformat(),
           'generated_at':stamp(generated_at).isoformat(), 'type':'SIMULATED_SIGNAL_REVIEW',
           'summary_by_currency':totals,'signals':rows}
    atomic_json(out/'report.json',doc)
    fields = ['ticker','mode','strategy','signal_version','alert_at','signal_price','entry_limit','entry_status',
              'entry_proxy','entry_at','shares','currency','deadline','status','last_price','last_at',
              'peak_price','peak_at','floor_price','floor_at','gross_proxy','estimated_cost','net_proxy',
              'net_return_pct','max_rise_pct','max_drop_pct','stop','stop_limit','target','first_touch',
              'stop_touch','target_touch','missing_bars','day_close','day_peak','day_floor','day_close_at','day_missing_bars']
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows);atomic_text(out/'signals.csv',buffer.getvalue())
    intro = [f'Daily signal review — {day} | 17:00 Eastern',
             'SIMULATED signal entries, not evidence of actual purchases or broker fills.',
             'Holding-period P/L holds to the last complete candle at/before the deadline (up to 5 minutes early), or latest when open.',
             'Day close/high/low are separate observations through session close, even after an hourly deadline.',
             'Costs use the saved estimate at alert time. No stop-limit execution is simulated.',
             'Totals sum independent scenarios, not portfolio returns; long-hold values are cumulative and must NOT be summed across dates.',
             'Peaks/lows are from available five-minute bars; missing paths withhold complete-period metrics.',
             'Charts and diagnostics describe observations and associations, not the cause of a price move.', '']
    for currency, total in totals.items():
        intro.append(f"{currency}: {total['signals']} signal plans, {total['priced_scenarios']} priced scenarios; net {total['net_proxy']:+.2f}")
    if not rows:intro.append('No BUY alerts or ongoing signal plans for this trading day.')
    text = list(intro); sections = []
    for r in rows:
        block = [f"{r['ticker']} | {r['mode']} | {r['status']} | {r['currency']}",
                 f"Signal: {r['alert_at']} at {fmt(r.get('signal_price'))}; buy limit {fmt(r['entry_limit'])}",
                 f"Simulated entry: {fmt(r.get('entry_proxy'))} at {fmt(r.get('entry_at'))}; shares {fmt(r.get('shares'))}; {r.get('entry_status')}",
                 f"Sell by: {r['deadline']}; holding-period finish/latest {fmt(r.get('last_price'))} at {fmt(r.get('last_at'))}",
                 f"Holding-period peak {fmt(r.get('peak_price'))} at {fmt(r.get('peak_at'))}; floor {fmt(r.get('floor_price'))} at {fmt(r.get('floor_at'))}",
                 f"Gross P/L {fmt(r.get('gross_proxy'))}; estimated costs {fmt(r.get('estimated_cost'))}; net P/L {fmt(r.get('net_proxy'))} ({fmt(r.get('net_return_pct'))}%)",
                 f"Day close {fmt(r.get('day_close'))}; day peak {fmt(r.get('day_peak'))}; day floor {fmt(r.get('day_floor'))}; missing day bars {fmt(r.get('day_missing_bars'))}",
                 f"Stop {fmt(r['stop'])}; stop-limit {fmt(r['stop_limit'])}; target {fmt(r['target'])}; first touch {fmt(r.get('first_touch'))}"]
        if r.get('prediction'):
            block.append(f"Prediction saved at alert: {r['prediction']['net_return_pct']:+.2f}% net (experimental shadow model {r['prediction']['model_id']})")
        block.extend(r['notes']);text.extend(block+[''])
        sections.append('<section><h2>'+html.escape(r['ticker'])+'</h2><pre>'+html.escape('\n'.join(block))+'</pre>'+
                        (f'<img src="{r["chart_file"]}" alt="Price path for {html.escape(r["ticker"])}">' if r['chart_file'] else '<p>No chart data available.</p>')+'</section>')
    learning = train_reports(tracker.directory, generated_at)
    learn_lines = ['Learning review — shadow mode (does not change alerts or risk rules)']
    for mode, info in learning['modes'].items():
        learn_lines.append(f"{mode}: {info['status']}; {info['usable_signals']} usable completed signals, {info['entry_sessions']} entry sessions.")
        if info.get('model'):
            learn_lines.append(f"Chronological holdout: model MSE {info['holdout_mse']:.4f}, baseline {info['baseline_mse']:.4f}; direction accuracy {info['direction_accuracy']:.1%}.")
            learn_lines.append('Largest standardized associations (not causes): '+', '.join(f"{a['feature']} {a['coefficient']:+.3f}" for a in info['associations'][:4]))
        else:learn_lines.append(info['requirement'])
    learn_lines.append('Cumulative completed outcomes (one ticker/mode/entry-date observation; descriptive, not proof of an edge):')
    for group in learning['outcome_groups']:
        learn_lines.append(f"{group['group']}: n={group['completed']}, win rate {group['win_rate']:.1%}, mean net return {group['mean_net_pct']:+.3f}%.")
    # Score only predictions made BEFORE the actual alert outcome became known.
    predicted = [r for r in rows if r.get('prediction') and r.get('status') == 'COMPLETE' and r.get('net_return_pct') is not None]
    if predicted:
        mae = sum(abs(r['prediction']['net_return_pct']-r['net_return_pct']) for r in predicted)/len(predicted)
        learn_lines.append(f'Previously saved shadow predictions: {len(predicted)} completed; mean absolute error {mae:.3f} percentage points.')
    text += learn_lines
    atomic_text(out/'report.txt','\n'.join(text)+'\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Daily signal review '+str(day)+'</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#172033;background:#f4f7fb}section{background:white;padding:20px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 system-ui}img{max-width:100%;height:auto}h1{font-size:28px}</style><h1>Daily signal review · '+str(day)+'</h1><pre>'+html.escape('\n'.join(intro))+'</pre>'+''.join(sections)+'<section><h2>Learning review</h2><pre>'+html.escape('\n'.join(learn_lines))+'</pre></section></html>'
    atomic_text(out/'report.html',page)
    # Written LAST: restart-safe completion marker.
    atomic_json(out/'complete.json',{'generated_at':stamp(generated_at).isoformat(),'schema':1,
                                  'needs_retry':any((r.get('missing_bars') or 0) > 0 or (r.get('day_missing_bars') or 0) > 0 for r in rows)})
    return out


class DailyReview:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_attempt = None
        with tracker.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS review_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def run_due(self, now, fetch=None):
        now = stamp(now)
        if (self.last_attempt is not None and (now-self.last_attempt).total_seconds() < 300
            and not self.last_attempt < stamp(str(now.date())+' 17:00') <= now):
            return []
        self.last_attempt = now
        with self.tracker.connect() as db:
            records = self.tracker.records()
            first = min([stamp(r['alert_at']).date() for r in records]+[now.date()])
            db.execute('INSERT OR IGNORE INTO review_state VALUES (?,?)',('start_date',str(first)))
            start = db.execute('SELECT value FROM review_state WHERE key=?',('start_date',)).fetchone()[0]
        last = now.date() if now.hour >= 17 else (now-pd.Timedelta(days=1)).date()
        def pending(day):
            marker = self.tracker.directory/'daily'/str(day)/'complete.json'
            if not marker.exists():return True
            saved = json.loads(marker.read_text(encoding='utf-8'))
            return (saved.get('needs_retry',False) and (now.date()-day).days <= 3
                    and (now-stamp(saved['generated_at'])).total_seconds() >= 1800)
        due = [d.date() for d in pd.date_range(start,str(last)) if trading_day(d.date()) and pending(d.date())]
        written = []
        for day in due:
            kwargs = {'force':True,'include_day':day}
            if fetch is not None:kwargs['fetch'] = fetch
            self.tracker.update(now,**kwargs)
            path = render_report(self.tracker,day,now)
            print(f'DAILY REPORT: {path / "report.html"}')
            written.append(path)
        return written
