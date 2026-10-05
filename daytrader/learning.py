"""Local outcome learning, in shadow mode. Never changes stops or sends orders.

Fixed ridge regression learns net hold-to-deadline percentage return from
features saved AT notification. Separate hourly/long models, chronological
holdout with overlapping labels purged. No chart-image AI or causal claims.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .trade_plan import eastern

FEATURES = ('stop_pct', 'target_pct', 'cost_pct', 'hour', 'weekday',
            'rsi', 'rvol', 'atr_pct', 'momentum_pct', 'distance_sma_pct',
            'is_meanrev', 'is_pattern')
VERSION = 'ridge-shadow-v1'


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def snapshot(row, order, now, cost_pct):
    """Only contemporaneous inputs: never derive features from future bars."""
    rec = row.get('rec') or {}
    risk = row.get('atr_stop') or row
    entry = order['entry']
    at = eastern(now)
    strategy = row.get('strategy', '')
    return dict(zip(FEATURES, [
        (entry-order['stop_price'])/entry*100,
        (order['target_price']-entry)/entry*100, cost_pct,
        at.hour + at.minute/60, at.weekday(),
        number(row.get('rsi', rec.get('rsi'))),
        number(row.get('relative_volume', rec.get('relative_volume'))),
        number(risk.get('single_bar_atr', risk.get('atr')))/entry*100
            if number(risk.get('single_bar_atr', risk.get('atr'))) is not None else None,
        number(row.get('mom_3m_pct')), number(row.get('dist_sma50_pct')),
        float('meanrev' in strategy), float('hs' in strategy),
    ]))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fit(rows):
    x = np.array([[number(r['features'].get(k)) for k in FEATURES] for r in rows], dtype=float)
    # Imputation/scaling are fitted ONLY on the training partition.
    med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    x = np.where(np.isfinite(x), x, med)
    scale = x.std(axis=0); scale[scale < 1e-9] = 1.
    x = np.clip((x-med)/scale, -10, 10)
    y = np.array([r['net_return_pct'] for r in rows])
    intercept = float(y.mean())
    coef = np.linalg.solve(x.T@x + 10*np.eye(len(FEATURES)), x.T@(y-intercept))
    return {'medians':med.tolist(), 'scales':scale.tolist(),
            'coefficients':coef.tolist(), 'intercept':intercept}


def predict(model, features):
    x = np.array([number(features.get(k)) for k in FEATURES], dtype=float)
    x = np.where(np.isfinite(x), x, model['medians'])
    x = np.clip((x-np.array(model['medians']))/np.array(model['scales']), -10, 10)
    return float(model['intercept']+x@np.array(model['coefficients']))


def training_rows(reports_dir, cutoff, require_features=True):
    """Read structured daily reports, deduplicating repeated long-hold snapshots."""
    rows = {}
    for path in sorted(Path(reports_dir).glob('*/report.json')):
        doc = json.loads(path.read_text(encoding='utf-8'))
        if doc.get('schema') != 1 or eastern(doc['as_of']) > eastern(cutoff):
            continue
        for r in doc['signals']:
            if (r.get('status') == 'COMPLETE' and r.get('entry_status') == 'OPEN_PRICE_PROXY'
                and r.get('missing_bars') == 0 and (r.get('features') or not require_features)
                and number(r.get('net_return_pct')) is not None
                and eastern(r['deadline']) <= eastern(cutoff)):
                rows[r['id']] = r
    # Multiple notifications for the same ticker/mode/session are correlated repeats.
    unique = {}
    for r in sorted(rows.values(), key=lambda r: eastern(r['alert_at'])):
        key = (r['ticker'], r['mode'], str(eastern(r['alert_at']).date()))
        unique.setdefault(key, r)
    return list(unique.values())


def train_reports(directory, now):
    root = Path(directory)
    rows = training_rows(root/'daily', now)
    observed = training_rows(root/'daily', now, require_features=False)
    groups = {}
    for r in observed:
        key = f"{r['mode']} | {r.get('strategy','unknown')} | {r.get('currency','unknown')}"
        groups.setdefault(key, []).append(r['net_return_pct'])
    review = []
    for key, values in sorted(groups.items()):
        gains = sum(max(0,v) for v in values)
        losses = -sum(min(0,v) for v in values)
        review.append({'group':key,'completed':len(values),
                       'win_rate':sum(v>0 for v in values)/len(values),
                       'mean_net_pct':sum(values)/len(values),
                       'profit_factor_equal_weight':gains/losses if losses else None})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    result = {'version':VERSION, 'trained_at':eastern(now).isoformat(), 'dataset_id':digest,
              'outcome_groups':review, 'operation':'SHADOW_ONLY', 'target':'net hold-to-deadline return percent; simulated', 'modes':{}}
    for mode in ('hourly', 'long'):
        data = [r for r in rows if r['mode'] == mode]
        days = sorted({str(eastern(r['alert_at']).date()) for r in data})
        info = {'usable_signals':len(data), 'entry_sessions':len(days), 'status':'COLLECTING',
                'requirement':'100 completed signals across 20 entry sessions; 60 train and 30 test after purging'}
        result['modes'][mode] = info
        if len(data) < 100 or len(days) < 20:
            continue
        split_day = days[int(len(days)*.7)]
        boundary = eastern(split_day)
        train = [r for r in data if eastern(r['alert_at']) < boundary and eastern(r['deadline']) < boundary]
        test = [r for r in data if eastern(r['alert_at']) >= boundary]
        info.update(train_n=len(train), test_n=len(test), test_start=split_day,
                    purged=len(data)-len(train)-len(test))
        if len(train) < 60 or len(test) < 30:
            info['status'] = 'COLLECTING_AFTER_PURGE'
            continue
        model = fit(train)
        y = np.array([r['net_return_pct'] for r in test])
        predicted = np.array([predict(model, r['features']) for r in test])
        baseline = np.mean([r['net_return_pct'] for r in train])
        mse = float(np.mean((y-predicted)**2))
        baseline_mse = float(np.mean((y-baseline)**2))
        info.update(status='SHADOW_TRAINED', holdout_mse=mse, baseline_mse=baseline_mse,
                    beats_baseline=bool(mse < baseline_mse*.95),
                    direction_accuracy=float(np.mean((predicted > 0) == (y > 0))),
                    train_ids=[r['id'] for r in train], test_ids=[r['id'] for r in test])
        # Final shadow predictor uses all available completed labels after validation.
        info['model'] = fit(data)
        info['associations'] = sorted(
            [{'feature':k, 'coefficient':float(v)} for k,v in zip(FEATURES,info['model']['coefficients'])],
            key=lambda a: -abs(a['coefficient']))
    path = root/'learning'/'runs'/f'{eastern(now).date()}-{digest}.json'
    atomic_json(path, result)
    atomic_json(root/'learning'/'latest.json', result)
    return result


def shadow_prediction(directory, mode, features, now):
    path = Path(directory)/'learning'/'latest.json'
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('version') != VERSION or eastern(doc['trained_at']) >= eastern(now):
        return None
    info = doc['modes'].get(mode, {})
    if 'model' not in info:
        return None
    return {'model_id':doc['dataset_id'], 'net_return_pct':predict(info['model'],features),
            'trained_at':doc['trained_at'], 'operation':'SHADOW_ONLY'}
