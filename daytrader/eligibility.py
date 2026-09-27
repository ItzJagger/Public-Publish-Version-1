"""Shared, fail-closed entry permission for CLI, AI text, API and alerts.

Held-position quotes are retained regardless of entry permission. No orders.
"""
import pandas as pd

from .clock import market_now, market_is_open
from .earnings import is_etf, next_earnings
from .evidence import tendency
from .policy import round_trip_cost, net_reward_risk, MIN_NET_RR, SHORT_MIN_NET_RR, SIGNAL_PROFILE
from .trade_plan import session_bounds, sell_by, size_example, previous_session_date, broker_levels


def permitted(result, mode):
    return result.get('eligibility', {}).get(mode, {}).get('eligible') is True


def attach_eligibility(result, now=None):
    now = market_now() if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('America/New_York')
    else:
        now = now.tz_convert('America/New_York')
    ticker = result['ticker']
    cost = round_trip_cost(ticker)
    common = []
    if result.get("liquidity_confirmed") is not True:
        common.append("insufficient historical trading liquidity")
    q = dict(result.get('quality') or {})
    if result.get('quote_as_of'):
        stamp=pd.Timestamp(result['quote_as_of'])
        stamp=stamp.tz_localize(now.tz) if stamp.tzinfo is None else stamp.tz_convert(now.tz)
        q['stale_minutes']=(now-stamp).total_seconds()/60
        q['is_stale']=q['stale_minutes']>45
        result['quality']=q
    if not q or q.get('stale_minutes') is None:
        common.append('data quality unavailable')
    if q.get('is_stale') or (q.get('stale_minutes') is not None and q['stale_minutes'] < -1):
        common.append('stale or future-dated quote')
    if q.get('verify_move'):
        common.append('unusual price move: verify in broker')
    if not market_is_open(now):
        common.append('outside regular market hours')
    # Price freshness applies to every mode. Intraday candle cadence and
    # entry-signal age are separate checks used only for hourly entries.
    signal_reasons = []
    signal_age_minutes = None
    try:
        stamp = pd.Timestamp(result['signal_as_of'])
        stamp = stamp.tz_localize(now.tz) if stamp.tzinfo is None else stamp.tz_convert(now.tz)
        durations = {'1m':1,'2m':2,'5m':5,'15m':15,'30m':30,'60m':60,'1h':60,'90m':90}
        duration = durations[result['interval']]
        signal_age_minutes = (now-stamp).total_seconds()/60-duration
        # completed_bars allows five seconds for publication. Until the next
        # candle closes plus that grace, this can be the newest completed bar.
        if signal_age_minutes < 0:
            signal_reasons.append('signal candle is unfinished or future-dated')
        elif signal_age_minutes > duration + 5/60:
            signal_reasons.append('completed signal feed is missing a newer candle')
    except (KeyError, ValueError, TypeError):
        signal_reasons.append('signal timestamp unavailable')

    bounds = session_bounds(ticker, now)
    if bounds is None or not bounds[0] <= now < bounds[1]:
        common.append('exchange closed (holiday, early close or outside session)')
    gates = {}
    for mode in ('buy-now', 'swing', 'position', 'weekday', 'overnight'):
        reasons = list(common)
        extra = {}
        horizon = 0
        raw_setup = False
        if mode == 'buy-now':
            reasons.extend(signal_reasons)
            patterns=result.get('patterns') or {}
            if patterns.get('bearish',{}).get('bearish'):
                reasons.append('confirmed bearish head-and-shoulders breakdown')
            bull=patterns.get('bullish',{})
            if result.get('strategy','').startswith('inverse-hs') and bull.get('max_entry') is not None:
                if float(result.get('last_price',0))>bull['max_entry']:
                    reasons.append('pattern breakout already extended: do not chase')
            extra['signal_age_minutes'] = signal_age_minutes
            rec = result.get('rec') or {}
            raw_setup = bool(result.get('raw_buy_signal', rec.get('status') == 'BUY_NOW'))
            if result.get('no_trade'):
                reasons.append('risk model: ' + (result.get('atr_stop') or {}).get('reason', 'unavailable'))
            if raw_setup and signal_age_minutes is not None and signal_age_minutes > 20:
                reasons.append('entry signal is older than the 20-minute action window')
            if raw_setup and not result.get('no_trade'):
                risk = result.get('atr_stop') or {}
                entry = float(result.get('last_price', rec['current_price']))
                extra['entry_quote'] = entry
                signal_price = float(rec['current_price'])
                floor = risk.get('floor', rec.get('floor_price', 0))
                ceiling = risk.get('ceiling', rec.get('ceiling_price', 0))
                extra['net_rr'] = net_reward_risk(entry, floor, ceiling, cost)
                if abs(entry-signal_price) > .25 * max(signal_price-floor, 0):
                    reasons.append('price moved too far from confirmed entry')
                if extra['net_rr'] < SHORT_MIN_NET_RR:
                    reasons.append(f'reward/risk below {SHORT_MIN_NET_RR} after costs')
                # Evaluate actual remaining session time, not historical best hour.
                minutes_left = 960 - (now.hour*60 + now.minute)
                if minutes_left < max(30, risk.get('hold_bars', 4)*15):
                    reasons.append('insufficient session time for planned hold')
        elif mode in ('swing', 'position'):
            block = result.get(mode) or {}
            raw_setup = bool(block.get('candidate'))
            if (block.get('patterns') or {}).get('bearish',{}).get('bearish'):
                reasons.append('confirmed bearish head-and-shoulders breakdown')
            bull=(block.get('patterns') or {}).get('bullish',{})
            if block.get('strategy','').startswith('inverse-hs') and bull.get('max_entry') is not None:
                if float(result.get('last_price',0))>bull['max_entry']:
                    reasons.append('pattern breakout already extended: do not chase')
            horizon = block.get('hold_days', 5) if mode == 'swing' else block.get('hold_weeks', 5)*5
            extra['confirmation'] = block.get('confirmation', {})
            extra['net_rr'] = block.get('net_rr')
            if raw_setup:
                quote = float(result.get('last_price', block['price']))
                extra['entry_quote'] = quote
                extra['net_rr'] = net_reward_risk(quote, block['floor'], block['ceiling'], cost)
                if extra['net_rr'] < MIN_NET_RR:
                    reasons.append('current quote no longer clears net reward/risk')
                if abs(quote-block['price']) > .5*block['atr']:
                    reasons.append('price moved too far from daily confirmation')
            if block.get('as_of'):
                # Weekday calendar is deliberately conservative on exchange holidays.
                stamp = pd.Timestamp(block['as_of']).date()
                if stamp != previous_session_date(ticker, now):
                    reasons.append('daily confirmation is not from prior weekday session')
            elif raw_setup:
                reasons.append('daily confirmation timestamp missing')
        else:
            block = result.get('season' if mode == 'weekday' else 'overnight') or {}
            values = (block.get('weekday_returns', {}).get(now.day_name(), []) if mode == 'weekday'
                      else block.get('gap_returns', []))
            extra['evidence'] = tendency(values, cost, min_samples=8 if mode == 'weekday' else 30)
            raw_setup = extra['evidence']['eligible']
            # These remain research watchlists, not buy alerts; no forced stop/target.
            if not raw_setup:
                reasons.append('tendency lacks stable positive evidence after costs')
            horizon = 1 if mode == 'overnight' else 0
            if mode == 'weekday' and now.hour >= 11:
                reasons.append('morning entry window has passed')
            if mode == 'overnight' and now.hour < 15:
                reasons.append('overnight entry window has not opened')
        if raw_setup and mode in ('buy-now', 'position', 'swing'):
            if mode == 'buy-now':
                risk = result.get('atr_stop') or {}
                deadline = sell_by(ticker, now, minutes=risk.get('hold_bars',4)*15)
                entry = float(result.get('last_price', result.get('current_price', (result.get('rec') or {}).get('current_price',0))))
            else:
                risk = result[mode]
                deadline = sell_by(ticker, now, sessions=int(horizon))
                entry = float(result.get('last_price', risk['price']))
            extra['sell_by'] = deadline.isoformat() if deadline is not None else None
            if deadline is None:
                reasons.append('full hold cannot fit before exchange close minus 5 minutes')
            if all(risk.get(k) is not None for k in ('floor','ceiling')):
                try:
                    order = broker_levels(entry, risk['floor'], risk['ceiling'])
                except (ValueError, TypeError) as exc:
                    reasons.append('invalid stop-limit plan: '+str(exc))
                    order = None
                if order is not None:
                    extra['order_plan'] = order
                    rounded_rr = net_reward_risk(order['entry'], order['limit_price'], order['target_price'], cost)
                    if rounded_rr < (SHORT_MIN_NET_RR if mode == 'buy-now' else MIN_NET_RR):
                        reasons.append('rounded order prices no longer clear net reward/risk')
                    sizing = size_example(ticker, order['entry'], order['limit_price'],order['target_price'])
                else:
                    sizing = {'eligible':False, 'reason':'invalid order levels'}
                extra['sizing'] = sizing
                if SIGNAL_PROFILE == 'active' and not sizing['eligible']:
                    reasons.append('position sizing: '+sizing['reason'])
        if not raw_setup:
            reasons.append('no confirmed setup')
        if raw_setup and not is_etf(ticker):
            date = next_earnings(ticker)
            if date is None:
                reasons.append('earnings date unknown')
            elif pd.Timestamp(date).date() <= (pd.Timestamp(extra['sell_by']).date() if extra.get('sell_by') else (now.normalize() + pd.offsets.BDay(horizon)).date()):
                reasons.append('earnings inside intended hold window or cached date expired')
        gates[mode] = {'eligible': not reasons, 'reasons': list(dict.fromkeys(reasons)),
                       'cost_pct': cost, **extra}
    result['eligibility'] = gates
    # Formation notices are not entries and never bypass buy eligibility.
    result['pattern_watches']=[]
    for mode, patterns in [('buy-now',result.get('patterns') or {}),
                            ('position',(result.get('position') or {}).get('patterns') or {})]:
        bull=patterns.get('bullish',{})
        if not bull.get('watch') or patterns.get('bearish',{}).get('bearish') or common:
            continue
        if mode=='buy-now' and (signal_reasons or signal_age_minutes is None or signal_age_minutes>5):
            continue
        if mode=='position':
            block=result.get('position') or {}
            if not block.get('as_of') or pd.Timestamp(block['as_of']).date()!=previous_session_date(ticker,now):continue
        quote=result.get('last_price')
        if quote is None or not bull['right_shoulder'] < float(quote) <= bull['trigger']:continue
        sessions=int((result.get('position') or {}).get('hold_weeks',4)*5)
        deadline=sell_by(ticker,now,minutes=60) if mode=='buy-now' else sell_by(ticker,now,sessions=sessions)
        if deadline is None:continue
        if not is_etf(ticker):
            earnings=next_earnings(ticker)
            if earnings is None or pd.Timestamp(earnings).date()<=deadline.date():continue
        result['pattern_watches'].append({'ticker':ticker,'mode':mode,**bull})
    rec = result.get('rec')
    if rec and rec.get('status') == 'BUY_NOW' and not permitted(result, 'buy-now'):
        rec['raw_status'] = 'BUY_NOW'
        rec['status'] = 'BLOCKED'
        rec.setdefault('rsi', float('nan'))
    return result
