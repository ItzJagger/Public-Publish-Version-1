"""Explicit exits and bounded whole-share examples. No orders or profit forecasts."""
from functools import lru_cache
import math
import pandas as pd
from .policy import POSITION_BUDGET, RISK_BUDGET, MIN_PROFIT, round_trip_cost


@lru_cache(maxsize=2)
def calendar(ticker):
    import exchange_calendars as xc
    return xc.get_calendar('XTSE' if ticker == 'CA' else 'XNYS')


def exchange(ticker):
    return calendar('CA' if ticker.upper().endswith('.TO') else 'US')


def eastern(now):
    ts = pd.Timestamp(now)
    return ts.tz_localize('America/New_York') if ts.tzinfo is None else ts.tz_convert('America/New_York')


def session_bounds(ticker, now):
    cal = exchange(ticker)
    date = str(eastern(now).date())
    if not cal.is_session(date):
        return None
    return cal.session_open(date).tz_convert('America/New_York'), cal.session_close(date).tz_convert('America/New_York')


def previous_session_date(ticker, now):
    cal = exchange(ticker)
    date = pd.Timestamp(eastern(now).date())
    return cal.date_to_session(date-pd.Timedelta(days=1), direction='previous').date()


def sell_by(ticker, now, minutes=None, sessions=20):
    now = eastern(now)
    cal = exchange(ticker)
    bounds = session_bounds(ticker, now)
    if bounds is None or not bounds[0] <= now < bounds[1]:
        return None
    if minutes is not None:
        end = now + pd.Timedelta(minutes=minutes)
        return end if end <= bounds[1]-pd.Timedelta(minutes=5) else None
    start = cal.date_to_session(str(now.date()))
    end_day = cal.session_offset(start, sessions)
    return cal.session_close(end_day).tz_convert('America/New_York')-pd.Timedelta(minutes=5)


def size_example(ticker, entry, floor, ceiling, budget=None, risk_budget=None, min_profit=None):
    budget = POSITION_BUDGET if budget is None else budget
    risk_budget = RISK_BUDGET if risk_budget is None else risk_budget
    min_profit = MIN_PROFIT if min_profit is None else min_profit
    if not ticker.upper().endswith('.TO'):
        return {'eligible':False, 'reason':'CAD sizing unavailable for US listing; FX conversion required'}
    vals = [entry, floor, ceiling, budget, risk_budget, min_profit]
    if not all(math.isfinite(float(x)) and x > 0 for x in vals) or not floor < entry < ceiling:
        return {'eligible':False, 'reason':'invalid sizing inputs'}
    cost = entry*round_trip_cost(ticker)/100
    gain, loss = ceiling-entry-cost, entry-floor+cost
    quantity = max(0, min(math.floor(budget/(entry+cost)), math.floor(risk_budget/loss)))
    minimum = math.ceil(min_profit/gain) if gain > 0 else None
    return {'eligible': quantity > 0 and gain*quantity >= min_profit,
            'reason':'target net profit below minimum within capital/risk limits',
            'shares':quantity, 'capital':quantity*entry, 'net_target_profit':quantity*gain,
            'planned_stop_loss':quantity*loss, 'minimum_shares':minimum,
            'budget':budget, 'risk_budget':risk_budget, 'minimum_profit':min_profit}


def broker_levels(entry, floor, ceiling):
    """Conservative cent prices for whole-share equity orders.

    Keep the existing technical stop (rounded down <1 cent). Equal stop/limit
    works for Wealthsimple CAD equities. No fill or current-bid guarantee.
    """
    from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
    vals = (entry, floor, ceiling)
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in vals):
        raise ValueError('Order prices must be positive and finite')
    if not float(floor) < float(entry) < float(ceiling):
        raise ValueError('Sell stop must be below entry and target above entry')
    def cents(value, rounding):
        return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=rounding))
    buy = cents(entry, ROUND_CEILING)
    stop = cents(floor, ROUND_FLOOR)
    target = cents(ceiling, ROUND_FLOOR)
    if not 0 < stop < buy < target:
        raise ValueError('Order levels collapse at cent precision')
    return {'entry':buy, 'stop_price':stop, 'limit_price':stop, 'target_price':target}
