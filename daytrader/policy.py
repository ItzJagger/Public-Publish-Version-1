"""Conservative, untuned defaults. Amounts use the listing's currency.

These are engineering hypotheses, not optimized/proven profitable parameters.
Edit here deliberately, record changes, and compare on untouched history.
"""
import math
import os

FX_ROUND_TRIP_PCT = 3.0
SLIPPAGE_ROUND_TRIP_PCT = 0.1
MIN_NET_RR = 1.5
SIGNAL_PROFILE = os.environ.get('SCAN_SIGNAL_PROFILE', 'balanced').strip().lower()
ENTRY_PROFILES = {
    'active': {'daily_shares': 500_000, 'trend_rvol': .8, 'meanrev_rvol': .8,
               'trend_rsi_max': 70, 'meanrev_rsi_max': 65},
    'strict': {'daily_shares': 1_000_000, 'trend_rvol': 1.1, 'meanrev_rvol': 1.0,
               'trend_rsi_max': 65, 'meanrev_rsi_max': 60},
    'balanced': {'daily_shares': 500_000, 'trend_rvol': 1.0, 'meanrev_rvol': .9,
                 'trend_rsi_max': 68, 'meanrev_rsi_max': 62},
}
if SIGNAL_PROFILE not in ENTRY_PROFILES:
    raise ValueError('SCAN_SIGNAL_PROFILE must be strict, balanced or active')
ENTRY_FILTERS = dict(ENTRY_PROFILES[SIGNAL_PROFILE])
MIN_DAILY_DOLLARS = 10_000_000
MIN_LIQUIDITY_DAYS = 10


def round_trip_cost(ticker, fx=FX_ROUND_TRIP_PCT, slip=SLIPPAGE_ROUND_TRIP_PCT):
    if not all(math.isfinite(x) and x >= 0 for x in (fx, slip)):
        raise ValueError('Costs must be finite and nonnegative')
    return slip + (0.0 if ticker.upper().endswith('.TO') else fx)


def net_reward_risk(entry, floor, ceiling, cost_pct):
    """Approximate round-trip cost charged on both winning and losing outcomes."""
    if not all(math.isfinite(float(x)) for x in (entry, floor, ceiling, cost_pct)):
        return 0.0
    if not 0 < floor < entry < ceiling or cost_pct < 0:
        return 0.0
    cost = entry * cost_pct / 100
    return (ceiling - entry - cost) / (entry - floor + cost)

# Active is an explicit research profile, not a claim of superior returns.
SHORT_MIN_NET_RR = 1.25 if SIGNAL_PROFILE == 'active' else MIN_NET_RR
DEFAULT_SIGNAL_INTERVAL = '5m' if SIGNAL_PROFILE == 'active' else '30m'

def positive_setting(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return value

# CAD examples only; no assumptions about USD/CAD conversion for sizing.
POSITION_BUDGET = positive_setting('SCAN_POSITION_CAD', '500')
RISK_BUDGET = positive_setting('SCAN_RISK_CAD', '5')
MIN_PROFIT = positive_setting('SCAN_MIN_PROFIT_CAD', '1.01')
