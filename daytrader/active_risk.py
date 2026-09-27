"""Untuned setup plan; targets are intended exits, not predicted price moves."""
import math
import pandas as pd
from .risk_levels import session_atr
from .policy import round_trip_cost, net_reward_risk, SHORT_MIN_NET_RR
from .trade_plan import sell_by


def active_risk(df, ticker, entry, now, hold_bars=4, pattern=None):
    out = {'status':'NO-TRADE', 'risk_model':'structure-v6-unvalidated' if pattern is None else 'inverse-hs-v8-unvalidated',
           'classification':'INTRADAY', 'hold_bars':hold_bars,
           'single_bar_atr':float('nan'), 'horizon_atr':None, 'mae_distance':None}
    def reject(reason):
        return {**out, 'reason':reason}
    if len(df) < 60 or not math.isfinite(entry) or entry <= 0:
        return reject('insufficient history or invalid price')
    deadline = sell_by(ticker, now, minutes=hold_bars*15)
    if deadline is None:
        return reject('insufficient time for full hold before exchange close minus 5 minutes')
    a = float(session_atr(df).iloc[-1])
    if not math.isfinite(a) or a <= 0:
        return reject('invalid ATR')
    # Invalidation beneath recent support; never tighten after testing R:R.
    support = float(df.Low.tail(3).min())
    distance = max(1.5*a, entry-support+.1*a)
    if pattern is not None:
        distance=max(distance, entry-float(pattern['right_shoulder'])+.1*a)
    floor = entry-distance
    past = df.loc[df.index.date < pd.Timestamp(now).date()]
    daily = past.groupby(past.index.date).agg(High=('High','max'),Low=('Low','min'))
    if len(daily) < 10:
        return reject('need 10 prior sessions for range sanity check')
    # A planned 2R exit bounded by 70% of recent daily range. This cap is not
    # evidence that the target can be reached in an hour.
    cap = .7*float((daily.High-daily.Low).tail(20).mean())
    reward = min(float(pattern['target'])-entry, cap) if pattern is not None else min(2*distance, cap)
    ceiling = entry+reward
    net = net_reward_risk(entry,floor,ceiling,round_trip_cost(ticker))
    out.update(floor=floor,ceiling=ceiling,floor_distance=distance,reward_distance=reward,
               floor_pct=distance/entry*100,ceiling_pct=reward/entry*100,rr=reward/distance,
               net_rr=net,single_bar_atr=a,sell_by=deadline.isoformat(),samples=len(daily),
               support=support,range_cap=cap)
    if floor <= 0 or net < SHORT_MIN_NET_RR:
        return reject(f'structure plan gives {net:.2f}:1 after costs; need {SHORT_MIN_NET_RR}:1')
    return {**out,'status':'TRADE','reason':'structure/ATR stop and planned target; NOT an estimated win probability'}
