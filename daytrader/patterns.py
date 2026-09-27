"""Causal head-and-shoulders geometry on COMPLETED bars only.

Pivots need two bars to their right. We never label the pivot bar as the time
we knew it. Numeric tolerances are untuned engineering choices, not win odds.
"""
import math
import pandas as pd
from .indicators import atr


def _one(df, inverse=True, intraday=False):
    none={'state':'none','entry':False,'watch':False,'bearish':False}
    if len(df)<25:return none
    work=df.tail(200).copy()
    volatility=float(atr(work).iloc[-1])
    if not math.isfinite(volatility) or volatility<=0:return none
    if intraday:work=work.loc[work.index.date==work.index[-1].date()]
    if len(work)<25:return none
    # Reflect a top so both shapes can use the same five-point bottom geometry.
    lows=work.Low.to_numpy() if inverse else -work.High.to_numpy()
    highs=work.High.to_numpy() if inverse else -work.Low.to_numpy()
    closes=work.Close.to_numpy() if inverse else -work.Close.to_numpy()
    pivots=[]
    for i in range(2,len(work)-2):
        lo=all(lows[i]<lows[j] for j in range(i-2,i+3) if j!=i)
        hi=all(highs[i]>highs[j] for j in range(i-2,i+3) if j!=i)
        if lo==hi:continue # ambiguous outside bar or no pivot
        point=(i,'L' if lo else 'H',float(lows[i] if lo else highs[i]))
        if pivots and pivots[-1][1]==point[1]:
            if (point[1]=='L' and point[2]<pivots[-1][2]) or (point[1]=='H' and point[2]>pivots[-1][2]):
                pivots[-1]=point
        else:pivots.append(point)
    for end in range(len(pivots),4,-1):
        pts=pivots[end-5:end]
        if [x[1] for x in pts]!=['L','H','L','H','L']:continue
        ls,n1,head,n2,rs=pts
        span=rs[0]-ls[0]
        if not 8<=span<=80 or len(work)-1-rs[0]>20 or ls[0]<8:continue
        left_gap=head[0]-ls[0];right_gap=rs[0]-head[0]
        if max(left_gap,right_gap)>3*min(left_gap,right_gap):continue
        slope=(n2[2]-n1[2])/(n2[0]-n1[0])
        def neck(i):return n1[2]+slope*(i-n1[0])
        height=neck(head[0])-head[2]
        if height<1.5*volatility:continue
        if min(ls[2],rs[2])-head[2]<.5*volatility:continue
        if abs(ls[2]-rs[2])>max(.5*volatility,.25*height):continue
        if abs(n2[2]-n1[2])>.5*height:continue
        if min(n1[2]-ls[2],n2[2]-rs[2])<.5*volatility:continue
        # A reversal needs an actual preceding trend into the left shoulder.
        if closes[ls[0]-8]-closes[ls[0]]<volatility:continue
        known=rs[0]+2
        last=len(work)-1
        # Reject invalidated structures, not just their final price.
        if any(lows[j]<head[2] for j in range(known,last+1)):continue
        if any(lows[j]<rs[2]-.1*volatility for j in range(known+1,last+1)):continue
        # Reject a breakout that happened before the right shoulder was knowable.
        if any(closes[j]>neck(j)+.05*volatility for j in range(rs[0]+1,known)):continue
        breaks=[j for j in range(known,last+1)
                if closes[j]>neck(j)+.05*volatility and closes[j-1]<=neck(j-1)+.05*volatility]
        neckline=neck(last);over=closes[last]-neckline
        first=breaks[0] if breaks else None
        fresh=first==last
        retest=(first is not None and 1<=last-first<=6
                and lows[last]<=neckline+.15*volatility and closes[last]>neckline+.05*volatility
                and lows[last-1]>neck(last-1)+.15*volatility
                and all(closes[j]>=neck(j)-.1*volatility for j in range(first,last)))
        if first is not None and not inverse:
            return {'state':'confirmed_bearish','entry':False,'watch':False,'bearish':True,
                    'neckline':-neckline,'right_shoulder':-rs[2], 'pattern_id':str(work.index[head[0]])+'-top',
                    'as_of':str(work.index[-1]),'known_at':str(work.index[known])}
        if first is not None and not fresh and not retest:continue
        direction=1 if inverse else -1
        eligible_shape=inverse and 0<over<=.5*volatility and (fresh or retest)
        watching=inverse and first is None and -.75*volatility<=over<=.05*volatility
        if not eligible_shape and not watching:continue
        return {'state':('breakout' if fresh else 'retest') if eligible_shape else 'forming',
                'entry':eligible_shape,'watch':watching,'bearish':False,
                'neckline':direction*neckline,'head':direction*head[2],
                'right_shoulder':direction*rs[2], 'height':height, 'atr':volatility,
                'target':direction*(neckline+height), 'max_entry':direction*(neckline+.5*volatility),
                'trigger':direction*(neckline+.05*volatility),
                'pattern_id':str(work.index[head[0]])+'-bottom',
                'as_of':str(work.index[-1]), 'known_at':str(work.index[known]),
                'pivot_times':[str(work.index[x[0]]) for x in pts]}
    return none


def head_shoulders(df, intraday=False):
    """Caller supplies only completed bars; return bullish setup + bearish veto."""
    return {'bullish':_one(df,True,intraday), 'bearish':_one(df,False,intraday)}


def complete_fifteen(df):
    """Build 15m risk bars only from all THREE consecutive completed 5m bars."""
    rows=[];indices=[]
    for stamp,g in df.groupby(df.index.floor('15min')):
        expected=pd.date_range(stamp,periods=3,freq='5min')
        if not g.index.equals(expected):continue
        rows.append({'Open':float(g.Open.iloc[0]),'High':float(g.High.max()),
                     'Low':float(g.Low.min()),'Close':float(g.Close.iloc[-1]),'Volume':float(g.Volume.sum())})
        indices.append(stamp)
    return pd.DataFrame(rows,index=pd.DatetimeIndex(indices),columns=['Open','High','Low','Close','Volume'])
