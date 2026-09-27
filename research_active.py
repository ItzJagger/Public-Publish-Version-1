"""Replay active hourly rules on completed 5m data; no alerts or orders.

A causal diagnostic, NOT a full portfolio/backtest: historical earnings calendars,
quote latency, spreads and portfolio overlap are not modeled. CSV trade rows
include stop/target ambiguity (stop first), gap losses and next-bar entry.
"""
import os
os.environ['SCAN_SIGNAL_PROFILE'] = 'active'
import argparse
from pathlib import Path
import json
from collections import Counter
import pandas as pd
from daytrader.data import fetch_intraday, completed_bars
from daytrader.signals import generate_signals
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.dispersion import compute_dispersion, classify_risk
from daytrader.active_risk import active_risk
from daytrader.entry_setup import choose_setup
from daytrader.patterns import complete_fifteen
from daytrader.policy import round_trip_cost, net_reward_risk, SHORT_MIN_NET_RR
from daytrader.trade_plan import size_example


def replay(ticker, df):
    trend=generate_signals(df);mean=generate_mean_reversion_signals(df)
    trades=[]; counts=Counter(); used_days=set()
    for i in range(780,len(df)-12):
        history=df.iloc[:i+1]
        risk=classify_risk(compute_dispersion(history))
        strategy,signal,raw,checks,patterns=choose_setup(history,risk,trend.iloc[:i+1],mean.iloc[:i+1])
        for c in signal.index:
            if c.startswith('check_') and not bool(signal[c]): counts[c]+=1
        if not raw:continue
        if patterns.get('bearish',{}).get('bearish'):counts['bearish_veto']+=1;continue
        counts['raw_setups']+=1
        now=df.index[i]+pd.Timedelta(minutes=5,seconds=5)
        # First executable following bar: optimistic open proxy, five-second delay
        # not resolvable in OHLCV. Costs remain charged; do not treat as fill proof.
        if df.index[i+1] != df.index[i]+pd.Timedelta(minutes=5):
            counts['missing_next_bar']+=1;continue
        plan=active_risk(complete_fifteen(history),ticker,float(signal.Close),now,pattern=patterns.get('bullish') if strategy.startswith('inverse-hs') else None)
        if plan['status']!='TRADE':counts['risk_or_time_blocked']+=1;continue
        entry=float(df.Open.iloc[i+1]);floor=plan['floor'];target=plan['ceiling']
        if strategy.startswith('inverse-hs') and entry>patterns['bullish']['max_entry']:
            counts['extended_pattern']+=1;continue
        if abs(entry-float(signal.Close)) > .25*(float(signal.Close)-floor):
            counts['entry_drift']+=1;continue
        if net_reward_risk(entry,floor,target,round_trip_cost(ticker)) < SHORT_MIN_NET_RR:
            counts['entry_cost_blocked']+=1;continue
        sizing=size_example(ticker,entry,floor,target)
        if not sizing['eligible']:counts['sizing_blocked']+=1;continue
        if now.date() in used_days:counts['daily_repeat']+=1;continue
        path=df.iloc[i+1:i+13]
        expected=pd.date_range(df.index[i+1],periods=12,freq='5min')
        if not path.index.equals(expected):counts['incomplete_path']+=1;continue
        exit_price=float(path.Close.iloc[-1]);reason='time';exit_at=path.index[-1]+pd.Timedelta(minutes=5)
        for stamp, bar in path.iterrows():
            if bar.Open <= floor: exit_price=float(bar.Open);reason='gap_stop';exit_at=stamp;break
            if bar.Open >= target: exit_price=target;reason='gap_target';exit_at=stamp;break
            if bar.Low <= floor: exit_price=floor;reason='stop';exit_at=stamp;break
            if bar.High >= target: exit_price=target;reason='target';exit_at=stamp;break
        used_days.add(now.date())
        net=(exit_price-entry-entry*round_trip_cost(ticker)/100)*sizing['shares']
        trades.append({'ticker':ticker,'signal':str(df.index[i]),'entry_time':str(path.index[0]),
                       'strategy':strategy,'entry':entry,'floor':floor,'target':target,
                       'exit':exit_price,'exit_time':str(exit_at),'reason':reason,'shares':sizing['shares'],'net_cad':net})
    values=[t['net_cad'] for t in trades]
    return trades,dict(counts, trades=len(trades),net_cad=sum(values),
                       win_rate=sum(x>0 for x in values)/len(values) if values else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tickers',nargs='*',default=['RY.TO','TD.TO','SHOP.TO','CNQ.TO','VFV.TO'])
    parser.add_argument('--data-dir',help='Use previously saved TICKER.csv files; no network')
    parser.add_argument('--out',default='research-output')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    summaries={};trades=[]
    for ticker in args.tickers:
        try:
            if args.data_dir:
                d=pd.read_csv(Path(args.data_dir)/(ticker+'.csv'),index_col=0)
                d.index=pd.to_datetime(d.index,utc=True).tz_convert('America/New_York')
            else:d=fetch_intraday(ticker,'60d','5m')
            d=completed_bars(d,'5m')
            d.to_csv(out/(ticker+'.csv'))
            rows,summary=replay(ticker,d);trades+=rows;summaries[ticker]=summary
            print(ticker,summary)
        except Exception as e:
            summaries[ticker]={'error':str(e)};print(ticker,'ERROR',e)
    pd.DataFrame(trades).to_csv(out/'trades.csv',index=False)
    report={'limits':'Diagnostic only. No historical earnings, portfolio capital, live spreads, delay or out-of-sample claim. Do not sum as portfolio returns.', 'tickers':summaries}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved raw histories and replay to',out)

if __name__=='__main__':main()
