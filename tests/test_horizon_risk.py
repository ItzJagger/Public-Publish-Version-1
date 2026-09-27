"""Empirical risk regression: same horizon, independent levels, honest rejection."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from daytrader.risk_levels import horizon_risk_from_frame, compute_legacy_atr_stop, session_atr
from daytrader.diagnostics import scan_diagnostics
from daytrader.eligibility import attach_eligibility
from test_signal_upgrade import candidate, NOW

ASOF=pd.Timestamp('2026-06-23 11:01',tz='America/New_York')


def history(direction=1):
    rows=[];idx=[]
    for day in pd.bdate_range(end='2026-06-23',periods=31):
        count=6 if day.date()==ASOF.date() else 26
        for j in range(count):
            prior=max(0,min(j-6,4))*.3*direction
            after=max(0,min(j-5,4))*.3*direction
            op=100+prior;close=100+after
            rows.append([op,max(op,close)+.15,min(op,close)-.15,close,100000])
            idx.append(day+pd.Timedelta(minutes=570+15*j))
    return pd.DataFrame(rows,index=pd.DatetimeIndex(idx).tz_localize('America/New_York'),
                        columns=['Open','High','Low','Close','Volume'])


class HorizonRisk(unittest.TestCase):
    def test_old_rejection_pattern_new_feasible_levels(self):
        df=history()
        with patch('daytrader.risk_levels.fetch_intraday',return_value=df):
            old=compute_legacy_atr_stop('TEST.TO',100,11,12,hold_bars=4)
        self.assertEqual(old['status'],'NO-TRADE')
        self.assertIn('target move',old['reason'])
        new=horizon_risk_from_frame(df,100,4,ASOF,.1)
        self.assertEqual(new['status'],'TRADE',new)
        self.assertEqual(new['samples'],29)  # first session has no ATR warmup
        self.assertGreaterEqual(new['net_rr'],1.5)
        self.assertGreaterEqual(new['floor_distance'],1.25*new['single_bar_atr'])

    def test_adverse_paths_are_included_and_rejected(self):
        out=horizon_risk_from_frame(history(-1),100,4,ASOF,.1)
        self.assertEqual(out['samples'],29)
        self.assertEqual(out['status'],'NO-TRADE')
        self.assertGreater(out['floor_distance'],out['reward_distance'])

    def test_costs_do_not_manipulate_stop_or_target(self):
        df=history()
        low=horizon_risk_from_frame(df,100,4,ASOF,.1)
        high=horizon_risk_from_frame(df,100,4,ASOF,3.1)
        self.assertEqual(low['floor'],high['floor'])
        self.assertEqual(low['ceiling'],high['ceiling'])
        self.assertEqual(high['status'],'NO-TRADE')

    def test_future_prices_do_not_rewrite_risk(self):
        df=history();before=horizon_risk_from_frame(df,100,4,ASOF,.1)
        df.loc[ASOF.normalize()+pd.Timedelta(hours=12)]=[200,201,199,200,100000]
        after=horizon_risk_from_frame(df,100,4,ASOF,.1)
        self.assertEqual(before,after)

    def test_one_window_per_day_not_overlapping_samples(self):
        out=horizon_risk_from_frame(history(),100,4,ASOF,.1)
        self.assertEqual(len(set(out['sample_dates'])),out['samples'])
        self.assertTrue(all(pd.Timestamp(d).date()<ASOF.date() for d in out['sample_dates']))

    def test_missing_bars_cannot_shorten_horizon(self):
        df=history();missing=df.index[(df.index.hour==11)&(df.index.minute==15)][:11]
        out=horizon_risk_from_frame(df.drop(missing),100,4,ASOF,.1)
        self.assertEqual(out['samples'],19)
        self.assertEqual(out['status'],'NO-TRADE')

    def test_horizon_changes_the_measured_target(self):
        df=history()
        hour=horizon_risk_from_frame(df,100,4,ASOF,.1)
        quarter=horizon_risk_from_frame(df,100,1,ASOF,.1)
        self.assertGreater(hour['reward_distance'],quarter['reward_distance'])

    def test_stale_risk_feed_rejected(self):
        out=horizon_risk_from_frame(history().iloc[:-3],100,4,ASOF,.1)
        self.assertEqual(out['status'],'NO-TRADE')
        self.assertIn('stale',out['reason'])

    def test_session_boundaries_and_invalid_values(self):
        for bars,price in [(0,100),(27,100),(4,-1),(4,float('nan'))]:
            self.assertEqual(horizon_risk_from_frame(history(),price,bars,ASOF,.1)['status'],'NO-TRADE')
        out=horizon_risk_from_frame(history(),100,4,ASOF,.1,entry_minute=931)
        self.assertEqual(out['status'],'NO-TRADE')

    def test_overnight_jump_excluded_from_intraday_atr(self):
        df=history();changed=df.copy();mask=changed.index.date==ASOF.date()
        changed.loc[mask,['Open','High','Low','Close']]+=50
        self.assertAlmostEqual(session_atr(df).iloc[-1],session_atr(changed).iloc[-1])

    def test_diagnostics_distinguish_signal_and_risk(self):
        r=candidate();r.update(raw_buy_signal=True,no_trade=True,atr_stop={'reason':'insufficient sample'})
        out=attach_eligibility(r,NOW)
        d=scan_diagnostics([out,{'ticker':'BAD','error':'feed failed'},
                            {'ticker':'QUIET','raw_buy_signal':False,'no_trade':True}])
        self.assertEqual(d['raw_setups'],1)
        self.assertEqual(d['setups_risk_blocked'],1)
        self.assertEqual(d['no_setup'],1)
        self.assertEqual(d['errors'],1)
        self.assertEqual(d['eligible'],0)



class RiskPipeline(unittest.TestCase):
    def test_raw_signal_survives_risk_rejection_and_valid_risk_can_alert(self):
        import daytrader.watchlist as wl
        import daytrader.eligibility as eg
        from daytrader.quality import check_quality
        from daytrader.playbook import buy_now_candidates
        for direction in [1,-1]:
            df=history(direction)
            scan_frame=df.resample('30min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            signals=scan_frame.assign(buy=True,sell=False,rsi=55.,vwap=scan_frame.Close)
            with patch.object(wl,'fetch_intraday',return_value=scan_frame),\
                 patch('daytrader.risk_levels.fetch_intraday',return_value=df),\
                 patch.object(wl,'market_now',return_value=ASOF),\
                 patch.object(eg,'market_now',return_value=ASOF),\
                 patch.object(wl,'check_quality',side_effect=lambda d:check_quality(d,now=ASOF)),\
                 patch('daytrader.entry_setup.generate_signals',return_value=signals),\
                 patch('daytrader.entry_setup.generate_mean_reversion_signals',return_value=signals):
                result=wl.scan_ticker('VFV.TO',hold_hours=1)
            self.assertTrue(result['raw_buy_signal'])
            self.assertEqual(result['no_trade'], direction<0)
            self.assertEqual(len(buy_now_candidates([result])),1 if direction>0 else 0)
            self.assertEqual(result['atr_stop']['buy_hour'],11)
            self.assertEqual(result['atr_stop']['buy_minute'],1)


if __name__=='__main__':unittest.main()
