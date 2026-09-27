import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from daytrader.data import completed_bars
from daytrader.signals import generate_signals
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.filters import relative_volume
from daytrader.policy import round_trip_cost, net_reward_risk, ENTRY_FILTERS
from daytrader.eligibility import attach_eligibility, permitted
from daytrader.evidence import tendency
from daytrader.playbook import buy_now_candidates, position_picks
from daytrader.swing import swing_from_frame
from daytrader.position import position_from_frame
import daytrader.earnings as earnings
from compare_signals import compare


def intraday(seed=0):
    idx = pd.DatetimeIndex([day+pd.Timedelta(minutes=570+30*b)
                           for day in pd.bdate_range('2026-01-05', periods=35) for b in range(13)])
    rng = np.random.default_rng(seed)
    c = 100+np.cumsum(rng.normal(0,.12,len(idx)))
    op = np.r_[c[0],c[:-1]]
    return pd.DataFrame(dict(Open=op,Close=c,High=np.maximum(op,c)+.08,
                             Low=np.minimum(op,c)-.08,Volume=rng.integers(1_000_000,3_000_000,len(idx))),index=idx)


def daily(closes):
    c = np.asarray(closes);op=c-.1
    return pd.DataFrame(dict(Open=op,Close=c,High=c+.15,Low=op-.15,Volume=2_000_000),
                        index=pd.bdate_range('2025-01-02',periods=len(c)))


def candidate(ticker='VFV.TO'):
    return {'ticker':ticker, 'quality':{'stale_minutes':1,'is_stale':False,'verify_move':False},
            'liquidity_confirmed':True,'signal_as_of':'2026-06-23 10:30:00-04:00','interval':'30m',
            'last_price':100.0,'rec':{'status':'BUY_NOW','current_price':100,'floor_price':99,
                                     'ceiling_price':102},
            'atr_stop':{'floor':99,'ceiling':102,'hold_bars':4}, 'no_trade':False}

NOW = pd.Timestamp('2026-06-23 11:01',tz='America/New_York')


class SignalRules(unittest.TestCase):
    def test_both_strategies_can_produce_entries(self):
        d=intraday()
        self.assertGreater(generate_signals(d)['buy'].sum(),0)
        self.assertGreater(generate_mean_reversion_signals(d)['buy'].sum(),0)

    def test_future_changes_do_not_rewrite_entries(self):
        d=intraday();cut=25*13
        for fn in (generate_signals,generate_mean_reversion_signals):
            before=fn(d.iloc[:cut])['buy']
            changed=d.copy();changed.iloc[cut:,changed.columns.get_loc('Close')]*=2
            pd.testing.assert_series_equal(before,fn(changed)['buy'].iloc[:cut])

    def test_mean_reversion_requires_recovery(self):
        out=generate_mean_reversion_signals(intraday())
        self.assertFalse(out.loc[out.Close < out.lower_band,'buy'].any())
        for i in np.flatnonzero(out.buy.to_numpy()):
            self.assertLess(out.Close.iloc[i-1],out.lower_band.iloc[i-1])
            self.assertGreaterEqual(out.Close.iloc[i],out.lower_band.iloc[i])
            self.assertEqual(out.index[i].date(),out.index[i-1].date())

    def test_thin_volume_blocks_both(self):
        d=intraday();d['Volume']=10
        self.assertFalse(generate_signals(d).buy.any())
        self.assertFalse(generate_mean_reversion_signals(d).buy.any())

    def test_relative_volume_excludes_current_bar(self):
        d=intraday();d['Volume']=100;d.iloc[-1,d.columns.get_loc('Volume')]=10000
        self.assertEqual(relative_volume(d).iloc[-1],100)

    def test_trend_confirmations_at_each_entry(self):
        out=generate_signals(intraday(1))
        picked=out.loc[out.buy]
        self.assertGreater(len(picked),0)
        self.assertTrue((picked.relative_volume>=ENTRY_FILTERS['trend_rvol']).all())
        self.assertTrue(picked.rsi.between(45,ENTRY_FILTERS['trend_rsi_max']).all())
        self.assertTrue(picked.not_extended.all())
        self.assertTrue(picked.trend_confirmed.all())

    def test_swing_rejects_bounce_in_downtrend(self):
        d=daily(list(np.linspace(120,100,79))+[100.6])
        r=swing_from_frame('VFV.TO',d)
        self.assertFalse(r['candidate'])
        self.assertFalse(r['confirmation']['trend_confirmed'])

    def test_long_hold_requires_full_trend_history(self):
        self.assertIsNone(position_from_frame('VFV.TO',daily(np.linspace(100,120,150))))

    def test_long_hold_positive_pullback_and_cost_rejection(self):
        rng=np.random.default_rng(3);n=252
        c=100+.25*np.arange(n)+rng.normal(0,.4,n)
        for k in range(12): c[n-12+k]-=9*np.sin(np.pi*k/11)
        c[-2]-=6.3;c[-1]=c[-2]+2.25
        d=daily(c)
        good=position_from_frame('VFV.TO',d,cost=.1)
        self.assertTrue(good['candidate'])
        self.assertFalse(position_from_frame('VFV.TO',d,cost=10)['candidate'])

    def test_tendency_rejects_cost_drag_and_recent_reversal(self):
        self.assertFalse(tendency([.2]*40,3.1)['eligible'])
        self.assertFalse(tendency([2.]*20+[-.2]*20,0)['eligible'])
        self.assertFalse(tendency([1.]*4,0)['eligible'])
        self.assertTrue(tendency([.3,.5]*20,.1)['eligible'])

    def test_comparison_uses_holdout_and_both_versions(self):
        result=compare(intraday(),'VFV.TO')
        self.assertEqual(set(result.version),{'original','revised'})
        self.assertEqual(len(result),4)
        self.assertTrue((result.holdout_from>'2026-01-05').all())


class CompletionAndGates(unittest.TestCase):
    def test_intraday_excludes_unfinished_bar(self):
        d=intraday().iloc[:3]
        now=pd.Timestamp('2026-01-05 10:15',tz='America/New_York')
        self.assertEqual(len(completed_bars(d,'30m',now)),1)

    def test_daily_excludes_today_even_after_close(self):
        d=daily([100,101,102])
        now=pd.Timestamp(d.index[-1]).tz_localize('America/New_York')+pd.Timedelta(hours=18)
        self.assertEqual(len(completed_bars(d,'1d',now)),2)

    def test_invalid_and_duplicate_data_are_rejected(self):
        d=intraday().iloc[:3].copy();d.iloc[-1,d.columns.get_loc('Close')]=np.nan
        with self.assertRaises(ValueError):completed_bars(d,'30m',NOW)
        d=intraday().iloc[:3];d.index=pd.DatetimeIndex([d.index[0]]*3)
        with self.assertRaises(ValueError):completed_bars(d,'30m',NOW)

    def test_valid_etf_signal_reaches_selector(self):
        r=attach_eligibility(candidate(),NOW)
        self.assertTrue(permitted(r,'buy-now'))
        self.assertEqual(len(buy_now_candidates([r])),1)

    def test_missing_gate_fails_closed(self):
        self.assertFalse(buy_now_candidates([candidate()]))
        self.assertFalse(position_picks([{'position':{'candidate':True}}]))

    def test_stale_price_blocks_but_keeps_holdings_quote(self):
        r=candidate();r['quality']['is_stale']=True
        out=attach_eligibility(r,NOW)
        self.assertFalse(buy_now_candidates([out]))
        self.assertEqual(out['rec']['status'],'BLOCKED')
        self.assertEqual(out['last_price'],100)

    def test_unknown_or_in_window_earnings_block(self):
        for date in (None,pd.Timestamp('2026-06-23')):
            with patch('daytrader.eligibility.next_earnings',return_value=date):
                self.assertFalse(permitted(attach_eligibility(candidate('RY.TO'),NOW),'buy-now'))
        with patch('daytrader.eligibility.next_earnings',return_value=pd.Timestamp('2026-08-01')):
            self.assertTrue(permitted(attach_eligibility(candidate('RY.TO'),NOW),'buy-now'))

    def test_us_cost_blocks_small_target(self):
        with patch('daytrader.eligibility.next_earnings',return_value=pd.Timestamp('2026-08-01')):
            out=attach_eligibility(candidate('AAPL'),NOW)
        self.assertIn('reward/risk below 1.5 after costs',out['eligibility']['buy-now']['reasons'])

    def test_late_session_blocks_new_hour_hold(self):
        r=candidate();r['signal_as_of']='2026-06-23 15:00:00-04:00'
        out=attach_eligibility(r,pd.Timestamp('2026-06-23 15:31',tz='America/New_York'))
        self.assertIn('insufficient session time for planned hold',out['eligibility']['buy-now']['reasons'])

    def test_unknown_liquidity_blocks(self):
        r=candidate();r.pop('liquidity_confirmed')
        self.assertFalse(permitted(attach_eligibility(r,NOW),'buy-now'))

    def test_cached_failed_earnings_refresh(self):
        with patch.dict(earnings._cache,{},clear=True),patch.dict(earnings._cached_at,{},clear=True),\
             patch.object(earnings,'_fetch_next_earnings',side_effect=[None,pd.Timestamp('2026-08-01')]) as fetch,\
             patch.object(earnings.time,'monotonic',side_effect=[0,10,3601,3601]):
            self.assertIsNone(earnings.next_earnings('TEST'))
            self.assertIsNone(earnings.next_earnings('TEST'))
            self.assertIsNotNone(earnings.next_earnings('TEST'))
            self.assertEqual(fetch.call_count,2)

    def test_net_rr_and_invalid_cost(self):
        self.assertGreater(net_reward_risk(100,99,102,.1),1.5)
        self.assertLess(net_reward_risk(100,99,102,3.1),0)
        self.assertEqual(net_reward_risk(100,-1,102,.1),0)
        with self.assertRaises(ValueError):round_trip_cost('AAPL',fx=-1)




class SharedPipeline(unittest.TestCase):
    def test_long_pick_and_gap_chasing_block(self):
        r=candidate()
        r['position']={'ticker':'VFV.TO','candidate':True,'price':100.,'floor':95.,'ceiling':110.,
                       'atr':2.,'hold_weeks':5,'as_of':'2026-06-22','net_rr':1.9,
                       'confirmation':{},'mom_3m_pct':5.,'dist_sma50_pct':0.}
        good=attach_eligibility(copy.deepcopy(r),NOW)
        self.assertEqual(len(position_picks([good])),1)
        r['last_price']=104
        bad=attach_eligibility(r,NOW)
        self.assertFalse(position_picks([bad]))
        self.assertIn('price moved too far from daily confirmation',bad['eligibility']['position']['reasons'])

    def test_in_window_earnings_blocks_long_pick(self):
        r=candidate('RY.TO')
        r['position']={'ticker':'RY.TO','candidate':True,'price':100.,'floor':95.,'ceiling':110.,
                       'atr':2.,'hold_weeks':5,'as_of':'2026-06-22','net_rr':1.9,'confirmation':{}}
        with patch('daytrader.eligibility.next_earnings',return_value=pd.Timestamp('2026-07-01')):
            out=attach_eligibility(r,NOW)
        self.assertFalse(position_picks([out]))
        self.assertIn('earnings inside intended hold window or cached date expired',out['eligibility']['position']['reasons'])

    def test_fresh_intraday_price_must_still_fit_levels(self):
        r=candidate();r['last_price']=100.8
        out=attach_eligibility(r,NOW)
        self.assertFalse(buy_now_candidates([out]))
        self.assertIn('price moved too far from confirmed entry',out['eligibility']['buy-now']['reasons'])

    def test_scan_attaches_same_gate_for_display_and_alerts(self):
        import daytrader.watchlist as wl
        import daytrader.eligibility as eg
        from daytrader.alerts import new_buy_signals
        d=intraday().iloc[:25*13+4]
        now=pd.Timestamp(d.index[-1]).tz_localize('America/New_York')+pd.Timedelta(minutes=31)
        price=float(d.Close.iloc[-1])
        risk={'status':'TRADE','floor':price-1,'ceiling':price+2,'floor_pct':100/price,
              'ceiling_pct':200/price,'floor_distance':1.,'reward_distance':2.,'rr':2.}
        rec={'status':'BUY_NOW','current_price':price,'entry_price':price,'floor_price':price-1,
             'ceiling_price':price+2,'unrealized_pct':0.,'as_of':d.index[-1]}
        with patch.object(wl,'fetch_intraday',return_value=d),\
             patch.object(wl,'market_now',return_value=now),\
             patch.object(eg,'market_now',return_value=now),\
             patch.object(wl,'check_quality',return_value={'is_stale':False,'verify_move':False,'stale_minutes':31}),\
             patch.object(wl,'compute_atr_stop',return_value=risk),\
             patch('daytrader.entry_setup.generate_signals',return_value=d.assign(buy=True,sell=False,rsi=55.,vwap=d.Close)),\
             patch('daytrader.entry_setup.generate_mean_reversion_signals',return_value=d.assign(buy=True,sell=False,rsi=55.,vwap=d.Close)):
            r=wl.scan_ticker('VFV.TO',interval='30m',hold_hours=.25)
        self.assertEqual(len(new_buy_signals(buy_now_candidates([r]),set())),1)
        self.assertEqual(r['last_price'],price)
        self.assertTrue(r['eligibility']['buy-now']['eligible'])


class SwingPositive(unittest.TestCase):
    def test_uptrend_pullback_can_qualify(self):
        closes=list(np.linspace(100,130,90))+list(np.linspace(129.8,127,6))+[127.5]
        out=swing_from_frame('VFV.TO',daily(closes))
        self.assertTrue(out['candidate'])
        self.assertGreaterEqual(out['net_rr'],1.5)
        self.assertFalse(swing_from_frame('VFV.TO',daily(closes),cost=5)['candidate'])


if __name__=='__main__':unittest.main()
