"""Regression for long-hold false rejections at :26/:57 on 30-minute bars."""
import unittest
from unittest.mock import patch
import pandas as pd
from test_signal_upgrade import candidate
from daytrader.eligibility import attach_eligibility, permitted


def long_candidate(ticker='VFV.TO'):
    r=candidate(ticker)
    r['signal_as_of']='2026-06-23 11:00:00-04:00'
    r['rec']['status']='FLAT'
    r['raw_buy_signal']=False
    r['position']={'ticker':ticker,'candidate':True,'price':100.,'floor':95.,'ceiling':110.,
                   'atr':2.,'hold_weeks':5,'as_of':'2026-06-22','net_rr':1.9,'confirmation':{}}
    return r


class TimeframeFreshness(unittest.TestCase):
    def test_long_hold_does_not_expire_between_half_hour_candles(self):
        now=pd.Timestamp('2026-06-23 11:57',tz='America/New_York')
        r=attach_eligibility(long_candidate(),now)
        self.assertTrue(permitted(r,'position'))
        self.assertFalse(any('candle' in reason or 'stale' in reason
                             for reason in r['eligibility']['position']['reasons']))

    def test_old_hourly_entry_still_blocked_without_false_feed_diagnosis(self):
        r=candidate();r['signal_as_of']='2026-06-23 11:00:00-04:00'
        out=attach_eligibility(r,pd.Timestamp('2026-06-23 11:57',tz='America/New_York'))
        reasons=out['eligibility']['buy-now']['reasons']
        self.assertIn('entry signal is older than the 20-minute action window',reasons)
        self.assertNotIn('completed signal feed is missing a newer candle',reasons)
        self.assertFalse(permitted(out,'buy-now'))

    def test_missed_new_candle_is_detected(self):
        r=candidate();r['signal_as_of']='2026-06-23 11:00:00-04:00'
        out=attach_eligibility(r,pd.Timestamp('2026-06-23 12:01',tz='America/New_York'))
        self.assertIn('completed signal feed is missing a newer candle',out['eligibility']['buy-now']['reasons'])

    def test_daily_earnings_and_quote_freshness_are_preserved(self):
        now=pd.Timestamp('2026-06-23 11:57',tz='America/New_York')
        with patch('daytrader.eligibility.next_earnings',return_value=None):
            out=attach_eligibility(long_candidate('TD.TO'),now)
        self.assertFalse(permitted(out,'position'))
        self.assertIn('earnings date unknown',out['eligibility']['position']['reasons'])
        r=long_candidate();r['quality']['is_stale']=True
        self.assertFalse(permitted(attach_eligibility(r,now),'position'))

    def test_daily_confirmation_date_still_required(self):
        r=long_candidate();r['position']['as_of']='2026-06-19'
        out=attach_eligibility(r,pd.Timestamp('2026-06-23 11:57',tz='America/New_York'))
        self.assertFalse(permitted(out,'position'))
        self.assertIn('daily confirmation is not from prior weekday session',out['eligibility']['position']['reasons'])


if __name__=='__main__':unittest.main()
