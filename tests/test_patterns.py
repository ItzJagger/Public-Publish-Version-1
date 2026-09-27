import unittest
import numpy as np
import pandas as pd
from daytrader.patterns import head_shoulders, complete_fifteen


def bottom():
    points={0:115,20:108,24:104,28:108,34:100,40:108,46:104.2,48:106.3,50:107.8,51:108.35}
    close=np.interp(np.arange(52),list(points),list(points.values()))
    return pd.DataFrame({'Open':close-.1,'High':close+.15,'Low':close-.15,'Close':close,'Volume':1_000_000},
                        index=pd.date_range('2026-09-24 09:30',periods=52,freq='5min',tz='America/New_York'))


class Patterns(unittest.TestCase):
    def test_watch_then_confirmed_breakout(self):
        d=bottom()
        watch=head_shoulders(d.iloc[:-1],True)['bullish']
        buy=head_shoulders(d,True)['bullish']
        self.assertEqual(watch['state'],'forming')
        self.assertFalse(watch['entry'])
        self.assertEqual(buy['state'],'breakout')
        self.assertTrue(buy['entry'])
        self.assertLess(pd.Timestamp(buy['known_at']),d.index[-1])
        self.assertGreater(buy['target'],d.Close.iloc[-1])
        self.assertEqual(watch['pattern_id'],buy['pattern_id'])

    def test_no_chasing_large_breakout(self):
        d=bottom();d.iloc[-1,d.columns.get_loc('Close')]=111;d.iloc[-1,d.columns.get_loc('High')]=111.2
        self.assertFalse(head_shoulders(d,True)['bullish']['entry'])

    def test_top_is_bearish_not_buy(self):
        d=bottom(); mirrored=d.copy()
        for col in ('Open','Close'):mirrored[col]=220-d[col]
        mirrored.High=220-d.Low;mirrored.Low=220-d.High
        p=head_shoulders(mirrored,True)
        self.assertTrue(p['bearish']['bearish'])
        self.assertFalse(p['bullish']['entry'])

    def test_retest_and_invalidation(self):
        d=bottom()
        extra=pd.DataFrame([[109,109.7,109.3,109.5,1e6],[108.2,108.5,108,108.35,1e6]],
               columns=d.columns,index=pd.date_range(d.index[-1]+pd.Timedelta(minutes=5),periods=2,freq='5min'))
        extra.iloc[0,0]=109.4
        p=head_shoulders(pd.concat([d,extra]),True)['bullish']
        self.assertEqual(p['state'],'retest')
        extra.iloc[0,extra.columns.get_loc('Low')]=99
        self.assertFalse(head_shoulders(pd.concat([d,extra]),True)['bullish']['entry'])

    def test_incomplete_or_missing_risk_bars_excluded(self):
        d=bottom().iloc[:8]
        self.assertEqual(len(complete_fifteen(d)),2)
        r=complete_fifteen(d.drop(d.index[1]))
        self.assertEqual(len(r),1)
        self.assertEqual(r.index[0],d.index[3])
        self.assertEqual(r.Volume.iloc[0],3e6)

    def test_unconfirmed_shoulder_cannot_signal(self):
        d=bottom().iloc[:48]
        self.assertFalse(head_shoulders(d,True)['bullish']['entry'])

    def test_no_pattern_in_monotonic_prices(self):
        d=bottom();x=np.arange(len(d))+100
        d.Open=x;d.Close=x;d.High=x+.1;d.Low=x-.1
        p=head_shoulders(d,True)
        self.assertFalse(p['bullish']['entry']);self.assertFalse(p['bearish']['bearish'])

if __name__=='__main__':unittest.main()
