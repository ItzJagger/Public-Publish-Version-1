import contextlib
import io
import unittest
from unittest.mock import patch
import pandas as pd
from test_patterns import bottom
from daytrader.entry_setup import choose_setup
from daytrader.position import position_from_frame
from daytrader.alerts import format_pattern_watch
import watch
import scan

class PatternIntegration(unittest.TestCase):
    def test_pattern_needs_volume_and_liquidity(self):
        d=bottom();signal=d.assign(buy=False,rsi=55,vwap=d.Close,relative_volume=1.2,check_liquidity=True)
        with patch('daytrader.entry_setup.SIGNAL_PROFILE','active'):
            self.assertEqual(choose_setup(d,'HIGH',trend=signal)[0],'inverse-hs-breakout')
            signal['relative_volume']=.9
            self.assertFalse(choose_setup(d,'HIGH',trend=signal)[2])
            signal['relative_volume']=1.2;signal['check_liquidity']=False
            self.assertFalse(choose_setup(d,'HIGH',trend=signal)[2])

    def test_daily_reversal_can_qualify_without_old_uptrend(self):
        d=bottom();prefix=pd.concat([d.iloc[[0]]]*180,ignore_index=True)
        d=pd.concat([prefix,d],ignore_index=True)
        d.index=pd.bdate_range('2025-01-01',periods=len(d))
        d.iloc[-1,d.columns.get_loc('Volume')]=1.3e6
        with patch('daytrader.position.SIGNAL_PROFILE','active'):
            result=position_from_frame('VFV.TO',d)
        self.assertTrue(result['candidate'])
        self.assertEqual(result['strategy'],'inverse-hs-breakout')
        self.assertLess(result['floor'],result['patterns']['bullish']['right_shoulder'])

    def test_watch_is_not_an_order(self):
        title,msg=format_pattern_watch({'ticker':'VFV.TO','mode':'buy-now','neckline':100})
        self.assertIn('WATCH ONLY',title)
        self.assertIn('Not a buy signal',msg)
        self.assertNotIn('Stop-limit SELL:',msg)

    def test_watch_does_not_consume_buy_cooldown(self):
        forming={'ticker':'TEST.TO','raw_buy_signal':False,'eligibility':{'buy-now':{'eligible':False}},
                 'pattern_watches':[{'ticker':'TEST.TO','mode':'buy-now','pattern_id':'x','neckline':100}]}
        buying={'ticker':'TEST.TO','raw_buy_signal':True,'rec':{'status':'BUY_NOW','current_price':100},
                'atr_stop':{'floor':99,'ceiling':102},'eligibility':{'buy-now':{'eligible':True}}}
        with patch.object(scan,'run_scan',side_effect=[{'results':[forming]},{'results':[forming]},{'results':[buying]}]), \
             patch.object(watch,'market_is_open',return_value=True), \
             patch.object(watch,'market_now',return_value=pd.Timestamp('2026-09-24 12:00',tz='America/New_York')), \
             patch.object(watch,'notification_config',return_value={'kind':'console'}), \
             patch.object(watch,'send_notification',return_value=True) as send, \
             patch.object(watch.time,'sleep',side_effect=[None,None,KeyboardInterrupt]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):watch.run_watch(alerts='short')
        self.assertEqual(send.call_count,2)
        self.assertIn('WATCH ONLY',send.call_args_list[0].args[0])
        self.assertIn('SHORT-TERM buy',send.call_args_list[1].args[0])

    def test_hourly_delivered_before_daily_fetch(self):
        events=[]
        r={'ticker':'TEST.TO','quality':{'is_stale':False},'raw_buy_signal':True,
           'rec':{'status':'BUY_NOW','current_price':100},'atr_stop':{'floor':99,'ceiling':102},
           'eligibility':{'buy-now':{'eligible':True}}}
        def send(*args):events.append('send');return True
        def daily(*args):events.append('daily');return None
        with patch.object(scan,'run_scan',return_value={'results':[r]}), \
             patch.object(watch,'attach_eligibility',side_effect=lambda r,n:r), \
             patch.object(watch,'market_is_open',return_value=True), \
             patch.object(watch,'market_now',return_value=pd.Timestamp('2026-09-24 12:00',tz='America/New_York')), \
             patch.object(watch,'notification_config',return_value={'kind':'console'}), \
             patch.object(watch,'send_notification',side_effect=send), \
             patch('daytrader.position.position_candidate',side_effect=daily), \
             contextlib.redirect_stdout(io.StringIO()):
            watch.run_watch(once=True)
        self.assertEqual(events,['send','daily'])

if __name__=='__main__':unittest.main()
