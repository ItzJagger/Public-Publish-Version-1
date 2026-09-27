import contextlib
import io
import unittest
from unittest.mock import patch
import pandas as pd
import watch
import scan


class AlertDelivery(unittest.TestCase):
    def run_cycles(self, deliveries):
        # Three same-day scans; stop after third sleep. No real transport/sleep.
        short={'ticker':'TEST.TO','raw_buy_signal':True,'rec':{'status':'BUY_NOW','current_price':100},
               'atr_stop':{'floor':99,'ceiling':102},
               'eligibility':{'buy-now':{'eligible':True}}}
        with patch.object(scan,'run_scan',return_value={'results':[short]}),\
             patch.object(watch,'market_is_open',return_value=True),\
             patch.object(watch,'market_now',return_value=pd.Timestamp('2026-06-23 11:00',tz='America/New_York')),\
             patch.object(watch,'notification_config',return_value={'kind':'console','target':None}),\
             patch.object(watch,'send_notification',side_effect=deliveries) as send,\
             patch.object(watch.time,'sleep',side_effect=[None,None,KeyboardInterrupt]),\
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(KeyboardInterrupt):
                watch.run_watch(alerts='short')
        return send.call_count,output.getvalue()

    def test_failed_send_can_retry_and_success_deduplicates(self):
        count,output=self.run_cycles([False,True])
        self.assertEqual(count,2)
        self.assertIn('cooldown not recorded',output)
        self.assertEqual(output.count('ALERTED:'),1)

    def test_each_stock_delivered_and_deduplicated_independently(self):
        rows=[{'ticker':tk,'raw_buy_signal':True,'rec':{'status':'BUY_NOW','current_price':100},
               'atr_stop':{'floor':99,'ceiling':102},'eligibility':{'buy-now':{'eligible':True}}}
              for tk in ('AAA.TO','BBB.TO')]
        with patch.object(scan,'run_scan',return_value={'results':rows}), \
             patch.object(watch,'market_is_open',return_value=True), \
             patch.object(watch,'market_now',return_value=pd.Timestamp('2026-06-23 11:00',tz='America/New_York')), \
             patch.object(watch,'notification_config',return_value={'kind':'console','target':None}), \
             patch.object(watch,'send_notification',side_effect=[False,True,True]) as send, \
             patch.object(watch.time,'sleep',side_effect=[None,None,KeyboardInterrupt]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):watch.run_watch(alerts='short')
        self.assertEqual(send.call_count,3)
        self.assertIn('AAA.TO',send.call_args_list[0].args[0])
        self.assertIn('BBB.TO',send.call_args_list[1].args[0])
        self.assertIn('AAA.TO',send.call_args_list[2].args[0])
        for call in send.call_args_list:
            self.assertEqual(call.args[1].count('Stop-limit SELL:'),1)

    def test_success_suppresses_duplicate_same_day(self):
        count,_=self.run_cycles([True])
        self.assertEqual(count,1)


if __name__=='__main__':unittest.main()
