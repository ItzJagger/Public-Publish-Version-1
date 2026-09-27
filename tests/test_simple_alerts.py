import unittest
import math
from unittest.mock import patch
import pandas as pd
from daytrader.trade_plan import broker_levels, size_example
from daytrader.alerts import format_categorized_alert

NOW=pd.Timestamp('2026-09-22 11:00',tz='America/New_York')

class SimpleAlerts(unittest.TestCase):
    def test_cad_stop_equals_limit_below_entry(self):
        x=broker_levels(100.001,98.999,103.009)
        self.assertEqual(x,dict(entry=100.01,stop_price=98.99,limit_price=98.99,target_price=103.0))
        self.assertLess(x['stop_price'],x['entry'])
        self.assertLessEqual(x['stop_price'],98.999)
        self.assertLess(98.999-x['stop_price'],.01)
    def test_rejects_screenshot_inverted_levels(self):
        with self.assertRaises(ValueError):broker_levels(4.09,4.16,5.16)
    def test_rejects_nonfinite_and_collapsed_prices(self):
        for levels in [(100,math.nan,105),(100,99,math.inf),(.009,.008,.0099),(100,0,101)]:
            with self.subTest(levels=levels),self.assertRaises(ValueError):broker_levels(*levels)
    def test_short_five_fields_and_date(self):
        r={'ticker':'TEST.TO','rec':{'current_price':100},'atr_stop':{'floor':99,'ceiling':102}}
        title,txt=format_categorized_alert([r],[],NOW)
        block=txt.split('\n\n')[0]
        self.assertEqual(len(block.splitlines()),5)
        for value in ['Buy: TEST.TO','Buy limit: CAD $100.00 max','Amount: 4 whole shares',
                      'Stop $99.00 | Limit $99.00','Take profit: CAD $102.00','2026-09-22 12:00 ET','expiry: Day']:
            self.assertIn(value,block)
        self.assertIn('live bid must exceed stop',txt)
        self.assertIn('not linked orders',txt)
    def test_long_uses_wider_original_floor_and_expiry(self):
        r={'ticker':'TEST.TO','price':100,'floor':97,'ceiling':106,'hold_weeks':4}
        _,txt=format_categorized_alert([],[r],NOW)
        self.assertIn('Stop $97.00 | Limit $97.00',txt)
        self.assertIn('Good until cancelled',txt)
        self.assertIn('Amount: 1 whole shares',txt)
        self.assertIn('Take profit: CAD $106.00',txt)
    def test_stale_quantity_is_not_reused(self):
        r={'ticker':'TEST.TO','rec':{'current_price':100},'atr_stop':{'floor':99,'ceiling':102},
           'eligibility':{'buy-now':{'sizing':{'shares':999,'eligible':True,'budget':250,'risk_budget':2,'minimum_profit':1.01}}}}
        _,txt=format_categorized_alert([r],[],NOW)
        self.assertIn('Amount: 1 whole shares',txt);self.assertNotIn('999',txt)
    def test_rounded_sizing_rechecked_in_gate(self):
        from test_signal_upgrade import candidate, NOW as fixture_now
        from daytrader.eligibility import attach_eligibility
        r=candidate();r['atr_stop'].update(floor=98.999,ceiling=102.009)
        with patch('daytrader.eligibility.SIGNAL_PROFILE','active'):
            out=attach_eligibility(r,fixture_now)
        g=out['eligibility']['buy-now']
        self.assertEqual(g['order_plan']['stop_price'],98.99)
        self.assertLessEqual(g['sizing']['planned_stop_loss'],g['sizing']['risk_budget'])
    def test_multiple_stocks_each_have_a_full_plan(self):
        rows=[{'ticker':tk,'rec':{'current_price':100},'atr_stop':{'floor':99,'ceiling':102}} for tk in ('AAA.TO','BBB.TO')]
        _,txt=format_categorized_alert(rows,[],NOW)
        self.assertEqual(txt.count('Stop-limit SELL:'),2)
        self.assertEqual(txt.count('Sell by:'),2)

if __name__=='__main__':unittest.main()
