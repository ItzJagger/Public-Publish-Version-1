"""Compare profiles on fixed synthetic fixtures, not profitability evidence."""
import unittest
from unittest.mock import patch
import pandas as pd
from test_signal_upgrade import intraday, daily, candidate, NOW
from daytrader.policy import ENTRY_PROFILES, ENTRY_FILTERS
from daytrader.filters import liquid_daily
from daytrader.signals import generate_signals
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.eligibility import attach_eligibility, permitted


class BalancedProfile(unittest.TestCase):
    def test_share_threshold_relaxes_but_dollar_floor_remains(self):
        d=daily([100]*30);d['Volume']=600_000
        with patch.dict(ENTRY_FILTERS,ENTRY_PROFILES['strict']):
            self.assertFalse(liquid_daily(d))
        with patch.dict(ENTRY_FILTERS,ENTRY_PROFILES['balanced']):
            self.assertTrue(liquid_daily(d))
            d['Close']=5
            self.assertFalse(liquid_daily(d))

    def test_balanced_retains_strict_signals_and_adds_some(self):
        counts={'strict':0,'balanced':0}
        for seed in range(10):
            d=intraday(seed)
            for fn in (generate_signals,generate_mean_reversion_signals):
                with patch.dict(ENTRY_FILTERS,ENTRY_PROFILES['strict']):
                    strict=fn(d)['buy']
                with patch.dict(ENTRY_FILTERS,ENTRY_PROFILES['balanced']):
                    balanced=fn(d)['buy']
                self.assertFalse((strict & ~balanced).any())
                counts['strict']+=int(strict.sum())
                counts['balanced']+=int(balanced.sum())
        self.assertGreater(counts['balanced'],counts['strict'])

    def test_both_profiles_keep_future_invariance(self):
        d=intraday(1);cut=25*13
        for profile in ENTRY_PROFILES.values():
            with patch.dict(ENTRY_FILTERS,profile):
                for fn in (generate_signals,generate_mean_reversion_signals):
                    pd.testing.assert_series_equal(fn(d.iloc[:cut])['buy'],fn(d)['buy'].iloc[:cut])

    def test_both_profiles_keep_cost_and_staleness_blocks(self):
        for profile in ENTRY_PROFILES.values():
            with patch.dict(ENTRY_FILTERS,profile):
                r=candidate();r['quality']['is_stale']=True
                self.assertFalse(permitted(attach_eligibility(r,NOW),'buy-now'))
                r=candidate('AAPL')
                with patch('daytrader.eligibility.next_earnings',return_value=pd.Timestamp('2026-08-01')):
                    self.assertFalse(permitted(attach_eligibility(r,NOW),'buy-now'))


if __name__=='__main__':unittest.main()
