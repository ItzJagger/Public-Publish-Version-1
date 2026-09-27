"""One active entry definition shared by live scans and research."""
from .signals import generate_signals
from .mean_reversion import generate_mean_reversion_signals
from .patterns import head_shoulders
from .policy import SIGNAL_PROFILE


def choose_setup(df, risk, trend=None, mean=None):
    strategy='meanrev' if risk=='LOW' else 'trend'
    signal=(mean if mean is not None else generate_mean_reversion_signals(df)) if risk=='LOW' else (trend if trend is not None else generate_signals(df))
    latest=signal.iloc[-1]
    raw=bool(latest['buy'])
    checks={name[6:]:bool(latest[name]) for name in signal.columns if name.startswith('check_')}
    patterns=head_shoulders(df,intraday=True) if SIGNAL_PROFILE=='active' else {}
    bull=patterns.get('bullish',{})
    if bull.get('entry'):
        threshold=.8 if bull['state']=='retest' else 1.1
        ready=(float(latest.get('relative_volume',0))>=threshold
               and bool(latest.get('check_liquidity',False)) and float(latest.Close)>float(latest.Open))
        patterns['bullish']['entry_confirmed']=bool(ready)
        if ready:
            raw=True;strategy='inverse-hs-'+bull['state']
            checks={'pattern':True,'volume':True,'liquidity':True,'bullish_bar':True}
    return strategy,latest,raw,checks,patterns
