"""Conservative tendency screen; its uncertainty margin is not a win probability."""
import numpy as np
import pandas as pd


def tendency(values, cost, min_samples=30):
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    net = s - cost
    n = len(net)
    # A simple two-standard-error penalty, not a calibrated confidence interval:
    # serial dependence and screening many symbols can make it too optimistic.
    lower = float(net.mean() - 2*net.std(ddof=1)/np.sqrt(n)) if n > 1 else None
    split = n//2
    stable = n >= min_samples and net.iloc[:split].mean() > 0 and net.iloc[split:].mean() > 0
    return {'samples': n, 'net_mean_pct': float(net.mean()) if n else None,
            'conservative_net_pct': lower, 'stable_halves': bool(stable),
            'eligible': bool(stable and lower is not None and lower > 0 and net.median() > 0)}
