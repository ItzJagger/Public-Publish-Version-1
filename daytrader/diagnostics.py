"""Explain quiet scans without treating missing data as an absence of setups."""
from .eligibility import permitted


def scan_diagnostics(results):
    valid = [r for r in results if not r.get('error')]
    raw = [r for r in valid if r.get('raw_buy_signal', r.get('rec', {}).get('raw_status', r.get('rec', {}).get('status')) == 'BUY_NOW')]
    risk_failed = [r for r in valid if r.get('no_trade')]
    risk_blocked = [r for r in raw if r.get('no_trade')]
    other_blocked = [r for r in raw if not r.get('no_trade') and not permitted(r, 'buy-now')]
    return {'total':len(results), 'errors':len(results)-len(valid), 'raw_setups':len(raw),
            'no_setup':len(valid)-len(raw), 'risk_rejected':len(risk_failed),
            'setups_risk_blocked':len(risk_blocked), 'setups_other_blocked':len(other_blocked),
            'eligible':sum(permitted(r,'buy-now') for r in valid),
            'long_setups':sum(bool((r.get('position') or {}).get('candidate')) for r in valid),
            'long_eligible':sum(bool(r.get('position')) and permitted(r,'position') for r in valid)}


def diagnostic_lines(results):
    d = scan_diagnostics(results)
    lines = [f"Scan coverage: {d['total']-d['errors']}/{d['total']} tickers evaluated; {d['errors']} errors",
             f"Hourly: {d['raw_setups']} raw setups; {d['no_setup']} without setup; "
             f"{d['setups_risk_blocked']} setups blocked by risk; "
             f"{d['setups_other_blocked']} by other gates; {d['eligible']} eligible before repeat-alert limits",
             f"Risk model rejected {d['risk_rejected']} tickers in total (with or without an entry signal)",
             f"Long hold: {d['long_setups']} technical setups; {d['long_eligible']} eligible (when requested)"]
    for r in [r for r in results if r.get('error')][:3]:
        lines.append(f"Data error {r['ticker']}: {r['error']}")
    for r in [r for r in results if r.get('no_trade')][:3]:
        lines.append(f"Risk rejection {r['ticker']}: {r.get('atr_stop', {}).get('reason','unknown')}")
    from collections import Counter
    failed = Counter(name for r in results for name, passed in r.get('entry_checks',{}).items() if not passed)
    if failed:
        lines.append('Entry conditions failed (overlap): '+', '.join(f'{k}={v}' for k,v in failed.most_common()))
    for r in results:
        gate = r.get('eligibility',{}).get('buy-now',{})
        if r.get('raw_buy_signal'):
            risk = r.get('atr_stop') or {}
            lines.append(f"Actual setup {r['ticker']} [{r.get('strategy')}] candle={r.get('signal_as_of')} "
                         f"entry={r.get('last_price')} floor={risk.get('floor')} target={risk.get('ceiling')} "
                         f"netRR={gate.get('net_rr',risk.get('net_rr'))} sell_by={gate.get('sell_by')} sizing={gate.get('sizing')} "
                         f"reasons={'; '.join(gate.get('reasons',[])) or 'ELIGIBLE'}")
        pos = r.get('position') or {}
        if pos.get('candidate'):
            g = r.get('eligibility',{}).get('position',{})
            lines.append(f"Long setup {r['ticker']}: {'; '.join(g.get('reasons',[])) or 'ELIGIBLE'}")
    for r in results:
        patterns=r.get('patterns') or {}
        bull=patterns.get('bullish',{})
        if bull.get('state','none')!='none':
            lines.append(f"Pattern {r['ticker']}: {bull['state']}; neckline={bull.get('neckline')}; known_at={bull.get('known_at')}")
        if patterns.get('bearish',{}).get('bearish'):
            lines.append(f"Bearish pattern {r['ticker']}: new hourly buys blocked")
        gate=r.get('eligibility',{}).get('buy-now',{})
        if r.get('raw_buy_signal'):
            lines.append(f"Timing {r['ticker']}: closed signal age={gate.get('signal_age_minutes')} min; scanned={r.get('scanned_at')}")
    return lines
