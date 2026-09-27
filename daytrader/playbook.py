"""Turn a watchlist scan into the three answers the trader asked for:

1. buy_now_candidates   - signal fired on the latest bar; rise expected over the
                          next intraday hold.
2. weekday_risers       - names whose strongest historical weekday is today,
                          for a morning buy held through the session.
3. overnight_gappers    - names that have tended to gap up overnight, for a
                          late-day buy held to the next morning.

All three are weak, noisy edges off limited history - this module just ranks
and filters; the honest framing is printed by the caller.
"""


from .eligibility import permitted


def is_us_listed(ticker: str) -> bool:
    """US tickers carry the ~1.5%-each-way FX cost on a CAD Wealthsimple account;
    TSX (.TO) names don't.
    """
    return not ticker.upper().endswith(".TO")


def buy_now_candidates(results: list[dict]) -> list[dict]:
    """Names flagged BUY_NOW (a fresh signal on the latest bar)."""
    out = []
    for r in results:
        if not permitted(r, "buy-now"):
            continue
        rec = r.get("rec")
        if rec and rec.get("status") == "BUY_NOW":
            out.append(r)
    return out


def weekday_risers(results: list[dict], today_weekday: str) -> list[dict]:
    """Names whose strongest historical weekday == today, ranked
    timing-agreement first (intraday pattern rises through the day), then by
    the size of the weekday tilt.
    """
    rows = []
    for r in results:
        if not permitted(r, "weekday"):
            continue
        season = r.get("season", {})
        if season.get("insufficient_data") or season.get("best_day") != today_weekday:
            continue
        tilt = season.get("best_day_return_pct")
        if tilt is None or tilt <= 0:  # "strongest" day can still be a losing day - skip those
            continue
        buy_h = season.get("best_buy_hour")
        sell_h = season.get("best_sell_hour")
        # "Rises through the day" = tends to sit at its low early, high later.
        timing_agrees = buy_h is not None and sell_h is not None and sell_h > buy_h
        rows.append({
            "ticker": r["ticker"],
            "tilt_pct": season.get("best_day_return_pct"),
            "timing_agrees": timing_agrees,
            "best_buy_hour": buy_h,
            "best_sell_hour": sell_h,
            "is_us": is_us_listed(r["ticker"]),
        })

    rows.sort(key=lambda x: (not x["timing_agrees"], -(x["tilt_pct"] or 0.0)))
    return rows


def overnight_gappers(results: list[dict], min_samples: int = 20) -> list[dict]:
    """Names that have tended to gap UP overnight, ranked by hit_rate x mean gap.

    Requires an attached `overnight` stat block (see overnight.overnight_edge),
    a positive average gap, and a better-than-coin-flip hit rate.
    """
    rows = []
    for r in results:
        if not permitted(r, "overnight"):
            continue
        ov = r.get("overnight")
        if not ov or ov["samples"] < min_samples:
            continue
        if ov["mean_gap_pct"] <= 0 or ov["hit_rate"] <= 0.5:
            continue
        rows.append({
            "ticker": r["ticker"],
            "score": r["eligibility"]["overnight"]["evidence"]["conservative_net_pct"],
            "is_us": is_us_listed(r["ticker"]),
            **ov,
        })

    rows.sort(key=lambda x: -x["score"])
    return rows


def swing_picks(results: list[dict]) -> list[dict]:
    """Names sitting near a recent low and turning up, for a multi-day hold.

    Requires an attached `swing` block (see swing.swing_candidate) flagged as a
    candidate. Ranked most-oversold first, then closest to the recent low.
    """
    rows = []
    for r in results:
        if not permitted(r, "swing"):
            continue
        sw = r.get("swing")
        if not sw or not sw.get("candidate"):
            continue
        gate = r['eligibility']['swing']
        rows.append({**sw, 'signal_price': sw['price'], 'price': gate.get('entry_quote', sw['price']),
                     'net_rr': gate.get('net_rr', sw.get('net_rr')), "is_us": is_us_listed(sw["ticker"])})

    rows.sort(key=lambda x: (x["rsi"], x.get("dist_from_low_pct", 999.0)))
    return rows


def position_picks(results: list[dict]) -> list[dict]:
    """Long-term-uptrend pullbacks, for a weeks-long hold.

    Requires an attached `position` block (see position.position_candidate)
    flagged as a candidate. Ranked strongest trend first (highest 3-month
    momentum), then closest to the 50-day average (cleanest pullback entry).
    """
    rows = []
    for r in results:
        if not permitted(r, "position"):
            continue
        pos = r.get("position")
        if not pos or not pos.get("candidate"):
            continue
        gate = r['eligibility']['position']
        rows.append({**pos, 'signal_price': pos['price'], 'price': gate.get('entry_quote', pos['price']),
                     'net_rr': gate.get('net_rr', pos.get('net_rr')), 'sell_by': gate.get('sell_by'), 'sizing': gate.get('sizing'), 'order_plan': gate.get('order_plan'), "is_us": is_us_listed(pos["ticker"])})

    rows.sort(key=lambda x: (-(x.get("mom_3m_pct") or 0.0), abs(x.get("dist_sma50_pct", 999.0))))
    return rows
