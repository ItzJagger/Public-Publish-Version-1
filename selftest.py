"""Offline self-test. Run: python selftest.py"""
"""Offline self-test: runs the whole pipeline on synthetic data with no
network and asserts the logic is coherent. Run from the project root:
    python selftest.py
"""
"""Offline self-test: runs the whole pipeline on synthetic data with no
network and asserts the logic is coherent. Run from the project root:
    python selftest.py
"""
"""Offline self-test: runs the whole pipeline on synthetic data with no
network and asserts the logic is coherent. Run from the project root:
    python selftest.py
"""
"""Offline test harness: feed synthetic OHLCV into the pipeline (no network)."""
import sys
import numpy as np
import pandas as pd

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(7)

# Trading-day calendar (weekdays only) ending "today".
def trading_days(n):
    days, d = [], pd.Timestamp("2026-06-19")
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.normalize())
        d -= pd.Timedelta(days=1)
    return list(reversed(days))


def intraday_shape(hour_frac, profile):
    """Return a price offset by time-of-day, controlling where low/high sit."""
    # hour_frac in [9.5, 16.0]
    t = (hour_frac - 9.5) / (16.0 - 9.5)  # 0..1 across the session
    if profile == "am_low_pm_high":   # bottoms ~late morning, tops ~mid afternoon
        return -np.cos(np.pi * t) * 0.6           # rises through the day
    if profile == "pm_low_am_high":   # tops early, bottoms late -> overnight pattern
        return np.cos(np.pi * t) * 0.6            # falls through the day
    return 0.0


def make_intraday(ticker, period, interval, base, vol, profile, daily_drift=0.0, gap_bias=0.0):
    n_days = int(period.replace("d", "")) if period.endswith("d") else 60
    if period.endswith("mo"):
        n_days = int(period.replace("mo", "")) * 21
    if period.endswith("y"):
        n_days = int(period.replace("y", "")) * 252
    step = {"15m": 15, "30m": 30, "1d": None}[interval]
    days = trading_days(n_days)

    rows, idx = [], []
    level = base
    for di, day in enumerate(days):
        level += daily_drift + rng.normal(0, base * 0.004)
        if interval == "1d":
            # daily bar with an overnight gap baked into Open
            gap = rng.normal(0, base * 0.003) + gap_bias * base
            op = level + gap
            cl = level + rng.normal(0, base * 0.004)
            hi = max(op, cl) + abs(rng.normal(0, base * vol * 0.5))
            lo = min(op, cl) - abs(rng.normal(0, base * vol * 0.5))
            rows.append((op, hi, lo, cl, int(rng.integers(1e6, 5e6))))
            idx.append(day + pd.Timedelta(hours=16))
            continue

        minutes = range(int(9.5 * 60), int(16 * 60), step)
        for m in minutes:
            hf = m / 60.0
            mid = level + intraday_shape(hf, profile) * base * vol * 2.0 + rng.normal(0, base * vol * 0.35)
            op = mid + rng.normal(0, base * vol * 0.3)
            cl = mid + rng.normal(0, base * vol * 0.3)
            hi = max(op, cl) + abs(rng.normal(0, base * vol * 0.2))
            lo = min(op, cl) - abs(rng.normal(0, base * vol * 0.2))
            rows.append((op, hi, lo, cl, int(rng.integers(1e5, 9e5))))
            ts = day + pd.Timedelta(minutes=m)
            idx.append(ts)

    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex(idx, name="datetime").tz_localize("America/New_York")
    return df


# Per-ticker synthetic profiles.
PROFILES = {
    "PG":  dict(base=150.0, vol=0.0030, profile="am_low_pm_high"),   # calm, intraday-friendly
    "KO":  dict(base=79.0,  vol=0.0028, profile="pm_low_am_high"),  # calm, overnight raw pattern
    "JNJ": dict(base=228.0, vol=0.0032, profile="am_low_pm_high"),
    "NVDA": dict(base=120.0, vol=0.012, profile="am_low_pm_high", daily_drift=0.05),  # volatile -> trend
    "GAPR": dict(base=90.0, vol=0.008, profile="am_low_pm_high", gap_bias=0.0025),    # tends to gap up overnight
}


def fake_fetch_intraday(ticker, period="5d", interval="5m"):
    if interval == "1d" and ticker in ("UPTR", "DNTR"):
        return build_daily_trend(ticker, period)
    p = PROFILES.get(ticker, PROFILES["PG"])
    return make_intraday(ticker, period, interval, p["base"], p["vol"], p["profile"],
                         p.get("daily_drift", 0.0), p.get("gap_bias", 0.0))


def build_daily_trend(ticker, period):
    """Deterministic daily series for the long-hold screen.
    UPTR: long uptrend that has pulled back to its 50-day and ticked up (a
    position candidate). DNTR: a downtrend (price below its 200-day -> not one).
    """
    n = int(period.replace("y", "")) * 252 if period.endswith("y") else 260
    n = max(n, 230)
    rng = np.random.default_rng(3 if ticker == "UPTR" else 11)
    base, slope = 100.0, (0.25 if ticker == "UPTR" else -0.20)
    days = trading_days(n)
    closes = np.array([base + slope * i + rng.normal(0, base * 0.004) for i in range(n)])
    if ticker == "UPTR":
        dip_depth, dip_len = 9.0, 12
        for k in range(dip_len):
            frac = k / (dip_len - 1)
            closes[n - dip_len + k] -= dip_depth * np.sin(np.pi * frac)
        closes[-2] -= dip_depth * 0.7        # trough near the end
        closes[-1] = closes[-2] + dip_depth * 0.25   # ... then tick up
    else:
        closes[-1] = closes[-2] + 0.1        # a tick up, but trend is still down
    rows, idx = [], []
    for i, day in enumerate(days):
        cl = closes[i]
        op = cl - abs(rng.normal(0, base * 0.003))
        hi = max(op, cl) + abs(rng.normal(0, base * 0.004))
        lo = min(op, cl) - abs(rng.normal(0, base * 0.004))
        rows.append((op, hi, lo, cl, 1_000_000))
        idx.append(day + pd.Timedelta(hours=16))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex(idx, name="datetime").tz_localize("America/New_York")
    return df


# Monkeypatch every module that imported fetch_intraday by name.
import daytrader.data as data_mod
data_mod.fetch_intraday = fake_fetch_intraday
import daytrader.risk_levels as rl
rl.fetch_intraday = fake_fetch_intraday
import daytrader.watchlist as wl
wl.fetch_intraday = fake_fetch_intraday
import daytrader.overnight as ov
ov.fetch_intraday = fake_fetch_intraday
import daytrader.swing as sw_mod
sw_mod.fetch_intraday = fake_fetch_intraday
import daytrader.position as pos_mod
pos_mod.fetch_intraday = fake_fetch_intraday

# Deterministic earnings dates (no network): AMZN inside a July window,
# CM.TO clear in late August, GAPR unknown.
FAKE_EARNINGS = {
    "AMZN": pd.Timestamp("2026-07-30"),
    "CM.TO": pd.Timestamp("2026-08-27"),
    "GOOGL": pd.Timestamp("2026-07-22"),
    "UPTR": pd.Timestamp("2026-12-01"),
    "PG": pd.Timestamp("2026-07-21"),
}
import daytrader.earnings as earn_mod
earn_mod._fetch_next_earnings = lambda t: FAKE_EARNINGS.get(t.upper())
earn_mod._cache.clear()

if __name__=="__main__":
    print("ok")


import math



"""Offline assertion tests (no network). Run: python3 tests.py"""



from daytrader import indicators
from daytrader.seasonality import plan_intraday_hold, SESSION_END_HOUR, BARS_PER_HOUR
from daytrader.risk_levels import compute_legacy_atr_stop as compute_atr_stop
from daytrader.watchlist import _reanchor, scan_ticker, scan_watchlist
from daytrader.mean_reversion import generate_mean_reversion_signals
from daytrader.recommend import current_recommendation

passed = 0
def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok: {name}")


print("indicators (Wilder's RSI / ATR):")
rising = pd.Series(np.linspace(10, 20, 60))
falling = pd.Series(np.linspace(20, 10, 60))
flat = pd.Series([15.0] * 60)
check("RSI rising -> high (>80)", indicators.rsi(rising).iloc[-1] > 80)
check("RSI falling -> low (<20)", indicators.rsi(falling).iloc[-1] < 20)
check("RSI flat -> 50", abs(indicators.rsi(flat).iloc[-1] - 50.0) < 1e-6)
check("RSI bounded 0..100", indicators.rsi(rising).dropna().between(0, 100).all())

ohlc = pd.DataFrame({
    "High": rising + 0.5, "Low": rising - 0.5, "Close": rising,
    "Open": rising, "Volume": [1000] * 60,
})
atr_series = indicators.atr(ohlc)
check("ATR positive", atr_series.dropna().gt(0).all())
check("ATR warmup is NaN", bool(np.isnan(atr_series.iloc[0])))


print("\nplan_intraday_hold (always a bounded intraday window):")
for season in [
    {"insufficient_data": True, "num_days": 3},
    {"insufficient_data": False, "best_buy_hour": 9, "best_sell_hour": 15},   # raw long
    {"insufficient_data": False, "best_buy_hour": 15, "best_sell_hour": 9},   # raw overnight
    {"insufficient_data": False, "best_buy_hour": 10, "best_sell_hour": 11},  # raw short
]:
    for hh in [0.25, 0.5, 1.0, 2.0, 5.0]:
        p = plan_intraday_hold(season, hh)
        entry_min = p["buy_hour"] * 60
        exit_min = p["sell_hour"] * 60 + p["sell_minute"]
        check(f"intraday buy<sell within session (season={season.get('best_buy_hour','est')}, hh={hh})",
              9 * 60 <= entry_min < exit_min <= SESSION_END_HOUR * 60)
        check(f"classify hour > buy hour (hh={hh})", p["classify_sell_hour"] > p["buy_hour"])
        check(f"hold_bars >=1 and respects cap (hh={hh})",
              1 <= p["hold_bars"] <= max(1, int(round(hh * BARS_PER_HOUR))))

# raw overnight is surfaced as info, but the plan is still intraday
p = plan_intraday_hold({"insufficient_data": False, "best_buy_hour": 15, "best_sell_hour": 9}, 1.0)
check("raw_overnight flagged when raw sell <= raw buy", p["raw_overnight"] is True)


print("\nlegacy risk-model regression (new model tested in tests/test_horizon_risk.py):")
# Short hold on a calm name should be reachable as a TRADE; sweep down.
got_trade = None
for hh_bars in [1, 2, 3, 4]:
    a = compute_atr_stop("PG", entry_price=150.0, buy_hour=10, sell_hour=11, hold_bars=hh_bars)
    if a["status"] == "TRADE":
        got_trade = a
        print(f"  (TRADE first reachable at hold_bars={hh_bars})")
        break
check("a TRADE is reachable at a short hold", got_trade is not None)
check("floor < entry < ceiling", got_trade["floor"] < 150.0 < got_trade["ceiling"])
check("reward ~= 2x risk", math.isclose(
    (got_trade["ceiling"] - 150.0) / (150.0 - got_trade["floor"]), 2.0, rel_tol=1e-6))
check("floor_pct/ceiling_pct present and consistent",
      math.isclose(got_trade["floor_pct"], (150.0 - got_trade["floor"]) / 150.0 * 100, rel_tol=1e-6))
check("carries dollar distances", "floor_distance" in got_trade and "reward_distance" in got_trade)

# A long hold on the same calm name should be rejected (not forced).
a_long = compute_atr_stop("PG", entry_price=150.0, buy_hour=10, sell_hour=15, hold_bars=20)
check("long hold on a calm name -> NO-TRADE", a_long["status"] != "TRADE")


print("\n_reanchor (single source of truth for IN_POSITION levels):")
a = dict(got_trade)
old_rr = (a["ceiling"] - 150.0) / (150.0 - a["floor"])
_reanchor(a, entry_price=148.0)
check("floor = entry - floor_distance after reanchor",
      math.isclose(a["floor"], 148.0 - got_trade["floor_distance"], rel_tol=1e-9))
check("ceiling = entry + reward_distance after reanchor",
      math.isclose(a["ceiling"], 148.0 + got_trade["reward_distance"], rel_tol=1e-9))
new_rr = (a["ceiling"] - 148.0) / (148.0 - a["floor"])
check("R:R preserved by reanchor", math.isclose(old_rr, new_rr, rel_tol=1e-9))
check("anchored_to flag set", a.get("anchored_to") == "entry")


print("\nrecommend BUY_NOW path (mean-reversion):")
# Two sessions so the 20-bar bands are defined; a monotonic ramp guarantees no
# premature signal, then a clear dip on the final ("now") bar of a mid-session
# day -> a fresh BUY_NOW (and the final bar is not an overnight boundary).
day1 = pd.date_range("2026-06-18 09:30", "2026-06-18 15:45", freq="15min", tz="America/New_York")
day2 = pd.date_range("2026-06-19 09:30", "2026-06-19 11:00", freq="15min", tz="America/New_York")
idx = day1.append(day2)
close = np.linspace(99.0, 101.0, len(idx) - 1).tolist() + [97.5]
close = np.array(close)
df = pd.DataFrame({
    "Open": close, "High": close + 0.1, "Low": close - 0.1,
    "Close": close, "Volume": [10000] * len(idx),
}, index=idx)
df.index.name = "datetime"
rec = current_recommendation(df, take_profit_pct=1.0, stop_loss_pct=0.5,
                             signal_fn=generate_mean_reversion_signals)
check("unconfirmed falling bar does not produce BUY_NOW", rec["status"] == "FLAT")
check("flat recommendation preserves the latest price", rec['current_price'] == close[-1])
check("flat recommendation preserves the bar timestamp", rec['as_of'] == idx[-1])


def eligible_fixture(r):
    """Selector/rendering unit fixture only. Real scans must pass live gates."""
    return {**r, 'eligibility': {m: {'eligible': True, 'evidence': {'conservative_net_pct': .2}}
             for m in ('buy-now','weekday','overnight','swing','position')}}


print("\nfull scan_ticker integration (no crashes, sane shapes):")
_saved_scan_clock = wl.market_now
wl.market_now = lambda: pd.Timestamp("2026-06-18 12:00", tz="America/New_York")
for tk in ["PG", "KO", "JNJ", "NVDA"]:
    r = scan_ticker(tk, interval="30m", period="60d", hold_hours=1.0)
    check(f"{tk}: has strategy+risk", "strategy" in r and "risk" in r)
    if r.get("atr_stop") and r["atr_stop"].get("classification"):
        check(f"{tk}: planned hold is INTRADAY", r["atr_stop"]["classification"] == "INTRADAY")


wl.market_now = _saved_scan_clock
print("\nclock (session phases, ET):")
from daytrader import clock
TZ = "America/New_York"
checks_phase = [("08:00", "pre-market"), ("09:45", "morning"), ("12:30", "midday"),
                ("15:30", "late-day"), ("16:30", "after-hours")]
for hhmm, exp in checks_phase:
    ts = pd.Timestamp(f"2026-06-23 {hhmm}", tz=TZ)  # a Tuesday
    check(f"phase at {hhmm} == {exp}", clock.session_phase(ts) == exp)
check("weekend detected", clock.session_phase(pd.Timestamp("2026-06-20 12:00", tz=TZ)) == "weekend")
check("morning -> morning view", clock.recommended_extra_views("morning") == {"morning"})
check("late-day -> overnight view", clock.recommended_extra_views("late-day") == {"overnight"})
check("midday -> no extra view", clock.recommended_extra_views("midday") == set())
check("minutes_since is positive for an old bar",
      clock.minutes_since(pd.Timestamp("2026-06-23 09:00", tz=TZ),
                          pd.Timestamp("2026-06-23 09:30", tz=TZ)) == 30)


print("\novernight edge + playbook selectors:")
from daytrader.overnight import overnight_edge
from daytrader.playbook import buy_now_candidates, overnight_gappers, weekday_risers, is_us_listed
ov = overnight_edge("GAPR")
check("overnight_edge returns stats for GAPR", ov is not None and ov["samples"] >= 20)
check("GAPR has a positive mean overnight gap", ov["mean_gap_pct"] > 0)
check("GAPR hit-rate is a probability", 0.0 <= ov["hit_rate"] <= 1.0)
check("is_us_listed: .TO is not US", is_us_listed("RY.TO") is False)
check("is_us_listed: AAPL is US", is_us_listed("AAPL") is True)

res = scan_watchlist(["PG", "KO", "NVDA", "GAPR"], interval="30m", period="60d",
                     hold_hours=1.0, include_overnight=True)
gappers = overnight_gappers(res)
check("stale/unknown-earnings scans produce no overnight picks", not gappers)
check("overnight_gappers sorted by score desc",
      all(gappers[i]["score"] >= gappers[i + 1]["score"] for i in range(len(gappers) - 1)))
# weekday riser must have a POSITIVE tilt (a losing 'best day' must be excluded)
fake = [{"ticker": "NEG", "season": {"insufficient_data": False, "best_day": "Tuesday",
         "best_day_return_pct": -0.5, "best_buy_hour": 10, "best_sell_hour": 14}},
        {"ticker": "POS", "season": {"insufficient_data": False, "best_day": "Tuesday",
         "best_day_return_pct": 0.5, "best_buy_hour": 10, "best_sell_hour": 14}}]
risers = weekday_risers([eligible_fixture(r) for r in fake], "Tuesday")
check("weekday_risers excludes a negative-tilt best day",
      [x["ticker"] for x in risers] == ["POS"])
check("weekday_risers marks timing agreement (low before high)", risers[0]["timing_agrees"] is True)
check("buy_now_candidates returns only BUY_NOW", all(
    c["rec"]["status"] == "BUY_NOW" for c in buy_now_candidates(res)))

print("\nswing screen (daily-timeframe dip finder):")
import daytrader.swing as swingmod
from daytrader.playbook import swing_picks

def _daily_frame(closes):
    idx = pd.bdate_range("2026-01-02", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": c.shift(1).fillna(c.iloc[0]),
        "High": pd.concat([c, c.shift(1).fillna(c.iloc[0])], axis=1).max(axis=1) + 0.4,
        "Low": pd.concat([c, c.shift(1).fillna(c.iloc[0])], axis=1).min(axis=1) - 0.4,
        "Close": c,
        "Volume": 1_000_000,
    }, index=idx)

_orig_fetch = swingmod.fetch_intraday
try:
    # A name that fell to a low then ticked up on the last bar -> candidate.
    dip = list(np.linspace(120, 100, 59)) + [100.6]
    swingmod.fetch_intraday = lambda t, period=None, interval=None: _daily_frame(dip)
    cand = swingmod.swing_candidate("DIPPER")
    check("swing: bounce in sustained downtrend is rejected", cand is not None and cand["candidate"] is False)
    check("swing: floor below price", cand["floor"] < cand["price"])
    check("swing: ceiling above price", cand["ceiling"] > cand["price"])
    check("swing: near the 20-day low", cand["dist_from_low_pct"] <= 3.0)
    check("swing: gives an exit-by date", bool(cand["exit_by"]))

    # Same decline but still falling on the last bar -> not a candidate.
    fall = list(np.linspace(120, 100, 59)) + [99.4]
    swingmod.fetch_intraday = lambda t, period=None, interval=None: _daily_frame(fall)
    falling = swingmod.swing_candidate("FALLER")
    check("swing: still-falling is NOT a candidate", falling is not None and falling["candidate"] is False)
finally:
    swingmod.fetch_intraday = _orig_fetch

# Integration: include_swing attaches a swing block, and the section builds.
res_sw = scan_watchlist(["PG", "KO", "NVDA"], interval="30m", period="60d",
                        hold_hours=1.0, include_swing=True)
check("swing block attached when include_swing=True", all("swing" in r for r in res_sw))
check("swing_picks returns a list", isinstance(swing_picks(res_sw), list))
import scan as _scan2
_sw_join = "\n".join(_scan2.swing_lines(res_sw))
check("swing section has a header", "4) SWING" in _sw_join)

print("\nprompt asks the AI to research online:")
from daytrader.ai_judge import build_batch_prompt as _bbp
_ptxt = _bbp(scan_watchlist(["PG"], interval="30m", period="60d", hold_hours=1.0))
check("prompt includes web-search instruction", "Web-search" in _ptxt)
check("prompt mentions earnings/gap risk", "earnings" in _ptxt.lower())

print("\nprompt includes the overnight section (regression for the copy bug):")
import scan as _scan
_sections = _scan.build_sections([eligible_fixture(r) for r in res], {"overnight"}, "Tuesday")
_joined = "\n\n".join("\n".join(s) for s in _sections)
check("build_sections includes a BUY NOW block", "1) BUY NOW" in _joined)
check("build_sections includes the OVERNIGHT block when requested", "3) OVERNIGHT" in _joined)
check("overnight block names the gapper", "GAPR" in _joined)
_no_overnight = "\n\n".join("\n".join(s) for s in _scan.build_sections(res, set(), "Tuesday"))
check("overnight block omitted when not requested", "3) OVERNIGHT" not in _no_overnight)

print("\nearnings awareness:")
from daytrader.earnings import earnings_note, next_earnings
check("next_earnings returns the faked date", str(next_earnings("AMZN").date()) == "2026-07-30")
check("unknown ticker -> None", next_earnings("KO") is None)
check("INSIDE window flagged", "INSIDE hold" in earnings_note("AMZN", "2026-08-06"))
check("clear of window flagged", "clear of window" in earnings_note("CM.TO", "2026-08-13"))
check("unknown date says UNKNOWN", "UNKNOWN" in earnings_note("KO", "2026-08-13"))
# ETFs: no single-company earnings gap (avoids a misleading UNKNOWN flag)
from daytrader.earnings import is_etf
check("known ETF recognised", is_etf("VFV.TO") and is_etf("SPY"))
check("non-ETF not flagged as ETF", not is_etf("AAPL"))
check("ETF earnings note has no gap risk", "no single-company earnings" in earnings_note("XEQT.TO", "2026-09-01"))

print("\nholdings file:")
from daytrader.holdings import attach_current, load_holdings
_hp = "/tmp/_test_holdings.txt"
open(_hp, "w").write(
    "# comment line\n"
    "AMZN 242.67 2026-07-06 long 218.03 291.95\n"
    "CM.TO 150.50 2026-07-09 long\n"
    "BAD_LINE_ONLY_TWO 1.0\n"
    "XX 10 2026-01-01 nonsense\n"
)
_holds = load_holdings(_hp)
check("parses exactly the two valid rows", [h["ticker"] for h in _holds] == ["AMZN", "CM.TO"])
check("optional floor/ceiling parsed", _holds[0]["floor"] == 218.03 and _holds[0]["ceiling"] == 291.95)
check("missing floor/ceiling -> None", _holds[1]["floor"] is None and _holds[1]["ceiling"] is None)
check("missing file -> empty list", load_holdings("/tmp/_does_not_exist.txt") == [])
_fake_results = [{"ticker": "AMZN", "rec": {"current_price": 246.0}}, {"ticker": "CM.TO", "last_price": 151.76}]
_att = attach_current(_holds, _fake_results)
check("attach_current computes pnl", abs(_att[0]["pnl_pct"] - (246.0/242.67-1)*100) < 1e-6)
check("attach_current keeps entry fields", _att[1]["entry"] == 150.50 and _att[1]["current"] == 151.76)

print("\ndata-quality guard:")
from daytrader.quality import check_quality, quality_warnings
_idx = pd.date_range("2026-07-09 10:00", periods=20, freq="30min", tz="America/New_York")
_qdf = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000}, index=_idx)
_fresh_now = _idx[-1] + pd.Timedelta(minutes=5)
_q_ok = check_quality(_qdf, now=_fresh_now)
check("fresh normal data -> no flags", not _q_ok["is_stale"] and not _q_ok["verify_move"])
_q_stale = check_quality(_qdf, now=_idx[-1] + pd.Timedelta(minutes=120))
check("old last bar -> stale flag", _q_stale["is_stale"] and "old" in _q_stale["note"])
_qdf2 = _qdf.copy()
_idx2 = list(_qdf2.index[:-1]) + [_qdf2.index[-1] + pd.Timedelta(days=1)]  # last bar on next day
_qdf2.index = pd.DatetimeIndex(_idx2)
_qdf2.iloc[-1, _qdf2.columns.get_loc("Close")] = 110.0                      # +10% vs prior day close
_q_move = check_quality(_qdf2, now=_qdf2.index[-1] + pd.Timedelta(minutes=5))
check("+10% print -> VERIFY flag", _q_move["verify_move"] and "VERIFY" in _q_move["note"])
_qw = quality_warnings([{"ticker": "WMT", "quality": _q_move}, {"ticker": "PG", "quality": _q_ok}])
check("quality_warnings lists only flagged names", len(_qw) == 1 and "WMT" in _qw[0])
_qw_closed = quality_warnings([{"ticker": "PG", "quality": _q_stale}], include_stale=False)
check("stale flag suppressed when market closed", _qw_closed == [])

print("\nholdings + summary in the prompt:")
import scan as _scan2
_hl = "\n".join(_scan2.holdings_lines(_att))
check("section 0 renders holdings", "YOUR OPEN POSITIONS" in _hl and "AMZN" in _hl and "CM.TO" in _hl)
check("holdings show earnings note", "earnings" in _hl)
check("US holding flags FX on exit", "US (FX on exit)" in _hl)
_sl = "\n".join(_scan2.summary_lines(_fake_results, _att))
check("summary names held positions", "AMZN (long)" in _sl and "CM.TO (long)" in _sl)
check("summary counts buy signals", "Fresh BUY signals" in _sl)
from daytrader.ai_judge import build_batch_prompt
_bp = build_batch_prompt([], holdings=_att)
check("prompt tells AI replays are not user positions", "NOT a position the user holds" in _bp)
check("prompt demands exit calls on held names", "HOLD or SELL call" in _bp)
check("no holdings -> no holdings instructions", "NOT a position the user holds" not in build_batch_prompt([]))
# compactness: one dense line per ticker, and a bounded overall size
from daytrader.ai_judge import _ticker_line
_line_nt = _ticker_line("KO", {"strategy": "meanrev", "no_trade": True, "current_price": 90.0, "atr_stop": {}})
check("NO-TRADE renders as a single short line", "\n" not in _line_nt and "NO-TRADE @ 90.00" in _line_nt)
_compact_results = [
    {"ticker": "PG", "strategy": "meanrev", "no_trade": True, "current_price": 149.0, "atr_stop": {}},
    {"ticker": "NVDA", "strategy": "trend", "rec": {"status": "FLAT", "current_price": 122.0, "rsi": 54.0}},
]
_batch_only = build_batch_prompt(_compact_results)
_lines_per_ticker = sum(1 for ln in _batch_only.splitlines() if ln.startswith("PG:"))
check("each ticker appears on exactly one line", _lines_per_ticker == 1)
check("compact prompt stays lean", len(_batch_only.split()) < 220)
# swing + long-hold lines carry an earnings clause
from daytrader.position import position_candidate
_up2 = pos_mod.position_from_frame("UPTR.TO", build_daily_trend("UPTR", "1y"))
check("long-hold line shows earnings note", "earnings" in "\n".join(_scan2.position_lines([eligible_fixture({"position": _up2})])))
_res_hold = _scan2.run_scan(view="intraday", tickers=["PG"], holdings_path=_hp)
check("run_scan returns holdings", [h["ticker"] for h in _res_hold["holdings"]] == ["AMZN", "CM.TO"])
check("held names get scanned even if not in watchlist",
      any(r["ticker"] == "AMZN" for r in _res_hold["results"]))
_pt = _scan2.build_prompt_text(_res_hold["results"], _res_hold["now"], _res_hold["extra_views"], holdings=_res_hold["holdings"])
check("prompt starts with decision summary", _pt.index("DECISION SUMMARY") < _pt.index("BUY NOW"))
check("prompt includes section 0 before section 1", _pt.index("YOUR OPEN POSITIONS") < _pt.index("BUY NOW"))
check("every scan result carries a quality dict", all("quality" in r for r in _res_hold["results"] if "error" not in r))

print("\nlong-hold (position) screen:")
from daytrader.position import position_candidate
from daytrader.playbook import position_picks
_up = pos_mod.position_from_frame("UPTR.TO", build_daily_trend("UPTR", "1y"))
check("UPTR returns a position block", _up is not None)
check("UPTR long trend is up (price > 200-day)", _up["uptrend"])
check("UPTR is pulled back to the 50-day", _up["pulled_back"])
check("UPTR RSI in the moderate band", _up["not_overbought"] and _up["not_crashing"])
check("UPTR has ticked up", _up["turning_up"])
check("UPTR is a candidate", _up["candidate"] is True)
check("UPTR stop below price, target above", _up["floor"] < _up["price"] < _up["ceiling"])
check("UPTR review date is weeks out", _up["hold_weeks"] >= 4)
_dn = position_candidate("DNTR")
check("DNTR returns a block", _dn is not None)
check("DNTR long trend is NOT up", _dn["uptrend"] is False)
check("DNTR is not a candidate", _dn["candidate"] is False)
# playbook picks: only candidates, and they carry an is_us flag
_picks = position_picks([
    eligible_fixture({"position": _up}), eligible_fixture({"position": _dn}), {"position": None}, {},
])
check("position_picks returns only the uptrend candidate", [p["ticker"] for p in _picks] == ["UPTR.TO"])
check("position_picks tags CAD FX flag", _picks[0]["is_us"] is False)
# a .TO candidate should be marked FX-free
_up_ca = dict(_up, ticker="RY.TO")
check("position_picks marks .TO as FX-free", position_picks([eligible_fixture({"position": _up_ca})])[0]["is_us"] is False)
# section renders
import scan as _scan
_sec = "\n".join(_scan.position_lines([eligible_fixture({"position": _up})]))
check("LONG HOLD section renders the pick", "LONG HOLD" in _sec and "UPTR" in _sec)
check("LONG HOLD section shows a review date", "review by" in _sec)
# scan_ticker attaches a position block when asked
_wl_pos = scan_ticker("PG", interval="30m", period="60d", include_position=True)
check("scan_ticker attaches position when included", "position" in _wl_pos)
check("scan_ticker omits position by default", scan_ticker("PG", interval="30m", period="60d").get("position") is None)
# 'all' and 'position' views resolve to include the position section
check("view 'all' includes position", "position" in _scan.resolve_views("all", "midday"))
check("view 'position' includes position", "position" in _scan.resolve_views("position", "midday"))

print("\nbacktest report (walk-forward, costs):")
import backtest_report as br
check("cost: US name carries FX", br.cost_pct("AAPL", 3.0, 0.1) == 3.1)
check("cost: .TO name has no FX", abs(br.cost_pct("TD.TO", 3.0, 0.1) - 0.1) < 1e-9)
_rep = br.run_report(["PG", "KO", "NVDA", "TD.TO"], daily_period="1y", intraday_period="60d")
check("report has all four modes", all(k in _rep for k in ["buy-now", "swing", "overnight", "weekday"]))
check("report has a buy&hold benchmark", "_buy_hold" in _rep)
_modes_with_trades = [k for k in ["buy-now", "swing", "overnight", "weekday"] if _rep.get(k)]
check("all modes report either statistics or no qualifying trades", all(v is None or isinstance(v, dict) for k,v in _rep.items() if not k.startswith("_")))
for _k in _modes_with_trades:
    check(f"{_k}: summary has avg + win + n", all(f in _rep[_k] for f in ("avg", "win", "n", "max_dd")))
# higher FX cost must lower the average net return (cost actually subtracts)
_low = br.run_report(["NVDA"], daily_period="1y", intraday_period="60d", fx=0.0, slip=0.0)
_high = br.run_report(["NVDA"], daily_period="1y", intraday_period="60d", fx=10.0, slip=0.0)
if _low.get("overnight") and _high.get("overnight"):
    check("higher FX cost lowers net avg", _high["overnight"]["avg"] < _low["overnight"]["avg"])
check("format_report renders a table", "BACKTEST REPORT" in br.format_report(
    _rep, ["PG", "KO"], "1y", "60d", 3.0, 0.1))

print("\nholdings hardening + tracking:")
from daytrader.holdings import price_status, validate_holdings
_hp2 = "/tmp/_test_holdings2.txt"
open(_hp2, "w").write(
    "# header\n"
    "AMZN 242.67 2026-07-06 long 218.03 291.95\n"     # clean
    "CM.TO 150.50 2026-07-09 long 170.00 140.00\n"    # floor above entry AND ceiling below entry AND floor>ceiling
    "TSLA 100 notadate swing\n"                        # bad date -> skipped
    "MSFT 400 2026-07-01 nonsense\n"                   # bad mode -> skipped
    "ONLYTWO 5\n"                                       # too few fields -> skipped
    "AMZN 250 2026-07-10 long\n"                       # duplicate ticker -> skipped
)
_pos, _iss = validate_holdings(_hp2)
check("validate keeps only the valid rows", [p["ticker"] for p in _pos] == ["AMZN", "CM.TO"])
check("bad date reported", any("not YYYY-MM-DD" in m for m in _iss))
check("bad mode reported", any("mode" in m and "must be one of" in m for m in _iss))
check("too-few-fields reported", any("at least TICKER" in m for m in _iss))
check("duplicate ticker reported", any("more than once" in m for m in _iss))
check("floor-above-entry warned", any("at/above entry" in m for m in _iss))
check("ceiling-below-entry warned", any("at/below entry" in m for m in _iss))
# price_status flags
check("ceiling hit -> take profit", "AT/ABOVE CEILING - take profit" in
      price_status({"current": 300.0, "floor": 218.0, "ceiling": 291.95}))
check("floor breached -> stop", "AT/BELOW FLOOR - stop hit, exit" in
      price_status({"current": 210.0, "floor": 218.0, "ceiling": 291.95}))
check("near floor -> watch", any("near floor" in f for f in
      price_status({"current": 219.0, "floor": 218.0, "ceiling": 291.95})))
check("mid-range -> no flag", price_status({"current": 255.0, "floor": 218.0, "ceiling": 291.95}) == [])
check("no quote -> no flag", price_status({"current": None, "floor": 1, "ceiling": 2}) == [])
# horizon flags via scan._horizon_flag
import scan as _scan3
_today = pd.Timestamp(_scan3.market_now()).normalize()
_old_intraday = {"ticker": "X", "mode": "intraday", "date": str((_today - pd.Timedelta(days=3)).date())}
check("stale day-trade flagged", "STALE DAY-TRADE" in _scan3._horizon_flag(_old_intraday, _scan3.market_now()))
_old_swing = {"ticker": "Y", "mode": "swing", "date": str((_today - pd.Timedelta(days=30)).date())}
check("past-horizon swing flagged", "PAST EXIT HORIZON" in _scan3._horizon_flag(_old_swing, _scan3.market_now()))
_fresh_long = {"ticker": "Z", "mode": "long", "date": str(_today.date())}
check("fresh long -> no horizon flag", _scan3._horizon_flag(_fresh_long, _scan3.market_now()) == "")
# missing-quote handling in attach_current
_held = [{"ticker": "NOSCAN", "entry": 100.0, "date": "2026-07-01", "mode": "long", "floor": 90.0, "ceiling": 120.0}]
_att2 = attach_current(_held, [{"ticker": "OTHER", "rec": {"current_price": 5.0}}])
check("missing quote flagged", _att2[0]["quote_missing"] is True and _att2[0]["current"] is None)
_hl2 = "\n".join(_scan3.holdings_lines(_att2))
check("missing quote shown in section 0", "no quote" in _hl2)
# validation issues surface in section 0
_hl_iss = "\n".join(_scan3.holdings_lines([], issues=["line 3: bad thing"]))
check("issues surface in holdings section", "NEEDS ATTENTION" in _hl_iss and "bad thing" in _hl_iss)

print("\nholdings-only + day views:")
_ho = _scan3.scan_holdings_only(holdings_path=_hp)   # AMZN + CM.TO fixture from earlier
check("scan_holdings_only scans only held names", sorted(r["ticker"] for r in _ho["results"]) == ["AMZN", "CM.TO"])
_htext = _scan3.build_holdings_text(_ho["now"], _ho["holdings"], _ho.get("holdings_issues"))
check("holdings text asks for hold/sell calls", "HOLD or SELL call" in _htext)
check("holdings text has section 0", "YOUR OPEN POSITIONS" in _htext)
check("empty holdings text is safe", "Positions check" in _scan3.build_holdings_text(_ho["now"], []))
check("day view resolves to short-term sections",
      _scan3.resolve_views("day", "midday") == {"morning", "overnight", "swing"})
check("day view excludes position", "position" not in _scan3.resolve_views("day", "midday"))
import server as _srv
check("server maps holdings alias", _srv.VIEW_ALIASES.get("holdings") == "holdings" and _srv.VIEW_ALIASES.get("positions") == "holdings")
check("server maps day alias", _srv.VIEW_ALIASES.get("day") == "day")
_pay_hold = _srv.scan_to_payload("holdings")
check("holdings payload ok + labelled", _pay_hold["ok"] and _pay_hold["view"] == "holdings")
check("holdings payload text is positions-only", "YOUR OPEN POSITIONS" in _pay_hold["text"] and "1) BUY NOW" not in _pay_hold["text"])
check("two-tab page has both tabs", "Day Trading" in _srv.PAGE and "Long Term" in _srv.PAGE and "showTab(" in _srv.PAGE)

print("\nbuy alerts + watcher:")
import os as _os
from daytrader import alerts as _al
# transport config selection from env
_saved = (_os.environ.pop(_al.NTFY_ENV, None), _os.environ.pop(_al.WEBHOOK_ENV, None))
check("no env -> console transport", _al.notification_config()["kind"] == "console")
_os.environ[_al.NTFY_ENV] = "my-topic"
check("ntfy env -> ntfy transport", _al.notification_config() == {"kind": "ntfy", "target": "my-topic"})
_os.environ.pop(_al.NTFY_ENV)
_os.environ[_al.WEBHOOK_ENV] = "https://example.com/hook"
check("webhook env -> webhook transport", _al.notification_config()["kind"] == "webhook")
_os.environ.pop(_al.WEBHOOK_ENV)
# console send never raises and reports success
check("console send returns True", _al.send_notification("t", "m", {"kind": "console", "target": None}) is True)
# dedupe: only not-yet-alerted names; canadian_only drops US
_cands = [
    {"ticker": "CM.TO", "rec": {"current_price": 162.0}, "atr_stop": {"floor": 160.0, "ceiling": 165.0}},
    {"ticker": "AAPL", "rec": {"current_price": 310.0}, "atr_stop": {"floor": 305.0, "ceiling": 318.0}},
]
_already = set()
_fresh = _al.new_buy_signals(_cands, _already, canadian_only=False)
check("all fresh when none alerted", [r["ticker"] for r in _fresh] == ["CM.TO", "AAPL"])
_already = {"CM.TO"}
check("alerted name filtered out", [r["ticker"] for r in _al.new_buy_signals(_cands, _already)] == ["AAPL"])
check("canadian_only drops US names", [r["ticker"] for r in _al.new_buy_signals(_cands, set(), canadian_only=True)] == ["CM.TO"])
# alert formatting
from daytrader.clock import market_now as _market_now
_t, _m = _al.format_buy_alert(_cands, _market_now())
check("alert title names the signal count", "2 SHORT-TERM buys" in _t)
check("alert message has levels", "Stop $160.00 | Limit $160.00" in _m and "Take profit: CAD $165.00" in _m)
check("alert message labels US currency", "Buy limit: USD" in _m)
check("alert message explains manual exits", "exits need your action" in _m)
# categorized short vs long labelling
_longs = [{"ticker": "CNR.TO", "price": 177.0, "floor": 169.0, "ceiling": 195.0, "hold_weeks": 5}]
_ct, _cm = _al.format_categorized_alert(_cands, _longs, _market_now())
check("categorized title counts both", "short-term" in _ct and "long-term" in _ct)
check("SHORT-TERM section labelled", "SHORT-TERM" in _cm and "CM.TO" in _cm)
check("LONG-TERM section labelled", "LONG-TERM" in _cm and "Buy: CNR.TO" in _cm and "Buy limit: CAD $177.00" in _cm)
check("long line shows exit deadline and expiry", "Sell by:" in _cm and "Good until cancelled" in _cm)
_lt, _lm = _al.format_categorized_alert([], _longs, _market_now())
check("long-only title", "LONG-TERM buy: CNR.TO" in _lt)
# single-pass watch cycle: fires the notification transport once for a fresh buy
import watch as _watch
_sent = []
class _FakeScan:
    def run_scan(self, view="auto"):
        return {"results": [
            eligible_fixture({"ticker": "CM.TO", "rec": {"status": "BUY_NOW", "current_price": 162.0}, "atr_stop": {"floor": 160.0, "ceiling": 165.0}}),
            {"ticker": "PG", "rec": {"status": "FLAT", "current_price": 149.0}},
        ]}
import daytrader.clock as _clk
_watch_scan_saved = None
try:
    import scan as _realscan
    _watch_scan_saved = _realscan
except Exception:
    pass
_al_send_saved = _al.send_notification
_clk_open_saved = _clk.market_is_open
try:
    sys.modules["scan"] = _FakeScan()
    _al.send_notification = lambda title, msg, cfg=None: (_sent.append((title, msg)) or True)
    # watch.run_watch imports 'scan' and calls alerts.send_notification (imported by name);
    # patch the reference used inside watch:
    _watch.send_notification = _al.send_notification
    _clk.market_is_open = lambda now=None: True
    _watch.market_is_open = lambda now=None: True
    _watch.run_watch(once=True)
    check("watcher sends one alert for a fresh BUY_NOW", len(_sent) == 1 and "CM.TO" in _sent[0][1])
finally:
    if _watch_scan_saved is not None:
        sys.modules["scan"] = _watch_scan_saved
    else:
        sys.modules.pop("scan", None)
    _al.send_notification = _al_send_saved
    _watch.send_notification = _al_send_saved
    _clk.market_is_open = _clk_open_saved
    _watch.market_is_open = _clk_open_saved

print(f"\nALL {passed} CHECKS PASSED")
