import math

import pandas as pd

from . import indicators
from .data import fetch_intraday, completed_bars
from .mean_reversion import generate_mean_reversion_signals
from .sessions import last_bar_of_day

ATR_INTERVAL = "15m"
ATR_BAR_MINUTES = 15
BARS_PER_HOUR = 60 / ATR_BAR_MINUTES
ATR_PERIOD = "10d"  # legacy regression reference only
ATR_WINDOW = 14
ATR_MULTIPLE = 1.25  # midpoint of the 1.0-1.5x band

GAP_PERIOD = "3mo"  # daily bars, for the overnight-gap distribution

SESSION_START_HOUR = 9   # market opens 9:30 ET; the "9" hour-bucket covers 9:30-9:59
SESSION_END_HOUR = 16    # market closes 16:00 ET

MIN_RR = 2.0
REALISTIC_MOVE_FRACTION = 0.7  # a single leg shouldn't need more than this share of the avg daily $ range


def compute_legacy_atr_stop(ticker: str, entry_price: float, buy_hour: int, sell_hour: int,
                     hold_bars: int | None = None) -> dict:
    """ATR-based floor/ceiling for a mean-reversion trade, scaled to the
    actual planned hold (buy_hour -> sell_hour), with an overnight-gap check
    when that hold crosses a market close.

    `hold_bars`, when given, sets the number of ATR-bar-equivalents the hold is
    exposed to directly (so the caller can express a fractional-hour hold);
    otherwise it's derived from the buy/sell hours. The INTRADAY/OVERNIGHT
    classification still comes from the buy/sell hours either way.

    A single-bar ATR understates the noise a multi-hour hold is exposed to —
    volatility scales with sqrt(time) — so the floor is sized off a
    horizon-scaled ATR (single-bar ATR * sqrt(bars in the hold)), not the raw
    single-bar value. A max-adverse-excursion check can widen it further. If
    the hold crosses a session close (OVERNIGHT), an intraday stop cannot
    protect against a gap: when the ticker's typical overnight gap exceeds
    the noise-based floor, this returns status="NO-TRADE-AS-INTRADAY" rather
    than reporting a stop that wouldn't actually cap the loss.
    """
    try:
        df = completed_bars(fetch_intraday(ticker, period=ATR_PERIOD, interval=ATR_INTERVAL), ATR_INTERVAL)
    except Exception as exc:
        return {
            "status": "NO-TRADE",
            "reason": f"could not fetch intraday data for ATR ({exc})",
            "single_bar_atr": float("nan"),
        }

    single_bar_atr = indicators.atr(df, window=ATR_WINDOW).iloc[-1]

    if pd.isna(single_bar_atr) or single_bar_atr <= 0:
        return {
            "status": "NO-TRADE",
            "reason": "not enough intraday bars yet for a reliable ATR",
            "single_bar_atr": float("nan"),
        }

    derived_bars, classification, consistency_bug = _hold_bars_and_classification(buy_hour, sell_hour)
    hold_bars = derived_bars if hold_bars is None else max(1, int(hold_bars))
    horizon_atr = single_bar_atr * math.sqrt(hold_bars)

    mae_distance = _max_adverse_excursion(df)
    floor_distance = ATR_MULTIPLE * horizon_atr
    used_mae = mae_distance is not None and mae_distance > floor_distance
    if used_mae:
        floor_distance = mae_distance

    result = {
        "hold_bars": hold_bars,
        "single_bar_atr": single_bar_atr,
        "horizon_atr": horizon_atr,
        "mae_distance": mae_distance,
        "floor_distance": floor_distance,
        "reward_distance": MIN_RR * floor_distance,
        "classification": classification,
        "consistency_bug": consistency_bug,
        "typical_gap": None,
        "worst_gap": None,
    }

    if classification == "OVERNIGHT":
        gap_stats = _overnight_gap_stats(ticker)
        if gap_stats:
            result["typical_gap"] = gap_stats["typical"]
            result["worst_gap"] = gap_stats["worst"]

            if gap_stats["typical"] > floor_distance:
                return {
                    **result,
                    "status": "NO-TRADE-AS-INTRADAY",
                    "reason": (
                        f"this hold crosses a market close and the typical overnight gap "
                        f"(${gap_stats['typical']:.2f}) exceeds the noise-based floor distance "
                        f"(${floor_distance:.2f}) - an intraday stop would not actually cap the "
                        f"loss here; the real risk is the gap, not intraday wiggle"
                    ),
                    "floor": entry_price - floor_distance,
                    "ceiling": entry_price + MIN_RR * floor_distance,
                    "rr": MIN_RR,
                }

    floor = entry_price - floor_distance
    reward_distance = MIN_RR * floor_distance
    ceiling = entry_price + reward_distance
    rr = reward_distance / floor_distance if floor_distance else 0.0

    daily = df.groupby(df.index.date).agg(high=("High", "max"), low=("Low", "min"))
    avg_daily_dollar_range = (daily["high"] - daily["low"]).mean()
    max_realistic_move = REALISTIC_MOVE_FRACTION * avg_daily_dollar_range

    if pd.isna(avg_daily_dollar_range) or reward_distance > max_realistic_move:
        return {
            **result,
            "status": "NO-TRADE",
            "reason": (
                f"target move (${reward_distance:.2f}) would exceed a realistic share "
                f"(${max_realistic_move:.2f}) of this stock's typical daily range "
                f"(${avg_daily_dollar_range:.2f}) - not shrinking the stop to force a fit"
            ),
            "floor": floor,
            "ceiling": ceiling,
            "rr": rr,
        }

    if rr < MIN_RR - 1e-9:
        return {
            **result,
            "status": "NO-TRADE",
            "reason": f"reward:risk of {rr:.2f}:1 is below the {MIN_RR:.1f}:1 minimum",
            "floor": floor,
            "ceiling": ceiling,
            "rr": rr,
        }

    reason = (
        f"floor sized off horizon ATR (single-bar ATR x sqrt({hold_bars} bars))"
        + (" widened by a max-adverse-excursion check" if used_mae else "")
        + f", clears {MIN_RR:.1f}:1 reward:risk"
    )
    if classification == "OVERNIGHT":
        if result["typical_gap"] is not None:
            reason += " (overnight hold, but typical gap is within the floor distance)"
        else:
            reason += " (overnight hold; not enough daily history to check the typical gap)"

    return {
        **result,
        "status": "TRADE",
        "reason": reason,
        "floor": floor,
        "ceiling": ceiling,
        "rr": rr,
        "floor_pct": floor_distance / entry_price * 100,
        "ceiling_pct": reward_distance / entry_price * 100,
        "mae_used": used_mae,
    }


def _hold_bars_and_classification(buy_hour: int, sell_hour: int) -> tuple[int, str, bool]:
    """How many ATR-bar-equivalents of intraday noise this hold is exposed
    to, and whether the buy->sell window stays inside one session
    (INTRADAY) or crosses a market close (OVERNIGHT).
    """
    if sell_hour > buy_hour:
        classification = "INTRADAY"
        exposure_hours = sell_hour - buy_hour
    else:
        classification = "OVERNIGHT"
        hours_to_close = max(SESSION_END_HOUR - buy_hour, 0.5)
        hours_from_open = max(sell_hour - SESSION_START_HOUR, 0.5)
        exposure_hours = hours_to_close + hours_from_open

    # Internal consistency gate: an INTRADAY hold must stay inside one
    # session. If the hours fall outside the trading session, that's a bug
    # in the caller's timing data — fail safe to OVERNIGHT and flag it.
    in_session = (
        SESSION_START_HOUR <= buy_hour < SESSION_END_HOUR
        and SESSION_START_HOUR <= sell_hour <= SESSION_END_HOUR
    )
    consistency_bug = classification == "INTRADAY" and not in_session
    if consistency_bug:
        classification = "OVERNIGHT"
        exposure_hours = max(exposure_hours, 1.0)

    hold_bars = max(1, round(exposure_hours * BARS_PER_HOUR))
    return hold_bars, classification, consistency_bug


def _overnight_gap_stats(ticker: str) -> dict | None:
    """Typical and worst-case overnight downside gap (prior close -> next
    open), in $, from daily bars. None if there isn't enough history yet.
    """
    try:
        daily = fetch_intraday(ticker, period=GAP_PERIOD, interval="1d")
    except Exception:
        return None

    if len(daily) < 10:
        return None

    gap = daily["Open"] - daily["Close"].shift(1)
    downside_gap = (-gap).clip(lower=0).dropna()

    if downside_gap.empty:
        return {"typical": 0.0, "worst": 0.0}

    return {"typical": float(downside_gap.mean()), "worst": float(downside_gap.max())}


def _max_adverse_excursion(df: pd.DataFrame, min_winning_trades: int = 5) -> float | None:
    """For past mean-reversion buy signals that ended up profitable, how far
    did price dip below entry before recovering? Returns the ~80th-percentile
    $ distance of that dip, or None if there isn't enough winning-trade
    history yet to trust the estimate.
    """
    signals = generate_mean_reversion_signals(df)
    eod = last_bar_of_day(signals.index)
    excursions = []
    position = None
    worst = None

    for ts, row in signals.iterrows():
        is_last_bar_of_day = ts in eod

        if position is None and row["buy"]:
            if is_last_bar_of_day:  # would be an overnight hold - skip, intraday only
                continue
            position = row["Close"]
            worst = row["Low"]
            continue

        if position is None:
            continue

        worst = min(worst, row["Low"])

        if row["sell"] or is_last_bar_of_day:
            if row["Close"] > position:
                excursions.append(position - worst)
            position = None
            worst = None

    if len(excursions) < min_winning_trades:
        return None

    excursions.sort()
    idx = min(len(excursions) - 1, int(0.8 * len(excursions)))
    return excursions[idx]


# Live model. The legacy function above is retained only for regression comparisons.
HORIZON_PERIOD = '60d'
MIN_HORIZON_SESSIONS = 20
STOP_EXCURSION_QUANTILE = .60
TARGET_EXCURSION_QUANTILE = .75


def session_atr(df):
    """Intraday true range excluding overnight jumps at each session's first bar."""
    previous = df.Close.shift(1)
    dates = pd.Series(df.index.date, index=df.index)
    previous = previous.where(dates.eq(dates.shift(1)), df.Open)
    tr = pd.concat([df.High-df.Low, (df.High-previous).abs(),
                    (df.Low-previous).abs()], axis=1).max(axis=1)
    return indicators._wilder(tr, ATR_WINDOW)


def horizon_risk_from_frame(df, entry_price, hold_bars, asof, cost_pct,
                            entry_minute=None):
    """Calibrate independent levels from one same-clock window per earlier session.

    Entry is each historical window's open. Include every usable window, not just
    profitable ones. Normalize excursions by ATR known before that entry, then
    scale to current session ATR. Samples never include today's/future paths.
    Quantiles are untuned heuristics, not confidence levels or win probabilities.
    """
    from .policy import net_reward_risk, MIN_NET_RR
    out = {'status':'NO-TRADE', 'risk_model':'empirical-horizon-v1',
           'classification':'INTRADAY', 'consistency_bug':False,
           'single_bar_atr':float('nan'), 'horizon_atr':None,
           'mae_distance':None, 'typical_gap':None, 'worst_gap':None,
           'hold_bars':hold_bars, 'samples':0}
    def reject(reason):
        return {**out, 'status':'NO-TRADE', 'reason':reason}
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(cost_pct) or cost_pct < 0:
        return reject('invalid entry price or cost')
    if not isinstance(hold_bars, int) or not 1 <= hold_bars <= 26:
        return reject('invalid intraday holding length')
    now = pd.Timestamp(asof)
    now = now.tz_localize('America/New_York') if now.tzinfo is None else now.tz_convert('America/New_York')
    # Validate and truncate before any indicator calculation.
    try:
        df = completed_bars(df, ATR_INTERVAL, now)
    except (ValueError, KeyError, TypeError) as exc:
        return reject(f'invalid risk history: {exc}')
    if df.empty:
        return reject('no completed risk bars')
    idx = df.index.tz_localize(now.tz) if df.index.tz is None else df.index.tz_convert(now.tz)
    df = df.copy(); df.index = idx
    minute = now.hour*60+now.minute if entry_minute is None else entry_minute
    if minute < 570 or minute+hold_bars*15 > 960:
        return reject('planned hold does not fit the remaining regular session')
    start_minute = int(minute//15)*15
    a = session_atr(df)
    current_atr = float(a.iloc[-1])
    out['single_bar_atr'] = current_atr
    if not math.isfinite(current_atr) or current_atr <= 0:
        return reject('insufficient ATR history')
    age = (now-df.index[-1]).total_seconds()/60-15
    if age > 20 or age < 0:
        return reject('risk history is stale or unfinished')
    adverse, favorable, sample_dates = [], [], []
    prior_atr = a.shift(1)
    for day, frame in df.loc[df.index.date < now.date()].groupby(df.loc[df.index.date < now.date()].index.date):
        starts = frame.index[(frame.index.hour*60+frame.index.minute) == start_minute]
        if not len(starts):
            continue
        start = starts[0]
        expected = pd.date_range(start, periods=hold_bars, freq='15min')
        if not expected.isin(frame.index).all():
            continue  # gaps must not shorten the sampled horizon
        base_atr = float(prior_atr.loc[start])
        if not math.isfinite(base_atr) or base_atr <= 0:
            continue
        path = frame.loc[expected]
        entry = float(path.Open.iloc[0])
        adverse.append(max(0., entry-float(path.Low.min()))/base_atr)
        favorable.append(max(0., float(path.High.max())-entry)/base_atr)
        sample_dates.append(str(day))
    out.update(samples=len(adverse), sample_dates=sample_dates,
               sample_start_minute=start_minute, cost_pct=cost_pct)
    if len(adverse) < MIN_HORIZON_SESSIONS:
        return reject(f'need {MIN_HORIZON_SESSIONS} prior same-time sessions; have {len(adverse)}')
    adverse_distance = float(pd.Series(adverse).quantile(STOP_EXCURSION_QUANTILE))*current_atr
    reward = float(pd.Series(favorable).quantile(TARGET_EXCURSION_QUANTILE))*current_atr
    # Size independently before inspecting reward/risk; never shrink to pass.
    distance = max(ATR_MULTIPLE*current_atr, adverse_distance)
    floor, ceiling = entry_price-distance, entry_price+reward
    rr = reward/distance
    net_rr = net_reward_risk(entry_price, floor, ceiling, cost_pct)
    out.update(floor=floor, ceiling=ceiling, floor_distance=distance,
               reward_distance=reward, floor_pct=100*distance/entry_price,
               ceiling_pct=100*reward/entry_price, rr=rr, net_rr=net_rr,
               mae_distance=adverse_distance, horizon_atr=adverse_distance,
               mae_used=adverse_distance > ATR_MULTIPLE*current_atr,
               stop_quantile=STOP_EXCURSION_QUANTILE, target_quantile=TARGET_EXCURSION_QUANTILE)
    if not 0 < floor < entry_price < ceiling:
        return reject('invalid independent stop/target levels')
    if net_rr < MIN_NET_RR:
        return reject(f'independent horizon levels give {net_rr:.2f}:1 after costs; need {MIN_NET_RR:.1f}:1')
    return {**out, 'status':'TRADE', 'reason':
            f'independent horizon levels from {len(adverse)} earlier sessions; '
            f'{net_rr:.2f}:1 after costs (historical estimates, not a win probability)'}


def compute_atr_stop(ticker, entry_price, buy_hour, sell_hour, hold_bars=None, asof=None):
    """Live risk API; no square-root multiplier or target derived from stop size."""
    from .policy import round_trip_cost
    asof = pd.Timestamp.now(tz='America/New_York') if asof is None else pd.Timestamp(asof)
    bars, classification, bug = _hold_bars_and_classification(buy_hour, sell_hour)
    if classification != 'INTRADAY':
        return {'status':'NO-TRADE-AS-INTRADAY','reason':'overnight risk requires a separate model',
                'classification':classification,'consistency_bug':bug,'single_bar_atr':float('nan')}
    try:
        df = fetch_intraday(ticker, period=HORIZON_PERIOD, interval=ATR_INTERVAL)
        return horizon_risk_from_frame(df, entry_price, bars if hold_bars is None else hold_bars,
                                       asof, round_trip_cost(ticker))
    except Exception as exc:
        return {'status':'NO-TRADE','reason':f'could not assess horizon risk: {exc}',
                'single_bar_atr':float('nan'),'risk_model':'empirical-horizon-v1'}
