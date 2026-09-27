from .data import fetch_intraday, completed_bars
from .clock import market_now
from .policy import SIGNAL_PROFILE, DEFAULT_SIGNAL_INTERVAL
from .filters import intraday_liquidity
from .eligibility import attach_eligibility
from .dispersion import classify_risk, compute_dispersion
from .mean_reversion import generate_mean_reversion_signals
from .overnight import overnight_edge
from .recommend import current_recommendation
from .risk_levels import compute_atr_stop
from .seasonality import analyze_seasonality, plan_intraday_hold
import math
import pandas as pd
from .signals import generate_signals
from .swing import swing_candidate
from .position import position_candidate
from .quality import check_quality

DEFAULT_HOLD_HOURS = 1.0  # planned intraday hold length (cap + fallback); the key dial


def _reanchor(atr_stop: dict, entry_price: float) -> None:
    """Move an ATR stop's floor/ceiling to a different entry price, keeping the
    same dollar risk/reward distances. Used when we're already IN_POSITION so
    the displayed levels are anchored to the real fill, not the latest price.
    """
    fd = atr_stop.get("floor_distance")
    rd = atr_stop.get("reward_distance")
    if fd is None or rd is None:
        return
    atr_stop["floor"] = entry_price - fd
    atr_stop["ceiling"] = entry_price + rd
    atr_stop["floor_pct"] = fd / entry_price * 100
    atr_stop["ceiling_pct"] = rd / entry_price * 100
    atr_stop["anchored_to"] = "entry"


def _scan_ticker(
    ticker: str,
    interval: str = DEFAULT_SIGNAL_INTERVAL,
    period: str = "60d",
    take_profit_pct: float = 0.75,
    stop_loss_pct: float = 0.4,
    hold_hours: float = DEFAULT_HOLD_HOURS,
    include_overnight: bool = False,
    include_swing: bool = False,
    include_position: bool = False,
) -> dict:
    """Completed-bar entries with horizon ATR levels for both strategies.

    Low dispersion uses confirmed mean reversion; other regimes use confirmed
    trend entries. Live percentage exit arguments remain accepted for API
    compatibility, but ATR now determines both strategies' live levels. The
    shared eligibility wrapper applies costs, freshness and earnings gates.
    """
    raw = fetch_intraday(ticker, period=period, interval=interval)
    df = completed_bars(raw, interval)
    if len(df) < 60:
        raise ValueError('Too few completed bars for confirmed signals')

    dispersion = compute_dispersion(df)
    risk = classify_risk(dispersion)
    season = analyze_seasonality(df.loc[df.index.date < market_now().date()])
    overnight = overnight_edge(ticker) if include_overnight else None
    swing = swing_candidate(ticker) if include_swing else None
    position = position_candidate(ticker) if include_position else None
    quality = check_quality(raw)
    last_price = float(raw["Close"].iloc[-1]) if len(raw) else None

    strategy = "meanrev" if risk == "LOW" else "trend"
    current_price = float(df["Close"].iloc[-1])

    # Evaluate entries before risk sizing so a rejection never hides raw signals.
    from .entry_setup import choose_setup
    from .patterns import complete_fifteen
    strategy,latest,raw_buy,checks,patterns=choose_setup(df,risk)
    now = market_now()
    if not math.isfinite(hold_hours) or hold_hours <= 0:
        raise ValueError('hold_hours must be positive and finite')
    bars = max(1, math.ceil(hold_hours*4))
    start_minute = now.hour*60+now.minute
    end_minute = start_minute+bars*15
    plan = {'buy_hour':now.hour, 'buy_minute':now.minute,
            'sell_hour':end_minute//60, 'sell_minute':end_minute%60,
            'classify_sell_hour':math.ceil(end_minute/60), 'hold_bars':bars,
            'hours_estimated':False, 'raw_best_buy_hour':season.get('best_buy_hour'),
            'raw_best_sell_hour':season.get('best_sell_hour'), 'raw_overnight':False}
    if SIGNAL_PROFILE == 'active':
        from .active_risk import active_risk
        if interval not in ('5m','15m'):
            raise ValueError('active profile requires 5m or 15m signal candles')
        risk_df=complete_fifteen(df) if interval=='5m' else df
        pattern=patterns.get('bullish') if strategy.startswith('inverse-hs') else None
        atr_stop = active_risk(risk_df, ticker, current_price, now, bars, pattern=pattern)
    else:
        atr_stop = compute_atr_stop(
            ticker,
            entry_price=current_price,
            buy_hour=plan["buy_hour"],
            sell_hour=plan["classify_sell_hour"],
            hold_bars=plan["hold_bars"],
            asof=now,
        )
    atr_stop.update(plan)  # buy/sell clock, hold_bars, raw_* hours, etc.

    if atr_stop["status"] != "TRADE":
        return {
            "ticker": ticker,
            "raw_buy_signal": raw_buy,
            "patterns": patterns,
            "quote_as_of": str(raw.index[-1]),
            "scanned_at": now.isoformat(),
            "entry_checks": checks,
            "strategy": strategy,
            "risk": risk,
            "dispersion": dispersion,
            "season": season,
            "overnight": overnight,
            "swing": swing,
            "position": position,
            "quality": quality,
            "liquidity_confirmed": bool(intraday_liquidity(df).iloc[-1]),
            "signal_as_of": str(df.index[-1]),
            "interval": interval,
            "last_price": last_price,
            "no_trade": True,
            "atr_stop": atr_stop,
            "current_price": current_price,
        }

    tp = atr_stop['ceiling_pct']
    sl = atr_stop['floor_pct']
    # An entry signal is not a hypothetical holding. Real positions live in holdings.txt.
    rec = {'status':'BUY_NOW' if raw_buy else 'FLAT', 'as_of':df.index[-1],
           'current_price':current_price, 'rsi':float(latest['rsi']), 'vwap':float(latest['vwap'])}
    if raw_buy:
        rec.update(entry_time=df.index[-1], entry_price=current_price, unrealized_pct=0.,
                   floor_price=atr_stop['floor'], ceiling_price=atr_stop['ceiling'])
    return {
        "ticker": ticker,
        "raw_buy_signal": raw_buy,
            "patterns": patterns,
            "quote_as_of": str(raw.index[-1]),
            "scanned_at": now.isoformat(),
            "entry_checks": checks,
        "strategy": strategy,
        "risk": risk,
        "dispersion": dispersion,
        "season": season,
        "overnight": overnight,
        "swing": swing,
        "position": position,
        "quality": quality,
            "liquidity_confirmed": bool(intraday_liquidity(df).iloc[-1]),
            "signal_as_of": str(df.index[-1]),
            "interval": interval,
        "last_price": last_price,
        "rec": rec,
        "take_profit_pct": tp,
        "stop_loss_pct": sl,
        "atr_stop": atr_stop,
        "no_trade": False,
    }



def scan_ticker(ticker, **kwargs):
    result = _scan_ticker(ticker, **kwargs)
    result["signal_version"] = "patterns-v8-" + SIGNAL_PROFILE + "-unvalidated"
    return attach_eligibility(result)


def scan_watchlist(tickers: list[str], **kwargs) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            results.append(scan_ticker(ticker, **kwargs))
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    return results
