import math

import numpy as np
import pandas as pd

SESSION_END_HOUR = 16        # 16:00 ET close
DEFAULT_BUY_HOUR = 10        # fallback entry hour when timing history is thin
MIN_WEEKDAY_SAMPLES = 4      # don't call a weekday "strongest" off 1-2 days
ATR_BAR_MINUTES = 15         # must match risk_levels.ATR_BAR_MINUTES
BARS_PER_HOUR = 60 // ATR_BAR_MINUTES


def analyze_seasonality(df: pd.DataFrame, min_days: int = 10) -> dict:
    """Find historically favorable local-market hours and weekdays to trade.

    "Best buy hour" = the hour where price has historically sat closest to
    the day's low; "best sell hour" = closest to the day's high. These are the
    raw historical extremes and are informational - the actual planned hold the
    risk engine sizes is a bounded, same-session window (see plan_intraday_hold).
    Day-of-week strength is the average open-to-close return on that weekday,
    reported only for weekdays with enough samples to be worth mentioning.
    """
    data = df.copy()
    data["date"] = data.index.date

    day_high = data.groupby("date")["High"].transform("max")
    day_low = data.groupby("date")["Low"].transform("min")
    day_range = (day_high - day_low).replace(0, np.nan)
    data["pos_in_range"] = (data["Close"] - day_low) / day_range

    num_days = data["date"].nunique()
    hourly = data.groupby(data.index.hour)["pos_in_range"].mean().dropna()

    if hourly.empty or num_days < min_days:
        return {"insufficient_data": True, "num_days": num_days}

    daily = data.groupby("date").agg(open=("Open", "first"), close=("Close", "last"))
    daily["weekday"] = pd.to_datetime(daily.index).day_name()
    daily["return_pct"] = (daily["close"] / daily["open"] - 1) * 100

    counts = daily.groupby("weekday")["return_pct"].count()
    weekday_avg = daily.groupby("weekday")["return_pct"].mean()
    weekday_avg = weekday_avg[counts >= MIN_WEEKDAY_SAMPLES]  # drop thin weekdays

    return {
        "weekday_returns": {name: group["return_pct"].tolist() for name, group in daily.groupby("weekday")},
        "insufficient_data": False,
        "num_days": num_days,
        "best_buy_hour": int(hourly.idxmin()),
        "best_sell_hour": int(hourly.idxmax()),
        "best_day": weekday_avg.idxmax() if not weekday_avg.empty else None,
        "best_day_return_pct": float(weekday_avg.max()) if not weekday_avg.empty else None,
        "worst_day": weekday_avg.idxmin() if not weekday_avg.empty else None,
        "worst_day_return_pct": float(weekday_avg.min()) if not weekday_avg.empty else None,
    }


def plan_intraday_hold(season: dict, hold_hours: float) -> dict:
    """Turn the timing stats into a concrete, same-session (INTRADAY) hold.

    Entry is the historically favorable buy hour (or a default when timing
    history is thin). The hold then runs for hold_hours, capped so the exit
    never crosses the close. This is the upstream fix for holds that would
    otherwise span the whole session or cross a market close: separating "when
    to enter" (from the data) from "how long to hold" (the dial) keeps the
    planned trade intraday by construction, so the stop the engine sizes can
    actually protect it. The raw historical best sell hour is still reported
    separately, including when that raw pattern was itself an overnight one.
    """
    estimated = bool(season.get("insufficient_data", True))
    buy_hour = DEFAULT_BUY_HOUR if estimated else int(season["best_buy_hour"])
    if buy_hour >= SESSION_END_HOUR:
        buy_hour = SESSION_END_HOUR - 1  # leave at least one bar before the close

    requested_bars = max(1, int(round(hold_hours * BARS_PER_HOUR)))
    bars_to_close = max(1, (SESSION_END_HOUR - buy_hour) * BARS_PER_HOUR)
    hold_bars = min(requested_bars, bars_to_close)

    exit_total_min = buy_hour * 60 + hold_bars * ATR_BAR_MINUTES
    sell_hour, sell_minute = divmod(exit_total_min, 60)
    # For INTRADAY/OVERNIGHT classification we need an hour strictly past the
    # entry hour even for a sub-hour hold, so round the exit hour up.
    classify_sell_hour = min(SESSION_END_HOUR, math.ceil(exit_total_min / 60))

    raw_best_buy = None if estimated else int(season["best_buy_hour"])
    raw_best_sell = None if estimated else int(season["best_sell_hour"])
    raw_overnight = raw_best_sell is not None and raw_best_sell <= raw_best_buy

    return {
        "buy_hour": buy_hour,
        "sell_hour": sell_hour,
        "sell_minute": sell_minute,
        "classify_sell_hour": classify_sell_hour,
        "hold_bars": hold_bars,
        "hours_estimated": estimated,
        "raw_best_buy_hour": raw_best_buy,
        "raw_best_sell_hour": raw_best_sell,
        "raw_overnight": raw_overnight,
    }


def format_hour(hour: int) -> str:
    return f"{hour % 24:02d}:00"


def format_clock(hour: int, minute: int) -> str:
    return f"{hour % 24:02d}:{minute:02d}"
