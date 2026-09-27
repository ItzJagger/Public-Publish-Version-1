import pandas as pd


def compute_dispersion(df: pd.DataFrame) -> dict:
    """Measure how far a stock's price swings per day, as a % of price."""
    daily = df.groupby(df.index.date).agg(high=("High", "max"), low=("Low", "min"), close=("Close", "last"))
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100

    return {
        "avg_daily_range_pct": daily["range_pct"].mean(),
        "daily_return_stdev_pct": daily["close"].pct_change().std() * 100,
        "num_days": len(daily),
    }


def classify_risk(dispersion: dict) -> str:
    """LOW dispersion = tends to stay in a tight range = better fit for mean-reversion."""
    range_pct = dispersion["avg_daily_range_pct"]
    if range_pct < 2.0:
        return "LOW"
    if range_pct < 3.5:
        return "MEDIUM"
    return "HIGH"
