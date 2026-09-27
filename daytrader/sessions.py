import pandas as pd


def last_bar_of_day(index: pd.DatetimeIndex) -> set:
    """Set of the final timestamp in each trading day.

    Precomputed once so the replay loops can test "is this the last bar of the
    day?" in O(1), instead of re-filtering the whole frame on every row (which
    was O(n^2) and slow on long/short-interval histories).
    """
    if len(index) == 0:
        return set()
    s = index.to_series()
    return set(s.groupby(index.date).last())
