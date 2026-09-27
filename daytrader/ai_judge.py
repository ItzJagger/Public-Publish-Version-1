from .seasonality import format_clock, format_hour

_MODEL = "claude-opus-4-8"

_INSTRUCTIONS = (
    "QUICK SECOND OPINION for a manual trader who places their own orders (you place none).\n"
    "- Web-search live price/news/sector AND earnings before the exit date for any name "
    "you'd act on; flag earnings inside a hold (un-stoppable gap). Don't rely on these numbers alone.\n"
    "- Levels are ATR-based, sized outside noise at >=1.5:1; NO-TRADE = correctly too calm to "
    "trade, don't override it or tighten a stop back inside the noise. Write levels as "
    "\"Floor: $X.XX | Ceiling: $Y.YY\".\n"
    "- Don't manufacture trades - \"nothing clean\" is a fine, useful answer. Be direct about "
    "uncertainty. Not financial advice."
)

_RESEARCH_INSTRUCTIONS = ""  # merged into _INSTRUCTIONS to cut length

_HOLDINGS_INSTRUCTIONS = (
    "Section 0 = the user's REAL holdings: give each an explicit HOLD or SELL call + one-line "
    "reason + exit date (earnings or horizon end, whichever sooner); verify earnings by web search. "
    "Any \"IN_POSITION\" in the ticker lines below is a REPLAYED signal, NOT a position the user "
    "holds - ignore it unless the name is also in section 0.\n"
)


def _num(value) -> str:
    return f"{value:.2f}" if value is not None and value == value else "n/a"


def _hold_line(atr_stop: dict) -> str:
    buy_hour = atr_stop.get("buy_hour")
    sell_hour = atr_stop.get("sell_hour")
    sell_minute = atr_stop.get("sell_minute", 0)
    est = " (entry hour estimated)" if atr_stop.get("hours_estimated") else ""
    return (
        f"Planned hold: ~{format_clock(buy_hour, atr_stop.get('buy_minute', 0))} -> ~{format_clock(sell_hour, sell_minute)} ET "
        f"({atr_stop.get('classification')}, {atr_stop.get('hold_bars')} x 15m bars){est}"
    )


def _ticker_summary(ticker: str, scan_result: dict) -> str:
    season = scan_result["season"]
    dispersion = scan_result["dispersion"]
    is_meanrev = scan_result.get("atr_stop") is not None

    lines = [
        f"Ticker: {ticker}",
        f"Strategy in use: {scan_result['strategy']} (risk: {scan_result['risk']}, "
        f"avg daily range {dispersion['avg_daily_range_pct']:.2f}%)",
    ]

    if scan_result.get("no_trade"):
        atr_stop = scan_result["atr_stop"]
        lines.append(f"Current price: {scan_result['current_price']:.2f}")
        if atr_stop.get("classification"):
            lines.append(_hold_line(atr_stop))
        lines.append(
            f"Single-bar ATR: {_num(atr_stop.get('single_bar_atr'))}, horizon ATR: "
            f"{_num(atr_stop.get('horizon_atr'))}, MAE floor distance: {_num(atr_stop.get('mae_distance'))}"
        )
        if atr_stop.get("classification") == "OVERNIGHT":
            lines.append(f"Typical overnight gap: {_num(atr_stop.get('typical_gap'))}, worst: {_num(atr_stop.get('worst_gap'))}")
        lines.append(f"Our risk gate says {atr_stop['status']}: {atr_stop['reason']}")
        lines.append("Do not propose a floor/ceiling for this ticker - just confirm or flag disagreement with skipping it.")
        return "\n".join(lines)

    rec = scan_result["rec"]
    if rec['status'] == 'BLOCKED':
        lines.append('ENTRY BLOCKED: ' + '; '.join(scan_result['eligibility']['buy-now']['reasons']))
        lines.append('Do not recommend an entry until these blocks are resolved.')
        return '\n'.join(lines)
    price = rec["current_price"]
    lines.append(f"Status: {rec['status']}")
    lines.append(f"Current price: {price:.2f}")

    if is_meanrev:
        atr_stop = scan_result["atr_stop"]
        lines.append(_hold_line(atr_stop))
        mae_note = " (widened by max-adverse-excursion check)" if atr_stop.get("mae_used") else ""
        lines.append(
            f"Single-bar ATR: {_num(atr_stop.get('single_bar_atr'))}, horizon ATR: "
            f"{_num(atr_stop.get('horizon_atr'))}{mae_note}"
        )
        if rec["status"] in ("IN_POSITION", "BUY_NOW"):
            lines.append(f"Entry price: {rec['entry_price']:.2f}, unrealized {rec['unrealized_pct']:+.2f}%")
            anchor = " (anchored to entry)" if atr_stop.get("anchored_to") == "entry" else ""
            lines.append(
                f"ATR floor (stop): {atr_stop['floor']:.2f}, ceiling (take profit): "
                f"{atr_stop['ceiling']:.2f}, R:R {atr_stop['rr']:.2f}:1{anchor}"
            )
        else:
            lines.append(
                f"No open position yet. If bought now at {price:.2f}, the ATR levels would be "
                f"floor (stop) {atr_stop['floor']:.2f}, ceiling (take profit) {atr_stop['ceiling']:.2f} "
                f"(R:R {atr_stop['rr']:.2f}:1). Wait for a mean-reversion buy signal first."
            )
    else:
        # Trend strategy: fixed-percentage targets.
        if rec["status"] in ("IN_POSITION", "BUY_NOW"):
            lines.append(f"Entry price: {rec['entry_price']:.2f}, unrealized {rec['unrealized_pct']:+.2f}%")
            lines.append(
                f"Our computed ceiling (take profit): {rec['ceiling_price']:.2f}, "
                f"floor (stop loss): {rec['floor_price']:.2f}"
            )
        else:
            lines.append(f"RSI: {rec['rsi']:.1f}")
            suggested_ceiling = price * (1 + scan_result["take_profit_pct"] / 100)
            suggested_floor = price * (1 - scan_result["stop_loss_pct"] / 100)
            lines.append(
                f"No open position - if bought now at {price:.2f}, our default targets would be "
                f"ceiling {suggested_ceiling:.2f} (+{scan_result['take_profit_pct']:.2f}%), "
                f"floor {suggested_floor:.2f} (-{scan_result['stop_loss_pct']:.2f}%)"
            )

    if not season.get("insufficient_data"):
        lines.append(
            f"Raw historical low hour: ~{format_hour(season['best_buy_hour'])} local market time; "
            f"high hour: ~{format_hour(season['best_sell_hour'])} (informational)"
        )
        if season.get("best_day"):
            lines.append(f"Strongest weekday: {season['best_day']} ({season['best_day_return_pct']:+.2f}% avg)")

    return "\n".join(lines)


def _ticker_line(ticker: str, r: dict) -> str:
    """One dense line per ticker: status, price, RSI, and levels if a signal fired."""
    if r.get('rec', {}).get('status') == 'BLOCKED':
        reasons = r.get('eligibility', {}).get('buy-now', {}).get('reasons', [])
        return f"{ticker}: BLOCKED - {'; '.join(reasons)}"
    strat = r.get("strategy", "?")
    if r.get("no_trade"):
        return f"{ticker}: NO-TRADE @ {r['current_price']:.2f} ({strat}) - {r.get('atr_stop', {}).get('reason', 'risk unavailable')}"

    rec = r["rec"]
    px = rec["current_price"]
    rsi = rec.get("rsi")
    rsi_s = f" RSI {rsi:.0f}" if isinstance(rsi, (int, float)) and rsi == rsi else ""
    st = rec["status"]
    parts = [f"{ticker}: {st} @ {px:.2f}{rsi_s} ({strat})"]

    live = st in ("BUY_NOW", "IN_POSITION")
    if r.get("atr_stop"):  # mean-reversion
        a = r["atr_stop"]
        if live:
            parts.append(f"floor {a['floor']:.2f} ceiling {a['ceiling']:.2f} RR {a['rr']:.1f}")
        else:
            parts.append(f"if bought: floor {a['floor']:.2f} ceiling {a['ceiling']:.2f}")
    elif live and rec.get("ceiling_price") is not None:  # trend, in a position
        parts.append(f"ceiling {rec['ceiling_price']:.2f} floor {rec['floor_price']:.2f}")
    if live and rec.get("entry_price") is not None:
        parts.append(f"entry {rec['entry_price']:.2f} ({rec['unrealized_pct']:+.1f}%)")
    return " | ".join(parts)


def build_batch_prompt(results: list[dict], holdings: list[dict] | None = None) -> str:
    """Format every scanned ticker into one compact prompt for manual paste into claude.ai.

    This is the primary, no-cost AI path: print it (scan.py --prompt), paste into
    a claude.ai chat, and read the response there. Needs no API key and does not
    import the anthropic package.
    """
    lines = [_ticker_line(r["ticker"], r) for r in results if "error" not in r]
    extra = _HOLDINGS_INSTRUCTIONS if holdings else ""
    table = "PER-TICKER (status @ price):\n" + "\n".join(lines)
    return _INSTRUCTIONS + "\n\n" + extra + "\n" + table


def get_ai_recommendation(ticker: str, scan_result: dict) -> str:
    """Ask Claude for a short narrative take on one ticker's current setup.

    Opt-in only (the --ai flag) - requires the anthropic package and an
    ANTHROPIC_API_KEY, and costs a small amount per call. Imported lazily so the
    clipboard prompt flow works without anthropic installed. Does not place
    trades; this is one more input to weigh manually.
    """
    import anthropic  # lazy: only the paid API path needs it

    client = anthropic.Anthropic()
    prompt = _INSTRUCTIONS + "\n\n" + _ticker_summary(ticker, scan_result)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((block.text for block in response.content if block.type == "text"), "")
