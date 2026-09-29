"""Push a phone notification when a fresh BUY signal fires.

Kept deliberately conservative and honest:
- Alerts fire only on BUY_NOW signals (not the watch-lists), so you get a ping
  when something actionable appears, not a constant stream.
- Each name alerts at most once per day (dedupe), so a signal that persists
  across several scans doesn't spam you.
- An alert says "come look and verify", never "trade" - it is not an order and
  not a substitute for the second-opinion check. It includes the price so you
  can sanity-check it against your broker (a bad tick can fire a false signal).

Notification transport (stdlib only, no extra packages), chosen by env var:
- SCAN_NTFY_TOPIC   -> POST to https://ntfy.sh/<topic> (free app, no account;
                       install ntfy on your phone and subscribe to the topic)
- SCAN_WEBHOOK_URL  -> POST {"title","message"} to any URL (Telegram/Discord/
                       Pushover bridges, Slack, etc.)
- neither set       -> just prints to the console (so you can test it dry)
"""
import json
import os
import urllib.request

NTFY_ENV = "SCAN_NTFY_TOPIC"
WEBHOOK_ENV = "SCAN_WEBHOOK_URL"


def notification_config() -> dict:
    """Read transport config from the environment. Console fallback if unset."""
    topic = os.environ.get(NTFY_ENV, "").strip()
    webhook = os.environ.get(WEBHOOK_ENV, "").strip()
    if topic:
        return {"kind": "ntfy", "target": topic}
    if webhook:
        return {"kind": "webhook", "target": webhook}
    return {"kind": "console", "target": None}


def send_notification(title: str, message: str, config: dict | None = None) -> bool:
    """Send one notification. Returns True on success. Never raises."""
    config = config or notification_config()
    kind = config.get("kind", "console")
    try:
        if kind == "ntfy":
            url = config["target"]
            if not url.startswith("http"):
                url = f"https://ntfy.sh/{url}"
            req = urllib.request.Request(
                url, data=message.encode("utf-8"),
                headers={"Title": title, "Priority": "high", "Tags": "chart_with_upwards_trend"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        if kind == "webhook":
            payload = json.dumps({"title": title, "message": message}).encode("utf-8")
            req = urllib.request.Request(
                config["target"], data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        print(f"\n[ALERT] {title}\n{message}\n")  # console fallback
        return True
    except Exception as e:
        print(f"[alert failed: {e}] {title}: {message}")
        return False


def new_buy_signals(candidates: list[dict], already_alerted: set,
                    canadian_only: bool = False) -> list[dict]:
    """Filter BUY_NOW candidates down to ones not yet alerted today.

    `already_alerted` is a set of tickers (mutated by the caller after sending).
    `canadian_only` drops FX-carrying US names to cut noise, if requested.
    """
    fresh = []
    for r in candidates:
        tk = r["ticker"].upper()
        if tk in already_alerted:
            continue
        if canadian_only and not tk.endswith(".TO"):
            continue
        fresh.append(r)
    return fresh


def format_buy_alert(candidates: list[dict], now) -> tuple[str, str]:
    """Build (title, message) for a set of fresh buy signals (short-term only)."""
    return format_categorized_alert(candidates, [], now)


def alert_plan(r, now, short):
    from .trade_plan import broker_levels, size_example, sell_by, eastern
    gate = r.get('eligibility', {}).get('buy-now', {}) if short else r
    rec = r.get('rec') or {}
    risk = (r.get('atr_stop') or rec) if short else r
    entry = gate.get('entry_quote', r.get('last_price', rec.get('current_price'))) if short else r.get('price')
    order = gate.get('order_plan')
    if order is None:
        order = broker_levels(entry, risk.get('floor',rec.get('floor_price')),
                              risk.get('ceiling',rec.get('ceiling_price')))
    # Recompute from displayed prices, never reuse a larger stale share count.
    prior_size = gate.get('sizing') or {}
    size = size_example(r['ticker'], order['entry'], order['limit_price'], order['target_price'],
                        budget=prior_size.get('budget'), risk_budget=prior_size.get('risk_budget'),
                        min_profit=prior_size.get('minimum_profit'))
    deadline = gate.get('sell_by')
    if not deadline:
        deadline = sell_by(r['ticker'], now, minutes=risk.get('hold_bars',4)*15) if short else sell_by(r['ticker'],now,sessions=r.get('hold_weeks',4)*5)
    return order, size, deadline


def _compact_stock(r, now, short):
    from .trade_plan import eastern
    order, size, deadline = alert_plan(r, now, short)
    when = eastern(deadline).strftime('%Y-%m-%d %H:%M ET') if deadline is not None else 'unavailable - do not enter'
    currency = 'CAD' if r['ticker'].upper().endswith('.TO') else 'USD'
    quantity = (f"{size['shares']} whole shares (~CAD ${size['capital']:.2f})" if size.get('eligible')
                else 'unavailable within sizing limits - do not enter')
    expiry = 'Day' if short else 'Good until cancelled (up to 90 days)'
    return [
        f"Buy: {r['ticker']} ({'SHORT-TERM' if short else 'LONG-TERM'})",
        f"Buy limit: {currency} ${order['entry']:.2f} max",
        f"Amount: {quantity}",
        f"Stop-limit SELL: Stop ${order['stop_price']:.2f} | Limit ${order['limit_price']:.2f} ({currency}); expiry: {expiry}",
        f"Take profit: {currency} ${order['target_price']:.2f} | Sell by: {when} if still held",
    ]


def format_categorized_alert(shorts: list[dict], longs: list[dict], now) -> tuple[str, str]:
    """Five compact fields per stock, with separate protective and profit exits."""
    ns, nl = len(shorts), len(longs)
    if ns and nl:
        title = f"Buy signals: {ns} short-term, {nl} long-term"
    elif ns:
        title = f"SHORT-TERM buy: {shorts[0]['ticker']}" if ns == 1 else f"{ns} SHORT-TERM buys"
    elif nl:
        title = f"LONG-TERM buy: {longs[0]['ticker']}" if nl == 1 else f"{nl} LONG-TERM buys"
    else:
        return 'No buy signals', 'No eligible entries.'
    blocks = ['\n'.join(_compact_stock(r, now, True)) for r in shorts]
    blocks += ['\n'.join(_compact_stock(r, now, False)) for r in longs]
    blocks.append('After buying: protect only filled shares; live bid must exceed stop. '
                  'Stop-limit may not fill. Profit/deadline exits need your action; these are not linked orders.')
    return title, '\n\n'.join(blocks)


def format_pattern_watch(item):
    """Explicitly non-actionable: no quantity or executable order suggestion."""
    currency='CAD' if item['ticker'].endswith('.TO') else 'USD'
    period='hourly' if item['mode']=='buy-now' else 'long-term'
    return (f"WATCH ONLY: {item['ticker']} ({period})",
            f"Inverse head-and-shoulders forming.\nWatch neckline near {currency} ${item['neckline']:.2f}.\n"
            "Not a buy signal. Wait for a confirmed breakout/retest and a separate BUY alert. Setup may fail.")
