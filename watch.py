"""Always-running watcher: scan on a schedule and push a phone notification
when a fresh BUY signal fires.

Run it on your laptop and leave it running during market hours:

    python watch.py                      # short-term every 1 min, long-term every 4 h
    python watch.py --alerts long        # only long-term (uptrend-pullback) buys
    python watch.py --alerts short       # only short-term (fired) buys
    python watch.py --canadian-only      # only alert on FX-free (.TO) signals
    python watch.py --interval-min 10 --long-interval-min 120

Set ONE of these environment variables first so alerts reach your phone
(otherwise alerts just print to this window so you can test it):

    export SCAN_NTFY_TOPIC=my-secret-topic-name     # free: install the ntfy app,
                                                    # subscribe to that topic
    export SCAN_WEBHOOK_URL=https://...             # or any webhook bridge

HONEST LIMITS:
- This runs on THIS computer. It only works while the laptop is on, awake, and
  running this script. Close the lid or let it sleep and alerts stop. (On a Mac,
  `caffeinate -s python watch.py` keeps it awake; on Windows set the power plan
  to never sleep. For true 24/7, run it on a small always-on box / cheap VPS.)
- An alert is a "come look and verify" ping, NOT an order and NOT a green light.
  Signals are often FX-drag or noise; most days the right move is still nothing.
- It fetches the same data as a normal scan, so a bad tick can fire a false
  alert - the message includes the price so you can sanity-check it.
"""
import argparse
import time
import json
from pathlib import Path
from daytrader.trade_plan import exchange, eastern
from daytrader.eligibility import attach_eligibility
from daytrader.policy import SIGNAL_PROFILE
from daytrader.tracker import SignalTracker
from daytrader.daily_review import DailyReview, next_report_seconds
from daytrader.diagnostics import diagnostic_lines

from daytrader.alerts import (
    format_categorized_alert, format_pattern_watch, new_buy_signals, notification_config, send_notification,
)
from daytrader.clock import describe_now, market_is_open, market_now
from daytrader.playbook import buy_now_candidates


def closed_wait_seconds(now, maximum_minutes=30):
    # Never sleep through the next exchange opening.
    waits = [maximum_minutes*60, next_report_seconds(now)]
    for ticker in ('SPY','XIU.TO'):
        cal = exchange(ticker)
        date = cal.date_to_session(str(now.date()), direction='next')
        opening = cal.session_open(date)
        if opening <= now:
            date = cal.next_session(date)
            opening = cal.session_open(date)
        waits.append(max(1., (opening-now).total_seconds()))
    return min(waits)


def run_watch(interval_min: int = 1, long_interval_min: int = 240, alerts: str = "both",
              canadian_only: bool = False, closed_sleep_min: int = 30,
              long_cooldown_days: int = 3, once: bool = False, early_watch: bool = True):
    """Hourly pass is delivered BEFORE slow daily work. WATCH is never BUY."""
    import scan
    from daytrader.playbook import buy_now_candidates, position_picks
    from daytrader.position import position_candidate
    if interval_min <= 0 or long_interval_min <= 0 or closed_sleep_min <= 0:
        raise ValueError('Scan intervals must be positive')
    config=notification_config()
    tracker=SignalTracker()
    daily_review=DailyReview(tracker)
    print('BUY tracker: delivered alerts tracked every 5 minutes; daily charts/reports at 17:00 Eastern in tracking/daily. Console-only alerts are not recorded.')
    def track(row, sent_at, short, message):
        if config['kind'] in ('ntfy','webhook'):
            try:tracker.record(row,sent_at,short,message)
            except Exception as exc:print(f'TRACKER RECORD ERROR: {exc}')
    def update_tracker(now):
        if tracker is not None:
            try:tracker.update(now)
            except Exception as exc:print(f'TRACKER UPDATE ERROR: {exc}')
        try:daily_review.run_due(now)
        except Exception as exc:print(f'DAILY REPORT ERROR (will retry): {exc}')
    want_short=alerts in ('both','short');want_long=alerts in ('both','long')
    print(f"Watcher started. Alerts via: {config['kind']} ({config.get('target')})")
    print(f"Signal profile: {SIGNAL_PROFILE}; short check {interval_min} min; long check {long_interval_min} min.")
    print(f"Early WATCH notices: {early_watch}. They are NOT buy signals. Ctrl-C to stop.")
    short_by_day={};long_last_alert={};watch_seen=set();last_long_check=None

    def refresh(results, now):
        return [attach_eligibility(r,now) if not r.get('error') and r.get('quality') else r for r in results]

    def save(results, now, stage):
        try:
            Path('logs').mkdir(exist_ok=True)
            with open(Path('logs')/f'scans-{now.date()}.jsonl','a',encoding='utf-8') as stream:
                stream.write(json.dumps({'as_of':str(now),'stage':stage,'profile':SIGNAL_PROFILE,
                                         'results':results},default=str,allow_nan=True)+'\n')
        except OSError as exc:print(f'Could not save diagnostic log: {exc}')
        for line in diagnostic_lines(results):print('  '+line)

    def notices(results, now, mode, bought=()):
        if not early_watch:return
        for result in results:
            for item in result.get('pattern_watches',[]):
                if item['mode']!=mode or item['ticker'] in bought:continue
                if canadian_only and not item['ticker'].endswith('.TO'):continue
                key=(str(now.date()),item['ticker'],mode,item['pattern_id'])
                if key in watch_seen:continue
                title,message=format_pattern_watch(item)
                if send_notification(title,message,config):
                    watch_seen.add(key)
                    print(f"{describe_now(now)} - WATCH ONLY: {item['ticker']}")

    while True:
        cycle_start=market_now()
        if not market_is_open(cycle_start):
            update_tracker(cycle_start)
            print(f"{describe_now(cycle_start)} - market closed, waiting.")
            if once:return
            time.sleep(closed_wait_seconds(cycle_start,closed_sleep_min));continue
        do_long=want_long and (last_long_check is None or
                 (cycle_start-last_long_check).total_seconds()>=long_interval_min*60)
        if not want_short and not do_long:
            update_tracker(cycle_start)
            if once:return
            time.sleep(max(1,min(interval_min*60,long_interval_min*60-(cycle_start-last_long_check).total_seconds())))
            continue
        try:
            # Don't request any daily downloads in the hourly pass.
            results=scan.run_scan(view='intraday' if want_short else 'position')['results']
            now=market_now();day=str(now.date());results=refresh(results,now)
            save(results,now,'hourly' if want_short else 'long')
            if want_short:
                already=short_by_day.setdefault(day,set())
                shorts=new_buy_signals(buy_now_candidates(results),already,canadian_only=canadian_only)
                for r in shorts:
                    title,message=format_categorized_alert([r],[],now)
                    if send_notification(title,message,config):
                        track(r,market_now(),True,message)
                        already.add(r['ticker'].upper())
                        print(f"{describe_now(now)} - ALERTED: {r['ticker']}(S)")
                    else:print(f"{r['ticker']}: delivery failed; cooldown not recorded. Will retry if still eligible.")
                notices(results,now,'buy-now',already)
                if not shorts:print(f"{describe_now(now)} - no fresh hourly signals.")
            if do_long:
                if want_short:
                    for r in results:
                        if not r.get('error') and r.get('quality'):
                            try:r['position']=position_candidate(r['ticker'])
                            except Exception as exc:print(f"Daily scan error {r['ticker']}: {exc}")
                    now=market_now();results=refresh(results,now);save(results,now,'long')
                last_long_check=now
                bought=set()
                for r in position_picks(results):
                    tk=r['ticker'].upper()
                    if canadian_only and not tk.endswith('.TO'):continue
                    last=long_last_alert.get(tk)
                    if last is not None and (now.date()-last).days<long_cooldown_days:continue
                    title,message=format_categorized_alert([],[r],now)
                    if send_notification(title,message,config):
                        track(r,market_now(),False,message)
                        long_last_alert[tk]=now.date();bought.add(tk)
                        print(f"{describe_now(now)} - ALERTED: {tk}(L)")
                    else:print(f"{tk}: delivery failed; cooldown not recorded.")
                notices(results,now,'position',bought)
        except Exception as exc:
            print(f"{describe_now(market_now())} - scan error (will retry): {exc}")
        update_tracker(market_now())
        if once:return
        # Count scan time in the interval, instead of adding it to every cycle.
        elapsed=(market_now()-cycle_start).total_seconds()
        time.sleep(max(1,interval_min*60-elapsed))


def main():
    parser = argparse.ArgumentParser(description="Always-running buy-signal watcher with phone alerts")
    parser.add_argument("--interval-min", type=int, default=1, help="Minutes between short-term checks")
    parser.add_argument("--long-interval-min", type=int, default=240, help="Minutes between long-term checks")
    parser.add_argument("--alerts", choices=["both", "short", "long"], default="both",
                        help="Which buy types to alert on")
    parser.add_argument("--canadian-only", action="store_true", help="Only alert on FX-free .TO signals")
    parser.add_argument("--closed-sleep-min", type=int, default=30, help="Sleep between checks when market closed")
    parser.add_argument("--no-early-watch", action="store_true", help="Disable non-buy pattern formation notices")
    args = parser.parse_args()
    try:
        run_watch(interval_min=args.interval_min, long_interval_min=args.long_interval_min,
                  alerts=args.alerts, canadian_only=args.canadian_only,
                  closed_sleep_min=args.closed_sleep_min, early_watch=not args.no_early_watch)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()
