"""Build/rebuild a local daily report, charts and shadow learning review. No orders."""
import argparse
import datetime
from daytrader.clock import market_now
from daytrader.tracker import SignalTracker, stamp
from daytrader.daily_review import render_report, trading_day


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Trading date YYYY-MM-DD; default latest completed 17:00 ET report day')
    parser.add_argument('--refresh', action='store_true', help='Retrieve available price history first')
    parser.add_argument('--directory', default='tracking', help='Local tracker directory')
    args = parser.parse_args()
    now = stamp(market_now())
    if args.date:
        day = datetime.date.fromisoformat(args.date)
        if not trading_day(day):parser.error('No US/Canadian trading session on this date')
    else:
        day = now.date() if now.hour >= 17 else now.date()-datetime.timedelta(days=1)
        while not trading_day(day):day -= datetime.timedelta(days=1)
    if stamp(str(day)+' 17:00') > now:parser.error('Daily reports are available after 17:00 Eastern')
    tracker = SignalTracker(args.directory)
    if args.refresh:tracker.update(now,force=True,include_day=day)
    path = render_report(tracker,day,now)
    print('Open:',(path/'report.html').resolve())
    print('Text, CSV, JSON and charts saved in:',path.resolve())


if __name__ == '__main__':main()
