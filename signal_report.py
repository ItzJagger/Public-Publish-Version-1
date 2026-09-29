"""View locally recorded BUY-alert outcomes. Does not send alerts or place orders."""
import argparse
from daytrader.clock import market_now
from daytrader.tracker import SignalTracker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',help='Alert date YYYY-MM-DD in Eastern time; defaults to today')
    parser.add_argument('--refresh',action='store_true',help='Fetch available historical bars before reporting')
    args=parser.parse_args()
    import datetime
    date=datetime.date.fromisoformat(args.date) if args.date else market_now().date()
    tracker=SignalTracker()
    if args.refresh:tracker.update(market_now(),force=True)
    else:tracker.write_reports()
    path=tracker.directory/'reports'/f'{date}.txt'
    print(path.read_text(encoding='utf-8') if path.exists() else f'No recorded BUY alerts for {date}. Tracking begins after installing this update.')
    if path.exists():print('Detailed CSV:',path.with_suffix('.csv'))

if __name__=='__main__':main()
