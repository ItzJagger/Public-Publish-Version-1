# Trading Research & Signal Scanner

**Public Publish Version 1**

A Personal Project of Mine using my skills in AI from my time at University utilizing my certificate in Generative AI to vibe code a personal project. This is version 1, the first version I feel happy uploading to the public. Setup details not included as I do want to keep that end private for now until I am fully confident and have tested the app fully.

A Python stock-screening and decision-support project for manual traders. It combines market data, technical analysis and risk checks to identify potential opportunities across U.S. and Canadian stocks and ETFs.

## What it does

- Screens for short-term and multi-week trading setups.
- Evaluates trend, mean-reversion and inverse head-and-shoulders patterns.
- Distinguishes early pattern watches from confirmed buy signals.
- Calculates entry levels, position sizes, stop-limit fields, profit targets and sell-by deadlines.
- Checks trading costs, currency conversion, liquidity, data freshness and earnings dates.
- Tracks user-recorded holdings and their planned exits.
- Presents results through a command-line interface, a local web dashboard and optional phone notifications.

## How it works

The scanner retrieves market data through `yfinance`, calculates indicators with pandas and NumPy, and evaluates completed candles against configurable strategy rules. Candidate trades then pass through shared risk and eligibility checks before appearing as buy alerts.

The active profile uses five-minute candles for entry signals and fifteen-minute volatility for hourly risk levels. Longer-term analysis uses daily candles. Research scripts and synthetic tests support evaluation of the calculations and signal pipeline.

## Project structure

| Component | Purpose |
| --- | --- |
| `daytrader/` | Data processing, indicators, strategies, risk checks and alerts |
| `scan.py` | Watchlist scanning and decision summaries |
| `watch.py` | Scheduled scans and notifications |
| `server.py` | Local web dashboard and JSON API |
| `mobile/` | Companion mobile interface source |
| `backtest_report.py`, `research_active.py` | Historical research and diagnostic replays |
| `tests/`, `selftest.py` | Automated checks using synthetic fixtures |

## Project status

Under active development and private evaluation. This public snapshot showcases the source code; installation and deployment instructions are intentionally omitted. Personal holdings, notification settings and private trading logs are excluded.

The program does not place orders. Signals and historical research are not guarantees of future returns, and stop-limit orders may not fill during rapid price changes. All trading decisions and execution remain with the user.
