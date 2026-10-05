# Trading Research & Signal Scanner

**Public Publish Version 1**

A Python stock-screening and decision-support project for manual traders. It combines market data, technical analysis and risk checks to identify potential opportunities across U.S. and Canadian stocks and ETFs.

## What it does

- Screens for short-term and multi-week trading setups.
- Evaluates trend, mean-reversion and inverse head-and-shoulders patterns.
- Distinguishes early pattern watches from confirmed buy signals.
- Calculates entry levels, position sizes, stop-limit fields, profit targets and sell-by deadlines.
- Checks trading costs, currency conversion, liquidity, data freshness and earnings dates.
- Tracks user-recorded holdings and their planned exits.
- Records delivered BUY alerts and follows their simulated performance through the sell-by deadline.
- Saves a 5:00 p.m. Eastern report each trading day: readable HTML with price charts, text, CSV and structured JSON.
- Reviews cumulative outcomes and trains separate hourly and long-term shadow models once enough completed observations exist.
- Presents results through a command-line interface, a local web dashboard and optional phone notifications.

## How it works

The scanner retrieves market data through `yfinance`, calculates indicators with pandas and NumPy, and evaluates completed candles against configurable strategy rules. Candidate trades then pass through shared risk and eligibility checks before appearing as buy alerts.

The active profile uses five-minute candles for entry signals and fifteen-minute volatility for hourly risk levels. Longer-term analysis uses daily candles. Research scripts and synthetic tests support evaluation of the calculations and signal pipeline.

## Project structure

| Component | Purpose |
| --- | --- |
| `daytrader/` | Data processing, indicators, strategies, risk checks and alerts |
| `scan.py` | Watchlist scanning and decision summaries |
| `watch.py` | Scheduled scans, notifications and automatic performance tracking |
| `signal_report.py` | Original reports grouped by alert date |
| `daily_report.py` | Detailed session reports, charts and cumulative learning review |
| `server.py` | Local web dashboard and JSON API |
| `mobile/` | Companion mobile interface source |
| `backtest_report.py`, `research_active.py` | Historical research and diagnostic replays |
| `tests/`, `selftest.py` | Automated checks using synthetic fixtures |

## Project status

Under active development and private evaluation. This public snapshot showcases the source code; installation and deployment instructions are intentionally omitted. Personal holdings, notification settings and private trading logs are excluded.

Performance records persist locally across restarts and are excluded from version control. Reports distinguish price-level touches from actual fills and flag missing or ambiguous data. The entry proxy uses an eligible completed five-minute bar opening within 15 minutes after notification; it does not assume intrabar fills. Estimated net results describe holding to the latest or deadline price, not execution at a touched stop or target. Daily reports feed a local experimental learning model. It learns in shadow mode and records predictions for evaluation; it does not automatically alter alerts, strategies or risk limits.

The program does not place orders. Signals and historical research are not guarantees of future returns, and stop-limit orders may not fill during rapid price changes. All trading decisions and execution remain with the user.

## Daily review and learning

Reports record the price at the signal, proposed buy limit, simulated entry time
and price, quantity, holding-period finish, peak and floor, stop/target touches,
estimated costs and gross/net profit or loss. Session-close observations are
shown separately from the hourly deadline. Long-term plans appear on each
relevant session until their deadline. Their profit figures are cumulative,
not daily changes or portfolio returns. CAD and USD totals remain separate.

The running watcher writes reports at the first loop at or after 17:00 Eastern.
Exchange calendars handle holidays and early closes. Restarting catches up
missed report dates using available cached/provider history; missing data is
explicitly marked. No signals still produces a report. Reports and model history
remain private under `tracking/`, which is excluded from Git.

The learner reads the structured reports and retains only completed, complete-data
simulated entries. It deduplicates repeated signal snapshots and trains separate
fixed ridge-regression models for hourly and long-term net holding-period returns.
Inputs were captured when the alert was sent: risk distances, costs, time,
strategy, RSI, relative volume and available trend/volatility measures. Missing
inputs are handled from training data only. Older alerts without those inputs
still appear in reports but are excluded from model training.

Training requires at least 100 completed observations across 20 entry sessions
per mode, with at least 60 training and 30 later test examples after removing
training outcomes that overlap the test period. These are minimum engineering
thresholds, not proof of statistical reliability. The final model saves shadow
predictions for new alerts. Holdout results and later prospective prediction
errors are reported. No alerts are blocked, promoted or rewritten by the model.

Charts show the numerical price history used for measurement. Diagnostic notes
identify observations such as costs exceeding a price gain, a stop touch, or a
favourable move that later faded. They cannot establish the cause of a price move.
Model coefficients describe associations, not causal explanations. Repeated
holdout reviews, correlated stocks and selection of only delivered alerts limit
what these results can establish. The system does not learn about rejected setups.

Technical reference: [scikit-learn cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
The local regression implementation uses NumPy; no paid AI service is required.
