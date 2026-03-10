# Stockholm — TODO & Roadmap

Structured backlog for all pending work. Items are grouped by theme, tagged by effort/impact, and marked with status.

Status: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` rejected/won't do

---

## Priority Queue (implement next)

These are bugs or near-complete features — low effort, high value.

### 1. Fix `extract_section` regex — `Geo-Risiko` breaks Bull/Bear parsing
**File:** `engine/agents.py`
**Problem:** The `extract_section` lookahead stops at known headers. Since `Geo-Risiko` now appears between `Risk Score` and `Bull Case` in the Gemini output, the regex may swallow the `Geo-Risiko` line into the `Risk Score` section, or skip `Bull Case` entirely.
**Fix:** Add `Geo-Risiko` to the negative lookahead in `extract_section`:
```python
pattern = rf"{header}:\s*(.*?)(?=\n(?:Risk Score|Geo-Risiko|Bull Case|Bear Case|Quellen|Zusammenfassung):|$)"
```
**Effort:** XS (1 line) · **Impact:** High (silent data corruption)
```
[x] Fix extract_section lookahead to include Geo-Risiko
```

---

### 2. Geopolitical alert — verify `should_notify` filter passes `GEOPOLITICAL_ALERT`
**File:** `core/notifications.py`, `scheduler.py`
**Problem:** `notifications.send_alert("GEO", "GEOPOLITICAL_ALERT", ...)` calls `should_notify(signal)` internally. That method filters by signal type (STRONG_BUY, STRONG_SELL, etc.). `GEOPOLITICAL_ALERT` likely doesn't pass → alerts silently dropped.
**Fix:** Check the `should_notify` logic. Either add `GEOPOLITICAL_ALERT` to the allowed set, or call the email/webhook layer directly for geo alerts.
**Effort:** XS · **Impact:** High (geo alerts are the main safety feature)
```
[x] Verify GEOPOLITICAL_ALERT bypasses signal-type filter in notifications
```

---

### 3. Priority re-analysis after high-severity geo event
**File:** `scheduler.py`
**Problem:** Step 7 of the original spec was partially implemented — the severity alert fires, but the watchlist is NOT re-queued for analysis after a ≥8 event.
**Fix:** In `run_geopolitical_scan()`, after sending the alert, call `pipeline.run_daily_cycle(force=True)` or individually queue each watchlist ticker.
**Note:** Add a cooldown (e.g. only re-trigger if last full scan was >2h ago) to avoid cascade.
**Effort:** S · **Impact:** High (main point of real-time geo monitoring)
```
[x] Queue full watchlist re-analysis when geo severity >= 8
[x] Add cooldown: skip re-trigger if last scan < 2h ago
```

---

### 4. Analysis detail page — render `geopolitical_context` and `geo_risk_score`
**File:** `templates/analysis_detail.html` (or equivalent stock detail template)
**Problem:** The two new DB columns are stored but never displayed anywhere except the dashboard heatmap chip.
**Fix:** Add a "Geopolitisches Risiko" section to the analysis detail view showing EXPOSURE/RICHTUNG/BEGRÜNDUNG and the Geo-Risiko score.
**Effort:** S · **Impact:** Medium
```
[x] Add geo fields to analysis detail template
[x] Show Geo-Risiko score next to Risk Score in history table
```

---

## Geopolitical System — Completion

### 5. Geo scan deduplication / delta detection
**File:** `core/database.py`, `scheduler.py`
**Problem:** Every 6h scan saves a full new row even if world events haven't changed. 120+ rows/month with 90% redundant content.
**Approach:** Before inserting, fetch the last scan and compare a hash or key phrases. Only save if materially different (or always save but add a `is_delta BOOLEAN` column).
**Effort:** M · **Impact:** Low (cosmetic/storage, not functional)
```
[x] Add is_delta flag to geopolitical_events table
[x] Hash-compare new scan vs last before saving full duplicate
```

### 6. Geo scan history page
**File:** `app.py`, `templates/` (new file)
**Description:** Simple `/geo-history` page showing last 30 scans from `geopolitical_events` with timestamps, severity averages, and collapsible full text. Useful for reviewing how geo risk evolved over time.
**Effort:** M · **Impact:** Medium
```
[ ] Add GET /geo-history route
[ ] Create templates/geo_history.html
[ ] Add navigation link in base.html
```

### 7. Staleness-aware geo context invalidation
**File:** `engine/staleness_tracker.py` or `engine/pipeline.py`
**Problem:** A stock analyzed 3 days ago may have a geo context from before a major new event. The staleness tracker knows signal age but not geo-event recency.
**Approach:** When geo scan is newer than the last analysis for a ticker, flag that ticker's geo context as stale and prioritize it in the next cycle.
**Effort:** M · **Impact:** Medium
```
[ ] Compare geo scan timestamp vs analysis timestamp per ticker
[ ] Flag tickers whose geo context predates the latest geo scan
[ ] Elevate stale-geo tickers in scan priority queue
```

---

## Data & Intelligence

### 8. RSS-based real-time geo trigger (replaces pure polling)
**File:** `scheduler.py`, new `clients/rss_client.py`
**Description:** Monitor RSS feeds from Reuters, AP, BBC every 15 minutes (zero API cost). If headlines contain trigger keywords (war, sanctions, coup, OPEC, blockade, escalation), immediately fire a full Perplexity geo scan instead of waiting up to 6h.
**Effort:** M · **Impact:** High (brings geo latency from 6h to ~15min)
**Dependencies:** `feedparser` library (add to requirements.txt)
```
[x] Create clients/rss_client.py with keyword scanner (stdlib xml, no feedparser)
[x] Add 15-min RSS check job to scheduler
[x] Trigger run_geopolitical_scan() on keyword hit (with 1h cooldown)
```

### 9. Central bank / FOMC event tracker
**File:** new `engine/macro_tracker.py`, `scheduler.py`
**Description:** Parse FOMC meeting dates, ECB decisions, and rate announcements. Flag all portfolio positions before and after events. Inject rate-change context into Stage 3 prompt alongside geopolitical context.
**Effort:** L · **Impact:** High (rate decisions move every sector)
```
[ ] Scrape/hardcode FOMC/ECB calendar for current year
[ ] Add macro_events table to DB
[ ] Inject upcoming rate events into Stage 3 prompt (like geo_block)
[ ] Alert if rate decision within 48h and portfolio has rate-sensitive tickers
```

### 10. Short squeeze probability scorer
**File:** `engine/quant_screener.py` or new `engine/squeeze_detector.py`
**Description:** Combine short interest %, days-to-cover, float %, borrow rate, and recent price momentum into a 0-100 squeeze probability score. Add to Stage 1 quant output and display as anomaly when score > 70.
**Note:** The system already fetches short interest data (`/api/short-interest/{ticker}`).
**Effort:** M · **Impact:** Medium
```
[ ] Add squeeze_score calculation to quant screener
[ ] Register as anomaly type when score >= 70
[ ] Display in Stage 3 Bear Case context
```

---

## Dashboard & UI

### 11. Risk Score vs Geo-Risiko trend chart per ticker
**File:** `templates/stock_detail.html`, `app.py`
**Description:** On the stock detail page, show a small time-series chart of `risk_score` and `geo_risk_score` from `analysis_history`. Useful to see if geo risk is rising while overall risk is stable.
**Effort:** M · **Impact:** Medium
```
[ ] Add API endpoint returning risk_score + geo_risk_score history for ticker
[ ] Add trend chart to stock_detail.html using existing Chart.js setup
```

### 12. Watchlist tier → adaptive scan frequency
**File:** `engine/pipeline.py`, `core/database.py`
**Description:** The watchlist already has a `tier` concept. Implement differential scan frequency: Tier 1 tickers scanned every cycle, Tier 2 every 2nd cycle, Tier 3 weekly only. Reduces API cost without losing coverage on priority holdings.
**Effort:** M · **Impact:** Medium (cost reduction)
```
[ ] Add last_scanned_at to watchlist table
[ ] In pipeline, filter tickers by tier and cycle count
[ ] Document tier behavior in Settings UI
```

### 13. AI-generated weekly letter
**File:** `scheduler.py`, `clients/gemini_client.py`
**Description:** Sunday-evening Gemini call that reads the week's analyses, geo events, and learning stats, then writes a 1-page "weekly letter" covering: portfolio changes, market regime shift, top risks, and top opportunities. Sent via existing email infrastructure.
**Effort:** M · **Impact:** High (synthesis of all subsystems into human-readable narrative)
```
[ ] Add weekly_letter job to scheduler (Sunday ~19:00)
[ ] Build Gemini prompt from DB: last 7d analyses, geo scans, learning stats
[ ] Format as HTML email using existing email template
[ ] Add toggle in Settings (weekly_letter_enabled)
```

---

## Notifications & Webhooks

### 14. Two-way Telegram bot
**File:** new `clients/telegram_bot.py`, `scheduler.py`
**Description:** A Telegram bot that receives commands and queries the analysis system. Example commands: `/analyze AAPL`, `/watchlist`, `/geo`, `/toppicks`. Responses formatted as Telegram messages with key metrics.
**Dependencies:** `python-telegram-bot` library
**Note:** Outbound Telegram webhooks already exist — this adds inbound command handling.
**Effort:** L · **Impact:** High (on-the-go access without opening dashboard)
```
[ ] Add python-telegram-bot to requirements.txt
[ ] Create clients/telegram_bot.py with command handlers
[ ] Register /analyze, /watchlist, /geo, /toppicks commands
[ ] Add bot polling job to scheduler (or webhook mode)
[ ] Add bot token setting in Settings UI
```

### 15. Price breakout → auto-trigger analysis
**File:** `scheduler.py`, new `engine/price_alert_engine.py`
**Description:** Every 15 minutes (job already exists: `check_price_alerts`), detect if any watchlist ticker moved ±3% intraday. Auto-queue that ticker for immediate Stage 2+3 analysis instead of waiting for the next scheduled cycle. Store the breakout as an anomaly.
**Note:** The existing `price_alert_check` job runs every 15min but only sends static alerts.
**Effort:** M · **Impact:** High (catches earnings reactions, gap-ups, etc.)
```
[x] In check_price_alerts, detect ±3% intraday move
[x] On trigger, call swarm.analyze_single_stock(ticker) in background thread
[x] Log breakout as anomaly in analysis result
[ ] Add intraday_trigger_pct setting in Settings UI
```

---

## Infrastructure & Quality

### 16. Multi-currency portfolio support
**File:** `engine/portfolio_manager.py`, `core/database.py`
**Description:** Allow trades to be logged in EUR/GBP/CHF/SEK. Track FX impact on portfolio returns separately from stock performance. Show currency-adjusted P&L.
**Effort:** L · **Impact:** Medium (important for European users with mixed currency holdings)
```
[ ] Add currency column to portfolio_trades table
[ ] Add FX rate fetching (yfinance has EUR/USD etc.)
[ ] Separate FX P&L from stock P&L in portfolio summary
[ ] Show currency exposure in portfolio page
```

### 17. DB backup rotation
**File:** `scheduler.py`, `core/database.py`
**Description:** The DB backup exists (`investment_monitor.db.backup`) but is likely overwritten every time. Add scheduled rotation: keep daily backups for 7 days, weekly for 4 weeks.
**Effort:** S · **Impact:** Medium (data safety)
```
[x] Add backup_db() method to Database class (SQLite online backup API)
[x] Schedule daily backup job at 03:30 (after health_check)
[x] Keep last 7 daily + 4 weekly backups, delete older
```

### 18. Test coverage for new geo subsystem
**File:** `tests/test_agents.py`, `tests/test_database.py`
**Description:** The new geo functions have zero test coverage. Add unit tests for: `save_geopolitical_scan`, `get_latest_geopolitical_scan`, `extract_section` with `Geo-Risiko` in response, severity parsing regex.
**Effort:** S · **Impact:** Medium (prevents regressions)
```
[x] Test save_geopolitical_scan — id, severity_avg, NULL severity, is_delta dedup
[x] Test get_latest_geopolitical_scan — recent returns row, >24h returns None
[x] Test extract_section doesn't swallow Bull Case when Geo-Risiko present
[x] Test SCHWEREGRAD regex — multi-score, slash variant, threshold 8, no-match
```

### 19. Smarter alert deduplication (direction + score change gate)
**File:** `core/notifications.py`, `engine/alert_manager.py`
**Description:** Current dedup is hash-based (same signal type = skip). Missing: if AAPL fired STRONG_BUY on Monday, suppress re-alert Tuesday unless signal flips direction, risk score changes by >2 points, or new geopolitical context appeared since last alert. "Alert fatigue" kills usefulness faster than anything else.
**Effort:** S · **Impact:** High (daily usability)
```
[ ] Track last_alert_signal + last_alert_score per ticker in DB
[ ] Gate re-alert: only fire if direction changed OR |score_delta| >= 2 OR new geo event since last alert
[ ] Add alert cooldown hours setting in Settings UI
```

### 20. Geo-risk overlay badge on watchlist table
**File:** `templates/watchlist.html`, `app.py`
**Description:** The geopolitical scan already generates per-ticker exposure scores. Show a small colored badge (1–10 scale) directly in the watchlist table row, so you can immediately see which holdings are geo-exposed without clicking into each analysis. Data already in `analysis_history.geo_risk_score`.
**Effort:** S · **Impact:** Medium (daily scan speed)
```
[ ] Expose geo_risk_score in GET /api/watchlist response
[ ] Add geo badge column to watchlist.html table
[ ] Color-code: green <4, yellow 4-7, red >7
```

---

## Trading & Execution

### 21. Real broker integration (IBKR / Alpaca)
**File:** new `clients/broker_client.py`, `engine/order_manager.py`, `app.py`
**Description:** The entire pipeline exists to generate signals but can only paper trade. Adding real order execution via Interactive Brokers (ib_insync) or Alpaca API would make this system genuinely actionable. Gate behind a confirmation UI — signals propose, user confirms. Position sizing, stop losses, and risk guards are already implemented.
**Effort:** XL · **Impact:** Critical (the missing last step)
**Dependencies:** `ib_insync` or `alpaca-trade-api`
```
[ ] Add broker_type setting (paper / ibkr / alpaca)
[ ] Create clients/broker_client.py with place_order / cancel / get_positions
[ ] Confirmation step: signal card shows "Execute?" button, requires user click
[ ] Sync open broker positions into portfolio_trades table
[ ] Add POST /api/orders/execute endpoint
[ ] Show live P&L from broker in portfolio page (vs simulated paper P&L)
```

---

## Data & Intelligence (continued)

### 22. Macro dashboard — yield curve, VIX term structure, credit spreads
**File:** new `engine/macro_tracker.py`, `templates/macro.html`, `app.py`
**Description:** The system tracks equities and geopolitics but not the macro backbone. Yield curve shape (2y/10y spread), VIX term structure (VIX/VIX3M), put/call ratio, credit spreads (HYG vs LQD), DXY. All free via yfinance tickers. These are leading indicators for regime changes and would make `market_regime.py` much more powerful.
**Effort:** L · **Impact:** High (macro context for every signal)
```
[ ] Create engine/macro_tracker.py: fetch ^TNX, ^IRX, ^VIX, HYG, LQD, DXY via yfinance
[ ] Compute: 2y/10y spread (inverted = recession signal), VIX term structure slope, HYG/LQD credit spread
[ ] Store daily snapshot in new macro_snapshots table
[ ] Inject macro context into Stage 3 Gemini prompt (alongside geo_block)
[ ] Add /macro page with charts: yield curve, VIX term, credit spread over 90d
[ ] Add macro badge to dashboard (e.g., "Yield curve inverted — recession watch")
```

### 23. Earnings calendar — pre/post earnings logic
**File:** `engine/earnings_tracker.py` (extend), `engine/pipeline.py`
**Description:** `earnings_tracker.py` exists but doesn't drive the pipeline. Should: auto-flag tickers with earnings in next 5 days, show implied move (options IV) vs historical earnings move, suppress STRONG_BUY signals within 48h of earnings unless explicitly overridden. Post-earnings drift (stocks that gap tend to continue 3-5 days) is an underused pattern.
**Effort:** M · **Impact:** High (biggest single-day risk event for any position)
```
[ ] Pull earnings dates from yfinance calendar or earningswhispers.com scrape
[ ] Add earnings_date column to watchlist
[ ] In pipeline: warn Stage 3 when earnings within 5 days (inject as risk factor)
[ ] Suppress STRONG_BUY for earnings-imminent tickers unless risk_score < 5
[ ] Track post-earnings drift: after earnings, flag ticker for re-analysis at +1d, +3d, +5d
[ ] Show earnings countdown badge on watchlist and stock_detail
```

### 24. Alternative sentiment — Reddit / Google Trends
**File:** new `engine/sentiment_reddit.py`, `engine/sentiment_trends.py`
**Description:** The system has no retail sentiment layer. Reddit (WSB, r/investing, r/stocks via PRAW) and Google Trends (pytrends) are especially predictive for retail-heavy stocks. A spike in search volume or Reddit mentions 1-2 days before a price move is a known leading signal.
**Effort:** L · **Impact:** Medium (high alpha for small/mid cap, lower for blue chip)
**Dependencies:** `praw` (Reddit), `pytrends` (Google Trends) — both free
```
[ ] Create engine/sentiment_reddit.py: fetch top posts + comment counts for ticker via PRAW
[ ] Create engine/sentiment_trends.py: pytrends relative search volume for ticker (7d)
[ ] Add reddit_sentiment_score + trends_score to Stage 1 quant output
[ ] Inject high-reddit-activity into Stage 3 prompt as anomaly
[ ] Show sentiment badges on watchlist (retail interest: low/medium/high)
```

### 25. 13F institutional holdings tracker
**File:** `engine/institutional_tracker.py` (extend), `clients/sec_edgar_client.py`
**Description:** The `institutional_tracker.py` exists but 13F filings (quarterly, free from SEC EDGAR) are the real source for big money moves. Tracking when Berkshire, Bridgewater, Citadel, or Tiger Global add/drop positions is a high-signal data point. The existing `sec_edgar_client.py` already handles EDGAR — extend it.
**Effort:** M · **Impact:** High (smart money tracking)
```
[ ] Add get_13f_filings(cik) to sec_edgar_client.py
[ ] Track top 20 institutional filers (hardcoded CIKs for Berkshire, Bridgewater, etc.)
[ ] Store holding changes in institutional_holdings table
[ ] Show "Smart Money" badge on stock_detail when a top-20 filer added position this quarter
[ ] Weekly job: refresh 13F data after EDGAR quarterly deadline (Feb/May/Aug/Nov 15)
```

### 26. Dark pool / unusual block trade detection
**File:** new `engine/dark_pool_tracker.py`
**Description:** Finviz and stockanalysis.com expose unusual block trades (>$1M single transactions) without requiring a paid feed. A simple scraper checking these for watchlist tickers would add an institutional signal layer with zero API cost. Complement to the insider tracker.
**Effort:** M · **Impact:** Medium
```
[ ] Scrape finviz.com/quote.ashx block trade section for watchlist tickers
[ ] Flag if block trade > $5M on a single ticker in last 48h
[ ] Add as anomaly type in Stage 1 quant output
[ ] Show in stock_detail as "Unusual block activity detected"
```

---

## Dashboard & UI (continued)

### 27. Watchlist groups / tags
**File:** `core/database.py`, `templates/watchlist.html`, `app.py`
**Description:** A flat list of 50 tickers becomes unmanageable. Groups (ETFs, Tech, Energy, Speculative) with per-group risk budgets, aggregate geo-exposure summaries, and group-level signal summaries. Also enables tiered scan frequency per group.
**Effort:** M · **Impact:** High (daily usability at scale)
```
[ ] Add group column to watchlist table (with default "Uncategorized")
[ ] CRUD endpoints: POST /api/watchlist/groups, PATCH /api/watchlist/{ticker}/group
[ ] Render watchlist grouped with collapsible sections
[ ] Show per-group aggregate: avg risk score, avg geo risk, signal distribution
[ ] Add group filter dropdown on watchlist and discoveries pages
```

### 28. Confidence decay visualization
**File:** `templates/watchlist.html`, `templates/stock_detail.html`
**Description:** `staleness_tracker.py` tracks signal age but the dashboard doesn't show it visually. A simple age indicator (green → yellow → red as signal ages past verification window) would make it immediately obvious which analyses need refreshing without reading timestamps.
**Effort:** S · **Impact:** Medium (reduces stale-analysis risk)
```
[ ] Add staleness_days computed field to watchlist API response
[ ] Color-code last_analyzed badge: green <2d, yellow 2-7d, red >7d
[ ] Add "Stale" filter on watchlist to show only tickers not analyzed in >5d
[ ] Tooltip: "Last analyzed X days ago — click to re-run"
```

### 29. API key budget health card (dashboard widget)
**File:** `templates/dashboard.html`, `app.py`
**Description:** `budget_tracker.py` has all the data but it's buried in Settings. Surface it as a small dashboard card: remaining monthly budget (Perplexity/Gemini), daily burn rate, estimated days until exhausted, cost per analysis. Prevents surprise budget lockouts mid-cycle.
**Effort:** S · **Impact:** Medium (operational awareness)
```
[ ] Add GET /api/budget/status endpoint (daily spend, monthly cap, days remaining, avg cost/analysis)
[ ] Add budget health card to dashboard.html (next to scheduler status)
[ ] Warn visually when >80% monthly budget consumed
```

### 30. Watchlist import from broker CSV
**File:** `app.py`, `templates/watchlist.html`
**Description:** Tickers must be added manually one by one. One-click import of your actual holdings from a broker CSV export (IBKR, Degiro, Schwab — all export position CSVs) would be the natural onboarding path and encourages users to track their real portfolio.
**Effort:** S · **Impact:** Medium (onboarding)
```
[ ] POST /api/watchlist/import endpoint: accept CSV upload
[ ] Parse common broker CSV formats (IBKR: "Symbol", Degiro: "Produkt", Schwab: "Symbol")
[ ] Preview screen showing parsed tickers before import
[ ] Add "Import from CSV" button to watchlist page
```

### 31. PWA — mobile push notifications
**File:** `static/`, `app.py`
**Description:** The web dashboard could be a Progressive Web App (add `manifest.json` + service worker with push subscription). This enables native-style push notifications without requiring a Telegram bot setup. The notification content already exists (alerts, geo events, breakouts).
**Effort:** M · **Impact:** Medium (mobile usability)
```
[ ] Add manifest.json (name, icons, theme_color, display: standalone)
[ ] Create static/service-worker.js with push notification handler
[ ] POST /api/push/subscribe endpoint (stores push subscription in DB)
[ ] Trigger push on: STRONG_BUY/SELL alert, geo severity >= 8, intraday breakout
[ ] Add "Enable notifications" button to dashboard
```

---

## Infrastructure & Quality (continued)

### 32. Vectorized backtesting with realistic execution costs
**File:** `engine/backtest_engine.py` (rewrite)
**Description:** Current backtesting accuracy numbers are likely optimistic because bid-ask spread, slippage, and commissions are not modeled. A clean vectorized backtest (pandas rolling) with realistic fill assumptions (0.1% slippage, €1 commission) would make signal validation credible enough to actually trust for real capital allocation.
**Effort:** L · **Impact:** High (credibility of all accuracy metrics)
```
[ ] Add slippage_pct and commission_eur parameters to backtest_engine
[ ] Use vectorized pandas operations instead of row-by-row loops
[ ] Walk-forward split: 70% in-sample fit, 30% out-of-sample test
[ ] Report: net-of-costs Sharpe, Sortino, max drawdown, win rate, avg hold
[ ] Compare gross vs net-of-costs returns side by side in backtest.html
```

### 33. Two-factor authentication (TOTP)
**File:** `core/auth.py`, `templates/login.html`, `app.py`
**Description:** The system stores encrypted API keys and is exposed via HTTPS. Adding TOTP (Google Authenticator / Authy compatible) via `pyotp` would be a meaningful security upgrade. Single-user setup: generate secret on first enable, show QR code, verify TOTP on each login.
**Effort:** M · **Impact:** High (security — API keys are valuable)
**Dependencies:** `pyotp`, `qrcode` (both small, pure-Python)
```
[ ] Add pyotp + qrcode to requirements.txt
[ ] Add totp_secret column to users table
[ ] GET /settings/2fa/setup — generate secret, render QR code
[ ] POST /settings/2fa/enable — verify TOTP code, store secret
[ ] Inject TOTP verification step into login flow after password check
[ ] Allow 2FA bypass via backup codes (generate 8 one-time codes on setup)
```

### 34. CLI mode — headless analysis without web server
**File:** new `cli.py`
**Description:** `python cli.py analyze AAPL --strategy balanced` should run a full 3-stage analysis and print results to terminal without starting the web server. Useful for scripting, cron jobs outside APScheduler, CI testing, or quick ad-hoc checks.
**Effort:** M · **Impact:** Medium (developer/power user usability)
```
[ ] Create cli.py using argparse or click
[ ] Commands: analyze <TICKER>, scan (run full pipeline), geo (latest geo scan), watchlist (list)
[ ] Reuse existing engine/ and clients/ directly — no HTTP layer
[ ] Output: colored terminal table (rich library) or plain JSON with --json flag
[ ] Add to README as usage example
```

### 35. Correlation-aware position sizing
**File:** `engine/position_sizing.py`, `engine/correlation_analyzer.py`
**Description:** `correlation_analyzer.py` computes the matrix but position sizing doesn't use it. A Kelly-adjusted position size that accounts for portfolio correlation (if AAPL and MSFT are 0.85 correlated, the second position should be smaller) would meaningfully improve the risk management math vs the current flat max-position-pct approach.
**Effort:** M · **Impact:** Medium (portfolio risk quality)
```
[ ] In position_sizing.py, fetch current portfolio correlation from correlation_analyzer
[ ] Scale position size down by correlation factor: size *= (1 - avg_corr_with_portfolio)
[ ] Cap: minimum 50% of base size (prevent over-dilution for very diversified portfolios)
[ ] Show adjusted position size in stock_detail alongside raw Kelly size
[ ] Document the formula in Settings help text
```

### 36. Export — PDF report and CSV history
**File:** `app.py`, `engine/report_generator.py`
**Description:** Useful the moment this informs real investment decisions: PDF export of the weekly report, CSV export of analysis history (all signals, dates, outcomes), and basic cost-basis tracking (FIFO) for paper trading positions. The HTML report already exists — PDF conversion is one library call.
**Effort:** M · **Impact:** Medium (record-keeping, tax prep)
**Dependencies:** `weasyprint` or `reportlab` for PDF
```
[ ] GET /report/weekly/pdf — render existing HTML report to PDF via weasyprint
[ ] GET /api/analysis/export.csv — all analysis_history rows as CSV
[ ] GET /api/portfolio/export.csv — paper trades with entry/exit/P&L as CSV
[ ] Add FIFO cost-basis calculation to portfolio_manager.py
[ ] Add Export buttons to history.html and portfolio.html
```

---

## AI & Intelligence (advanced)

### 37. Natural language portfolio queries
**File:** new `engine/portfolio_qa.py`, `app.py`, `templates/analyze.html`
**Description:** "Which of my holdings are most exposed to a Chinese tariff escalation?" or "What's my portfolio's effective sector tilt?" answered by Gemini against live DB data. Not a chatbot — a structured query interface on the analyze page. The data is all there (geo exposure, sector weights, correlation, risk scores, analysis history). This would be the highest-leverage UX addition because it collapses 10 minutes of dashboard clicking into one question.
**Effort:** M · **Impact:** Very High (unique, differentiating feature)
```
[ ] Create engine/portfolio_qa.py: builds context from DB (current holdings, latest analyses, geo scores, sector weights) and passes to Gemini with user query
[ ] POST /api/portfolio/ask endpoint: accepts free-text question, returns Gemini answer
[ ] Add query input box to analyze.html ("Ask about your portfolio...")
[ ] Inject structured context: holdings JSON, sector distribution, top geo exposures, last geo scan summary
[ ] Rate-limit: 1 query per 30s, deducted from Gemini budget
[ ] Show source data used in answer (which tickers/analyses informed the response)
```

### 38. Continuous news NLP scoring (free, no LLM)
**File:** new `engine/nlp_scorer.py`, `scheduler.py`
**Description:** Currently news is only fetched per-ticker at analysis time via Perplexity. A parallel background job running FinBERT or VADER (both free, local, no API cost) on RSS headline streams would score sentiment for all watchlist tickers every hour without burning any API budget. The delta between the NLP score and the last Gemini signal score flags which tickers need re-analysis most urgently.
**Effort:** L · **Impact:** High (continuous monitoring, zero API cost)
**Dependencies:** `transformers` + `torch` for FinBERT, or `vaderSentiment` (pure Python, lighter)
```
[ ] Create engine/nlp_scorer.py with VADER sentiment scorer (zero dependency on GPU)
[ ] Fetch RSS feeds (Reuters, AP, MarketWatch) and match headlines to watchlist tickers by name/ticker mention
[ ] Score each mention: compound score -1.0 to +1.0
[ ] Store hourly snapshot in ticker_sentiment table (ticker, score, headline_count, timestamp)
[ ] In pipeline Stage 1: inject NLP score delta as anomaly when |delta_vs_last_analysis| > 0.3
[ ] Dashboard widget: "Sentiment Movers" — tickers with biggest NLP score shift in last 24h
[ ] Weekly job: compare NLP scores to actual signal outcomes for calibration
```

### 39. Geopolitical scenario stress testing
**File:** new `engine/geo_scenario.py`, `templates/portfolio.html`, `app.py`
**Description:** Distinct from the generic `scenario_analyzer.py`: predefine 5-6 named geo scenarios (China blockades Taiwan, OPEC production cut -20%, Russia expands conflict, US-Iran escalation, EU energy crisis) with estimated sector impact vectors. When the geo scanner detects a relevant event, automatically run the matching scenario against the current portfolio and show estimated P&L impact. Scenario definitions are static config — the hard part (correlations, sector exposures) already exists.
**Effort:** M · **Impact:** High (most useful during actual crises, which is when it matters most)
```
[ ] Create engine/geo_scenario.py with 6 hardcoded scenarios as dicts:
    { name, keywords, sector_impacts: {energy: +0.15, tech: -0.10, ...}, historical_analog }
[ ] Scenario runner: cross with portfolio sector weights → estimated portfolio impact %
[ ] Auto-trigger: when geo scan severity >= 8, find matching scenario by keyword overlap, run it
[ ] Show scenario result card on dashboard when triggered: "Taiwan scenario match — estimated portfolio impact: -4.2%"
[ ] Manual trigger: POST /api/scenarios/run?name=taiwan_blockade
[ ] Add /scenarios page showing all presets with portfolio impact preview
```

### 40. Pairs trading / statistical arbitrage
**File:** new `engine/pairs_trader.py`, `app.py`
**Description:** `correlation_analyzer.py` computes the matrix but doesn't look for cointegration. Detecting pairs that are historically cointegrated (XOM/CVX, AAPL/MSFT, GLD/GDX, etc.) and monitoring the spread for mean-reversion entry is a natural zero-cost-data extension of the existing correlation work. When the spread exceeds 2σ, flag as a long/short opportunity.
**Effort:** L · **Impact:** Medium (new strategy class, complements directional signals)
**Dependencies:** `statsmodels` (for Engle-Granger cointegration test — likely already installed or trivial to add)
```
[ ] Create engine/pairs_trader.py: run Engle-Granger cointegration test on all watchlist pairs
[ ] Filter: only pairs with p-value < 0.05 and >90d of shared history
[ ] Monitor spread z-score daily: flag when |z| > 2.0
[ ] Signal: z > +2 = short spread (long B, short A), z < -2 = long spread (long A, short B)
[ ] Store cointegrated pairs + current z-score in pairs_signals table
[ ] Add /pairs page: show active pairs with spread chart, current z-score, entry/exit levels
[ ] Weekly retest: recheck cointegration (pairs break down over time)
```

### 41. Self-hosted LLM fallback (Ollama)
**File:** new `clients/ollama_client.py`, `clients/gemini_client.py` (extend)
**Description:** When monthly Gemini/Perplexity budget is exhausted, automatically fall back to a local LLM via Ollama (llama3, mistral) for Stage 3 synthesis. Quality is lower but "budget-resilient degraded analysis" is far better than "no analysis from the 15th of the month onwards." The budget tracker already detects exhaustion — it just has no fallback. Increasingly practical as local models improve.
**Effort:** M · **Impact:** High (eliminates hard budget cutoffs mid-month)
**Dependencies:** `ollama` Python library (calls local Ollama daemon — user installs Ollama separately)
```
[ ] Create clients/ollama_client.py: generate(prompt, model='llama3') via Ollama REST API (localhost:11434)
[ ] In gemini_client.py: when monthly budget exhausted, fall back to ollama_client
[ ] Add ollama_enabled and ollama_model settings
[ ] Label fallback analyses in DB: source='ollama' vs source='gemini'
[ ] Show "(local model)" badge on analysis cards generated via Ollama
[ ] Health check: ping Ollama at startup, warn in settings if not available
```

### 42. Programmatic API access (personal API key)
**File:** `core/auth.py`, `app.py`, `templates/settings.html`
**Description:** The dashboard has REST endpoints but they're session-auth only. A personal API key system (generated in Settings, bearer token on requests) would allow integrating Stockholm into external tools — Notion automations, Obsidian daily notes, custom scripts, Raycast plugins, n8n workflows. Costs almost nothing to implement and completely changes the integration story.
**Effort:** S · **Impact:** High (ecosystem / integration potential)
```
[ ] Add api_keys table (key_hash, label, created_at, last_used_at, scopes)
[ ] POST /api/keys/generate endpoint (returns key once, then stores hash only)
[ ] GET /api/keys → list user's keys (label + last_used)
[ ] DELETE /api/keys/{id} → revoke
[ ] Middleware: accept Authorization: Bearer <key> header on all /api/* endpoints
[ ] Scope system: read-only vs full access (start with read-only only)
[ ] Show API key management section in settings.html with copy button
[ ] Document example: curl /api/watchlist -H "Authorization: Bearer sk_..."
```

---

## Portfolio Analytics (advanced)

### 43. Dividend and corporate action ledger
**File:** `engine/portfolio_manager.py`, `core/database.py`
**Description:** Splits, dividends, mergers, and spin-offs are currently not tracked. A position opened at $150 pre-split is worth different things pre/post 3:1 split — the paper trading P&L and cost basis math breaks silently. This is a correctness issue more than a feature, and becomes critical the moment real money is involved.
**Effort:** M · **Impact:** High (correctness — silent P&L errors)
```
[ ] Add corporate_actions table: ticker, action_type (split/dividend/merger), date, factor/amount
[ ] Fetch splits from yfinance Ticker.splits and dividends from Ticker.dividends
[ ] Weekly job: check for new corporate actions on all watchlist tickers
[ ] Retroactively adjust cost basis in portfolio_trades for pre-split entries
[ ] Add dividend income tracking: when ex-date passes, credit dividend to paper portfolio cash
[ ] Show corporate action timeline on stock_detail.html
[ ] Alert via webhook when a watchlist ticker announces a split or special dividend
```

---

## Completed

```
[x] Geopolitical scan (get_geopolitical_scan) — perplexity_client.py
[x] Per-ticker exposure assessment (get_ticker_geopolitical_exposure) — perplexity_client.py
[x] geopolitical_events table + save/get methods — database.py
[x] geopolitical_context + geo_risk_score columns in analysis_history — database.py
[x] Stage 2: geo exposure injection — agents.py
[x] Stage 3: geo_block in Gemini prompt + Geo-Risiko parsing — agents.py
[x] 6-hour geopolitical_scan scheduler job — scheduler.py
[x] High-severity (>=8) email/webhook alert — scheduler.py
[x] Geopolitisches Radar card + exposure heatmap — dashboard.html
[x] GET /api/geopolitical + GET /api/geopolitical/exposure — app.py
[x] README rewrite — README.md
```

---

## Won't Do (documented for future reference)

```
[-] CrewAI multi-agent rewrite — current single-swarm approach is simpler, cheaper, and more debuggable
[-] Real-time WebSocket price feed — yfinance polling is sufficient; adds complexity without meaningful gain
[-] Options chain deep analysis — data quality from free sources is unreliable; high false-signal risk
```
