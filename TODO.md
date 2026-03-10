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
[ ] Fix extract_section lookahead to include Geo-Risiko
```

---

### 2. Geopolitical alert — verify `should_notify` filter passes `GEOPOLITICAL_ALERT`
**File:** `core/notifications.py`, `scheduler.py`
**Problem:** `notifications.send_alert("GEO", "GEOPOLITICAL_ALERT", ...)` calls `should_notify(signal)` internally. That method filters by signal type (STRONG_BUY, STRONG_SELL, etc.). `GEOPOLITICAL_ALERT` likely doesn't pass → alerts silently dropped.
**Fix:** Check the `should_notify` logic. Either add `GEOPOLITICAL_ALERT` to the allowed set, or call the email/webhook layer directly for geo alerts.
**Effort:** XS · **Impact:** High (geo alerts are the main safety feature)
```
[ ] Verify GEOPOLITICAL_ALERT bypasses signal-type filter in notifications
```

---

### 3. Priority re-analysis after high-severity geo event
**File:** `scheduler.py`
**Problem:** Step 7 of the original spec was partially implemented — the severity alert fires, but the watchlist is NOT re-queued for analysis after a ≥8 event.
**Fix:** In `run_geopolitical_scan()`, after sending the alert, call `pipeline.run_daily_cycle(force=True)` or individually queue each watchlist ticker.
**Note:** Add a cooldown (e.g. only re-trigger if last full scan was >2h ago) to avoid cascade.
**Effort:** S · **Impact:** High (main point of real-time geo monitoring)
```
[ ] Queue full watchlist re-analysis when geo severity >= 8
[ ] Add cooldown: skip re-trigger if last scan < 2h ago
```

---

### 4. Analysis detail page — render `geopolitical_context` and `geo_risk_score`
**File:** `templates/analysis_detail.html` (or equivalent stock detail template)
**Problem:** The two new DB columns are stored but never displayed anywhere except the dashboard heatmap chip.
**Fix:** Add a "Geopolitisches Risiko" section to the analysis detail view showing EXPOSURE/RICHTUNG/BEGRÜNDUNG and the Geo-Risiko score.
**Effort:** S · **Impact:** Medium
```
[ ] Add geo fields to analysis detail template
[ ] Show Geo-Risiko score next to Risk Score in history table
```

---

## Geopolitical System — Completion

### 5. Geo scan deduplication / delta detection
**File:** `core/database.py`, `scheduler.py`
**Problem:** Every 6h scan saves a full new row even if world events haven't changed. 120+ rows/month with 90% redundant content.
**Approach:** Before inserting, fetch the last scan and compare a hash or key phrases. Only save if materially different (or always save but add a `is_delta BOOLEAN` column).
**Effort:** M · **Impact:** Low (cosmetic/storage, not functional)
```
[ ] Add is_delta flag to geopolitical_events table
[ ] Hash-compare new scan vs last before saving full duplicate
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
[ ] Add feedparser to requirements.txt
[ ] Create clients/rss_client.py with keyword scanner
[ ] Add 15-min RSS check job to scheduler
[ ] Trigger run_geopolitical_scan() on keyword hit (with 1h cooldown)
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
[ ] In check_price_alerts, detect ±3% intraday move
[ ] On trigger, call swarm.analyze_single_stock(ticker) in background thread
[ ] Log breakout as anomaly in analysis result
[ ] Add intraday_trigger_pct setting in Settings
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
[ ] Add backup_db() method to Database class
[ ] Schedule daily backup job at 03:30 (after health_check)
[ ] Keep last 7 daily + 4 weekly backups, delete older
[ ] Log backup size + status to scheduler_log
```

### 18. Test coverage for new geo subsystem
**File:** `tests/test_agents.py`, `tests/test_database.py`
**Description:** The new geo functions have zero test coverage. Add unit tests for: `save_geopolitical_scan`, `get_latest_geopolitical_scan`, `extract_section` with `Geo-Risiko` in response, severity parsing regex.
**Effort:** S · **Impact:** Medium (prevents regressions)
```
[ ] Test save_geopolitical_scan with mock severity scores
[ ] Test get_latest_geopolitical_scan with >24h old record (expect None)
[ ] Test extract_section doesn't swallow Bull Case when Geo-Risiko present
[ ] Test run_geopolitical_scan severity alert threshold
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
