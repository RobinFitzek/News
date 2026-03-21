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
[x] Add GET /geo-history route
[x] Create templates/geo_history.html
[x] Add navigation link in base.html
```

### 7. Staleness-aware geo context invalidation
**File:** `engine/staleness_tracker.py` or `engine/pipeline.py`
**Problem:** A stock analyzed 3 days ago may have a geo context from before a major new event. The staleness tracker knows signal age but not geo-event recency.
**Approach:** When geo scan is newer than the last analysis for a ticker, flag that ticker's geo context as stale and prioritize it in the next cycle.
**Effort:** M · **Impact:** Medium
```
[x] Compare geo scan timestamp vs analysis timestamp per ticker
[x] Flag tickers whose geo context predates the latest geo scan
[x] Elevate stale-geo tickers in scan priority queue
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
[x] Scrape/hardcode FOMC/ECB calendar for current year
[x] Add macro_events table to DB
[x] Inject upcoming rate events into Stage 3 prompt (like geo_block)
[x] Alert if rate decision within 48h and portfolio has rate-sensitive tickers
```

### 10. Short squeeze probability scorer
**File:** `engine/quant_screener.py` or new `engine/squeeze_detector.py`
**Description:** Combine short interest %, days-to-cover, float %, borrow rate, and recent price momentum into a 0-100 squeeze probability score. Add to Stage 1 quant output and display as anomaly when score > 70.
**Note:** The system already fetches short interest data (`/api/short-interest/{ticker}`).
**Effort:** M · **Impact:** Medium
```
[x] Add squeeze_score calculation to quant screener
[x] Register as anomaly type when score >= 70
[x] Display in Stage 3 Bear Case context
```

---

## Dashboard & UI

### 11. Risk Score vs Geo-Risiko trend chart per ticker
**File:** `templates/stock_detail.html`, `app.py`
**Description:** On the stock detail page, show a small time-series chart of `risk_score` and `geo_risk_score` from `analysis_history`. Useful to see if geo risk is rising while overall risk is stable.
**Effort:** M · **Impact:** Medium
```
[x] Add API endpoint returning risk_score + geo_risk_score history for ticker
[x] Add trend chart to stock_detail.html using existing Chart.js setup
```

### 12. Watchlist tier → adaptive scan frequency
**File:** `engine/pipeline.py`, `core/database.py`
**Description:** The watchlist already has a `tier` concept. Implement differential scan frequency: Tier 1 tickers scanned every cycle, Tier 2 every 2nd cycle, Tier 3 weekly only. Reduces API cost without losing coverage on priority holdings.
**Effort:** M · **Impact:** Medium (cost reduction)
```
[x] Add last_scanned_at to watchlist table
[x] In pipeline, filter tickers by tier and cycle count
[ ] Document tier behavior in Settings UI
```

### 13. AI-generated weekly letter
**File:** `scheduler.py`, `clients/gemini_client.py`
**Description:** Sunday-evening Gemini call that reads the week's analyses, geo events, and learning stats, then writes a 1-page "weekly letter" covering: portfolio changes, market regime shift, top risks, and top opportunities. Sent via existing email infrastructure.
**Effort:** M · **Impact:** High (synthesis of all subsystems into human-readable narrative)
```
[x] Add weekly_letter job to scheduler (Sunday ~19:00)
[x] Build Gemini prompt from DB: last 7d analyses, geo scans, learning stats
[x] Format as HTML email using existing email template
[x] Add toggle in Settings (weekly_letter_enabled)
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
[x] Add currency column to portfolio_trades table
[x] Add FX rate fetching (yfinance has EUR/USD etc.)
[x] Separate FX P&L from stock P&L in portfolio summary
[x] Show currency exposure in portfolio page
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
[x] Expose geo_risk_score in GET /api/watchlist response
[x] Add geo badge column to watchlist.html table
[x] Color-code: green <4, yellow 4-7, red >7
```

---

## Trading & Execution

### 21. Auto-Trading Integration (paper → real broker)

**Files:**
- `engine/auto_paper_trader.py` (extend)
- new `engine/order_manager.py`
- new `clients/broker_client.py`
- `app.py` (new routes)
- `core/config.py` (new defaults)
- `templates/settings.html` (new Auto-Trade section)
- `templates/paper_trading.html` (new auto-trade dashboard)
- `templates/dashboard.html` (signal card execute button)

**Description:**
The `AutoPaperTrader` (`engine/auto_paper_trader.py`) already silently enters/exits trades on STRONG signals with hardcoded parameters (+8% TP, -4% SL, 30-day timeout). It runs with no UI and no user feedback. This item makes it a first-class, optional feature with: configurable risk parameters, a trust gate that must be earned before real execution, an optional confirmation step, broker abstraction (Alpaca or IBKR), and live position sync. Feature is OFF by default and can be used purely in paper mode indefinitely.

**Effort:** XL · **Impact:** Critical (the missing last step)

---

#### Phase 1 — Make existing AutoPaperTrader configurable (S effort)

Current implementation has hardcoded TP (8%), SL (4%), max_days (30), signal threshold.
These must come from DB settings.

**Settings to add to `core/config.py` DEFAULT_SETTINGS:**
```python
"auto_trade_enabled": False,             # master switch
"auto_trade_signal_filter": "STRONG",    # "STRONG" = STRONG_BUY/SELL only, "ALL" = BUY/SELL too
"auto_trade_take_profit_pct": 8.0,       # % gain to auto-close long
"auto_trade_stop_loss_pct": 4.0,         # % loss to auto-close (positive number)
"auto_trade_max_days_open": 30,          # force-close after N days
"auto_trade_position_size_pct": 5.0,     # % of paper portfolio per trade
"auto_trade_max_open_positions": 10,     # cap concurrent positions
"auto_trade_require_confirm": True,      # user must click confirm before entry
"auto_trade_mode": "paper",              # "paper" | "alpaca" | "ibkr"
"auto_trade_min_trust_trades": 20,       # closed trades needed before live mode unlocks
"auto_trade_min_trust_win_rate": 55.0,   # min win-rate % needed before live mode unlocks
```

**Changes to `engine/auto_paper_trader.py`:**
```
[x] _init_table, process_new_signals, check_open_positions, get_performance_summary
[ ] Read TP/SL/max_days/signal_filter/max_positions from db.get_setting() instead of hardcoded
[ ] Add position_size_pct support: shares = (portfolio_value * pos_pct) / entry_price
[ ] Add max_open_positions guard: skip entry if open count >= limit
[ ] Add dedup guard: skip entry if ticker already has an open auto-paper trade
[ ] Expose get_trade_log(limit=50) → list of closed trades with all fields
[ ] Expose manual_close(trade_id) → force-close a specific open position at current price
```

**Scheduler wiring:**
```
[ ] Call auto_paper_trader.process_new_signals() at end of every main scan
[ ] Call auto_paper_trader.check_open_positions() on every price-alert tick (15-min)
```

---

#### Phase 2 — Settings UI (S effort)

Add **Auto-Trade** section to `templates/settings.html` in a new
`<form action="/settings/save-auto-trade">` block.

**UI layout:**
```
[ ] Master toggle: "Enable Auto-Trading" (checkbox → auto_trade_enabled)
    ⚠ Warning banner when enabled: "This enters trades automatically.
      Use paper mode until you trust the signals."

[ ] Mode selector (radio):
    ○ Paper only   ← safe, default
    ○ Alpaca       ← locked with 🔒 icon until trust gate met
    ○ IBKR         ← locked with 🔒 icon until trust gate met

[ ] Signal filter (radio):
    ○ STRONG only (STRONG_BUY / STRONG_SELL)
    ○ All signals  (BUY / SELL / STRONG_*)

[ ] Risk parameters (number inputs, collapsed when disabled):
    Take-profit %     (default 8,  range 1–50)
    Stop-loss %       (default 4,  range 1–25)
    Max days open     (default 30, range 1–90)
    Position size %   (default 5,  range 1–20)
    Max open positions (default 10, range 1–50)

[ ] Confirmation toggle:
    "Require confirmation before each trade" (→ auto_trade_require_confirm)
    Sub-label: "Sends Telegram/email with Approve/Skip. Expires in 5 min."

[ ] Trust gate progress bar (JS-loaded via GET /api/auto-trade/trust-gate):
    ████████░░  18 / 20 trades  ·  Win rate: 58.3% ✓
    "Live mode unlocks at 20 trades with ≥55% win rate"
    🔒 red when not met, 🔓 green when met

[ ] Broker credentials (hidden unless mode = alpaca/ibkr):
    Alpaca:  API Key (pw field), API Secret (pw field),
             Base URL (default https://paper-api.alpaca.markets)
    IBKR:    TWS Host (default 127.0.0.1), Port (7497 paper/7496 live),
             Client ID (default 1)

[ ] Save + status message
```

**New routes:**
```
[ ] POST /settings/save-auto-trade   — save all auto_trade_* settings + broker creds
[ ] GET  /api/auto-trade/trust-gate  — {closed, win_rate, trusted, needed_trades, needed_win_rate}
```

---

#### Phase 3 — Paper Trading Page UI (M effort)

Extend `templates/paper_trading.html` with a dedicated auto-trade section.

**Auto-Trade Status Card (shown only if auto_trade_enabled=True):**
```
┌──────────────────────────────────────────────────────────────┐
│  ⚡ Auto-Trade  [MODE: PAPER]   [● ENABLED  toggle]          │
├──────────────────────────────────────────────────────────────┤
│  Win Rate: 61.2%  ·  Avg PnL: +2.4%  ·  Total: +18.7%      │
│  Trust Gate: ✅ UNLOCKED  (23 trades · 61% win rate)         │
│  Open: 4 / 10 positions                                       │
└──────────────────────────────────────────────────────────────┘
```

**Open Auto-Positions Table:**
```
Ticker  Dir   Entry Date  Entry $  Current $  P&L%   Days  [×]
AAPL    LONG  2026-03-08  182.40   189.20     +3.7%  3d    [×]
NVDA    LONG  2026-03-07  892.10   918.00     +2.9%  4d    [×]
```
- `[×]` → POST /api/auto-trade/close/<id> (force-close at current price)
- Row background green/red by P&L sign
- Faint TP line (+8%) and SL line (-4%) as visual reference in P&L column

**Auto-Trade Log Table (last 20 closed, paginated):**
```
Date        Ticker  Dir   Entry $  Exit $   P&L%   Closed By
2026-03-10  TSLA    LONG  244.00   263.52   +8.0%  take_profit ✅
2026-03-09  AMZN    LONG  188.00   180.48   -4.0%  stop_loss   ❌
2026-03-08  MSFT    LONG  412.00   423.11   +2.7%  time_limit  ⏰
```
- Color: green for take_profit, red for stop_loss, gray for time_limit

**New API endpoints:**
```
[ ] GET  /api/auto-trade/status           — {enabled, mode, open_positions, performance, trust_gate}
[ ] GET  /api/auto-trade/positions        — list of open auto_paper_trades with live P&L
[ ] GET  /api/auto-trade/log?page=1       — paginated closed trade history
[ ] POST /api/auto-trade/close/<id>       — force-close specific position
[ ] POST /api/auto-trade/toggle           — flip enabled/disabled master switch
[ ] GET  /api/auto-trade/pending-confirm  — list pending confirmation requests
[ ] POST /api/auto-trade/confirm/<token>  — approve a pending trade (executes entry)
[ ] POST /api/auto-trade/skip/<token>     — skip/decline a pending trade
```

---

#### Phase 4 — Signal Card Confirmation Flow (M effort)

When `auto_trade_require_confirm=True`, signals create a pending confirmation request
instead of executing immediately. User approves via Telegram or email link.

**New DB table `auto_trade_pending`:**
```sql
CREATE TABLE auto_trade_pending (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token       TEXT UNIQUE NOT NULL,      -- 32-char random hex, used in confirm URLs
    analysis_id INTEGER NOT NULL,
    ticker      TEXT NOT NULL,
    direction   TEXT NOT NULL,             -- LONG / SHORT
    signal      TEXT NOT NULL,
    score       INTEGER,
    proposed_entry_price REAL,
    proposed_shares      REAL,
    proposed_size_usd    REAL,
    risk_tp_price        REAL,             -- entry * (1 + tp_pct/100)
    risk_sl_price        REAL,             -- entry * (1 - sl_pct/100)
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,             -- created_at + 5 minutes
    status      TEXT DEFAULT 'pending',    -- pending | approved | skipped | expired
    decided_at  TEXT
)
```

**Notification format (Telegram + email):**
```
⚡ Auto-Trade Proposal: AAPL LONG
Signal: STRONG_BUY  ·  Score: 82/100
Entry: $182.40  |  Size: $913 (5% of portfolio)
Target: $197.00 (+8%)  ·  Stop: $175.10 (-4%)
Risk gate: ✅ All clear
────────────────────────────────
[✅ Approve]   [❌ Skip]
Expires in 5 minutes
```
- Telegram: inline keyboard buttons → POST /api/auto-trade/confirm/<token>
- Email: itsdangerous TimestampSigner links (5-min max_age)

**Dashboard signal card UI:**
```
┌─ AAPL  STRONG BUY  82 ─────────────────────────────┐
│  (summary…)                                          │
│                                                      │
│  [📊 View]   [⚡ Auto-Execute →]                     │
└──────────────────────────────────────────────────────┘
```
- Button states: idle → "⏳ Pending…" → "✅ Queued" or "❌ Skipped"
- If require_confirm=False + mode=paper: executes inline, shows "✅ Auto-executed"

**Changes:**
```
[ ] Add auto_trade_pending table (init in AutoPaperTrader._init_table)
[ ] Add create_pending_confirmation(analysis_id, ticker, ...) method
[ ] Add confirm_trade(token) → execute entry, mark approved
[ ] POST /api/auto-trade/propose endpoint (creates pending + fires notification)
[ ] Telegram bot: handle callback_query for Approve/Skip inline buttons
[ ] Email: render confirm/skip signed URLs in notification template
[ ] Dashboard JS: Auto-Execute button state machine
[ ] Scheduler job every 5 min: mark expired where expires_at < now()
```

---

#### Phase 5 — Risk Guard Integration (M effort)

Before ANY auto-trade entry (paper or real), run a fail-fast gate sequence.

**Gate checks (in order):**

1. **Global loss gate** — `portfolio_manager.get_risk_gate_status()`
   If triggered → block all entries, log, send ≤1 notification per cooldown

2. **Position concentration** — `(proposed_size_usd / portfolio_value) * 100 > max_position_pct`?
   If yes → shrink to fit, or skip if result below min viable size

3. **Sector concentration** — adding this ticker pushes sector > `portfolio_max_sector_pct`?
   Fetch sector via yfinance info['sector'], compare current weights. If yes → skip

4. **Duplicate position** — ticker already open in auto_paper_trades?
   If yes → skip

5. **Signal age** — triggering analysis older than 24h?
   If yes → skip (stale signal)

**Gate result surfaced in confirmation notification:**
```
Risk gate: ✅ All clear
  · Global loss: -1.2% (limit -10%) ✅
  · AAPL position: 4.8% (limit 10%) ✅
  · Tech sector: 22% (limit 30%) ✅
```
Or blocked:
```
Risk gate: ❌ BLOCKED — Tech sector 31% > limit 30%
```

**Changes:**
```
[ ] Add _run_risk_gate(ticker, proposed_size_usd) → {allowed, reason, checks[]}
[ ] Call gate before every entry in process_new_signals() and confirm_trade()
[ ] Extend auto_paper_trades table: add blocked_reason TEXT column
[ ] Log blocked entries to auto_paper_trades with status='blocked'
[ ] Inject gate summary into confirmation notification
[ ] GET /api/auto-trade/risk-gate-status — live gate check result for UI
```

---

#### Phase 6 — Real Broker Execution (XL effort — ⚠️ Claude Opus)

Thin abstraction over brokers. Paper mode uses AutoPaperTrader. Real modes route through
OrderManager → BrokerClient.

**`clients/broker_client.py`:**
```python
class BrokerClient(ABC):
    def place_order(ticker, qty, side, order_type='market') -> dict
    def cancel_order(order_id) -> bool
    def get_positions() -> list[dict]
    def get_account() -> dict       # cash, buying_power, equity

class AlpacaBrokerClient(BrokerClient)   # pure requests, no new dep needed
class IBKRBrokerClient(BrokerClient)     # uses ib_insync
class PaperBrokerClient(BrokerClient)    # delegates to auto_paper_trader
```

**`engine/order_manager.py`:**
```python
class OrderManager:
    execute_entry(ticker, direction, size_usd) -> OrderResult
        # 1. run _run_risk_gate()
        # 2. shares = size_usd / current_price
        # 3. broker_client.place_order()
        # 4. write to portfolio_trades (origin='auto')
        # 5. send execution notification

    execute_exit(trade_id, reason) -> OrderResult
        # 1. fetch open trade
        # 2. broker_client.place_order(side='sell'/'cover')
        # 3. update portfolio_trades: exit_price, exit_date, pnl
        # 4. send close notification

    sync_broker_positions()
        # pull open positions from broker
        # upsert into portfolio_trades (origin='broker-sync')
        # mark closed in portfolio_trades if absent at broker
```

**New app.py routes:**
```
[x] POST /api/orders/execute            — execute confirmed trade ({token} or {ticker, direction, size_usd})
[x] POST /api/orders/close/<trade_id>   — close position at broker
[x] GET  /api/orders/status/<order_id>  — poll fill status
[x] GET  /api/broker/account            — equity, buying power, cash
[x] GET  /api/broker/positions          — live broker positions
[x] POST /api/broker/sync               — manual position sync trigger
```

**Portfolio page additions:**
```
[x] Toggle: "Live P&L" (broker) vs "Paper P&L" columns
[x] Origin badge per row: 🤖 auto | ✋ manual | 🔄 broker-sync
[x] Header bar: "🟢 Alpaca  ·  $12,340 buying power  ·  synced 2m ago"
```

**Scheduler:**
```
[x] Broker position sync every 5 min (market hours only, uses _is_market_open())
[x] Broker P&L snapshot every 15 min → store in paper_snapshots (add broker_value col)
```

**Trust gate enforcement:**
```
[x] Block mode switch to alpaca/ibkr in settings if trust gate not met
[x] UI: "2 more paper trades needed (18/20 · 61% win rate)"
[x] Admin override: auto_trade_trust_override setting (for dev testing)
```

**New dependencies (optional, guarded imports):**
```
alpaca-trade-api>=3.0.0   # only imported when mode=alpaca
ib_insync>=0.9.86          # only imported when mode=ibkr
```

---

#### Phase 7 — Observability & Performance Tracking (S effort)

```
[ ] Auto-trade equity curve: separate orange line on paper_trading.html equity chart
[ ] Weekly AI Letter digest: "Auto-trader: 3 opened, 2 closed, avg +1.8%"
[ ] Dashboard mini-widget: "⚡ Auto this week: 2 open · 1 closed +5.2%"
[ ] GET /api/export/auto-trades — CSV (id, ticker, direction, entry, exit, pnl, reason, date)
[ ] Telegram bot /autostatus command: "4 open · 23 closed · 62% win · +18.3% total"
```

---

#### Full checklist:

**Phase 1 — Configurable core:**
```
[x] Add 10 auto_trade_* settings to core/config.py DEFAULT_SETTINGS
[x] Update auto_paper_trader.py: read all params from DB
[x] Add position_size_pct calculation + max_positions guard
[x] Add dedup guard (skip if ticker already open)
[x] Add get_trade_log() and manual_close() methods
[x] Wire process_new_signals() + check_open_positions() into scheduler
```

**Phase 2 — Settings UI:**
```
[x] Auto-Trade section in settings.html (all controls above)
[x] Mode selector with trust gate lock on broker options
[x] POST /settings/save (extended with auto_trade_* params)
[x] GET /api/auto-trade/trust-gate
```

**Phase 3 — Paper trading page:**
```
[x] Auto-trade status card (mode, trust gate, open count, perf)
[x] Open auto-positions table with [×] close button
[x] Closed trade log table (paginated, color-coded)
[x] All /api/auto-trade/* endpoints (status, positions, log, close, toggle)
```

**Phase 4 — Confirmation flow:**
```
[x] auto_trade_pending table (DB migration)
[x] create_pending_confirmation() + confirm_trade() in auto_paper_trader.py
[x] POST /api/auto-trade/confirm/<token>, /skip/<token>
[x] Scheduler: expire pending after 5 min (via run_auto_paper_exit)
[x] Telegram inline keyboard for Approve/Skip callbacks
[x] Email signed-URL confirm/skip links
[x] Dashboard signal card Auto-Execute button + state machine
```

**Phase 5 — Risk guards:**
```
[x] _run_risk_gate(ticker, size_usd) method
[x] Gate checks: global loss, position concentration, sector, dedup
[x] Log blocked entries to auto_paper_trades (status='blocked', blocked_reason col)
[x] Gate summary in confirmation notification
[x] GET /api/auto-trade/risk-gate-status
```

**Phase 6 — Real broker:**
```
[x] clients/broker_client.py (ABC + Alpaca + IBKR + Paper impls)
[x] engine/order_manager.py (execute_entry, execute_exit, sync_broker_positions)
[x] POST /api/orders/execute + /api/orders/close/<id> + GET /api/orders/status/<id>
[x] GET /api/broker/account + /api/broker/positions + POST /api/broker/sync
[x] Portfolio page: live P&L toggle, origin badge, broker connection header widget
[x] Broker sync scheduler job (5 min, market hours)
[x] Trust gate enforcement before mode switch
[x] alpaca-trade-api + ib_insync in requirements.txt (optional/guarded)
```

**Phase 7 — Observability:**
```
[x] GET /api/export/auto-trades CSV
[x] Auto-trade line on equity curve chart
[x] Weekly letter auto-trade digest
[x] Dashboard auto-trade summary widget
[x] Telegram /autostatus command
```

---

## Data & Intelligence (continued)

### 22. Macro dashboard — yield curve, VIX term structure, credit spreads
**File:** new `engine/macro_tracker.py`, `templates/macro.html`, `app.py`
**Description:** The system tracks equities and geopolitics but not the macro backbone. Yield curve shape (2y/10y spread), VIX term structure (VIX/VIX3M), put/call ratio, credit spreads (HYG vs LQD), DXY. All free via yfinance tickers. These are leading indicators for regime changes and would make `market_regime.py` much more powerful.
**Effort:** L · **Impact:** High (macro context for every signal)
```
[x] Create engine/macro_tracker.py: fetch ^TNX, ^IRX, ^VIX, HYG, LQD, DXY via yfinance
[x] Compute: 2y/10y spread (inverted = recession signal), VIX term structure slope, HYG/LQD credit spread
[x] Store daily snapshot in new macro_snapshots table
[x] Inject macro context into Stage 3 Gemini prompt (alongside geo_block)
[ ] Add /macro page with charts: yield curve, VIX term, credit spread over 90d
[ ] Add macro badge to dashboard (e.g., "Yield curve inverted — recession watch")
```

### 23. Earnings calendar — pre/post earnings logic
**File:** `engine/earnings_tracker.py` (extend), `engine/pipeline.py`
**Description:** `earnings_tracker.py` exists but doesn't drive the pipeline. Should: auto-flag tickers with earnings in next 5 days, show implied move (options IV) vs historical earnings move, suppress STRONG_BUY signals within 48h of earnings unless explicitly overridden. Post-earnings drift (stocks that gap tend to continue 3-5 days) is an underused pattern.
**Effort:** M · **Impact:** High (biggest single-day risk event for any position)
```
[x] Pull earnings dates from yfinance calendar or earningswhispers.com scrape
[x] Add earnings_date column to watchlist
[x] In pipeline: warn Stage 3 when earnings within 5 days (inject as risk factor)
[x] Suppress STRONG_BUY for earnings-imminent tickers unless risk_score < 5
[x] Track post-earnings drift: after earnings, flag ticker for re-analysis at +1d, +3d, +5d
[ ] Show earnings countdown badge on watchlist and stock_detail
```

### 24. Alternative sentiment — Reddit / Google Trends
**File:** new `engine/sentiment_reddit.py`, `engine/sentiment_trends.py`
**Description:** The system has no retail sentiment layer. Reddit (WSB, r/investing, r/stocks via PRAW) and Google Trends (pytrends) are especially predictive for retail-heavy stocks. A spike in search volume or Reddit mentions 1-2 days before a price move is a known leading signal.
**Effort:** L · **Impact:** Medium (high alpha for small/mid cap, lower for blue cap)
**Dependencies:** `praw` (Reddit), `pytrends` (Google Trends) — both free
```
[x] Create engine/sentiment_reddit.py: fetch top posts + comment counts for ticker via PRAW
[x] Create engine/sentiment_trends.py: pytrends relative search volume for ticker (7d)
[x] Add reddit_sentiment_score + trends_score to Stage 1 quant output
[x] Inject high-reddit-activity into Stage 3 prompt as anomaly
[x] Show sentiment badges on watchlist (retail interest: low/medium/high)
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
[x] Add GET /api/budget/status endpoint (daily spend, monthly cap, days remaining, avg cost/analysis)
[x] Add budget health card to dashboard.html (next to scheduler status)
[x] Warn visually when >80% monthly budget consumed
```

### 30. Watchlist import from broker CSV
**File:** `app.py`, `templates/watchlist.html`
**Description:** Tickers must be added manually one by one. One-click import of your actual holdings from a broker CSV export (IBKR, Degiro, Schwab — all export position CSVs) would be the natural onboarding path and encourages users to track their real portfolio.
**Effort:** S · **Impact:** Medium (onboarding)
```
[x] POST /api/watchlist/import endpoint: accept CSV upload
[x] Parse common broker CSV formats (IBKR: "Symbol", Degiro: "Produkt", Schwab: "Symbol")
[x] Preview screen showing parsed tickers before import
[x] Add "Import from CSV" button to watchlist page
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
[x] Create cli.py using argparse or click
[x] Commands: analyze <TICKER>, scan (run full pipeline), geo (latest geo scan), watchlist (list)
[x] Reuse existing engine/ and clients/ directly — no HTTP layer
[x] Output: colored terminal table (rich library) or plain JSON with --json flag
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
[x] GET /report/weekly/pdf — render existing HTML report to PDF via weasyprint
[x] GET /api/analysis/export.csv — all analysis_history rows as CSV
[x] GET /api/portfolio/export.csv — paper trades with entry/exit/P&L as CSV
[x] Add FIFO cost-basis calculation to portfolio_manager.py
[x] Add Export buttons to history.html and portfolio.html
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
[x] Create engine/geo_scenario.py with 6 hardcoded scenarios as dicts:
    { name, keywords, sector_impacts: {energy: +0.15, tech: -0.10, ...}, historical_analog }
[x] Scenario runner: cross with portfolio sector weights → estimated portfolio impact %
[x] Auto-trigger: when geo scan severity >= 8, find matching scenario by keyword overlap, run it
[ ] Show scenario result card on dashboard when triggered: "Taiwan scenario match — estimated portfolio impact: -4.2%"
[x] Manual trigger: POST /api/scenarios/run?name=taiwan_blockade
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
42.5. External Plugin Manager UI in Settings  
    - Files: UI components, settings API  
    - Description: Create a user-friendly interface for managing external plugins in the settings area.  
    - Effort: 5  
    - Impact: Medium  
    - Dependencies: Requires completion of the new settings design.  
    - Checklist: [ ] 

---

## Portfolio Analytics (advanced)

### 43. Dividend and corporate action ledger
**File:** `engine/portfolio_manager.py`, `core/database.py`
**Description:** Splits, dividends, mergers, and spin-offs are currently not tracked. A position opened at $150 pre-split is worth different things pre/post 3:1 split — the paper trading P&L and cost basis math breaks silently. This is a correctness issue more than a feature, and becomes critical the moment real money is involved.
**Effort:** M · **Impact:** High (correctness — silent P&L errors)
```
[x] Add corporate_actions table: ticker, action_type (split/dividend/merger), date, factor/amount
[x] Fetch splits from yfinance Ticker.splits and dividends from Ticker.dividends
[x] Weekly job: check for new corporate actions on all watchlist tickers
[x] Retroactively adjust cost basis in portfolio_trades for pre-split entries
[x] Add dividend income tracking: when ex-date passes, credit dividend to paper portfolio cash
[ ] Show corporate action timeline on stock_detail.html
[x] Alert via webhook when a watchlist ticker announces a split or special dividend
```

---

## AI & Intelligence (advanced, continued)

### 44. Supply chain risk mapping
**File:** new `engine/supply_chain.py`, `core/database.py`
**Description:** Deeper than sector-level geo exposure: for each watchlist ticker, map its top 5 suppliers and key customers. If a supplier appears in the geo scan (e.g., TSMC when a Taiwan event fires), automatically flag the downstream holding even if it's not directly mentioned. One-time Perplexity call per ticker ("Who are the key suppliers and customers of {company}?"), cached in DB, refreshed quarterly.
**Effort:** M · **Impact:** High (catches indirect geo exposure missed by direct sector heuristics)
```
[ ] Create engine/supply_chain.py: query Perplexity for supplier/customer graph per ticker
[ ] Add supply_chain_map table: ticker, supplier_ticker, relationship_type, cached_at
[ ] In geo scan: after scoring direct exposure, cross-check supplier list against flagged regions
[ ] Elevate geo_risk_score if a key supplier is in a flagged region (+2 pts, max 10)
[ ] Show supply chain dependencies on stock_detail.html
[ ] Quarterly refresh job: re-fetch for tickers not updated in >90 days
```

### 45. Economic moat scoring
**File:** `engine/agents.py` (Stage 3 sub-prompt), `core/database.py`
**Description:** A dedicated Gemini sub-prompt within Stage 3 that explicitly scores 1–10 on the five Buffett moat types: network effects, switching costs, cost advantage, intangible assets, efficient scale. A stock with a high risk score but a 9/10 moat is a very different hold decision than one with the same risk score and no moat. The bull/bear synthesis gestures at this but doesn't score it systematically.
**Effort:** M · **Impact:** High (fundamental quality filter — separates "risky but great business" from "risky and weak business")
```
[ ] Add moat_scoring sub-prompt to Stage 3 Gemini call (or as a Stage 2.5 step)
[ ] Parse 5 moat scores from response: network_effects, switching_costs, cost_advantage, intangibles, efficient_scale
[ ] Add moat_score (composite 1–10) + moat_breakdown (JSON) columns to analysis_history
[ ] Show moat radar chart on stock_detail.html (pentagon/spider chart via Chart.js)
[ ] Use moat_score as signal modifier: dampen STRONG_SELL for moat_score > 8 (quality override)
[ ] Add moat column to watchlist table for at-a-glance view
```

### 46. Portfolio-level anomaly detection
**File:** new `engine/portfolio_anomaly.py`, `scheduler.py`, `app.py`
**Description:** Current anomaly detection is per-ticker. Missing: detecting when the portfolio as a whole shows unusual patterns — all holdings down simultaneously (systemic event vs idiosyncratic), correlation spike across normally uncorrelated positions (regime change signal), or portfolio beta suddenly doubling. This is a different signal class than any individual ticker analysis. `var_calculator.py` and `correlation_analyzer.py` have the ingredients.
**Effort:** M · **Impact:** High (systemic risk detection is blind spot of per-ticker approach)
```
[ ] Create engine/portfolio_anomaly.py: compute rolling portfolio-level metrics (correlation dispersion, average beta, common factor exposure)
[ ] Detect: >70% of holdings in same direction intraday (systemic flag)
[ ] Detect: rolling 20d average pairwise correlation spikes >0.2 above 90d baseline (regime change)
[ ] Detect: effective portfolio beta > 1.5 * target_beta (leverage creep)
[ ] Store portfolio_anomalies table: type, severity, triggered_at, description
[ ] Alert via webhook/email when portfolio anomaly fires
[ ] Show portfolio anomaly banner on dashboard when active
```

### 47. Cross-asset composite signals
**File:** `engine/macro_tracker.py` (extend), `core/notifications.py`
**Description:** When gold rises + VIX spikes + equities fall + credit spreads widen simultaneously, that's a "flight to safety" regime that calls for a very different response than any single indicator firing. The macro tracker (#22) will collect all these series — this is the synthesis layer on top. A named composite alert type: "Flight-to-Safety pattern — 4/4 indicators aligned."
**Dependencies:** Requires macro_tracker.py (#22) to be implemented first.
**Effort:** S · **Impact:** High (once macro data exists, composites are pattern-matching on top)
```
[ ] Define composite signal patterns as config dicts: { name, conditions: [{series, direction, threshold}], min_match }
[ ] Patterns to implement: "Flight to Safety" (gold+, VIX+, equities-), "Risk-On Surge" (equities+, VIX-, credit spreads-), "Inflation Spike" (gold+, bonds-, DXY-)
[ ] After each macro snapshot, evaluate all patterns and fire if min_match conditions met
[ ] Store composite signals in macro_composite_signals table
[ ] Alert with composite name when triggered (distinct alert type from individual macro alerts)
[ ] Show active composite signal badge on dashboard
```

### 48. Feature importance from meta-labeler
**File:** `engine/meta_labeler.py`, `templates/learning.html`, `app.py`
**Description:** The Random Forest meta-labeler is trained and predicts signal confidence — but which features does it weight most heavily? Surfacing `feature_importances_` (e.g., "RSI weight 23%, momentum_3m 18%, D/E ratio 12%") on the Learning page tells which of the 20+ quant metrics are actually predictive vs noise. Already have the trained model — this is a one-liner in sklearn, plus a UI to display it.
**Effort:** XS · **Impact:** Medium (insight into which quant metrics matter)
```
[x] After model.fit() in meta_labeler.py, extract feature_importances_ with feature names
[x] Store as JSON in learning_stats or a separate model_metadata table
[x] Add GET /api/learning/feature-importance endpoint
[x] Add feature importance bar chart to learning.html (sorted descending)
[ ] Show "Top 3 predictive features" summary card on dashboard learning section
```

---

## Infrastructure & Quality (continued)

### 49. Market holiday skip logic
**File:** `scheduler.py`
**Description:** The scheduler currently runs on US market holidays (Memorial Day, 4th July, Thanksgiving, Christmas, etc.) and burns Perplexity/Gemini budget analyzing stocks with zero price movement. A simple calendar check before each scan cycle prevents ~10 wasted API calls/year and avoids misleading "no news" analyses on closed-market days.
**Effort:** XS · **Impact:** Low (operational correctness, minor cost saving)
```
[x] Add US market holiday list for current year to scheduler.py (or use pandas_market_calendars / hardcoded set)
[x] Wrap run_daily_cycle() with: if today in market_holidays: skip and log "Market closed — skipping cycle"
[x] Also skip intraday breakout check (#15) and price alert check on holidays
[x] Add holiday_skip_enabled setting (default True) in Settings UI
```

---

## UI Features — Missing Pages & Views (Phase 4+)

These features have backend API endpoints but no dedicated frontend pages. Listed by effort & impact.

### 50. Dividend & Corporate Actions Ledger Page
**File:** `app.py`, `templates/corporate_actions.html`
**Description:** Show dividend history, stock splits, mergers, and other corporate actions for all watchlist stocks. Useful for tracking cost-basis adjustments and understanding portfolio events. Database table exists (`corporate_actions`), API endpoint works (`/api/stock/{ticker}/corporate-actions`), just need the UI page.
**Dependencies:** `engine/corporate_actions.py`, database
**Effort:** XS · **Impact:** Medium (financial accuracy)
```
[ ] Create GET /corporate-actions route in app.py
[ ] Create templates/corporate_actions.html with table (date, ticker, action_type, details)
[ ] Add navigation link in base.html
[ ] Add filters: by ticker, by type (dividend/split/merger), by date range
[ ] Show dividend sum by ticker for tax reporting
```

### 51. Watchlist Groups & Tags
**File:** `app.py`, `templates/watchlist.html`, `core/database.py`
**Description:** Current watchlist only supports tier (1/2/3). Add ability to tag stocks by strategy (e.g., "Tech Growth", "Dividend", "Value Turnaround", "Momentum"). Improves organization for traders managing >20 stocks. Database schema ready, just need UI.
**Effort:** S · **Impact:** Medium (user experience)
```
[ ] Add watchlist_tags table: (tag_id, user, tag_name, color_hex, created_at)
[ ] Add watchlist_stock_tags junction table: (stock_id, tag_id)
[ ] Add tag management UI to /watchlist (add/delete/rename tags)
[ ] Show tag filters in watchlist view (multi-select, color-coded)
[ ] Save selected tags in localStorage for persistence
```

### 52. Dark Pool & Institutional Block Trade Activity Page
**File:** `app.py`, `templates/dark_pool.html`
**Description:** Show unusual institutional accumulation (block trades, dark pool activity) indicating insider confidence. API exists (`/api/stock/{ticker}/dark-pool-activity`), needs visualization. Good for identifying early institutional moves.
**Dependencies:** `engine/institutional_tracker.py`
**Effort:** S · **Impact:** Medium (signal generation)
```
[ ] Create GET /dark-pool route in app.py (list all stocks with recent block activity)
[ ] Create GET /dark-pool/{ticker} route for ticker detail
[ ] Create templates/dark_pool.html with time-series chart (cumulative blocks)
[ ] Show volume vs typical, date, price at time of block
[ ] Add filter: min block size (1M shares, 5M, 10M+)
[ ] Add "Alert if block exceeds threshold" feature
```

### 53. Confidence Decay Visualization
**File:** `templates/analysis_detail.html`, `app.py`
**Description:** Add timeline showing when signal confidence drops over time. Shows "This analysis is 8 days old — confidence decayed from 92% → 71%. Recommend re-analysis." Helps traders know when signals become stale.
**Dependencies:** `engine/staleness_tracker.py`
**Effort:** S · **Impact:** Low (UX improvement)
```
[ ] Add staleness_metadata to analysis detail page
[ ] Show confidence decay curve (analysis_date → today with 50% decay at 5 days)
[ ] Add "Re-analyze" button if stale
[ ] Show visual indicator (green/yellow/red) based on age
[ ] Add to analysis history table: "Age" column with decay %
```

### 54. Economic Moat Scoring Dashboard
**File:** `engine/moat_scorer.py`, `app.py`, `templates/moat_analysis.html`
**Description:** Rank watchlist by competitive durability (brand strength, switching costs, network effects, cost advantages). Uses low-cost heuristics (P/E stability, gross margin consistency, free cash flow trends). Not ML-based, but useful for long-term investor focus.
**Dependencies:** New `engine/moat_scorer.py`
**Effort:** M · **Impact:** Medium (stock selection)
```
[ ] Create engine/moat_scorer.py with moat_score() function
[ ] Score factors: P/E stability (3yr stddev), margin consistency, FCF trend, dividend history
[ ] Combine into single 0-100 moat score
[ ] Add GET /api/moat/{ticker} endpoint
[ ] Create templates/moat_analysis.html (ranked list with breakdown)
[ ] Show moat score on stock detail pages
[ ] Add "Moat Score" column to watchlist view
```

### 55. Portfolio Anomaly Detection Dashboard
**File:** `app.py`, `templates/portfolio_anomaly.html`
**Description:** Detect when entire portfolio moves together vs idiosyncratic moves (e.g., "Market down 2%, my portfolio down 1% → good diversification" vs "Market flat, my portfolio down 3% → something wrong"). Helps traders spot portfolio-level risk.
**Dependencies:** `engine/portfolio_analyzer.py`, real-time price data
**Effort:** M · **Impact:** Medium (risk awareness)
```
[ ] Calculate daily portfolio correlation vs SPY
[ ] Detect when beta > expected (systemic risk)
[ ] Detect sector concentration (e.g., 60% in Tech)
[ ] Create GET /api/portfolio/anomaly-detection endpoint
[ ] Create templates/portfolio_anomaly.html
[ ] Show: correlation chart, beta trend, sector concentration pie chart
[ ] Alert if portfolio correlation > 0.9 or < 0.3 (unexpected)
```

### 56. Natural Language Q&A for Portfolio
**File:** `app.py`, `engine/agents.py`
**Description:** Ask questions like "What's my best performer?" or "Which tech stock lost the most?" and get instant answers. Uses Claude/Gemini to parse the question and query portfolio data.
**Dependencies:** `clients/provider_registry.py` (Claude/Gemini access)
**Effort:** M · **Impact:** Medium (user experience)
```
[ ] Create POST /api/portfolio/ask endpoint
[ ] Parse user question with LLM: "get_best_performer" vs "get_sector_concentration" vs "get_loss_leaders"
[ ] Query portfolio_history and construct answer
[ ] Return formatted response + supporting data
[ ] Add Q&A widget to dashboard sidebar or portfolio page
[ ] Store questions in audit log for training
```

### 57. Real-time News & Sentiment Dashboard
**File:** `app.py`, `templates/news_sentiment.html`
**Description:** Continuous VADER sentiment scoring of news headlines (no LLM cost). Shows sentiment trend per ticker without burning API budget. Helps identify when macro narratives shift (e.g., "AI boom" → "AI bubble" shift).
**Dependencies:** New `engine/news_sentiment.py`, RSS feeds or newsapi.org (free tier)
**Effort:** M · **Impact:** Medium (narrative tracking)
```
[ ] Create engine/news_sentiment.py with VADER-based scorer
[ ] Fetch news via newsapi.org or RSS feeds (free sources)
[ ] Score each headline: positive/neutral/negative
[ ] Track rolling 7-day sentiment trend per ticker
[ ] Create GET /api/sentiment/{ticker} endpoint
[ ] Create templates/news_sentiment.html (sentiment trend chart)
[ ] Add "Sentiment Score" to stock detail page
[ ] Alert if sentiment flips sharply (bullish → bearish)
```

### 58. Supply Chain Risk Mapper (Advanced)
**File:** `app.py`, `engine/supply_chain_mapper.py`, `templates/supply_chain.html`
**Description:** Map second-order exposure: "TSMC down → check exposure to NVDA, AMD, QCOM" and "China tariffs → check materials costs for auto suppliers". Complex but high signal value for institutional traders. Requires supply chain data (expensive sources or manual curation).
**Dependencies:** Supply chain graph (manually curated or from commercial sources), `engine/supply_chain_mapper.py`
**Effort:** L · **Impact:** High (institutional edge)
```
[ ] Curate supply chain relationships (TSMC → NVDA, AAPL, AMD, etc.)
[ ] Store in database or JSON (supply_chain_relationships table)
[ ] Create engine/supply_chain_mapper.py
[ ] For each watchlist ticker, trace upstream/downstream exposure
[ ] Create GET /api/supply-chain/{ticker} endpoint
[ ] Create templates/supply_chain.html with network graph visualization
[ ] Show: "If TSMC falls 5%, estimated NVDA impact: -2.3%"
[ ] Add supply chain alerts to geopolitical scanner
```

### 59. Mobile PWA (Push Notifications & Offline)
**File:** `static/service_worker.js`, `app.py`, `templates/base.html`
**Description:** Convert app to Progressive Web App (PWA) with offline support and push notifications. Enables installing app on home screen and getting browser notifications for alerts.
**Effort:** L · **Impact:** Medium (user experience)
```
[ ] Create service_worker.js for offline caching (cache analysis pages, watchlist)
[ ] Add manifest.json (app name, icons, theme color)
[ ] Implement push notifications via Web Push API
[ ] Backend: store subscription endpoints, send push for signals
[ ] Add "Install app" prompt in base.html
[ ] Test on iOS (PWA support is limited) and Android
```

### 60. Watchlist Smart Sorting & Filtering
**File:** `templates/watchlist.html`
**Description:** Add sorting/filtering: by signal recency, by volatility, by sector, by performance (YTD, 1M), by tier. Current view is just a simple list. Makes watchlist management easier.
**Effort:** XS · **Impact:** Low (UX improvement)
```
[ ] Add sort dropdowns: "By Tier", "By Performance", "By Signal Age", "By Volatility"
[ ] Add filter toggles: "Show Only Tier 1", "Show Only Alerts", "Show Only Discovered"
[ ] Save sort preference in localStorage
[ ] Add "Quick Actions" column: Add Note, Archive, Remove
```

---

## Frontend — React SPA Rewrite

> **Status:** Complete. All 28 pages ported from Jinja2 to React 18 + TypeScript. Jinja2 templates remain for legacy/fallback.

### Stack

| Layer | Technology |
|---|---|
| Framework | React 18.3 + TypeScript 5.5 |
| Routing | React Router 6 (client-side SPA) |
| State | Zustand 5 (theme, toasts) |
| Data Fetching | TanStack Query 5 (React Query) |
| HTTP | Axios with CSRF interceptor |
| Animation | Framer Motion 11 |
| Charts | Chart.js 4 + react-chartjs-2 |
| Build | Vite 5 → `/static/react/` |
| Styling | CSS Modules + glassmorphic design tokens |

### Pages — Implementation Status

```
[x] LoginPage — session auth with TOTP support
[x] TotpPage — 6-digit TOTP + backup code entry
[x] TwoFactorSetupPage — QR code, backup codes, enable/disable
[x] DashboardPage — system command center, market regime, benchmarks, geo radar, economic calendar
[x] SettingsPage — 7 tabbed panels (API, Scheduler, Analysis, Budget, Security, Appearance, Plugins)
[x] WatchlistPage — full CRUD, tier filter, sort, notes, sparklines, CSV import
[x] AnalyzePage — ticker submission form with CSRF
[x] StockDetailPage — multi-tab (overview, chart, earnings, peers, sentiment, patterns)
[x] HistoryPage — analysis history table with CSV/JSON export
[x] DiscoveriesPage — auto-discovery grid, status filters, promote/dismiss, stats
[x] DiscoverPage — manual AI discovery (sector, focus, limit), result cards, add to watchlist
[x] TopPicksPage — ranked picks table, win streaks, recent signals, learning stats
[x] InsiderActivityPage — insider signals table, scan trigger
[x] PortfolioPage — holdings, trade log, add trade modal, summary metrics, CSV export
[x] PaperTradingPage — positions, trades, equity metrics, CSV export
[x] TrustPage — trust gate status, 4 check cards, signal accuracy breakdown
[x] LearningPage — weight suggestions, feature importance, signal accuracy, apply weights
[x] CrosscheckPage — crosscheck history with verdicts and confidence
[x] GeoHistoryPage — severity summary, events table, region/sector breakdown
[x] SectorScreenPage — sector cards with momentum, signals, risk, top picks
[x] BacktestPage — date range form, progress bar, results, apply weights, export
[x] JournalPage — trade journal entries, add/close/delete
[x] CompareStocksPage — side-by-side comparison of up to 5 tickers with normalized chart
[x] MacroPage — yield curve, VIX, DXY, regime, 90-day chart, economic calendar events
[x] CorporateActionsPage — dividends, splits, mergers with ticker/type filters, dividend summary
[x] GraveyardPage — removed tickers, post-removal performance, win rate
[x] ScenariosPage — geopolitical stress-test cards with per-sector impact bars, run scenario, portfolio impact
[x] ArchitecturePage — pipeline flow, AI agents, data sources, risk gates, auto-trade flow, tech stack
[x] NotFoundPage — 404 with back-to-dashboard link
```

### Components

```
[x] Layout: RootLayout, Sidebar (animated collapse), Navbar, PageHeader, StatusPill
[x] UI: Button, Card (glassmorphic + glow), Badge, Modal, Spinner, Toast, ProgressBar
[x] UI: MetricCard, Delta, StatusDot, Divider, Kbd, CopyToast
[x] Dashboard: SystemCommandCenter, GeoRadarCard, MarketRegimeCard, IntelStrip, EconomicCalendarCard, BenchmarkCard
[x] Settings: PanelAppearance, PanelSecurity, PanelAPIConnections, PanelScheduler, PanelPlugins, PanelAnalysis, PanelBudget
[x] Keyboard overlay (KbdOverlay), Luminary theme toggle
```

### API Layer

```
[x] Axios client with session auth (withCredentials) + CSRF interceptor
[x] CSRF singleton with in-flight deduplication
[x] TanStack Query client with stale/gc configuration
[x] 24 endpoint modules: auth, watchlist, portfolio, stock, discovery, insider, learning, backtest, journal, macro, geopolitical, graveyard, corporateActions, settings, budget, plugins, providers, scheduler, status, logs, topPicks, history, paperTrading, personalKeys
```

### Router & Navigation

```
[x] 28 routes in React Router (SPA catch-all in FastAPI serves index.html)
[x] AuthGuard wrapper for protected routes
[x] Animated sidebar with 5 nav groups + settings/logout
[x] Vite dev server proxies /api to FastAPI (port 8000)
[x] Production build to /static/react/ with code splitting (react-vendor, motion, query, charts)
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
[x] React SPA frontend — 28 pages, 25 API endpoint modules, full component library
```

---

## Frontend — React SPA Rewrite

Full migration from Jinja2 templates to a React 18 Single Page Application.

**Tech Stack:** React 18, TypeScript, Vite, TanStack Query (React Query), Zustand, Framer Motion, Chart.js, Axios, CSS Modules

**Build:** `cd frontend && npm run build` → outputs to `static/react/`

### Pages — Completed
```
[x] LoginPage — username/password + TOTP redirect, split layout, market indices strip
[x] TotpPage — 6-digit TOTP entry, backup code toggle, auto-submit
[x] TwoFactorSetupPage — QR code display, backup codes, verification, disable 2FA
[x] DashboardPage — System command center, market regime, benchmarks, intelligence strip, geo radar
[x] WatchlistPage — Full CRUD, tier filters, sort, export, notes modal, CSV import
[x] AnalyzePage — Ticker submission form, CSRF handling
[x] HistoryPage — Analysis history table, ticker filter, CSV/JSON export
[x] SettingsPage — 7 tabbed panels (API, Scheduler, Analysis, Budget, Security, Appearance, Plugins)
[x] LogsPage — Active alerts, scheduler logs, login security stats
[x] DiscoveriesPage — Auto-discovery results grid, status filtering, promote/dismiss, strategy stats
[x] TopPicksPage — Rankings table, recent signals, learning stats
[x] InsiderActivityPage — Signals table, scan trigger, significance scores
[x] PortfolioPage — Holdings table, trade log, add trade modal, export
[x] PaperTradingPage — Positions, trades, summary metrics, export
[x] TrustPage — Gate status, 4 check cards, signal accuracy breakdown
[x] LearningPage — Weight suggestions, feature importance, apply weights
[x] CrosscheckPage — Crosscheck history with verdicts
[x] GeoHistoryPage — Severity summary, events table
[x] SectorScreenPage — Sector cards with momentum/signals
[x] BacktestPage — Date range form, progress bar, results metrics, apply weights
[x] JournalPage — Add entry, entries list, close/delete, P&L tracking
[x] StockDetailPage — Multi-tab (Overview, Chart, Earnings, Peers, Sentiment, Patterns)
[x] DiscoverPage — AI stock discovery form (sector/focus/count), results grid, add to watchlist
[x] CompareStocksPage — Side-by-side comparison (2-5 tickers), normalized price chart, metrics table
[x] GraveyardPage — Removed tickers, post-removal performance, win rate analysis
[x] ArchitecturePage — Visual system overview (pipeline, AI agents, data sources, risk gates, tech stack)
[x] MacroPage — Macro dashboard, yield spread/VIX chart, upcoming central bank events
[x] CorporateActionsPage — Dividends & actions ledger, filters, dividend income summary
[x] NotFoundPage — 404 with navigation
```

### Infrastructure — Completed
```
[x] React Router with AuthGuard — protected routes, SPA catch-all in FastAPI
[x] Axios API client — session auth, CSRF interceptor, 401/403 error handling
[x] TanStack Query — data fetching with stale times, cache invalidation
[x] Zustand stores — theme (dark/light/system, sidebar state), toast notifications
[x] CSS Modules — component-scoped styles following glassmorphic design tokens
[x] Vite build — code splitting (react-vendor, motion, query, charts), proxy dev server
[x] Sidebar navigation — 5 nav groups, animated collapse, SVG icons, status pill
[x] 23 API endpoint modules — typed hooks for all backend endpoints
```

### Remaining Work
```
[ ] Settings panel parity — Settings page covers ~50% of Jinja2 settings.html features
[ ] Architecture page customization — drag-and-drop rearrangeable sections (future)
[ ] Design Rewrite "Breathe" integration — glassmorphic design language from CSS TODO section
[ ] ScenariosPage — standalone geopolitical scenario stress-testing (currently embedded in Portfolio)
```

---

## Won't Do (documented for future reference)

```
[-] CrewAI multi-agent rewrite — current single-swarm approach is simpler, cheaper, and more debuggable
[-] Real-time WebSocket price feed — yfinance polling is sufficient; adds complexity without meaningful gain
[-] Options chain deep analysis — data quality from free sources is unreliable; high false-signal risk
```

---

## Design Rewrite — "Breathe" (Glasmorphic Design Language)

> **Philosophy:** Three-dimensional glassmorphism rooted in Scandinavian restraint.
> Inspired by Oxygen 16 "Breathe With You" — light that breathes, glass that floats,
> data that speaks without shouting. Sharp edges. Clean structure. Everything earns its place.
>
> **Effort:** L (full design system rewrite) · **Impact:** Very High (complete visual identity)
> **Status:** Spec complete — ready for implementation

---

### BREATHE-1 · Design Language Foundation

Establish the complete token system before touching any component. This is the ground truth
all later work references. Incomplete tokens will cascade into broken components.

#### 1.1 CSS Token Architecture — `static/css/modern.css`

- [ ] **Depth tokens** — Define the Z-space vocabulary used by every component layer:
  - `--z-luminary: 0` — The light source (background)
  - `--z-shell: 1` — The frosted site background
  - `--z-void: 2` — Air between shell and cards
  - `--z-glass: 3` — Primary card/panel layer
  - `--z-elevated: 4` — Modals, dropdowns, tooltips
  - `--z-overlay: 5` — Loading screen, full-screen overlays

- [ ] **Glass material tokens** — Precision values for every glass surface:
  - `--glass-blur-near: blur(8px)` · `--glass-blur-mid: blur(20px)` · `--glass-blur-far: blur(36px)`
  - `--glass-saturation: saturate(160%)` — Color amplification through glass
  - `--glass-tint-dark: rgba(255,255,255,0.05)` — Dark mode panel fill
  - `--glass-tint-light: rgba(255,255,255,0.62)` — Light mode panel fill
  - `--glass-border-highlight: rgba(255,255,255,0.14)` — Top/left specular rim
  - `--glass-border-shadow: rgba(0,0,0,0.18)` — Bottom/right depth rim
  - `--glass-specular: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 50%)` — Surface shine

- [ ] **Glow tokens** — The Oxygen light color system:
  - Signal glows: `--glow-positive: #6BFF9E` · `--glow-negative: #FF6B6B` · `--glow-neutral: #6BB8FF`
  - Light sources dark: `--glow-gold: #FFD87A` (warm) · `--glow-ice: #A8D8FF` (cool)
  - Light sources light: `--glow-amber: #FFB347` (warm) · `--glow-sky: #B8DAFF` (cool)
  - `--glow-intensity: 0.6` — User-adjustable multiplier (controlled by Settings slider)
  - `--glow-radius-near: 120px` · `--glow-radius-far: 400px`

- [ ] **Motion tokens** — The Breathe timing manifesto:
  - `--ease-breathe: cubic-bezier(0.34, 1.2, 0.64, 1)` — Spring overshoot, the signature curve
  - `--ease-defuse: cubic-bezier(0.16, 1, 0.3, 1)` — Mercury dissolution (noise → form)
  - `--ease-sink: cubic-bezier(0.4, 0, 1, 1)` — Elements settling into depth
  - `--duration-diffuse: 600ms` · `--duration-breathe: 4000ms` · `--duration-float: 6000ms`
  - `--parallax-strength: 8px` — Max card depth shift on mouse movement

- [ ] **Typography tokens** — Add display weight precision to existing font stack:
  - Enforce Source Serif 4 only for hero numbers and major section headers
  - DM Sans for all body/UI — tighten tracking to `letter-spacing: -0.01em`
  - JetBrains Mono for all financial data: tickers, percentages, timestamps, deltas
  - Add `--text-display: clamp(3rem, 6vw, 5rem)` — KPI hero values scale to viewport

#### 1.2 New Color Palette

- [ ] **Dark Mode "The Deep"** — Void with ultraviolet undertone, not pure black:
  - `--bg-primary: #080810` · `--bg-secondary: #0E0E1A` · `--bg-tertiary: #141426`
  - `--text-primary: #EEE8DC` — Warm off-white, readable through smoke glass
  - `--text-secondary: #9990A0` — Muted lavender-grey
  - `--border-primary: rgba(255,255,255,0.07)` · `--border-highlight: rgba(255,255,255,0.14)`
  - Signal: positive `#4EE88A` · negative `#E86060` · neutral `#60A8E8` — restrained, not neon

- [ ] **Light Mode "The Breath"** — Nordic morning paper, warm cream:
  - `--bg-primary: #F5F0E8` · `--bg-secondary: #EDE8DF` · `--bg-tertiary: #E4DFD4`
  - `--text-primary: #18141E` — Near-black with violet depth
  - `--text-secondary: #5C5468` — Muted plum-grey
  - `--border-primary: rgba(0,0,0,0.07)` · `--border-highlight: rgba(255,255,255,0.80)` — bright frost edge
  - Signal: positive `#0D7A3C` (forest) · negative `#A01818` (deep red) · neutral `#1456A0` (Nordic blue)

---

### BREATHE-2 · Three-Dimensional Layer Architecture

The structural core. Every visual decision flows from this Z-space system.
Implement in strict order — each layer must be visually confirmed before building the next.

#### 2.1 Layer 0 — The Luminary (Background Light Source)

The deepest layer. A living light source everything above floats inside.
Fixed position — does not scroll with content.

- [ ] **DOM structure** (`templates/base.html`):
  - Insert `<div class="luminary" aria-hidden="true">` as first child of `<body>`
  - `position: fixed; inset: 0; z-index: var(--z-luminary); pointer-events: none`

- [ ] **Dual light orbs** — Two radial glow sources, slowly drifting:
  - Primary orb: top-right position, `var(--glow-gold)` / `var(--glow-amber)` — 800px diameter
    — dark: 25% opacity / light: 15% opacity
  - Secondary orb: bottom-left, `var(--glow-ice)` / `var(--glow-sky)` — 600px diameter
    — dark: 18% / light: 10%
  - Both animated: `luminary-drift var(--duration-float) ease-in-out infinite alternate`
  - Drift keyframes: ±80px translation along each orb's axis — slow breathing movement

- [ ] **Parallax response** (`static/js/dashboard.js` — new `ParallaxManager` class):
  - Listen to `mousemove` on `document`, shift orb positions by `mouseX * 0.04, mouseY * 0.04`
  - Use `requestAnimationFrame` with lerp (linear interpolation factor 0.08) for silky motion
  - Disable when `prefers-reduced-motion` is active or Settings Parallax toggle is OFF
  - Auto-disable on touch devices via `'ontouchstart' in window` detection

- [ ] **Particle field** — 40–60 SVG micro-dots drifting in the light:
  - `<circle>` elements, radius 1–3px, scattered across luminary layer
  - Opacity 0.06–0.15 in dark / 0.03–0.08 in light — barely visible texture
  - Each drifts on a unique loop (8–20s) via `transform: translate` keyframes
  - Particles near light orbs receive `filter: blur(1px)` for a soft corona effect
  - Not interactive — pure ambient depth texture

- [ ] **Glow bleed** — Luminary bleeds into upper layers via `mix-blend-mode: screen`
  on Shell and glass layers so card positions affect their warm/cool tint

#### 2.2 Layer 1 — The Shell (Frosted Site Background)

The material quality of the site itself. Not a card — the entire page as a surface.

- [ ] **DOM** (`templates/base.html`): `<div class="shell" aria-hidden="true">` above Luminary
  - `position: fixed; inset: 0; z-index: var(--z-shell); pointer-events: none`

- [ ] **Frosted material** (`static/css/modern.css`):
  - `backdrop-filter: blur(2px) saturate(120%)` — Minimal, just enough for material feel
  - Background: `rgba(var(--bg-primary-rgb), 0.88)` — Semi-transparent over Luminary
  - SVG grain noise overlay at 3% opacity via `::before` pseudo — frosted texture vs. flat

- [ ] **Refractive edge distortion** (`static/js/liquid-glass.js`):
  - SVG `<filter id="shell-refraction">` using `feTurbulence` + `feDisplacementMap`
  - Applied to a thin `::before` element along page edges (20px inset border zone)
  - `feTurbulence`: `baseFrequency="0.65"` `numOctaves="3"` — fine grain, not noisy
  - `feDisplacementMap`: `scale="8"` — subtle lens bend at viewport edge

- [ ] **Ambient tint absorption** — Shell hue shifts slightly based on time of day:
  - CSS custom property `--luminary-hue-offset` set by JS at page load
  - Morning: `+6deg` toward amber; evening: `-4deg` toward cool blue
  - Applied via `filter: hue-rotate(var(--luminary-hue-offset))` on Shell
  - Transition: `60s linear` — imperceptibly gradual

#### 2.3 Layer 2 — The Void (Space Between Shell and Cards)

The invisible depth that gives the design its 3D quality — the perception of air.

- [ ] **Shadow architecture** — Cards cast downward shadows into the Void:
  - Base shadow: `0 8px 32px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.2)`
  - Signal-tinted shadow: `0 16px 48px rgba(var(--card-glow-rgb), 0.15)` — subtle color bleed
  - Ensures cards appear to genuinely float above the shell surface

- [ ] **Micro-parallax on scroll** (`static/js/dashboard.js` — `ParallaxManager`):
  - `IntersectionObserver` tracks cards entering viewport for baseline Y position
  - On scroll: `translateY` shifts by `(scrollY - cardEntryY) * 0.06` — imperceptible depth
  - Max shift: `var(--parallax-strength)` = 8px — grounded, not gimmicky
  - Disable on mobile and when Parallax setting is OFF

- [ ] **Liquid shimmer in gap zones** — Pure CSS sweep on Shell `::after`:
  - Very faint gradient `rgba(255,255,255,0.02)` sweeps left-to-right across the page
  - `shimmer-sweep 12s linear infinite` — seamless loop, suggests moving light on a surface
  - Dark mode only — hidden in light mode where it would collapse the depth illusion

#### 2.4 Layer 3 — The Glass (Cards and Components)

The primary interaction layer. All cards, panels, tables, and widgets live here.
Sharp-edged, precise, luminous. Swedish engineering meets Oxygen aesthetics.

- [ ] **Glass material base** (`static/css/modern.css` — `.card` class overhaul):
  - `backdrop-filter: blur(var(--glass-blur-mid)) var(--glass-saturation)` — the core material
  - Background: `var(--glass-tint-dark/light)` per theme — transparent tint over blurred content
  - Border: 1px composed — top+left use `var(--glass-border-highlight)`, bottom+right
    use `var(--glass-border-shadow)` — achieved via `border-image` gradient or pseudo-element
  - `border-radius: 0` — Sharp edges are the design statement. No rounding on cards.
  - Specular highlight: `::before` pseudo with `var(--glass-specular)` covering top-left quadrant,
    `pointer-events: none`, opacity fades from corner outward

- [ ] **The Bulb — glow source behind each card** (`static/css/modern.css`):
  - `::after` pseudo-element behind each card (`z-index: -1`)
  - `background: radial-gradient(ellipse, var(--card-glow-color) 0%, transparent 70%)`
  - `--card-glow-color` defaults to `var(--glow-neutral)` — overridden per data context:
    - Positive metric cards → `var(--glow-positive)`
    - Negative metric cards → `var(--glow-negative)`
    - Charts, tables, info panels → `var(--glow-neutral)`
  - Opacity: `calc(0.12 * var(--glow-intensity))` — scales with Settings slider
  - Animation: `breathing-glow var(--duration-breathe) ease-in-out infinite alternate`
    keyframes: opacity 0.08 → 0.18, scale 0.95 → 1.05 — a slow living pulse

- [ ] **Hover state** — Card responds to presence:
  - `transition: all var(--duration-fast) var(--ease-breathe)`
  - `backdrop-filter` blur reduces: 20px → 12px — card "clears" as focus approaches
  - Specular `::before` opacity increases — surface brightens toward viewer
  - `translateY(-3px)` — card lifts toward the viewer
  - Glow `::after` opacity increases — the bulb brightens behind it
  - `box-shadow` deepens — Void shadow grows as card lifts away from Shell

- [ ] **Liquid bulge on hover** (`static/js/liquid-glass.js` upgrade):
  - On `mouseenter`: animate SVG `feDisplacementMap` scale from 0 → 6 → 0 over 800ms
  - Effect: glass membrane appears to breathe inward then recover — responds to presence
  - One `<filter>` per card reused via `filter: url(#card-bulge-N)`, N = card index

---

### BREATHE-3 · Data Visualization Language — "Sharp Clarity"

Data must be immediately understood before it is read. Every number and chart
should communicate its meaning graphically first, textually second.

#### 3.1 Chart.js Global Theme Overhaul (`static/js/dashboard.js`)

- [ ] **Global defaults override** — Set Chart.js defaults before any chart initializes:
  - `Chart.defaults.font.family = 'JetBrains Mono'`
  - `Chart.defaults.color = getComputedStyle(root).getPropertyValue('--text-secondary')`
  - Grid lines: `rgba(255,255,255,0.04)` dark / `rgba(0,0,0,0.06)` light — structural ghost only
  - No chart borders, no legend boxes — text labels only, no boxing
  - Tooltip style: glass card — `backdrop-filter`, sharp corners, `--glass-tint` background

- [ ] **Line charts** — Replace solid fills with direction-coded color fields:
  - `createLinearGradient` from signal color at 30% opacity → 0% at bottom of chart area
  - The color field communicates trend direction before the line is read
  - Line itself: 1.5px weight, signal color (positive/negative based on trend direction)
  - Remove point dots from line — the line is pure geometry

- [ ] **Bar charts** — Swedish precision geometry:
  - 2px gap between bars, `borderRadius: 0` — sharp
  - Per-bar gradient from `--glow-neutral` base to signal color based on bar value
  - Hover: bar gets inner glow, slight lift effect via Chart.js `onHover` callback

#### 3.2 New Micro-Chart Types (`static/js/dashboard.js`)

- [ ] **SparkBar class** — Inline 8-bar charts replacing sparkline lines in table cells:
  - SVG-drawn, 40×16px, value labels hidden until hover (tooltip only)
  - Bar color shifts from neutral → positive/negative based on most recent value
  - Implemented as `class SparkBar { render(container, data) }` injected into `<td>` cells

- [ ] **Signal dot matrix** — 10-dot grid replacing percentage text for strength indicators:
  - 10 sharp 6×6px squares in a horizontal row, 3px gap
  - Filled count proportional to value (0–100%), filled squares use signal color
  - Renders at a glance: 10 green squares = maximum confidence, 3 = low
  - No library — pure SVG `<rect>` elements

- [ ] **Horizon chart** — Single-row color-banded history in 20px height:
  - Positive bands above baseline, negative below, encoded via CSS gradient bands
  - Implemented as a `<div>` with `background: linear-gradient()` computed from data
  - Extremely space-efficient: shows full history where a line chart would need 120px

- [ ] **Constellation graph** — Correlation data as floating nodes with edges:
  - SVG-only: `<rect>` nodes (sharp, not circles), `<line>` edges
  - Node size encodes market cap weight, edge thickness encodes correlation strength
  - Node glow color encodes performance signal
  - Force layout: simple iterative JS repulsion loop (no D3 dependency)

#### 3.3 Typography-Forward KPI Numbers (`static/css/modern.css` + `dashboard.js`)

- [ ] **Hero number styling** — The number IS the graphic:
  - `var(--text-display)` size, `font-family: JetBrains Mono`, `font-weight: 700`
  - Positive: `color: var(--glow-positive)` + `text-shadow: 0 0 20px currentColor`
  - Negative: `color: var(--glow-negative)` + matching glow

- [ ] **CountUp class** — Numbers defuse in from noise on page load:
  - Scrambled digits → final value over 1.2s, easing matches `--ease-defuse` curve
  - Mercury aesthetic: each digit resolves independently with slight stagger
  - Each KPI card number initializes its `CountUp` when entering viewport

#### 3.4 Signal Glyphs — SVG Icon Language (`templates/base.html`)

- [ ] **Four glyph symbols** as `<symbol>` definitions in `<defs>`, referenced via `<use>`:
  - `BUY`: Sharp upward chevron, filled, `var(--glow-positive)`
  - `SELL`: Sharp downward chevron, filled, `var(--glow-negative)`
  - `HOLD`: Horizontal double-bar (geometric pause), `var(--glow-neutral)`
  - `WATCH`: Geometric eye outline, `var(--glow-gold)`
- [ ] Each glyph hover: scale 1.15×, glow intensifies — `transition: all 200ms var(--ease-breathe)`
- [ ] Text label hidden by default, appears as tooltip on hover — never clutters the view
- [ ] All glyphs include `aria-label` and `<title>` for accessibility

#### 3.5 Directional Arrows as Data Graphics (`static/css/modern.css`)

- [ ] Replace `▲ +2.4%` text patterns with `.delta` component:
  - Sharp SVG arrowhead (8×10px triangle) + 2px shaft, length proportional to magnitude
  - Arrow + percentage figure are one self-contained `<span class="delta delta--positive">`
  - Scanning a table of deltas becomes instantaneous — eyes read direction before numbers

---

### BREATHE-4 · Loading Screen — Mercury Diffusion

> **Critical design note:** The diffusion effect lives IN the page text itself — not as a
> separate overlay, laser, or layer imposed on top of content. The website's own text —
> nav labels, card titles, KPI numbers, table headers, every visible text node — starts
> as shimmering noise characters and resolves in place into the real content.
> There is no separate "loading screen" sitting above the UI. The UI IS the loading animation.
> The structure, glassmorphic cards, and layout are visible from the first frame;
> only the text within them is still resolving. This makes the page feel alive as it loads
> rather than blocked behind a gate.

#### 4.1 Page-Text Diffusion System — The Core Effect

The Mercury diffusion runs directly on the site's real text content. Every text node
on the page is wrapped and treated as a diffusion target. No overlay, no separate screen.

- [ ] **`TextDiffuser` class** (`static/js/loading-screen.js`) — wraps all page text in diffusion targets:
  - On `DOMContentLoaded` (before paint): traverse all visible text nodes in `<body>`
    — nav links, card headings, KPI values, table cells, labels, badges
  - For each text node: store the real string, replace characters with random noise chars
    from the set `░▒▓█╔╗╝╚║═╠╣╦╩╬│─┼@#$%&?!`
  - Node characters resolve probabilistically, each character independent:
    `p = baseP + (1 - normalizedPositionInPage) * 0.04` — top of page resolves faster
  - Nav and headings resolve first (0–400ms), body text mid (300–800ms),
    table data and sub-labels last (600–1200ms) — a wave from top to bottom
  - Unresolved characters re-randomize every 2–3 frames — shimmering noise texture
  - **The layout, cards, and glass layers are fully visible from frame zero** — only
    the text inside them is diffusing. The glassmorphic depth effect is seen immediately.

- [ ] **Character resolution per node:**
  - Wrap each character in `<span data-real="X" data-resolved="false">`
  - On resolve: `span.textContent = realChar`, add `.resolved` class
  - CSS: `.resolved` transitions `color: var(--glow-gold) → var(--text-primary)` over 80ms
  - Unresolved spans: `color: var(--glow-gold)` at 35% opacity dark /
    `var(--glow-amber)` at 45% opacity light — hot embers that cool as they settle
  - Number characters (0–9) on KPI values: resolve last within their node — the numbers
    crystallize after surrounding text, making the data reveal feel intentional

- [ ] **Dark mode text diffusion:**
  - Unresolved: `var(--glow-gold)` embers — glowing noise in the void
  - Resolved: `var(--text-primary)` warm off-white — cooling from forge temperature
  - KPI number glow at resolution moment: brief `text-shadow: 0 0 12px var(--glow-gold)`
    flashing off as the value settles — each number "sparks" when it finalizes

- [ ] **Light mode text diffusion:**
  - Unresolved: `var(--glow-amber)` at 45% — wet ink not yet dried on Nordic paper
  - Resolved: `var(--text-primary)` deep ink — ink drying, settling into the page
  - No glow flash — the ink-dry metaphor is silent and precise

- [ ] **Exclusions** — these text nodes are never diffused, they appear instantly:
  - Interactive input fields, form labels, error messages
  - Any text inside `[aria-live]` regions
  - Text injected after initial page load (live data updates use the number diffuse
    from Section 7.2 instead — a different, shorter micro-diffusion)

- [ ] **Completion:**
  - When all text nodes are resolved: fire `CustomEvent('textDiffuseComplete')`
  - Page is now fully readable — no follow-up action needed, no overlay to remove
  - Total diffusion window: 800ms–1400ms depending on page length and device speed

#### 4.2 ASCII Sword — Centerpiece During Diffusion

The sword is not a blocking overlay. It is a centerpiece element rendered inside the
page's hero area (above the fold, center of the dashboard) that exists only during the
diffusion window, then dissolves as the last text nodes resolve around it.
It sits in the same Z-space as content — surrounded by diffusing text, not above it.

The sword materializes through the same Mercury diffusion algorithm as the text —
it is the first thing to fully resolve (~600ms), becoming the visual anchor while
surrounding page text is still resolving. It draws the eye while the page loads.

```
                    ║
                   /║\
                  / ║ \
                 /  ║  \
                / ══╬══ \
               /    ║    \
              /     ║     \
             /      ║      \
            /       ║       \
           /________|________\
                    ║
                    ║
                    ║
                    ║
              ══════╬══════
                    ║
              ┌─────╨─────┐
              │     ║     │
              └───────────┘
```

- [ ] **Sword DOM placement** (`templates/base.html`):
  - Injected as a child of the main dashboard hero section — same layer as content
  - `position: absolute` within the hero, centered, `z-index: 1` — not fixed, not overlay
  - `aria-hidden="true"` — purely decorative
  - `<pre class="diffuse-sword" aria-hidden="true">` with character `<span>` children

- [ ] **Sword diffusion** — uses the same `TextDiffuser` probabilistic engine:
  - Center column (blade axis) resolves first — blade materializes before guard and pommel
  - Probability: `p = (1 - normalizedDistanceFromBladeAxis) * 0.10 + 0.02`
  - Fully resolved by ~600ms — stands clear while surrounding page text still diffuses
  - Hold fully resolved until page text diffusion completes, then fade out:
    `opacity: 1 → 0` over 400ms via `var(--ease-defuse)` — dissolves like vapor

- [ ] **Interactions:**
  - Click or keypress: immediately resolve ALL remaining text nodes and sword, skip wait
  - `prefers-reduced-motion`: all text shows final values instantly, sword skipped entirely
  - Escape key: force-complete diffusion immediately

#### 4.3 Wiring and Settings (`templates/base.html` + `static/js/loading-screen.js`)

- [ ] Check `localStorage.getItem('stockholm-loading-screen') !== 'false'` on
  `DOMContentLoaded` — if disabled, skip `TextDiffuser` entirely, show all text instantly
- [ ] `TextDiffuser` must run before first paint — wrap text nodes synchronously in
  `DOMContentLoaded` handler, before any `requestAnimationFrame` loop starts
- [ ] `prefers-reduced-motion` check at init — bypass all diffusion if active
- [ ] Sword `<pre>` element injected via JS only when loading screen is enabled —
  never hardcoded in HTML (keeps HTML clean for SSR/SEO)

---

### BREATHE-5 · Settings Page — Appearance & Effects (`templates/settings.html`)

Add a new "Appearance & Effects" section. These are all client-side only —
no server config changes needed.

- [ ] **Loading Screen toggle**
  - Label: "Intro Loading Screen" · Description: "Show ASCII sword animation on page load"
  - Toggle switch, ON by default — `localStorage` key `stockholm-loading-screen`

- [ ] **Depth Effects toggle**
  - Label: "Glass Depth Effects" · Description: "Enables backdrop-blur and glass materials.
    Disable on low-end devices for better performance."
  - Toggle ON by default — sets `data-depth="off"` on `<html>` when disabled
  - CSS: `[data-depth="off"] .card { backdrop-filter: none; background: var(--bg-secondary); }`

- [ ] **Glow Intensity slider**
  - Label: "Glow Intensity" · Range 0–100, default 60, step 5
  - Live preview: updates `--glow-intensity` on `:root` in real-time as slider moves
  - `localStorage` key `stockholm-glow-intensity`

- [ ] **Parallax toggle**
  - Label: "Parallax Depth" · Description: "Cards shift subtly with cursor movement"
  - Default ON desktop / OFF mobile (auto-detected) — `localStorage` key `stockholm-parallax`

- [ ] **Settings preview thumbnail** (200×120px CSS-only miniature):
  - Contains mini glass card, mini glow blobs, mini grid lines — purely illustrative
  - All elements use actual CSS variables — updates live when settings change
  - Not interactive, no click target

---

### BREATHE-5b · Server Efficiency — Deep Sleep Optimization (`templates/settings.html` + backend)

A separate settings section: "Server & Performance". Controls how aggressively the
server conserves resources during idle periods. Designed for always-on deployments
where the user may leave the monitor running overnight or over weekends.

#### 5b.1 Deep Sleep Mode Toggle

- [ ] **UI control** (`templates/settings.html`):
  - Section: "Server & Performance"
  - Toggle label: "Deep Sleep Mode"
  - Description: "During inactive hours, reduce background polling and AI scan frequency
    to minimum to save server resources, API budget, and energy."
  - Toggle switch, default OFF — stored in DB via existing settings API (`/api/settings`)
  - Setting key: `deep_sleep_enabled` (boolean)
  - Visual indicator: when ON, show a faint "sleeping" pulse badge in the nav bar
    (a tiny dim dot, not intrusive — just visible status)

- [ ] **Backend — `core/config.py`**:
  - Add `deep_sleep_enabled: bool = False` to default settings
  - Add `deep_sleep_poll_interval: int = 120` (minutes, default — 2 hours between scans)
  - Add `deep_sleep_start: str = "22:00"` — when deep sleep activates
  - Add `deep_sleep_end: str = "07:00"` — when full operation resumes
  - Add `deep_sleep_min_checks: int = 1` — minimum scans per deep sleep window
    (never goes fully dark — at least one check per night)

- [ ] **Backend — `scheduler.py`**:
  - On each scheduled job trigger: check `deep_sleep_enabled` setting AND current time
  - If within deep sleep window: skip the job OR reschedule to run once at the window midpoint
  - APScheduler `pause()` / `resume()` on non-critical jobs during sleep window:
    - **Pause during deep sleep:** discovery scan, social/news scraping, non-urgent re-analysis
    - **Never pause during deep sleep:** price alert checks, hard stop-loss guards,
      critical portfolio monitors — these always run at full frequency
  - Log a single "entering deep sleep" event at window start, "resuming" at window end
    (no per-job spam in logs)

- [ ] **Backend — `core/database.py`**:
  - Store `last_deep_sleep_entry` timestamp in settings table
  - Expose `GET /api/server/sleep-status` endpoint → `{ sleeping: bool, resumes_at: str }`
    for the UI indicator badge

#### 5b.2 Deep Sleep Intensity Selector

- [ ] Three-level selector (radio group) below the toggle, enabled only when toggle is ON:
  - **Light** — Scan interval ×2 during sleep window. Alert checks unchanged.
    - Use case: light reduction, still nearly responsive
  - **Deep** *(default when enabled)* — Scan interval ×6. Alerts at ×2.
    Non-critical background tasks suspended entirely.
    - Use case: overnight idle, balanced protection vs. resource savings
  - **Hibernate** — All scans suspended. Only hard alert guards run (price breach,
    stop-loss). Minimum 1 health-check per hour.
    - Use case: extended absence (weekend), maximum savings
    - Warning displayed: "AI analysis will be stale until Deep Sleep ends."
  - Stored as `deep_sleep_intensity: str` — `"light"` / `"deep"` / `"hibernate"`

#### 5b.3 Sleep Window Configuration

- [ ] **Time pickers** for sleep window start/end (only shown when toggle ON):
  - "Sleep from:" time input, default `22:00`
  - "Wake at:" time input, default `07:00`
  - Handles overnight window correctly (start > end = crosses midnight)
  - Changes saved to DB immediately on blur — no separate Save button needed

- [ ] **Weekend full-day option:**
  - Checkbox: "Extend Deep Sleep across full weekends (Sat–Sun)"
  - When checked: Deep Sleep runs all day Saturday and Sunday regardless of time window
  - Stored as `deep_sleep_full_weekends: bool`

#### 5b.4 Live Status Display

- [ ] In the Server & Performance section header: show current scheduler state:
  - Active: `● Active — next scan in 14 min` (green dot, JetBrains Mono)
  - Sleeping: `◌ Deep Sleep — resumes at 07:00` (dim dot, muted color)
  - Hibernate: `○ Hibernating — alerts only` (empty dot, `var(--text-tertiary)`)
  - Fetched from `GET /api/server/sleep-status` on settings page load, no polling

---

### BREATHE-6 · Component Redesigns

Apply Layer 3 glass and Breathe language to every major component.

#### 6.1 Navigation Bar (`templates/base.html` + `static/css/modern.css`)

- [ ] Full-width glass band: `backdrop-filter: blur(var(--glass-blur-far)) var(--glass-saturation)`
- [ ] Background: `var(--glass-tint-dark/light)` — transparent over page content
- [ ] Height: reduce to 48px — ultra-thin, more content space above fold
- [ ] Bottom edge: `1px solid var(--glass-border-highlight)` — the only visible border
- [ ] Active nav link: 2px `var(--glow-neutral)` left accent, no background fill
- [ ] Hover nav link: small glass pill background, no color change — restraint

#### 6.2 All Cards (`static/css/modern.css`)

- [ ] Apply full Layer 3 glass material (Section 2.4) to all `.card` elements
- [ ] Inner padding: `var(--space-6)` minimum — glass needs breathing room inside
- [ ] Section headers inside cards: `var(--text-xs)`, uppercase, `letter-spacing: 0.12em`
- [ ] Financial values: `font-family: JetBrains Mono`, `var(--text-2xl)` minimum for KPIs
- [ ] `border-radius: 0` — enforce everywhere cards are defined

#### 6.3 Tables (`static/css/modern.css`)

- [ ] Header row: glass material, `position: sticky`, `top: 48px` (below nav)
- [ ] Even rows: `rgba(255,255,255,0.02)` tint — barely visible structure, not zebra
- [ ] Row hover: glass tint + 2px left accent line in `var(--glow-neutral)`
- [ ] Sort indicators: replace text arrows with sharp chevron SVG glyphs
- [ ] Delta cells: inject `.delta` directional arrow components (Section 3.5)
- [ ] Sparkline cells: inject `SparkBar` instances (Section 3.2)

#### 6.4 Buttons (`static/css/modern.css`)

- [ ] **Primary (pill shape, `border-radius: 9999px`):**
  - Glass fill, signal-color border glow on hover, `translateY(-2px)` lift
- [ ] **Toggle/mode (sharp rectangle, `border-radius: 0`):**
  - Active state: glass fill + inner glow — no heavy background color
- [ ] **Danger:**
  - `border: 1px solid rgba(var(--glow-negative-rgb), 0.3)` → 80% on hover
  - Glow bulb behind activates with `var(--glow-negative)` on hover

#### 6.5 Modals (`static/css/modern.css` + `templates/base.html`)

- [ ] Backdrop: `backdrop-filter: blur(12px)` — see-through blur, not solid overlay
  - `rgba(var(--bg-primary-rgb), 0.6)` background — not opaque
- [ ] Modal card: Layer 4 glass, sharp corners, `box-shadow: 0 32px 80px rgba(0,0,0,0.6)`
- [ ] Enter animation: `scale(0.96) → scale(1)` + fade, 300ms `var(--ease-breathe)`
- [ ] Close `×`: top-right, JetBrains Mono, hover glows faint `var(--glow-negative)`

#### 6.6 Toast Notifications (`static/css/modern.css` + `static/js/dashboard.js`)

- [ ] Glass material, sharp corners, right-rail bottom-anchored stack
- [ ] Left border: 3px solid signal color + inset glow `rgba(var(--signal-rgb), 0.3)`
- [ ] Enter: slide from right + fade, `var(--ease-breathe)` — Exit: slide right + fade

#### 6.7 Login Page (`templates/login.html`)

- [ ] Full Breathe treatment: Luminary layer visible, login card is centered glass slab
- [ ] Input fields: `background: rgba(255,255,255,0.04)`, `border-radius: 0`
  - Focus: left border accent `var(--glow-neutral)`, no ring glow
- [ ] Mobile: Luminary reduced to single static gradient — performance first

---

### BREATHE-7 · Animation System — "The Breathe Curve"

#### 7.1 Page Load Sequence

- [ ] Implement timed sequence with `setTimeout` chains on `DOMContentLoaded`:
  - `t=0ms`: Loading screen (if enabled) — Mercury diffusion sword (1800ms total)
  - `t=1800ms` (or `t=0` if disabled): Shell fades in (200ms)
  - `t=2000ms`: Luminary orbs drift in from opacity 0 (600ms, `ease-defuse`)
  - `t=2200ms`: Navigation glass materializes (300ms, `ease-breathe`)
  - `t=2400ms`: Cards stagger in — 80ms apart, each 500ms `ease-breathe` spring settle
  - `t=3000ms+`: Glow bulbs begin their breathing animation cycle

#### 7.2 Data Update Animations (`static/js/dashboard.js`)

- [ ] Number change: current value defuses to noise → resolves to new value (400ms total)
  — unmistakable live data refresh signal
- [ ] Chart line update: new point draws in via `stroke-dashoffset` SVG animation
- [ ] Signal change (BUY → SELL): glyph `rotateY(180deg)` flip, color transition 800ms
- [ ] Positive → Negative shift: bulb color transitions from green → red over 800ms

#### 7.3 Scroll-Triggered Entrance

- [ ] `IntersectionObserver` on all cards: animate in when entering viewport, not on page load
  - Stagger eliminated — each card animates freshly as it enters view
- [ ] Charts: initialize only when entering viewport (no off-screen rendering waste)

#### 7.4 Reduced Motion Compliance (non-negotiable)

- [ ] Every animation has a `@media (prefers-reduced-motion: reduce)` override:
  - Diffusion effects → instant `opacity` transitions only
  - Parallax → completely disabled
  - Loading screen → completed sword shown 600ms, then fade (no diffusion)
  - Count-up → final value shown immediately
  - Card enter → `opacity` only, no `translateY`
  - Breathing glow → static opacity, no keyframe animation
  - Luminary drift → static centered position, no movement

---

### BREATHE-8 · Performance and Accessibility

#### 8.1 Performance Guards

- [ ] `backdrop-filter` toggle: disabled when `data-depth="off"` (Settings, Section 5)
- [ ] Particle field: capped at 40 particles, hidden on mobile
- [ ] `will-change: transform` added only to parallax-active elements, removed when disabled
- [ ] SVG displacement filters: single shared `<defs>` block per page, not per card
- [ ] Charts: lazy-init via `IntersectionObserver` — off-screen charts don't render
- [ ] `content-visibility: auto` on below-fold page sections where browser support exists

#### 8.2 Mobile Adaptations

- [ ] Luminary: single static gradient, no drift animation
- [ ] Glass blur: 20px → 8px reduced radius
- [ ] Parallax: always OFF on touch devices
- [ ] Particle field: hidden entirely
- [ ] Loading screen: 800ms max, simpler diffusion (fewer character positions)
- [ ] SparkBars: 4 bars instead of 8 for narrow viewports

#### 8.3 Accessibility

- [ ] Color contrast ≥ 4.5:1 for all text in both themes — verify with DevTools
- [ ] Glow colors never sole indicator — always paired with text/icon
- [ ] Focus states: `outline: 2px solid var(--glow-neutral)`, `outline-offset: 4px`, sharp corners
- [ ] All signal glyphs: `aria-label` + `<title>` attributes (Section 3.4)
- [ ] Loading screen: `role="dialog"`, `aria-live="polite"` for screen readers

---

### BREATHE-9 · File Execution Checklist

> Implement phases in order — each phase depends on the previous being stable.

**Phase A — Foundation** (no visible changes until complete)
- [ ] `static/css/modern.css` — Replace all CSS variables with Breathe token system
- [ ] `static/js/dashboard.js` — Skeleton: `ParallaxManager`, `LoadingScreen` stubs, new settings keys

**Phase B — Layer System** (the structural rewrite)
- [ ] `templates/base.html` — Add `<div class="luminary">`, `<div class="shell">`,
  loading screen HTML, SVG `<defs>` for filters and glyphs
- [ ] `static/css/modern.css` — Implement all 4 layers (Sections 2.1–2.4)
- [ ] `static/js/liquid-glass.js` — Upgrade SVG filter; add per-card bulge animation

**Phase C — Data Visualization**
- [ ] `static/js/dashboard.js` — Chart.js global overrides, `CountUp`, `SparkBar`, delta arrows

**Phase D — Loading Screen**
- [ ] `static/js/loading-screen.js` — New file: full `LoadingScreen` class with Mercury algorithm
- [ ] `static/css/modern.css` — Loading screen styles for dark + light mode
- [ ] `templates/base.html` — Wire loading screen init from `localStorage` on `DOMContentLoaded`

**Phase E — Settings and Components**
- [ ] `templates/settings.html` — Appearance & Effects section with 4 new controls + preview
- [ ] `static/css/modern.css` — All component redesigns: nav, cards, tables, buttons,
  modals, toasts, login

**Phase F — Polish and Verification**
- [ ] Dark mode: full visual review — void depth, glow presence, readability
- [ ] Light mode: full visual review — frosted morning light, signal legibility
- [ ] `prefers-reduced-motion`: every animation confirmed compliant
- [ ] Mobile: all adaptations functioning at 375px and 768px breakpoints
- [ ] All 4 Settings toggles persist across page reload and browser restart
- [ ] Loading screen: ON/OFF works; click-skip works; dark + light modes verified
- [ ] Performance: no scroll jank; Chrome Layers panel shows expected GPU compositing
- [ ] Accessibility: tab order intact; ARIA labels on all new SVG elements

---

### Reference

**Design Inspirations:**
- Oxygen 16 "Breathe With You" — spatial light depth, glass over glow, breathing life
- Mercury LLM diffusion — probabilistic token resolution from noise to form
- Swedish design language — restraint, sharpness, purposeful space, nothing excess

**Existing Code — Reuse, Do Not Rewrite:**
- `ThemeManager` in `dashboard.js` — extend only
- `liquid-glass.js` SVG architecture — upgrade, keep displacement map structure
- CSS variable `[data-theme]` selector pattern in `modern.css` — keep, rename tokens
- `ChartManager` chart init in `dashboard.js` — wrap with global defaults
- All existing `IntersectionObserver` patterns — reuse for scroll-triggered animations
