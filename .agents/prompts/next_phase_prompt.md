# Stockholm Investment Monitor — Next Phase Implementation

You are working on an automated investment monitoring system called **Stockholm** that runs 24/7 on a Linux home server. It uses FastAPI (`app.py`), APScheduler (`scheduler.py`), SQLite (`core/database.py`), and engine modules in `engine/`.

## Codebase Layout

```
app.py                    — FastAPI routes (3200+ lines), all UI + API endpoints
scheduler.py              — APScheduler jobs: daily scan, discovery, weekly report, hit rates, price alerts
core/database.py          — SQLite wrapper, all tables and queries
core/config.py            — .env-based configuration
engine/pipeline.py        — Daily analysis pipeline orchestrator
engine/quant_screener.py  — Quantitative screening (yfinance data, 4-factor scoring)
engine/auto_discovery.py  — Auto-discovery of new stocks
engine/discovery_hit_rate.py — Tracks 30/60/90-day outcomes of discoveries
engine/learning_optimizer.py — Weight optimization based on backtest
engine/data_freshness.py  — Per-ticker yfinance fetch health tracking
engine/api_fallback.py    — Yahoo JSON fallback when yfinance fails
engine/report_generator.py — Weekly HTML email reports
engine/webhook_notifier.py — Telegram/Discord alerts
templates/dashboard.html  — Main dashboard with system status bar, intel strip, cards
templates/base.html       — Base template with nav, keyboard shortcuts, copy-to-clipboard
static/css/modern.css     — Full design system ("Nordisch Klar" dark theme)
start.sh                  — Startup script
```

## Existing Patterns You Must Follow

1. **Engine modules are singletons**: each file ends with `module_name = ClassName()` (e.g., `quant_screener = QuantScreener()`)
2. **Database access**: always use `from core.database import db` — call `db.execute()`, `db.query()`, `db.query_one()`, `db.get_setting()`, `db.set_setting()`
3. **Tables are created in `core/database.py`** `_initialize_tables()` method, OR in individual engine modules via `self._init_table()` using `CREATE TABLE IF NOT EXISTS`
4. **API endpoints** use lazy imports: `from engine.module import singleton` inside the route function
5. **Scheduler jobs** are registered in `scheduler.py` `start()` method with APScheduler `CronTrigger` or `IntervalTrigger`
6. **Dashboard widgets** use AJAX: add a `<div>` in `dashboard.html`, fetch from `/api/...` in the `{% block extra_scripts %}` section
7. **Logging**: use `import logging; logger = logging.getLogger(__name__)`
8. **Error handling**: every yfinance/external call must be wrapped in try/except, never let background jobs crash

---

## YOUR TASKS

Implement all three tasks below. For each task, create the engine module first, then the API endpoint, then the scheduler job, then the dashboard widget.

---

### TASK 1: Self-Maintenance & Health Monitor

Create `engine/health_monitor.py` with class `HealthMonitor`:

**Database table** `system_health`:
```sql
CREATE TABLE IF NOT EXISTS system_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_type TEXT NOT NULL,     -- 'disk', 'memory', 'db_size', 'error_rate', 'uptime'
    value REAL,
    unit TEXT,
    status TEXT DEFAULT 'ok',     -- 'ok', 'warning', 'critical'
    checked_at TEXT DEFAULT (datetime('now'))
)
```

**Methods**:
- `check_disk_usage()` → returns `{path, total_gb, used_gb, free_gb, percent, status}`. Warning at 85%, critical at 95%.
- `check_memory_usage()` → returns `{total_mb, used_mb, percent, status}`. Use `psutil` if available, fall back to reading `/proc/meminfo`.
- `check_db_size()` → returns `{size_mb, table_count, largest_tables: [{name, rows, size_estimate}]}`. Query `sqlite_master` for tables, count rows per table.
- `cleanup_old_data(days=180)` → delete analysis_history older than N days, compact scheduler_logs to last 1000 entries, delete data_freshness entries for tickers no longer in watchlist. Return `{deleted_analyses, deleted_logs, deleted_freshness}`.
- `vacuum_database()` → run `VACUUM` on SQLite. Only run if last vacuum was >7 days ago (track in settings).
- `get_full_health_report()` → aggregate all checks into one dict with overall status.
- `get_error_rate(hours=24)` → count error entries in scheduler_logs in the last N hours.
- `get_uptime()` → track process start time (set in `__init__`), return uptime in hours.

**Scheduler job**: Add `health_check` job in `scheduler.py` — runs daily at 03:00. Calls `cleanup_old_data()` and `vacuum_database()` weekly (check day of week). Sends webhook alert if any check is 'critical'.

**API endpoint**: `GET /api/health` — returns `get_full_health_report()`.

**Dashboard widget**: Add a small health summary in the system status bar on `dashboard.html` showing disk and DB size. AJAX-loaded from `/api/health`.

---

### TASK 2: Signal Performance Feedback Loop

Create `engine/signal_grader.py` with class `SignalGrader`:

**Database table** `signal_grades`:
```sql
CREATE TABLE IF NOT EXISTS signal_grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    ticker TEXT NOT NULL,
    signal TEXT NOT NULL,           -- 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    confidence INTEGER,
    signal_date TEXT NOT NULL,
    price_at_signal REAL,
    price_30d REAL,
    price_60d REAL,
    price_90d REAL,
    return_30d REAL,
    return_60d REAL,
    return_90d REAL,
    grade TEXT,                     -- 'correct', 'partially_correct', 'incorrect', 'pending'
    graded_at TEXT,
    UNIQUE(analysis_id)
)
```

**Methods**:
- `grade_pending_signals()` → find all ungraded signals older than 30 days. For each, fetch current price via yfinance. Calculate returns. Grade: if signal was BUY/STRONG_BUY and return_30d > 0, it's 'correct'. If signal was SELL and return < 0, 'correct'. Otherwise 'incorrect'. 'partially_correct' if 30d was wrong but 60d or 90d was right. Return count graded.
- `get_accuracy_by_signal()` → `{STRONG_BUY: {total, correct, accuracy_pct}, BUY: {...}, ...}`
- `get_accuracy_by_month()` → monthly accuracy trend for the last 12 months
- `get_weight_recommendations()` → based on which factors correlate with correct signals, suggest weight adjustments. Compare average factor scores of correct vs incorrect signals.
- `auto_tune_weights()` → if accuracy data is sufficient (>50 graded signals), automatically update `quant_weights_override` in DB settings. Log the change. Use conservative adjustments (max ±10% per factor per tuning cycle).
- `get_monthly_self_report()` → return summary dict: signals this month, accuracy, best/worst performing signal, weight changes applied.

**Scheduler job**: Add `grade_signals` job — runs daily at 22:00. Calls `grade_pending_signals()`. Once per month (1st of month), calls `auto_tune_weights()` and sends the self-report via webhook.

**API endpoint**: `GET /api/signal-accuracy` — returns accuracy by signal type + monthly trend.

**Dashboard widget**: Add an "AI Accuracy" card on dashboard showing overall accuracy percentage and a trend indicator (improving/declining).

---

### TASK 3: Automated Paper Trading Validation

Create `engine/auto_paper_trader.py` with class `AutoPaperTrader`:

**Database table** `auto_paper_trades`:
```sql
CREATE TABLE IF NOT EXISTS auto_paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    exit_price REAL,
    exit_date TEXT,
    shares REAL DEFAULT 100,
    pnl REAL,
    pnl_pct REAL,
    status TEXT DEFAULT 'open',      -- 'open', 'closed_profit', 'closed_loss', 'closed_timeout'
    source_analysis_id INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
)
```

**Methods**:
- `process_new_signals()` → query recent analyses (last 24h) with signal STRONG_BUY or BUY. For each, if no existing open position for that ticker, open a paper trade at current price. Use fixed $10,000 notional per trade (shares = 10000 / price). Return list of opened trades.
- `check_open_positions()` → for each open trade, fetch current price. Close if: (a) signal was BUY and price dropped >10% (stop loss), (b) price rose >15% (take profit), (c) trade is >30 days old (timeout). Calculate PnL. Return list of closed trades.
- `get_performance_summary()` → total trades, win rate, avg return, total PnL, vs SPY benchmark over same period. Sharpe ratio if enough data.
- `get_open_positions()` → list of currently open paper trades with current unrealized PnL.
- `should_trust_signals()` → return True if win rate >55% and avg return >0 over last 50+ trades. This is the "confidence gate" — only when this is True should the system's alerts be taken seriously.

**Scheduler job**: Add two jobs:
1. `auto_paper_entry` — runs daily at 10:00 (after market open). Calls `process_new_signals()`.
2. `auto_paper_exit` — runs daily at 15:30 (before market close). Calls `check_open_positions()`.

**API endpoint**: `GET /api/paper-trading/auto` — returns `get_performance_summary()` + `get_open_positions()`.

**Dashboard widget**: Add a "Signal Validation" card showing paper trading win rate, total PnL, and whether signals are currently "trusted" (the confidence gate).

---

## IMPORTANT CONSTRAINTS

- **No new pip dependencies** except `psutil` for health monitoring (wrap in try/except with fallback)
- **Never crash background jobs** — every external call in try/except
- **Use existing CSS classes** from `modern.css`: `.card`, `.badge`, `.badge-success`, `.badge-danger`, `.metric-value`, `.metric-label`, `.sys-status-item`
- **Respect the design**: minimal, monochrome, "Nordisch Klar" aesthetic. No bright colors except signal green/red.
- All new tables use `CREATE TABLE IF NOT EXISTS`
- All new API endpoints require `username: str = Depends(require_auth)`
- Dashboard AJAX calls use `{credentials: 'same-origin'}`
