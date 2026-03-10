# Stockholm — AI Investment Monitor

Autonomous, self-hosted investment intelligence platform. Runs a continuous analysis pipeline combining quantitative screening, real-time market intelligence, and AI research synthesis — all on your own server with full control over data and costs.

---

## Architecture

Three-stage analysis funnel with adaptive budget management:

```
Stage 1 — Quant Screen       (zero API cost, pure math)
          ↓ top candidates
Stage 2 — Market Intelligence (Perplexity: real-time news + geopolitical exposure)
          ↓ enriched candidates
Stage 3 — Research Synthesis  (Gemini: Bull/Bear/Risk + Geo-Risiko score)
          ↓
       SQLite DB → Dashboard → Email/Webhook Alerts
```

All AI calls are budget-gated. Monthly EUR limits are set in Settings and enforced per-call via a token cost estimator.

---

## Features

### Analysis Pipeline
- **Quantitative Screener** — P/E vs sector, PEG, P/B, RSI(14), SMA cross, Bollinger, momentum vs SPY, ROE, FCF yield, D/E, current ratio — computed locally from yfinance data, zero API cost
- **Real-time News** — Perplexity AI queries for breaking news, analyst consensus, risk flags, price targets, upcoming catalysts
- **Geopolitical Analysis** — Independent 6-hour scan of active conflicts, sanctions, political instability, energy geopolitics; per-ticker exposure reasoning (supply chain, revenue geography, regulatory risk) injected into every analysis
- **AI Research Notes** — Gemini synthesizes all inputs into structured Bull Case / Bear Case / Risk Score (1-10) / Geo-Risiko (1-10) with mandatory source citations
- **Insider Activity** — SEC Form 4 filings parsed and scored; high-significance trades ("Follow The Money") injected into Stage 3 prompt
- **AI Cross-Check** — Validates AI-generated metrics (P/E, market cap, revenue growth) against yfinance ground truth; flags hallucinated numbers

### Automation & Scheduling
- **14+ background jobs** — stock scans, geopolitical scans, weekly/monthly cycles, discovery runs, paper trade entries/exits, signal grading, ML retraining, health checks
- **Geopolitical scan** — runs every 6 hours, 24/7, independent of market hours; fires email alert if any event scores ≥ 8/10 severity
- **Active-hours gate** — main stock scans only run within configured window (e.g. 08:00–22:00 weekdays)
- **Auto-start** on boot via systemd

### Self-Learning System
- **Prediction tracking** — every AI signal recorded with confidence, timestamp, strategy
- **Verification windows** — momentum signals verified at 20 days, value at 180 days, balanced at 60 days
- **Accuracy kill switch** — pipeline pauses automatically if hit rate drops below 50% with 20+ verified predictions
- **Meta-Labeler** — Random Forest trained on historical signal outcomes; blends 70% quant score + 30% RF confidence
- **Monte Carlo Permutation Test (MCPT)** — weekly statistical significance validation
- **Adaptive weight tuning** — screener weights auto-adjusted based on verified P&L

### Discovery Engine
- **Daily free discovery** — S&P 500 momentum scan, no API cost
- **Weekly AI discovery** — Perplexity finds trending stocks by sector and strategy focus (growth / value / dividend / balanced)
- **Promote → Watchlist** — one-click promotion with full quant pre-screen
- **Hit rate tracking** — discovery accuracy tracked over time

### Portfolio & Risk
- **Portfolio management** — trade log, position tracking, sector concentration limits, max position sizing
- **Paper trading** — simulated trade execution with equity curve and Sharpe ratio
- **Backtesting** — walk-forward with per-ticker accuracy, alpha vs SPY
- **Value at Risk, drawdown, correlation** — real-time portfolio risk metrics
- **Risk guard** — global kill switch if portfolio drawdown exceeds configured threshold

### Dashboard & Alerts
- **Geopolitisches Radar** — collapsible card showing latest geo scan with severity badge + portfolio exposure heatmap (HOCH/MITTEL/GERING/KEINE color-coded)
- **Market Regime** — SPY/VIX/10Y yield + SMA50/200 regime detection
- **Economic Calendar** — upcoming market-moving events
- **Sector Momentum** — heat grid + rotation signals
- **Email alerts** — Strong Buy/Sell signals, daily summary, geopolitical severity alerts
- **Telegram/Discord webhooks** — configurable webhook notifications
- **Trading Journal** — structured trade log with entry/exit tracking

### Security
- **Session-based authentication** with bcrypt password hashing
- **Encrypted API key storage** — AES encryption in SQLite, never in logs or env files
- **CSRF protection** on all state-changing forms
- **Rate limiting** on login (5/min with exponential backoff lockout)
- **Security headers** — X-Frame-Options DENY, CSP, HSTS (if HTTPS), referrer policy
- **Audit log** — login/logout/password changes recorded
- **HTTPS support** — optional TLS with cert/key files

---

## Quick Start

```bash
# 1. Clone and set up
git clone <repo>
cd News
./setup.sh

# 2. Start
python main.py
```

Dashboard: **http://localhost:8443**

On first login you'll be prompted to change the default password.

---

## Installation (manual)

**Requirements:** Python 3.10+

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## API Keys

Configure in **Settings → API Keys** (stored encrypted in SQLite):

| Service | Cost | Purpose |
|---------|------|---------|
| **Perplexity** | ~$5/month | Real-time news, geopolitical scan, discovery |
| **Gemini** | Free tier available | Research synthesis, AI notes |

Both keys are optional — the system runs with quant-only mode if neither is configured. Perplexity alone enables news + geo scanning; Gemini alone enables research synthesis from cached news.

Monthly EUR budgets are enforced per-API in Settings. The system won't make API calls once the monthly limit is reached.

---

## Configuration

All settings are managed through the dashboard (Settings page). Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `scan_interval_hours` | 2 | How often the stock pipeline runs |
| `active_hours_start/end` | 08:00–22:00 | Active window for stock scans |
| `timezone` | Europe/Berlin | Scheduler timezone |
| `monthly_budget_perplexity` | €5 | Monthly Perplexity spend cap |
| `monthly_budget_gemini` | €5 | Monthly Gemini spend cap |
| `stage1_min_score` | 40 | Minimum quant score to advance to Stage 2 |
| `analysis_depth` | standard | quick / standard / deep |
| `alert_threshold` | STRONG_BUY,STRONG_SELL | Which signals trigger email |
| `daily_summary_enabled` | true | Daily email digest at configured time |
| `discovery_enabled` | true | Enable auto-discovery jobs |
| `max_position_pct` | 10% | Portfolio position size limit |
| `stop_loss_pct` | 15% | Paper trading stop loss |
| `kill_switch_accuracy` | 50% | Accuracy threshold to pause pipeline |

---

## Project Structure

```
News/
├── main.py                    # Entry point — starts server + scheduler
├── app.py                     # FastAPI application (131 API endpoints)
├── scheduler.py               # APScheduler — 14+ background jobs
│
├── core/
│   ├── database.py            # SQLite manager (20+ tables)
│   ├── config.py              # Configuration, model pricing, defaults
│   ├── budget_tracker.py      # Adaptive API cost management
│   ├── notifications.py       # Email + webhook alerts
│   ├── auth.py                # Session auth + bcrypt
│   ├── encryption.py          # AES key encryption
│   ├── audit_log.py           # Security audit log
│   └── csrf.py                # CSRF token management
│
├── engine/
│   ├── pipeline.py            # 3-stage analysis orchestrator
│   ├── agents.py              # Stage 1/2/3 agent logic
│   ├── quant_screener.py      # Quantitative screening (zero-cost)
│   ├── discovery_engine.py    # New opportunity discovery
│   ├── learning_optimizer.py  # Self-learning + accuracy tracking
│   ├── meta_labeler.py        # Random Forest signal filter
│   ├── insider_tracker.py     # SEC Form 4 parser + scorer
│   ├── market_regime.py       # SPY/VIX regime detection
│   ├── backtest_engine.py     # Walk-forward backtesting
│   ├── paper_trading.py       # Simulated trade execution
│   ├── portfolio_manager.py   # Position + risk management
│   ├── ai_crosscheck.py       # AI hallucination detector
│   ├── staleness_tracker.py   # Signal age + confidence decay
│   └── [30+ additional modules]
│
├── clients/
│   ├── perplexity_client.py   # News, geo scan, discovery, exposure
│   ├── gemini_client.py       # Research synthesis (adaptive model tier)
│   └── sec_edgar_client.py    # SEC EDGAR filings
│
├── templates/                 # 26 Jinja2 HTML templates (dark theme)
├── static/                    # CSS + JS assets
├── data/                      # SQLite DB, encryption keys, ML model cache
├── tests/                     # Unit tests
└── systemd/                   # Systemd service file
```

---

## Systemd (Auto-start on boot)

```bash
sudo cp systemd/investment-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable investment-monitor
sudo systemctl start investment-monitor

# Monitor
sudo systemctl status investment-monitor
journalctl -u investment-monitor -f
```

---

## Database

SQLite at `data/investment_monitor.db`. Key tables:

| Table | Purpose |
|-------|---------|
| `analysis_history` | All analysis results with Bull/Bear/Risk/Geo scores |
| `geopolitical_events` | Global geo scans (raw summary, severity avg) |
| `watchlist` | Tracked tickers |
| `discovered_stocks` | Discovery queue with promote/dismiss status |
| `insider_transactions` | SEC Form 4 filings |
| `portfolio_trades` | Trade history |
| `backtest_results` | Per-ticker backtest outcomes |
| `alerts` | Alert log with dedup hash |
| `api_cost_log` | Per-call token + cost tracking |
| `users` / `user_sessions` | Auth tables |

---

## Geopolitical Analysis

The geopolitical subsystem runs independently of stock analysis:

1. **Global scan** (every 6h, 24/7) — queries Reuters, BBC, FT, Al Jazeera, AP for active conflicts, sanctions, political instability, energy/resource geopolitics. Each event is rated `SCHWEREGRAD 1-10`. Severity ≥ 8 triggers an immediate email/webhook alert.

2. **Per-ticker exposure** (each Stage 2 call) — Perplexity assesses how a specific company is affected by the current events, reasoning from revenue geography, supply chain dependencies, raw material exposure, regulatory environment. No hardcoded sector maps — the AI reasons from company fundamentals.

3. **Stage 3 injection** — Gemini receives the exposure assessment and outputs a separate `Geo-Risiko: [1-10]` score alongside the main Risk Score.

4. **Dashboard** — "Geopolitisches Radar" card shows the latest scan with severity badge and a portfolio-level exposure heatmap.

---

## API Endpoints

131 REST endpoints. Key groups:

- `GET/POST /` `/watchlist` `/analyze` `/history` `/settings` — core dashboard pages
- `GET /api/health` `/api/budget` `/api/algo-status` — system monitoring
- `GET /api/geopolitical` `/api/geopolitical/exposure` — geopolitical scan + exposure
- `GET /api/market-regime` `/api/sector-momentum` `/api/economic-calendar` — market context
- `GET /api/portfolio/*` — concentration, VaR, correlation, rebalancing, benchmark
- `GET /api/paper-trading/*` — paper trade summary, equity curve, risk metrics
- `POST /backtest/run` `GET /api/backtest/*` — backtesting
- `GET /api/signal-accuracy` `/api/ab-comparison` `/api/calibration` — learning analytics
- `GET /api/patterns` `/api/sentiment` `/api/options-flow` `/api/institutional` — per-ticker data
- `POST /scheduler/start` `/scheduler/stop` `/scheduler/run-now` — scheduler control

---

## Troubleshooting

**Scheduler doesn't start**
- Both API keys must be configured in Settings
- Check: Settings → API Keys → status shows ✅

**Perplexity rate limit**
- Wait 60 seconds (automatic retry with backoff is built in)
- Reduce scan frequency or lower monthly budget to trigger earlier throttling

**Dashboard not reachable**
```bash
# Check if running
ps aux | grep "python main.py"
# Check port
sudo lsof -i :8443
```

**Accuracy kill switch active (pipeline paused)**
- Go to Settings → Clear Kill Switch
- Investigate recent signal quality in the Learning page before re-enabling

**Gemini API errors**
```bash
pip install --upgrade google-genai
```
The project uses `google-genai >= 1.0.0` (not the deprecated `google-generativeai`).

---

## Cost Reference

Estimated monthly costs at default settings (2h scan interval, 20-ticker watchlist):

| Component | Calls/month | Est. Cost |
|-----------|-------------|-----------|
| Perplexity — news scans | ~300 | ~$3–5 |
| Perplexity — geo scans | ~120 (6h × 30d) | ~$0.50 |
| Perplexity — geo exposure | ~300 (per analysis) | ~$1–2 |
| Gemini Flash — synthesis | ~300 | ~$0.05 |

Both APIs have a monthly EUR cap enforced in-app. The system stops calling APIs once the cap is reached — there are no surprise bills.

---

## License

Private use. Not for commercial distribution.
