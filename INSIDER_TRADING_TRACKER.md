# 🕵️ Insider Trading Tracker

## Overview

Track insider buying and selling activity across your watchlist. When executives buy their own stock, it's often one of the **strongest bullish signals** available. This professional-grade feature fetches official SEC Form 4 filings and analyzes them with AI-powered insights.

## Why Insider Trading Matters

### Professional traders watch insider activity because:

1. **Insiders know best** - Executives have non-public information about their company's prospects
2. **Putting money where mouth is** - A CEO buying $1M of stock is a strong conviction signal
3. **Predictive power** - Studies show insider buying often precedes price increases
4. **Legal and public** - All insider trades must be reported to SEC within 2 business days

### Signal Strength Hierarchy:

```
🟢 STRONGEST SIGNALS:
├─ CEO/CFO buying large amounts
├─ Multiple executives buying simultaneously (cluster)
├─ Purchases after stock decline
└─ Consistent buying over time

🟡 MODERATE SIGNALS:
├─ Director purchases
├─ VP-level purchases
└─ Small executive purchases

🔴 CAUTION SIGNALS:
├─ Cluster selling by executives
├─ CEO selling large % of holdings
└─ Heavy selling at highs

⚪ NEUTRAL (Usually):
├─ Single routine sales (often for taxes/diversification)
├─ Automatic selling plans (10b5-1)
└─ Option exercises followed by sale
```

## Features

### 1. **Automatic SEC Edgar Integration**
- Fetches official Form 4 filings (insider transaction reports)
- Parses XML data from SEC's public database
- Free, official source - no paid API needed
- Real-time data (1-2 day lag from transaction to filing)

### 2. **Intelligent Significance Scoring (0-100)**

Each transaction gets scored based on:
- **Transaction type**: Purchase (+30), Sale (-20), Award (0)
- **Transaction size**: >$5M (+20), >$1M (+15), >$500K (+10)
- **Insider title**: CEO (+15), CFO (+12), Director (+8)
- **Pattern detection**: Clusters, consistency, timing

### 3. **Pattern Detection**

Automatically detects:
- **Cluster Buying**: 3+ insiders buying within 30 days
- **Cluster Selling**: 3+ insiders selling within 30 days
- **Executive Buying**: CEO/CFO purchases (highest signal)
- **Unusual Size**: Transactions over $1M
- **Consistent Direction**: 80%+ same direction (strong consensus)

### 4. **AI-Powered Analysis**

- **Gemini Flash**: Interprets patterns and provides actionable insights
- **Perplexity** (optional): Adds market context - "Why are insiders buying NOW?"
- Combines technical signal with fundamental context

### 5. **Net Signal Calculation**

Aggregates all transactions into single signal:
- **BULLISH** (score > 30): Net buying, strong conviction
- **SLIGHTLY_BULLISH** (score 10-30): Moderate buying
- **NEUTRAL** (score -10 to 10): Mixed or no clear direction
- **SLIGHTLY_BEARISH** (score -30 to -10): Moderate selling
- **BEARISH** (score < -30): Net selling, potential concern

## Usage

### Web Interface

#### 1. **Main Overview Page** (`/insider-activity`)

```
Features:
- Top insider signals across watchlist
- Quick scan of all stocks
- Sortable by signal strength, transaction count, value
- One-click detail view
```

**Actions:**
- Click "SCAN WATCHLIST" to fetch latest SEC data
- Click ticker or "VIEW DETAILS" for comprehensive analysis
- Table shows: Signal, Transaction count, Buy/Sell values, Net position

#### 2. **Detailed Ticker View** (`/insider-activity/{TICKER}`)

```
Sections:
1. Summary Stats
   - Total transactions, purchases, sales
   - Net signal and score
   - Buy/sell values and net value

2. AI Interpretation (Gemini)
   - What the activity means
   - Signal strength assessment
   - What investors should watch

3. Market Context (Perplexity - if available)
   - Why insiders are trading now
   - Recent catalysts or events
   - Broader market context

4. Patterns Detected
   - Visual display of detected patterns
   - Explanations of significance

5. Most Significant Transaction
   - Highlighted key transaction
   - Link to official SEC Form 4

6. Full Transaction Table
   - All transactions in period
   - Sortable, filterable
   - Direct links to SEC filings
```

### Python API

#### Basic Usage:

```python
from engine.insider_tracker import insider_tracker
from clients.sec_edgar_client import sec_client

# Get insider summary for a ticker
summary = sec_client.get_insider_summary('NVDA', days_back=90)

print(f"Signal: {summary['net_signal']}")
print(f"Score: {summary['signal_score']}")
print(f"Transactions: {summary['transactions_count']}")
```

#### Comprehensive Analysis:

```python
# Get full analysis with AI insights
analysis = insider_tracker.get_insider_analysis('AAPL', days_back=180)

if analysis['has_activity']:
    print(f"Summary: {analysis['summary']}")
    print(f"Patterns: {analysis['patterns']}")
    print(f"AI Analysis: {analysis['gemini_analysis']}")
```

#### Scan Entire Watchlist:

```python
# Scan all watchlist stocks for insider activity
results = insider_tracker.scan_watchlist_insiders(days_back=90)

for result in results:
    if result['net_signal'] == 'BULLISH':
        print(f"{result['ticker']}: {result['signal_score']} score")
```

#### Get Top Signals:

```python
# Get highest-significance insider trades
top_signals = insider_tracker.get_top_insider_signals(
    limit=10,
    min_significance=70
)

for signal in top_signals:
    print(f"{signal['ticker']}: {signal['most_significant']}")
```

## Testing

```bash
# Run comprehensive test suite
python test_insider_tracker.py

# Tests:
# 1. SEC Edgar API connection
# 2. CIK lookup (company identifier)
# 3. Form 4 parsing
# 4. Significance scoring
# 5. Pattern detection
# 6. Database storage
# 7. Watchlist scanning (optional - makes many API calls)
```

Expected output:
```
✅ SEC Edgar Client: PASSED
✅ Insider Tracker Engine: PASSED
✅ Database Integration: PASSED
✅ Watchlist Scan: PASSED (if run)
```

## Data Source

### SEC Edgar Database
- **Official source**: U.S. Securities and Exchange Commission
- **Form 4**: Statement of Changes in Beneficial Ownership
- **Filing deadline**: 2 business days after transaction
- **Public data**: Free, no API key required
- **Rate limit**: 10 requests/second
- **Coverage**: All U.S. public companies

### What Gets Reported:
- **Who**: Officer name and title
- **What**: Buy, Sell, Award, Exercise, etc.
- **When**: Transaction date and filing date
- **How much**: Number of shares and price per share
- **Type**: Direct ownership vs. indirect (trusts, etc.)

## Architecture

```
┌─────────────────────────────────────────┐
│         SEC Edgar Database              │
│      (Form 4 Filings - Official)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      SECEdgarClient                     │
│  • Fetches Form 4 XML files             │
│  • Parses insider transactions          │
│  • Calculates significance scores       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      InsiderTracker                     │
│  • Detects patterns (clusters, etc.)    │
│  • Calculates net signals               │
│  • Integrates AI analysis               │
└──────────────┬──────────────────────────┘
               │
               ├──────────────┐
               ▼              ▼
    ┌─────────────────┐  ┌──────────────┐
    │ Gemini Flash    │  │ Perplexity   │
    │ (Interpretation)│  │ (Context)    │
    └─────────────────┘  └──────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Database Storage                   │
│  • insider_transactions table           │
│  • Historical tracking                  │
│  • Quick retrieval                      │
└─────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Web Interface                      │
│  • /insider-activity (overview)         │
│  • /insider-activity/{ticker} (detail)  │
└─────────────────────────────────────────┘
```

## Database Schema

```sql
CREATE TABLE insider_transactions (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    insider_name TEXT,
    title TEXT,
    transaction_date TEXT,
    filing_date TEXT,
    transaction_type TEXT,        -- 'Purchase', 'Sale', 'Award', etc.
    transaction_code TEXT,         -- SEC code: P, S, A, etc.
    shares REAL,
    price REAL,
    value REAL,                    -- shares * price
    significance_score INTEGER,    -- 0-100
    form4_url TEXT,               -- Link to SEC filing
    created_at TIMESTAMP,
    UNIQUE(ticker, insider_name, transaction_date, shares, price)
);
```

## Performance & Rate Limiting

### SEC Edgar Rate Limits:
- **Limit**: 10 requests per second
- **Enforcement**: IP-based throttling
- **Client handling**: Built-in delays (150ms between requests)

### Performance Expectations:
- **Single ticker**: 5-15 seconds (fetches + parses ~20 filings)
- **Watchlist scan (10 stocks)**: 1-3 minutes
- **Watchlist scan (50 stocks)**: 5-10 minutes

### Optimization Tips:
1. **Cache results** - SEC data only updates when new filings arrive
2. **Scan during off-hours** - Less contention on SEC servers
3. **Focus on high-priority tickers** - Not all stocks need real-time monitoring
4. **Use database storage** - Query DB first before hitting SEC API

## Interpretation Guide

### When to Act on Insider Signals

#### Strong BUY Signals:
```
✅ Act on:
- CEO buying >$1M (especially after decline)
- 3+ executives buying simultaneously
- Consistent buying over 90 days
- First purchase in years

⚠️ Investigate further:
- Single small purchase
- Director buying (less conviction)
- Buying near highs (could be overconfident)
```

#### Strong SELL Signals:
```
⚠️ Caution on:
- CEO selling >20% of holdings
- Multiple execs selling at once
- Pattern of sales before bad news

✅ Usually OK:
- Single sale for diversification
- Scheduled 10b5-1 plan sales
- Options exercise + immediate sale
- Sales to pay taxes on awards
```

### Common Pitfalls:

1. **Not all sales are bearish** - Executives often sell for personal reasons unrelated to company prospects
2. **Timing lag** - 2-day filing delay means trade happened days ago
3. **Context matters** - Buying after 50% drop is different than buying at ATH
4. **Volume matters** - $100K purchase is different than $10M

## Integration with Existing Features

### 1. **Watchlist Integration**
```python
# Insider activity automatically tracked for watchlist stocks
# Scan button fetches latest SEC data
```

### 2. **Analysis Pipeline**
```python
# Add insider signals to Stage 2 analysis
if insider_summary['net_signal'] == 'BULLISH':
    analysis['insider_boost'] = +10  # Boost score
```

### 3. **Alert System** (Future)
```python
# Alert when high-significance insider buying detected
if transaction['significance_score'] >= 85:
    send_alert(f"🟢 {ticker}: CEO bought ${value:,.0f}")
```

## Troubleshooting

### "No CIK found for ticker"
- Ticker might be incorrect or delisted
- Try variations (e.g., BRK.A vs BRK-A)
- SEC database uses CIK (10-digit number), not ticker

### "No transactions found"
- Many stocks have no recent insider activity (normal)
- Try increasing `days_back` parameter (90 → 180)
- Check if company has insiders (SPACs often don't)

### "Rate limited by SEC"
- Wait 10 seconds and retry
- Client has built-in rate limiting (should be rare)
- Don't scan large watchlists too frequently

### "Failed to parse Form 4"
- Form 4 XML structure varies slightly
- Parser uses heuristics - some edge cases may fail
- Check SEC filing manually for accuracy

## Future Enhancements

1. **Real-time monitoring** - Check for new filings hourly
2. **Historical backtesting** - "How predictive were past insider buys?"
3. **Insider portfolios** - Track what specific executives own
4. **Form 3/5 integration** - Initial ownership + annual statements
5. **Insider reputation scoring** - Which insiders have best track record?
6. **Cross-sector comparison** - Relative insider activity by industry
7. **10b5-1 detection** - Flag automatic trading plans

## Resources

- **SEC Edgar Search**: https://www.sec.gov/edgar/searchedgar/companysearch
- **Form 4 Guide**: https://www.sec.gov/files/form4data.pdf
- **Transaction Codes**: https://www.sec.gov/about/forms/form4data.pdf
- **OpenInsider** (reference): http://openinsider.com/

## Files Added/Modified

### New Files:
- `clients/sec_edgar_client.py` - SEC Edgar API client
- `engine/insider_tracker.py` - Insider analysis engine
- `templates/insider_activity.html` - Main overview page
- `templates/insider_detail.html` - Detailed ticker view
- `test_insider_tracker.py` - Comprehensive test suite
- `INSIDER_TRADING_TRACKER.md` - This documentation

### Modified Files:
- `core/database.py` - Added insider_transactions table + methods
- `app.py` - Added /insider-activity routes
- `templates/base.html` - Added navigation link

## Quick Start

```bash
# 1. Run tests (optional but recommended)
python test_insider_tracker.py

# 2. Start application
python app.py

# 3. Navigate to insider tracking
http://localhost:8000/insider-activity

# 4. Click "SCAN WATCHLIST" to fetch latest SEC data

# 5. View detailed analysis for any ticker
http://localhost:8000/insider-activity/NVDA
```

---

**Status**: ✅ **READY FOR USE**
**Data Source**: SEC Edgar (Official, Free)
**API Key Required**: None
**Cost**: Free (SEC data is public)
**Rate Limit**: 10 requests/second (SEC enforced)
