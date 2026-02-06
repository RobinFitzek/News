# 🌐 Perplexity-Powered Stock Discovery

## Overview

The system now uses **Perplexity AI** with real-time internet access to discover new interesting stocks based on current market trends, breaking news, and momentum. This replaces the previous hardcoded stock universe approach with intelligent, context-aware discovery.

## What's New

### Before ❌
- Discovery limited to hardcoded stock lists (30-40 predefined tickers)
- Simple momentum-based ranking
- No awareness of breaking news or market catalysts
- Static recommendations regardless of market conditions

### After ✅
- **AI-powered discovery** using Perplexity's real-time internet access
- **Context-aware recommendations** based on:
  - Breaking news and catalysts (last 24-48 hours)
  - Analyst upgrades and sentiment shifts
  - Technical momentum and volume spikes
  - Sector rotation and macro trends
- **Structured output parsing** for automated integration
- **Fallback mechanism** to momentum-based discovery if API unavailable

## New Features

### 1. **Direct Perplexity Discovery** (`clients/perplexity_client.py`)

```python
from clients.perplexity_client import pplx_client

# Discover stocks with customizable parameters
result = pplx_client.discover_trending_stocks(
    sector="Technology",      # Optional: filter by sector
    focus="growth",           # "growth", "value", "dividend", or "balanced"
    limit=5                   # Number of stocks to discover
)

# Result structure:
{
    "success": True,
    "sector": "Technology",
    "focus": "growth",
    "raw_analysis": "Full analysis text from Perplexity...",
    "stocks": [
        {
            "ticker": "NVDA",
            "reason": "AI-Marktführer mit starkem Wachstum",
            "score": 92,
            "catalyst": "Q4 Earnings am 21. Feb",
            "confidence": "high"  # high/medium/low
        },
        # ... more stocks
    ],
    "timestamp": "2026-02-04T...",
    "type": "stock_discovery"
}
```

### 2. **Discovery Engine Integration** (`engine/discovery_engine.py`)

```python
from engine.discovery_engine import discovery_engine

# Enhanced trending discovery (uses Perplexity if available)
trending = discovery_engine.discover_trending(limit=5)
# Returns: ['NVDA', 'PLTR', 'COIN', 'AMD', 'AVGO']

# Advanced discovery with full details
result = discovery_engine.discover_with_perplexity(
    sector=None,              # All sectors
    focus="balanced",         # Balanced approach
    limit=5
)

# Dashboard suggestions (includes Perplexity picks)
suggestions = discovery_engine.get_discovery_suggestions()
```

### 3. **Structured Output Format**

Perplexity responses follow this exact structure for easy parsing:

```
[Full analysis and reasoning...]

RECOMMENDED_STOCKS:
- TICKER: [Reason] | Score: [0-100] | Catalyst: [Event]
- TICKER: [Reason] | Score: [0-100] | Catalyst: [Event]
...
```

**Score Interpretation:**
- **90-100**: Exceptional opportunity
- **70-89**: Very interesting
- **50-69**: Solid opportunity
- **30-49**: Speculative
- **0-29**: High risk

## Focus Modes

The discovery system adapts its recommendations based on the selected focus:

| Focus | Target Stocks | Criteria |
|-------|---------------|----------|
| **growth** | High-growth, disruptive companies | Strong revenue growth, market leadership, innovation |
| **value** | Undervalued quality stocks | Low P/E, solid fundamentals, catch-up potential |
| **dividend** | Income-generating stocks | Stable dividends, consistent payouts, sustainable cash flow |
| **balanced** | Mixed approach | Balance of growth, valuation, and quality |

## API Budget Management

- **Cost**: 1 API call per discovery request
- **Daily limit**: 33 calls (~$5/month budget)
- **Auto-fallback**: If limit reached, uses momentum-based discovery
- **Smart integration**: Only called when needed (weekly cycle)

## Testing

Run the test script to verify the new functionality:

```bash
# Run parsing tests (no API calls)
python test_discovery.py

# Run full live tests (uses 2-3 API calls)
python test_discovery.py
# Answer 'y' when prompted
```

## Integration Points

### Weekly Cycle
The weekly analysis cycle automatically uses Perplexity discovery:

```python
# In engine/cycle_processor.py, line 177
discoveries = discovery_engine.discover_trending(limit=5)
```

This now intelligently uses Perplexity if available, falling back to momentum-based discovery otherwise.

### Dashboard
Discovery suggestions on the dashboard now include Perplexity picks:

```python
suggestions = discovery_engine.get_discovery_suggestions()

# Returns structure:
{
    "trending": [...],
    "sectors": {...},
    "categories": {...},
    "perplexity_picks": [  # NEW!
        {
            "ticker": "NVDA",
            "score": 92,
            "reason": "AI-Marktführer..."
        }
    ]
}
```

## Error Handling

The system gracefully handles failures:

1. **API not configured**: Falls back to momentum-based discovery
2. **Daily limit reached**: Uses fallback mechanism
3. **Parsing errors**: Attempts to extract tickers from raw text
4. **Network errors**: Retries with exponential backoff (built into client)

## Example Workflow

```python
# 1. Check if Perplexity is available
if pplx_client.is_configured():
    usage = pplx_client.get_usage()
    print(f"Remaining calls: {usage['remaining']}")

# 2. Discover stocks
result = pplx_client.discover_trending_stocks(
    sector="Healthcare",
    focus="value",
    limit=5
)

# 3. Process results
if result['success']:
    for stock in result['stocks']:
        print(f"{stock['ticker']}: {stock['reason']} (Score: {stock['score']})")

        # Add to watchlist for analysis
        db.add_to_watchlist(stock['ticker'])

# 4. The next cycle will automatically analyze these new stocks
```

## Benefits

1. **Real-time awareness**: Discovers stocks based on TODAY's news and trends
2. **Context-rich**: Includes catalysts and specific reasons for recommendations
3. **Automated**: Structured output integrates seamlessly with existing pipeline
4. **Cost-effective**: 1 discovery call can replace scanning 30+ predefined tickers
5. **Adaptive**: Recommendations change based on market conditions and focus mode

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Perplexity API (Real-time Web)            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│     EnhancedPerplexityClient.discover_trending()    │
│  • Structured prompts with focus/sector filters     │
│  • Parses RECOMMENDED_STOCKS format                 │
│  • Returns scored, ranked recommendations           │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│      DiscoveryEngine.discover_with_perplexity()     │
│  • Filters out watchlist duplicates                 │
│  • Integrates with existing pipeline                │
│  • Falls back to momentum if unavailable            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│          CycleProcessor (Weekly/Daily)              │
│  • Adds discovered stocks to analysis queue         │
│  • Runs Stage 1/2/3 analysis on discoveries         │
│  • Saves results to database                        │
└─────────────────────────────────────────────────────┘
```

## Configuration

Make sure your `.env` file has:

```bash
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxx
```

Check configuration:
```python
from clients.perplexity_client import pplx_client

print(f"Configured: {pplx_client.is_configured()}")
print(f"Usage: {pplx_client.get_usage()}")
```

## Future Enhancements

Potential improvements:

1. **Sector-specific discovery**: Dedicated prompts for each sector
2. **Catalyst tracking**: Monitor announced catalysts and alert before events
3. **Discovery history**: Track which discoveries performed well
4. **Learning feedback**: Adjust discovery prompts based on success rate
5. **Multi-focus discovery**: Combine multiple focus modes in one call

## See Also

- `clients/perplexity_client.py` - Core Perplexity client implementation
- `engine/discovery_engine.py` - Discovery engine with Perplexity integration
- `test_discovery.py` - Test script for verifying functionality
- `QUICK_START_TESTING.md` - General testing guide

---

**Status**: ✅ Implemented and ready for testing
**Version**: 1.0
**Date**: 2026-02-04
