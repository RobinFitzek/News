# 🚀 Quick Start: Perplexity Stock Discovery

## ✅ What's Been Implemented

1. **Perplexity Client Enhancement** (`clients/perplexity_client.py`)
   - New method: `discover_trending_stocks()` - AI-powered stock discovery
   - Structured output parsing with scores, catalysts, and confidence levels

2. **Discovery Engine Upgrade** (`engine/discovery_engine.py`)
   - Enhanced `discover_trending()` - Now uses Perplexity when available
   - New method: `discover_with_perplexity()` - Full control over discovery parameters
   - Smart fallback to momentum-based discovery if API unavailable

3. **Web Interface** (NEW!)
   - Route: `/discover` - Interactive discovery page
   - Real-time stock discovery from the browser
   - One-click add to watchlist
   - API usage tracking

4. **Test Script** (`test_discovery.py`)
   - Comprehensive testing without using production API calls
   - Live API testing mode (optional)

## 🧪 Testing Options

### Option 1: Command Line (Recommended for first test)

```bash
# 1. Run the test script
python test_discovery.py

# First it runs parsing tests (no API calls)
# Then asks if you want to run live tests

# Press 'y' to use 2-3 API calls and test live discovery
```

### Option 2: Web Interface (User-friendly)

```bash
# 1. Start the application
python app.py

# 2. Open browser and navigate to:
http://localhost:8000/discover

# 3. Select parameters:
#    - Sector: Technology, Healthcare, etc. (or "All Sectors")
#    - Focus: Growth, Value, Dividend, or Balanced
#    - Limit: 1-10 stocks

# 4. Click "DISCOVER STOCKS"
# Wait 20-30 seconds for results
```

### Option 3: Python Console (For developers)

```python
# Interactive testing in Python console
from clients.perplexity_client import pplx_client
from engine.discovery_engine import discovery_engine

# Check API status
print(pplx_client.get_usage())

# Discover stocks directly with Perplexity
result = pplx_client.discover_trending_stocks(
    sector="Technology",
    focus="growth",
    limit=5
)

# Print results
for stock in result['stocks']:
    print(f"{stock['ticker']}: {stock['reason']} (Score: {stock['score']})")

# Or use the discovery engine
stocks = discovery_engine.discover_trending(limit=5)
print(stocks)  # Returns list of tickers
```

## 📊 Expected Output Examples

### Command Line Test Output:
```
🧪 TESTING PERPLEXITY-POWERED STOCK DISCOVERY
======================================================================

📊 API Status:
   Used today: 5/33
   Remaining: 28

======================================================================
TEST 1: Direct Perplexity Discovery Call
======================================================================

🔍 Discovering trending stocks (balanced focus)...

✅ Discovery successful!

📊 Parsed Stocks (5):

   🟢 NVDA - Score: 92/100 (high confidence)
      Reason: AI-Marktführer mit starkem Wachstum
      Catalyst: Q4 Earnings am 21. Feb

   🟢 PLTR - Score: 85/100 (high confidence)
      Reason: Regierungsverträge + KI-Expansion
      Catalyst: Army Contract Announcement
...
```

### Web Interface Output:
- Clean card-based UI showing each discovered stock
- Score badges (green for high scores, yellow for medium, red for low)
- Reason and catalyst information
- One-click "Add to Watchlist" buttons
- "Analyze Now" links for immediate deep analysis
- Full raw analysis in collapsible section

## 🔍 How to Interpret Results

### Score Ranges
- **90-100**: Exceptional opportunity - Strong buy signal
- **70-89**: Very interesting - Consider for watchlist
- **50-69**: Solid opportunity - Research further
- **30-49**: Speculative - High risk/reward
- **0-29**: High risk - Proceed with caution

### Confidence Levels
- **High**: Multiple strong indicators, clear catalyst
- **Medium**: Good fundamentals, some uncertainty
- **Low**: Speculative, limited data

### Catalysts
Look for specific upcoming events:
- **Earnings reports**: Company financial results
- **FDA approvals**: Drug/device regulatory decisions
- **Product launches**: New product announcements
- **Acquisitions**: M&A activity
- **Regulatory changes**: Policy shifts affecting sector

## 🛠️ Troubleshooting

### "Perplexity API not configured"
```bash
# Add to your .env file:
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxx

# Restart the application
```

### "Daily limit reached"
- Wait until tomorrow (limit resets at midnight)
- Uses 1 API call per discovery
- Daily limit: 33 calls

### "No stocks found"
- All discovered stocks might already be in your watchlist
- Try a different sector or focus mode
- Check if API call succeeded (look for error messages)

### Parsing errors
- The system attempts to extract tickers even if parsing fails
- Check the "Full Analysis" section for raw Perplexity output
- Report parsing issues for improvement

## 📈 Next Steps After Discovery

1. **Review discovered stocks** - Read reasons and catalysts carefully
2. **Add to watchlist** - Click "Add to Watchlist" for interesting stocks
3. **Run full analysis** - Use "Analyze Now" for deep dive with all 4 agents
4. **Monitor catalysts** - Set reminders for upcoming events
5. **Track performance** - See if predictions materialize

## 🎯 Best Practices

### When to Use Discovery
- **Weekly**: Find new opportunities based on market changes
- **After major news**: Sector-specific discovery after big events
- **Portfolio rebalancing**: Discover stocks in underrepresented sectors
- **Earnings season**: Find stocks with upcoming catalysts

### Parameter Selection
- **Balanced focus**: Best for general discovery
- **Growth focus**: For aggressive portfolios, tech-heavy
- **Value focus**: For conservative approaches, undervalued stocks
- **Dividend focus**: For income-oriented portfolios

### API Budget Tips
- Discovery uses 1 call, analysis uses 3-5 calls
- Prioritize weekly discovery over daily
- Use discovery results to feed into normal analysis cycle
- Set limit to 3-5 stocks to conserve budget

## 📚 Documentation

- **Full details**: See `PERPLEXITY_DISCOVERY.md`
- **API reference**: See `clients/perplexity_client.py` docstrings
- **Architecture**: See `PERPLEXITY_DISCOVERY.md` - Architecture section

## 🎉 Success Criteria

You'll know it's working when:
1. Test script completes without errors ✓
2. Web interface loads and shows API usage ✓
3. Discovery returns 3-5 stocks with scores ✓
4. Each stock has reason, score, and catalyst ✓
5. Stocks can be added to watchlist ✓

## 💡 Tips for First Use

1. **Start with test script** - Validate functionality first
2. **Use balanced focus** - Most reliable for first test
3. **Try "All Sectors"** - Wider discovery pool
4. **Limit to 3 stocks** - Conserve API budget
5. **Review full analysis** - Learn how Perplexity reasons

---

**Ready to discover?** Run `python test_discovery.py` or visit `/discover` in your browser! 🚀
