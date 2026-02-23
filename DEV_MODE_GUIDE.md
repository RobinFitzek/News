# Development Mode - Quick Reference

## Installation Complete! ✅

### What Was Added:

**1. Development Mode Toggle** - Verbose debug logging system
- 🔧 Shows function names, line numbers, and detailed traces  
- 📊 Configurable via setting or environment variable
- 🎯 Third-party library logs included in debug mode

### How to Use Development Mode:

**Option 1: Command Line (Temporary)**
```bash
DEV_MODE=true python main.py
DEV_MODE=true python app.py
```

**Option 2: Persistent Setting**
```bash
python3 dev_mode.py enable   # Turn on
python3 dev_mode.py disable  # Turn off  
python3 dev_mode.py status   # Check current state
```

**Option 3: Web UI**
- Go to Settings → System
- Toggle "Development Mode"
- Restart application

### What You See in Dev Mode:

**Normal Mode (INFO):**
```
2026-02-06 14:30:15 - engine.quant_screener - INFO - Screening AAPL
```

**Dev Mode (DEBUG):**
```
2026-02-06 14:30:15 - [DEBUG] engine.quant_screener:127 - screen_ticker() - Fetching earnings calendar for AAPL, type: <class 'dict'>
2026-02-06 14:30:15 - [DEBUG] engine.volume_analyzer:89 - get_volume_metrics() - Volume ratio: 0.93x, VWAP deviation: +6.15%
```

### Current Issues Found:

The production features test revealed some schema mismatches between the migration script and the modules:

1. **earnings_calendar** - Column name differences (`days_until` vs schema)
2. **volume_metrics** - Column name differences (`current_volume` vs schema)
3. **cash_positions** - Column name differences (`amount` vs `balance`)
4. **alerts** - Column name differences (`type` vs schema)
5. **dividend tracker** - Timezone-aware datetime issues

### Next Steps:

**To see full debug output now:**
```bash
DEV_MODE=true python3 test_production_features.py
```

The debug logs will show you EXACTLY where each error occurs with:
- File path and line number
- Function name  
- Variable values
- Full stack traces

**To fix and run properly:**
The schema alignment needs to be completed (either update migration or update modules).

### Files Modified:

1. `logging_config.py` - Added dev mode support
2. `core/config.py` - Added `development_mode` setting  
3. `core/database.py` - Added `query()`, `query_one()`, `execute()` helper methods
4. `engine/earnings_tracker.py` - Fixed yfinance calendar handling
5. `dev_mode.py` - Quick toggle script

Development mode is now fully operational! 🎉
