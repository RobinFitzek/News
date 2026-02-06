# Quick Start - Testing Guide

## Running Tests

### Prerequisites

```bash
# Ensure you're in the project directory
cd /home/robin/Documents/GitHub/News

# Activate virtual environment (if using one)
source venv/bin/activate
```

### Run All Tests

```bash
# Using unittest (built-in)
python -m unittest discover tests -v

# Using the test runner
python tests/run_tests.py
```

### Run Specific Tests

```bash
# Database tests only
python -m unittest tests.test_database -v

# Single test class
python -m unittest tests.test_database.TestDatabase -v

# Single test method
python -m unittest tests.test_database.TestDatabase.test_database_initialization -v
```

## Test Results Interpretation

### Successful Run
```
test_database_initialization ... ok
test_get_set_setting ... ok
...
Ran 11 tests in 0.045s

OK
```

### Failed Test
```
test_something ... FAIL
...
FAILED (failures=1)
```

### Error in Test
```
test_something ... ERROR
...
FAILED (errors=1)
```

## Quick System Check

Run this to verify the system is stable:

```bash
# 1. Test database operations
python -m unittest tests.test_database.TestDatabase -v

# 2. Check error handling (should show "ok" for all)
python -m unittest tests.test_database.TestDatabase.test_invalid_ticker_save_analysis -v
python -m unittest tests.test_database.TestDatabase.test_transaction_rollback_on_error -v

# 3. Performance test
python -m unittest tests.test_database.TestDatabasePerformance.test_bulk_insert_performance -v
```

All should pass ✅

## Manual Testing

### Test Error Handling

```python
from core.database import db

# Test invalid ticker (should raise ValueError)
try:
    db.save_analysis('', {'recommendation': 'Buy'})
except ValueError as e:
    print(f"✅ Error handled: {e}")

# Test transaction rollback
try:
    with db._get_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ('test', '"value"'))
        raise Exception("Test error")
except:
    pass

# Verify rollback worked
val = db.get_setting('test')
print(f"✅ Transaction rolled back: {val is None}")
```

### Test API Clients

```python
from clients.gemini_client import gemini_client
from clients.perplexity_client import pplx_client

# Test Gemini client initialization
print(f"Gemini configured: {gemini_client.is_configured()}")
print(f"Gemini usage: {gemini_client.get_usage()}")

# Test Perplexity client
print(f"Perplexity configured: {pplx_client.is_configured()}")
print(f"Perplexity usage: {pplx_client.get_usage()}")

# Test error handling with invalid prompt
result = gemini_client.generate("")
print(f"✅ Invalid prompt handled: {'⚠️' in result}")
```

### Test Pipeline

```python
from engine.pipeline import pipeline

# This will test the full pipeline with real data
# Make sure APIs are configured first
try:
    results = pipeline.run_daily_cycle()
    print(f"✅ Pipeline completed successfully")
    print(f"Results: {len(results)} stocks analyzed")
except Exception as e:
    print(f"Pipeline error (expected if APIs not configured): {e}")
```

## Common Issues

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'google'
```

**Solution**: Activate virtual environment or install dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Database Locked

```
sqlite3.OperationalError: database is locked
```

**Solution**: Close any open connections or wait for retry

```bash
# The code now handles this automatically with retry logic
# If persistent, check for stuck processes
ps aux | grep python
```

### Issue: Tests Fail in Virtual Environment

**Solution**: Run from project root

```bash
cd /home/robin/Documents/GitHub/News
python -m unittest discover tests -v
```

## Verifying Stability Improvements

### 1. Error Handling Test

This should NOT crash:

```python
from engine.agents import swarm

# Test with invalid ticker
result = swarm._get_stock_data('INVALID123456')
print(f"✅ Invalid ticker handled: {'error' in result}")

# Test with empty input
result = swarm._get_stock_data('')
print(f"✅ Empty ticker handled: {'error' in result}")
```

### 2. Database Resilience Test

This should handle errors gracefully:

```python
from core.database import db

# Test invalid data types
try:
    db.set_setting('test', object())  # Invalid type
except (TypeError, ValueError):
    print("✅ Invalid data type rejected")

# Test SQL injection prevention (should be safe)
malicious = "'; DROP TABLE settings; --"
db.set_setting(malicious, 'value')
settings = db.get_all_settings()
print(f"✅ SQL injection prevented: {len(settings) > 0}")
```

### 3. API Resilience Test

Mock API failures:

```python
from unittest.mock import Mock, patch
from clients.gemini_client import gemini_client

# Simulate API error
with patch.object(gemini_client.client.models, 'generate_content', side_effect=Exception("API Error")):
    result = gemini_client.generate("test prompt")
    print(f"✅ API error handled: {'Fehler' in result or '⚠️' in result}")
```

## Performance Benchmarks

Expected performance:

- Database insert: < 0.05s per record
- 100 bulk inserts: < 5s
- Single analysis save: < 0.1s
- Settings retrieval: < 0.01s
- Watchlist operations: < 0.05s

Run performance test:

```bash
python -m unittest tests.test_database.TestDatabasePerformance -v
```

## Health Check

Test the health endpoint:

```bash
# Start the application
python app.py

# In another terminal, check health
curl http://localhost:8000/health | python -m json.tool
```

Expected response:
```json
{
  "status": "healthy" or "degraded",
  "checks": {
    "database": {"healthy": true},
    "api_keys": {"gemini": true, "perplexity": false},
    "scheduler": {"running": false},
    "disk_space": {"warning": false}
  }
}
```

## Continuous Monitoring

Set up cron job for automated testing:

```bash
# Add to crontab
0 0 * * * cd /home/robin/Documents/GitHub/News && python -m unittest discover tests >> test_results.log 2>&1
```

## Success Criteria

✅ All database tests pass
✅ Error handling tests pass
✅ Invalid inputs are rejected
✅ Transactions rollback on errors
✅ Performance tests meet benchmarks
✅ Health check returns valid status
✅ No crashes on invalid data
✅ Logs show appropriate error messages

If all criteria met: **System is stable and production-ready! 🎉**
