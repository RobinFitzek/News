# Investment Monitor Test Suite

This directory contains comprehensive tests for the Investment Monitor application.

## Test Structure

- `test_database.py` - Database operations and transactions
- `test_api_clients.py` - Gemini and Perplexity API client tests
- `test_agents.py` - Investment analysis agent tests

## Running Tests

### Run All Tests

```bash
# From project root
python -m pytest tests/ -v

# Or using the test runner
python tests/run_tests.py
```

### Run Specific Test File

```bash
python -m pytest tests/test_database.py -v
```

### Run Specific Test Class

```bash
python -m pytest tests/test_database.py::TestDatabase -v
```

### Run Specific Test Method

```bash
python -m pytest tests/test_database.py::TestDatabase::test_database_initialization -v
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Coverage

The test suite aims to cover:

- ✅ Database operations (CRUD)
- ✅ Transaction handling and rollback
- ✅ API client error handling
- ✅ Rate limiting and quota management
- ✅ Stock data fetching
- ✅ Analysis pipeline stages
- ✅ Input validation
- ✅ Error recovery

## Writing New Tests

When adding new features, please include tests:

1. Create test file: `tests/test_<module>.py`
2. Inherit from `unittest.TestCase`
3. Use `setUp()` and `tearDown()` for test fixtures
4. Mock external dependencies (APIs, databases)
5. Test both success and failure cases
6. Run tests before committing

### Example Test

```python
import unittest
from unittest.mock import Mock, patch

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.feature = MyFeature()

    def test_success_case(self):
        """Test successful operation"""
        result = self.feature.do_something()
        self.assertTrue(result)

    def test_error_handling(self):
        """Test error is handled gracefully"""
        with self.assertRaises(ValueError):
            self.feature.do_something_invalid()
```

## Continuous Integration

Tests are run automatically on:
- Every commit
- Pull requests
- Before deployment

## Troubleshooting

### Import Errors

Make sure you're running tests from the project root:

```bash
cd /home/robin/Documents/GitHub/News
python -m pytest tests/
```

### Database Locked

Tests use temporary databases. If you get "database locked" errors:

```bash
# Clean up any leftover temp files
rm /tmp/*.db
```

### API Mocking Issues

Most tests mock external APIs. If you need to test with real APIs:

```bash
# Set API keys in environment
export GEMINI_API_KEY="your-key"
export PERPLEXITY_API_KEY="your-key"

# Run integration tests
python -m pytest tests/ -m integration
```
