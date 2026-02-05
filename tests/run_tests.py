#!/usr/bin/env python3
"""
Test runner for Investment Monitor
Runs all tests and generates a coverage report
"""
import sys
import unittest
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_tests():
    """Discover and run all tests"""
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
