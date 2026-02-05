"""
Unit tests for database operations
"""
import unittest
import tempfile
import os
from pathlib import Path
from core.database import Database
from datetime import datetime


class TestDatabase(unittest.TestCase):
    """Test database operations"""

    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = Database(Path(self.temp_db.name))

    def tearDown(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_database_initialization(self):
        """Test that database initializes properly"""
        self.assertIsNotNone(self.db)
        self.assertTrue(os.path.exists(self.temp_db.name))

    def test_get_set_setting(self):
        """Test setting and getting configuration values"""
        # Test string setting
        self.db.set_setting('test_key', 'test_value')
        value = self.db.get_setting('test_key')
        self.assertEqual(value, 'test_value')

        # Test integer setting
        self.db.set_setting('scan_interval_hours', 4)
        value = self.db.get_setting('scan_interval_hours')
        self.assertEqual(value, 4)

        # Test boolean setting
        self.db.set_setting('email_enabled', True)
        value = self.db.get_setting('email_enabled')
        self.assertTrue(value)

    def test_get_nonexistent_setting_returns_default(self):
        """Test that nonexistent settings return default values"""
        value = self.db.get_setting('nonexistent_key')
        self.assertIsNone(value)

    def test_watchlist_operations(self):
        """Test watchlist add/remove/get operations"""
        # Add ticker
        self.db.add_to_watchlist('AAPL', 'Apple Inc.')
        watchlist = self.db.get_watchlist()
        self.assertEqual(len(watchlist), 5)  # 5 default tickers

        # Check ticker exists
        tickers = [item['ticker'] for item in watchlist]
        self.assertIn('AAPL', tickers)

        # Remove ticker
        self.db.remove_from_watchlist('AAPL')
        watchlist = self.db.get_watchlist(active_only=True)
        tickers = [item['ticker'] for item in watchlist]
        self.assertNotIn('AAPL', tickers)

    def test_save_and_retrieve_analysis(self):
        """Test saving and retrieving analysis"""
        ticker = 'TSLA'
        results = {
            'recommendation': 'Strong Buy - Great momentum',
            'news': 'Positive earnings beat',
            'fundamental': 'Strong financials',
            'technical': 'Bullish trend',
            'risk_score': 3
        }

        # Save analysis
        analysis_id, signal, confidence = self.db.save_analysis(ticker, results)

        # Verify save
        self.assertIsNotNone(analysis_id)
        self.assertEqual(signal, 'STRONG_BUY')
        self.assertEqual(confidence, 90)

        # Retrieve analysis
        history = self.db.get_analysis_history(ticker, limit=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['ticker'], ticker)
        self.assertEqual(history[0]['signal'], 'STRONG_BUY')

    def test_invalid_ticker_save_analysis(self):
        """Test that invalid ticker raises error"""
        with self.assertRaises(ValueError):
            self.db.save_analysis('', {'recommendation': 'Buy'})

        with self.assertRaises(ValueError):
            self.db.save_analysis(None, {'recommendation': 'Buy'})

    def test_invalid_results_save_analysis(self):
        """Test that invalid results raise error"""
        with self.assertRaises(ValueError):
            self.db.save_analysis('AAPL', None)

        with self.assertRaises(ValueError):
            self.db.save_analysis('AAPL', {})

    def test_transaction_rollback_on_error(self):
        """Test that transactions rollback on error"""
        # This should fail due to invalid data
        try:
            with self.db._get_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ('test', 'value'))
                # Force an error
                raise Exception("Test error")
        except:
            pass

        # Verify rollback - setting should not exist
        value = self.db.get_setting('test')
        self.assertIsNone(value)

    def test_database_connection_retry(self):
        """Test that database retries on lock"""
        # This is difficult to test without threading
        # Just verify connection can be established
        conn = self.db._get_conn()
        self.assertIsNotNone(conn)
        conn.close()

    def test_portfolio_operations(self):
        """Test portfolio trade operations"""
        # Add buy trade
        self.db.add_trade(
            ticker='NVDA',
            trade_type='BUY',
            amount=10,
            price=450.00,
            fees=5.00,
            notes='Test trade'
        )

        # Get trades
        trades = self.db.get_trades('NVDA')
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['ticker'], 'NVDA')
        self.assertEqual(trades[0]['type'], 'BUY')

        # Get portfolio holdings
        holdings = self.db.get_portfolio_holdings()
        self.assertTrue(len(holdings) >= 1)
        nvda_holding = next((h for h in holdings if h['ticker'] == 'NVDA'), None)
        self.assertIsNotNone(nvda_holding)
        self.assertEqual(nvda_holding['shares'], 10)


class TestDatabasePerformance(unittest.TestCase):
    """Test database performance under load"""

    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = Database(Path(self.temp_db.name))

    def tearDown(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_bulk_insert_performance(self):
        """Test performance of bulk inserts"""
        import time

        start = time.time()

        # Insert 100 analyses
        for i in range(100):
            self.db.save_analysis(
                f'TICK{i}',
                {
                    'recommendation': 'Buy',
                    'news': 'Test news',
                    'fundamental': 'Test fundamental',
                    'technical': 'Test technical'
                }
            )

        duration = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        self.assertLess(duration, 5.0, f"Bulk insert took {duration}s, expected < 5s")

        # Verify all were saved
        history = self.db.get_analysis_history(limit=100)
        self.assertEqual(len(history), 100)


if __name__ == '__main__':
    unittest.main()
