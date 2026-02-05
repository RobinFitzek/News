"""
Unit tests for investment agents
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from engine.agents import InvestmentSwarm


class TestInvestmentSwarm(unittest.TestCase):
    """Test investment analysis agents"""

    def setUp(self):
        """Set up test swarm"""
        with patch('engine.agents.db'), \
             patch('engine.agents.pplx_client'), \
             patch('engine.agents.gemini_client'):
            self.swarm = InvestmentSwarm()

    def test_swarm_initialization(self):
        """Test swarm initializes properly"""
        self.assertIsNotNone(self.swarm)
        self.assertIsNotNone(self.swarm.pplx)
        self.assertIsNotNone(self.swarm.gemini)

    @patch('engine.agents.yf.Ticker')
    def test_get_stock_data_with_valid_ticker(self, mock_ticker):
        """Test getting stock data with valid ticker"""
        # Mock yfinance response
        mock_stock = Mock()
        mock_stock.info = {
            'longName': 'Apple Inc.',
            'trailingPE': 25.5,
            'marketCap': 2500000000000,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        mock_stock.history.return_value.empty = True
        mock_ticker.return_value = mock_stock

        result = self.swarm._get_stock_data('AAPL')

        self.assertIn('ticker', result)
        self.assertEqual(result['ticker'], 'AAPL')
        self.assertEqual(result['name'], 'Apple Inc.')
        self.assertNotIn('error', result)

    @patch('engine.agents.yf.Ticker')
    def test_get_stock_data_with_invalid_ticker(self, mock_ticker):
        """Test getting stock data with invalid ticker"""
        mock_ticker.side_effect = ValueError("Invalid ticker")

        result = self.swarm._get_stock_data('INVALID')

        self.assertIn('error', result)
        self.assertEqual(result['ticker'], 'INVALID')

    def test_get_stock_data_with_empty_ticker(self):
        """Test getting stock data with empty ticker"""
        result = self.swarm._get_stock_data('')

        self.assertIn('error', result)

    def test_get_stock_data_with_none_ticker(self):
        """Test getting stock data with None ticker"""
        result = self.swarm._get_stock_data(None)

        self.assertIn('error', result)

    @patch('engine.agents.yf.Ticker')
    def test_stage1_scan_with_valid_tickers(self, mock_ticker):
        """Test stage 1 scan with valid tickers"""
        # Mock yfinance
        mock_stock = Mock()
        mock_stock.info = {'longName': 'Test Corp', 'sector': 'Technology'}
        mock_stock.history.return_value.empty = True
        mock_ticker.return_value = mock_stock

        # Mock Gemini response
        self.swarm.gemini.generate = Mock(return_value="Score: 75 | Grund: Strong fundamentals")

        results = self.swarm.stage1_scan(['AAPL', 'MSFT'], variant='balanced')

        self.assertEqual(len(results), 2)
        self.assertIn('ticker', results[0])
        self.assertIn('score', results[0])

    def test_stage1_scan_with_empty_list(self):
        """Test stage 1 scan with empty ticker list"""
        results = self.swarm.stage1_scan([], variant='balanced')

        self.assertEqual(len(results), 0)

    @patch('engine.agents.yf.Ticker')
    def test_stage1_scan_handles_api_errors(self, mock_ticker):
        """Test stage 1 scan handles API errors gracefully"""
        # Mock yfinance
        mock_stock = Mock()
        mock_stock.info = {'longName': 'Test Corp'}
        mock_stock.history.return_value.empty = True
        mock_ticker.return_value = mock_stock

        # Mock Gemini error
        self.swarm.gemini.generate = Mock(return_value="⚠️ API Error")

        results = self.swarm.stage1_scan(['AAPL'], variant='balanced')

        # Should handle error and return empty or partial results
        self.assertIsInstance(results, list)

    @patch('engine.agents.yf.Ticker')
    def test_stage1_scan_with_mixed_valid_invalid(self, mock_ticker):
        """Test stage 1 scan with mix of valid and invalid tickers"""
        def ticker_side_effect(symbol):
            if symbol == 'INVALID':
                raise ValueError("Invalid ticker")
            mock_stock = Mock()
            mock_stock.info = {'longName': f'{symbol} Corp'}
            mock_stock.history.return_value.empty = True
            return mock_stock

        mock_ticker.side_effect = ticker_side_effect

        self.swarm.gemini.generate = Mock(return_value="Score: 50 | Grund: Average")

        results = self.swarm.stage1_scan(['AAPL', 'INVALID', 'MSFT'], variant='balanced')

        # Should process valid tickers and skip invalid ones
        self.assertGreater(len(results), 0)
        tickers = [r['ticker'] for r in results]
        self.assertNotIn('INVALID', tickers)

    def test_stage1_scan_score_validation(self):
        """Test that stage 1 scan validates score ranges"""
        # Mock dependencies
        with patch('engine.agents.yf.Ticker') as mock_ticker:
            mock_stock = Mock()
            mock_stock.info = {'longName': 'Test'}
            mock_stock.history.return_value.empty = True
            mock_ticker.return_value = mock_stock

            # Mock score outside valid range
            self.swarm.gemini.generate = Mock(return_value="Score: 150 | Grund: Test")

            results = self.swarm.stage1_scan(['AAPL'], variant='balanced')

            # Score should be clamped to valid range (0-100)
            if results:
                self.assertLessEqual(results[0]['score'], 100)
                self.assertGreaterEqual(results[0]['score'], 0)


if __name__ == '__main__':
    unittest.main()
