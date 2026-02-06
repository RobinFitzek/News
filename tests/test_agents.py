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
             patch('engine.agents.gemini_client'), \
             patch('engine.agents.quant_screener'):
            self.swarm = InvestmentSwarm()

    def test_swarm_initialization(self):
        """Test swarm initializes properly"""
        self.assertIsNotNone(self.swarm)
        self.assertIsNotNone(self.swarm.pplx)
        self.assertIsNotNone(self.swarm.gemini)
        self.assertIsNotNone(self.swarm.screener)

    @patch('engine.agents.yf.Ticker')
    def test_get_stock_data_with_valid_ticker(self, mock_ticker):
        """Test getting stock data with valid ticker"""
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

    def test_stage1_scan_uses_quant_screener(self):
        """Test stage 1 scan uses quant screener (not Gemini)"""
        self.swarm.screener.screen_batch = Mock(return_value=[
            {'ticker': 'AAPL', 'score': 75, 'composite_score': 75, 'signal': 'Opportunity',
             'initial_reason': 'Quant Score 75/100'},
            {'ticker': 'MSFT', 'score': 60, 'composite_score': 60, 'signal': 'Neutral',
             'initial_reason': 'Quant Score 60/100'},
        ])

        results = self.swarm.stage1_scan(['AAPL', 'MSFT'], variant='balanced')

        self.assertEqual(len(results), 2)
        self.swarm.screener.screen_batch.assert_called_once_with(['AAPL', 'MSFT'], 'balanced')
        # Verify Gemini was NOT called
        self.swarm.gemini.generate.assert_not_called()

    def test_stage1_scan_with_empty_list(self):
        """Test stage 1 scan with empty ticker list"""
        results = self.swarm.stage1_scan([], variant='balanced')

        self.assertEqual(len(results), 0)

    def test_stage1_scan_returns_signals_not_buy_sell(self):
        """Test that stage 1 returns Opportunity/Caution/Neutral, not Buy/Sell"""
        self.swarm.screener.screen_batch = Mock(return_value=[
            {'ticker': 'AAPL', 'score': 80, 'composite_score': 80, 'signal': 'Opportunity',
             'initial_reason': 'Test'},
        ])

        results = self.swarm.stage1_scan(['AAPL'], variant='balanced')

        self.assertEqual(results[0]['signal'], 'Opportunity')
        self.assertNotIn('Buy', results[0].get('signal', ''))
        self.assertNotIn('Sell', results[0].get('signal', ''))


if __name__ == '__main__':
    unittest.main()
