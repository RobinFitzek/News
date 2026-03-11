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


class TestExtractSection(unittest.TestCase):
    """Unit tests for the extract_section regex inside InvestmentSwarm."""

    # Minimal Gemini-style response containing all sections
    SAMPLE_RESPONSE = (
        "Risk Score: 7/10\n"
        "Geo-Risiko: 6 — Exportbeschränkungen durch US-China-Konflikt\n"
        "Bull Case: Starkes Wachstum in Cloud-Segment, Margensteigerung erwartet.\n"
        "Bear Case: Zölle könnten Margen um 3–5% drücken.\n"
        "Quellen: Reuters, Bloomberg\n"
        "Zusammenfassung: Neutral — abwarten."
    )

    def _run_extract(self, text: str, header: str) -> str:
        """Run extract_section via the private helper inside analyze_ticker."""
        import re

        def extract_section(t: str, h: str) -> str:
            pattern = rf"{h}:\s*(.*?)(?=\n(?:Risk Score|Geo-Risiko|Bull Case|Bear Case|Quellen|Zusammenfassung):|$)"
            m = re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        return extract_section(text, header)

    def test_bull_case_not_swallowed_by_geo_risiko(self):
        """Geo-Risiko in lookahead must prevent Bull Case from being eaten by Risk Score."""
        bull = self._run_extract(self.SAMPLE_RESPONSE, "Bull Case")
        self.assertIn("Cloud-Segment", bull)
        self.assertNotIn("Geo-Risiko", bull)

    def test_risk_score_stops_at_geo_risiko(self):
        """Risk Score section must end before Geo-Risiko line."""
        risk = self._run_extract(self.SAMPLE_RESPONSE, "Risk Score")
        self.assertIn("7/10", risk)
        self.assertNotIn("Exportbeschränkungen", risk)

    def test_geo_risiko_extracted_correctly(self):
        """Geo-Risiko section must contain its own content and not bleed into Bull Case."""
        geo = self._run_extract(self.SAMPLE_RESPONSE, "Geo-Risiko")
        self.assertIn("Exportbeschränkungen", geo)
        self.assertNotIn("Starkes Wachstum", geo)

    def test_bear_case_extracted_correctly(self):
        """Bear Case must be extracted and not bled into Quellen."""
        bear = self._run_extract(self.SAMPLE_RESPONSE, "Bear Case")
        self.assertIn("Zölle", bear)
        self.assertNotIn("Reuters", bear)

    def test_empty_section_returns_empty_string(self):
        """Missing section header must return empty string, not raise."""
        result = self._run_extract(self.SAMPLE_RESPONSE, "Nonexistent Section")
        self.assertEqual(result, "")


class TestSeverityParsing(unittest.TestCase):
    """Unit tests for the SCHWEREGRAD regex used in scheduler and database."""

    def _parse_scores(self, text: str):
        import re
        return [int(m) for m in re.findall(r'SCHWEREGRAD[:\s/]+(\d+)', text)]

    def test_parses_multiple_scores(self):
        text = "SCHWEREGRAD: 8\nSCHWEREGRAD: 5\nSCHWEREGRAD: 9"
        self.assertEqual(self._parse_scores(text), [8, 5, 9])

    def test_parses_score_with_slash(self):
        text = "SCHWEREGRAD/10: 7"
        scores = self._parse_scores(text)
        self.assertIn(7, scores)

    def test_threshold_8_triggers_alert(self):
        text = "SCHWEREGRAD: 8 — Kritischer Konflikt"
        scores = self._parse_scores(text)
        self.assertTrue(max(scores) >= 8)

    def test_threshold_7_does_not_trigger(self):
        text = "SCHWEREGRAD: 7 — Erhöhte Spannung"
        scores = self._parse_scores(text)
        self.assertFalse(max(scores) >= 8)

    def test_no_scores_returns_empty(self):
        text = "Alles ruhig, keine Ereignisse."
        self.assertEqual(self._parse_scores(text), [])


if __name__ == '__main__':
    unittest.main()
