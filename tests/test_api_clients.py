"""
Unit tests for API clients (Gemini and Perplexity)
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from clients.gemini_client import AdaptiveGeminiClient
from clients.perplexity_client import EnhancedPerplexityClient
from datetime import datetime


class TestGeminiClient(unittest.TestCase):
    """Test Gemini API client"""

    def setUp(self):
        """Set up test client"""
        with patch('clients.gemini_client.db'):
            self.client = AdaptiveGeminiClient()

    def test_client_initialization(self):
        """Test client initializes properly"""
        self.assertIsNotNone(self.client)
        self.assertIsNotNone(self.client.models)
        self.assertIsNotNone(self.client.daily_budget)

    def test_is_configured_with_no_key(self):
        """Test is_configured returns False when no API key"""
        self.client.api_key = None
        self.assertFalse(self.client.is_configured())

    def test_is_configured_with_key(self):
        """Test is_configured returns True with valid key"""
        self.client.api_key = "test_key_123456789"
        self.client.client = Mock()
        self.assertTrue(self.client.is_configured())

    def test_get_usage_structure(self):
        """Test get_usage returns proper structure"""
        usage = self.client.get_usage()

        self.assertIn('flash', usage)
        self.assertIn('pro', usage)
        self.assertIn('is_configured', usage)

        # Check flash structure
        self.assertIn('used_today', usage['flash'])
        self.assertIn('daily_limit', usage['flash'])
        self.assertIn('remaining', usage['flash'])

    def test_daily_reset(self):
        """Test daily usage resets"""
        # Simulate usage
        self.client.requests['flash-8b']['day'] = [datetime.now()]
        self.assertEqual(len(self.client.requests['flash-8b']['day']), 1)

        # Force reset by changing date
        from datetime import timedelta
        self.client.last_reset_date = datetime.now().date() - timedelta(days=1)

        # Check reset
        self.client._check_daily_reset()
        self.assertEqual(len(self.client.requests['flash-8b']['day']), 0)

    def test_fallback_tier_selection(self):
        """Test fallback tier selection"""
        # Test pro -> flash-2.5 fallback
        self.client.requests['pro']['day'] = list(range(10))  # Exhaust pro
        fallback = self.client._get_fallback_tier('pro')
        self.assertEqual(fallback, 'flash-2.5')

    @patch('clients.gemini_client.genai.Client')
    def test_generate_with_invalid_prompt(self, mock_client):
        """Test generate handles invalid prompts"""
        self.client.api_key = "test_key"
        self.client.client = Mock()

        # Test empty prompt
        result = self.client.generate("")
        self.assertIn("⚠️", result)

        # Test None prompt
        result = self.client.generate(None)
        self.assertIn("⚠️", result)

    @patch('clients.gemini_client.genai.Client')
    def test_generate_handles_api_errors(self, mock_client):
        """Test generate handles API errors gracefully"""
        self.client.api_key = "test_key"
        self.client.client = Mock()
        self.client.client.models.generate_content = Mock(side_effect=Exception("API Error"))

        result = self.client.generate("test prompt", tier='flash-8b')
        self.assertIn("Fehler", result)


class TestPerplexityClient(unittest.TestCase):
    """Test Perplexity API client"""

    def setUp(self):
        """Set up test client"""
        self.client = EnhancedPerplexityClient()
        self.client.api_key = "test_key_123456789"

    def test_client_initialization(self):
        """Test client initializes properly"""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.requests_used_today, 0)
        self.assertIsNotNone(self.client.session)

    def test_is_configured_with_no_key(self):
        """Test is_configured returns False when no API key"""
        self.client.api_key = None
        self.assertFalse(self.client.is_configured())

    def test_is_configured_with_key(self):
        """Test is_configured returns True with valid key"""
        self.client.api_key = "test_key_123456789"
        self.assertTrue(self.client.is_configured())

    def test_get_usage_structure(self):
        """Test get_usage returns proper structure"""
        usage = self.client.get_usage()

        self.assertIn('used_today', usage)
        self.assertIn('daily_limit', usage)
        self.assertIn('remaining', usage)
        self.assertIn('is_configured', usage)

    def test_daily_limit_enforcement(self):
        """Test daily limit is enforced"""
        self.client.requests_used_today = self.client.daily_limit

        result = self.client._call_api("test system", "test query")
        self.assertIsNone(result)

    def test_call_api_with_invalid_inputs(self):
        """Test _call_api handles invalid inputs"""
        # Empty prompt
        result = self.client._call_api("", "test query")
        self.assertIsNone(result)

        # Empty query
        result = self.client._call_api("test system", "")
        self.assertIsNone(result)

    @patch('clients.perplexity_client.requests.Session.post')
    def test_call_api_handles_timeout(self, mock_post):
        """Test _call_api handles timeout errors"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        result = self.client._call_api("test system", "test query")
        self.assertIsNone(result)

    @patch('clients.perplexity_client.requests.Session.post')
    def test_call_api_handles_connection_error(self, mock_post):
        """Test _call_api handles connection errors"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = self.client._call_api("test system", "test query")
        self.assertIsNone(result)

    @patch('clients.perplexity_client.requests.Session.post')
    def test_call_api_handles_rate_limit(self, mock_post):
        """Test _call_api handles rate limiting"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '1'}
        mock_post.return_value = mock_response

        result = self.client._call_api("test system", "test query")
        # Should retry but eventually fail
        self.assertIsNone(result)

    @patch('clients.perplexity_client.requests.Session.post')
    def test_call_api_success(self, mock_post):
        """Test successful API call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response

        result = self.client._call_api("test system", "test query")
        self.assertEqual(result, 'Test response')
        self.assertEqual(self.client.requests_used_today, 1)


class TestAPIClientIntegration(unittest.TestCase):
    """Integration tests for API clients"""

    @patch('clients.gemini_client.genai.Client')
    @patch('clients.gemini_client.db')
    def test_gemini_with_multiple_tiers(self, mock_db, mock_genai):
        """Test Gemini client with multiple tier requests"""
        client = AdaptiveGeminiClient()
        client.api_key = "test_key"
        client.client = Mock()

        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Test response"
        client.client.models.generate_content.return_value = mock_response

        # Make requests at different tiers
        result1 = client.generate("test prompt 1", tier='flash-8b')
        result2 = client.generate("test prompt 2", tier='flash')
        result3 = client.generate("test prompt 3", tier='pro')

        self.assertNotIn("⚠️", result1)
        self.assertNotIn("⚠️", result2)
        self.assertNotIn("⚠️", result3)


if __name__ == '__main__':
    unittest.main()
