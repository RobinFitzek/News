"""
Google Gemini Adapter
Uses the official google-genai SDK for Gemini models.
Supports adaptive model selection and budget-aware operation.
"""
import logging
import time
import json
from typing import Optional, Dict
from clients.adapters.base import BaseProviderAdapter


class GoogleGeminiAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini API using the native genai SDK."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._clients = {}  # Cache genai clients by api_key hash
        self._budget_tracker = None
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    @property
    def budget_tracker(self):
        if self._budget_tracker is None:
            from core.budget_tracker import budget_tracker
            self._budget_tracker = budget_tracker
        return self._budget_tracker

    def _get_client(self, api_key: str):
        """Get or create a genai client for this api_key."""
        key_hash = hash(api_key) if api_key else 0
        if key_hash not in self._clients:
            try:
                from google import genai
                self._clients[key_hash] = genai.Client(api_key=api_key)
            except Exception as e:
                self.logger.error(f"Failed to create Gemini client: {e}")
                return None
        return self._clients[key_hash]

    def _get_model_name(self, provider: Dict) -> str:
        """Resolve model name from provider config."""
        model = provider.get('model', 'gemini-2.5-flash')
        # Allow tier aliases
        extra = json.loads(provider.get('extra_config', '{}') or '{}')
        tier_map = extra.get('tier_map', {})
        if model in tier_map:
            return tier_map[model]
        return model

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        if not provider or not provider.get('api_key'):
            return None

        api_key = provider['api_key']
        client = self._get_client(api_key)
        if not client:
            return None

        model_name = self._get_model_name(provider)
        provider_id = provider.get('id', 0)

        # Combine system + user prompt for Gemini (single-prompt interface)
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                )

                if not response or not hasattr(response, 'text') or not response.text:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    return None

                response_text = response.text

                # Extract token counts
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    um = response.usage_metadata
                    input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(um, 'candidates_token_count', 0) or 0
                if input_tokens == 0:
                    input_tokens = len(full_prompt) // 4
                if output_tokens == 0:
                    output_tokens = len(response_text) // 4

                # Log cost
                try:
                    self.db.log_provider_call(provider_id, model_name, input_tokens, output_tokens)
                except Exception:
                    pass

                self.logger.info(f"Gemini adapter OK: {model_name} ({input_tokens}in/{output_tokens}out)")
                return response_text

            except Exception as e:
                error_name = type(e).__name__
                self.logger.error(f"Gemini adapter error ({error_name}): {e}")

                # Check for specific Google API exceptions
                if 'PermissionDenied' in error_name:
                    try:
                        self.db.raise_system_alert(
                            'gemini_auth', 'Gemini API Key Invalid',
                            'API key rejected. AI analysis disabled until key is updated.',
                            severity='error', service='gemini',
                            action_url='/settings', action_label='Update API Key')
                    except Exception:
                        pass
                    return None

                if 'ResourceExhausted' in error_name:
                    self.logger.error(f"Gemini quota exhausted for {model_name}")
                    return None

                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None

        return None

    def test_connection(self, provider: Dict) -> Dict:
        if not provider or not provider.get('api_key'):
            return {"status": "error", "error": "no_api_key", "message": "No API key configured"}

        result = self.generate(
            provider,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly: OK",
            temperature=0,
            max_tokens=12,
        )

        if result:
            return {"status": "ok", "message": f"Gemini reachable ({self._get_model_name(provider)})"}

        return {
            "status": "error",
            "error": "request_failed",
            "message": "Gemini API test failed. Check API key and model name.",
        }

    def is_configured(self, provider: Dict) -> bool:
        return bool(provider.get('api_key') and provider.get('model'))
