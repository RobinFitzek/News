"""
Anthropic Adapter — Claude models via the Anthropic Messages API.

The Anthropic API is not OpenAI-compatible:
- Uses x-api-key header (not Authorization: Bearer)
- Uses /v1/messages endpoint with a dedicated 'system' field
- Response: {"content": [{"type": "text", "text": "..."}], "usage": {...}}
"""
import requests
import logging
from typing import Optional, Dict
from clients.adapters.base import BaseProviderAdapter

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    def _endpoint(self, provider: Dict) -> str:
        base = (provider.get('base_url') or ANTHROPIC_API_URL).rstrip('/')
        if base.endswith('/v1/messages'):
            return base
        if base.endswith('/v1'):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        api_key = provider.get('api_key') or ''
        model = provider.get('model') or 'claude-sonnet-4-6'
        provider_id = provider.get('id', 0)
        endpoint = self._endpoint(provider)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                text = self._extract_content(data)
                if text:
                    usage = data.get('usage', {})
                    in_tok = usage.get('input_tokens', len(system_prompt + user_prompt) // 4)
                    out_tok = usage.get('output_tokens', len(text) // 4)
                    try:
                        self.db.log_provider_call(provider_id, model, in_tok, out_tok)
                    except Exception:
                        pass
                return text

            err = self._parse_error(resp)
            self.logger.error("Anthropic HTTP %s for model %s: %s", resp.status_code, model, err)
            return None

        except requests.exceptions.Timeout:
            self.logger.error("Anthropic request timed out (model: %s)", model)
            return None
        except Exception as e:
            self.logger.error("Anthropic request failed: %s", e)
            return None

    def _extract_content(self, data: Dict) -> Optional[str]:
        try:
            for block in data.get('content', []):
                if block.get('type') == 'text':
                    return block['text'].strip() or None
        except (KeyError, TypeError):
            pass
        return None

    def _parse_error(self, resp) -> str:
        try:
            body = resp.json()
            return body.get('error', {}).get('message') or resp.text[:200]
        except Exception:
            return resp.text[:200] if resp.text else f"HTTP {resp.status_code}"

    def test_connection(self, provider: Dict) -> Dict:
        api_key = provider.get('api_key') or ''
        if not api_key:
            return {"status": "error", "error": "missing_api_key", "message": "API key required for Anthropic."}

        result = self.generate(
            provider=provider,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly: OK",
            temperature=0,
            max_tokens=10,
        )
        if result:
            return {"status": "ok", "message": "Anthropic reachable."}

        return {
            "status": "error",
            "error": "request_failed",
            "message": "Could not reach Anthropic. Check your API key at console.anthropic.com.",
        }

    def is_configured(self, provider: Dict) -> bool:
        return bool(provider.get('api_key') and provider.get('model'))
