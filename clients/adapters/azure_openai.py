"""
Azure OpenAI Adapter

Azure uses a different authentication header (api-key instead of
Authorization: Bearer) and a deployment-specific URL format:
  https://{resource}.openai.azure.com/openai/deployments/{deployment}

Set base_url to the full deployment URL — the adapter appends
/chat/completions?api-version=... automatically.
"""
import requests
import logging
from typing import Optional, Dict, List
from clients.adapters.base import BaseProviderAdapter

# Default API version — can be overridden via provider extra_config
DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIAdapter(BaseProviderAdapter):
    """Adapter for Azure-hosted OpenAI models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    def _build_endpoint(self, base_url: str, api_version: str) -> str:
        base = base_url.rstrip('/')
        if '/chat/completions' in base:
            # Already a full endpoint — just ensure api-version is present
            sep = '&' if '?' in base else '?'
            return f"{base}{sep}api-version={api_version}" if 'api-version' not in base else base
        return f"{base}/chat/completions?api-version={api_version}"

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        api_key = provider.get('api_key') or ''
        model = provider.get('model') or ''
        provider_id = provider.get('id', 0)
        base_url = (provider.get('base_url') or '').rstrip('/')

        extra = provider.get('extra_config') or {}
        if isinstance(extra, str):
            import json
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        api_version = extra.get('api_version', DEFAULT_API_VERSION)

        endpoint = self._build_endpoint(base_url, api_version)
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Azure ignores the model field in the body (it's baked into the deployment URL)
        # but some deployments accept it — include it if provided
        if model:
            payload["model"] = model

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get('choices', [])
                text = choices[0].get('message', {}).get('content', '').strip() if choices else None
                if text:
                    usage = data.get('usage', {})
                    in_tok = usage.get('prompt_tokens', len(system_prompt + user_prompt) // 4)
                    out_tok = usage.get('completion_tokens', len(text) // 4)
                    try:
                        self.db.log_provider_call(provider_id, model or 'azure', in_tok, out_tok)
                    except Exception:
                        pass
                return text

            err = resp.text[:300] if resp.text else f"HTTP {resp.status_code}"
            self.logger.error("Azure OpenAI HTTP %s: %s", resp.status_code, err)
            return None

        except requests.exceptions.Timeout:
            self.logger.error("Azure OpenAI timed out for %s", base_url)
            return None
        except Exception as e:
            self.logger.error("Azure OpenAI request failed: %s", e)
            return None

    def test_connection(self, provider: Dict) -> Dict:
        if not provider.get('base_url'):
            return {
                "status": "error",
                "error": "missing_base_url",
                "message": "Set base_url to your Azure deployment URL: https://{resource}.openai.azure.com/openai/deployments/{deployment}",
            }
        result = self.generate(
            provider=provider,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly: OK",
            temperature=0,
            max_tokens=10,
        )
        if result:
            return {"status": "ok", "message": "Azure OpenAI reachable."}
        return {
            "status": "error",
            "error": "request_failed",
            "message": "Could not reach Azure OpenAI. Check your resource URL, deployment name, and api-key.",
        }

    def is_configured(self, provider: Dict) -> bool:
        return bool(provider.get('base_url') and provider.get('api_key'))
