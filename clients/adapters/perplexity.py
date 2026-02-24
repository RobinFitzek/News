"""
Perplexity Adapter
Uses Perplexity's native API with web search capabilities.
Supports search_domain_filter and search_recency_filter.
"""
import requests
import json
import logging
import time
from typing import Optional, Dict
from clients.adapters.base import BaseProviderAdapter


class PerplexityAdapter(BaseProviderAdapter):
    """Adapter for Perplexity API with web search integration."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db = None
        self._session = self._create_session()

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    def _create_session(self) -> requests.Session:
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        if not provider or not provider.get('api_key'):
            return None

        api_key = provider['api_key']
        base_url = (provider.get('base_url') or 'https://api.perplexity.ai').rstrip('/')
        model = provider.get('model') or 'sonar'
        provider_id = provider.get('id', 0)

        # Parse extra_config for Perplexity-specific features
        extra = json.loads(provider.get('extra_config', '{}') or '{}')
        domains = extra.get('search_domains', [
            "finance.yahoo.com", "bloomberg.com", "reuters.com",
            "seekingalpha.com", "marketwatch.com"
        ])
        recency = extra.get('search_recency', 'day')

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Add Perplexity-specific search features if domains given
        if domains:
            payload["search_domain_filter"] = domains
        if recency:
            payload["search_recency_filter"] = recency

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self._session.post(
                    f"{base_url}/chat/completions",
                    headers=headers, json=payload, timeout=30,
                )

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 2))
                    time.sleep(retry_after)
                    continue

                if resp.status_code in (401, 403):
                    self.logger.error(f"Perplexity auth error: {resp.status_code}")
                    try:
                        self.db.raise_system_alert(
                            'perplexity_auth', 'Perplexity API Key Invalid',
                            'API key rejected. News analysis disabled until key is updated.',
                            severity='error', service='perplexity',
                            action_url='/settings', action_label='Update API Key')
                    except Exception:
                        pass
                    return None

                resp.raise_for_status()
                data = resp.json()

                choices = data.get('choices', [])
                if not choices:
                    return None

                content = choices[0].get('message', {}).get('content', '').strip()
                if not content:
                    return None

                # Log tokens
                usage = data.get('usage', {})
                in_tok = usage.get('prompt_tokens', 800)
                out_tok = usage.get('completion_tokens', 400)
                try:
                    self.db.log_provider_call(provider_id, model, in_tok, out_tok)
                except Exception:
                    pass

                # Clear auth alert on success
                try:
                    self.db.clear_system_alert('perplexity_auth')
                except Exception:
                    pass

                return content

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None
            except requests.exceptions.HTTPError:
                return None
            except Exception as e:
                self.logger.error(f"Perplexity adapter error: {e}")
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
            temperature=0, max_tokens=12,
        )

        if result:
            return {"status": "ok", "message": f"Perplexity reachable ({provider.get('model', 'sonar')})"}

        return {
            "status": "error", "error": "request_failed",
            "message": "Perplexity API test failed. Check API key.",
        }

    def is_configured(self, provider: Dict) -> bool:
        return bool(provider.get('api_key'))
