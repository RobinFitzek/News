"""
OpenAI-Compatible Adapter
Works with any provider exposing an OpenAI-like /v1/chat/completions endpoint.
(OpenAI, Groq, Together, Ollama, OpenRouter, etc.)
"""
import requests
import json
import logging
import time
from typing import Optional, Dict, List
from clients.adapters.base import BaseProviderAdapter


class OpenAICompatibleAdapter(BaseProviderAdapter):
    """Adapter for any OpenAI-compatible API (the most common format)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    def _build_candidate_endpoints(self, base_url: str) -> List[str]:
        """Build list of candidate endpoints to try."""
        base = base_url.rstrip('/')
        if '/v1/chat/completions' in base or '/chat/completions' in base:
            return [base]
        # If the URL already includes /v1 (e.g. https://integrate.api.nvidia.com/v1),
        # appending /v1/chat/completions would produce a double /v1/v1/ path.
        # Append /chat/completions directly in that case.
        if base.endswith('/v1'):
            return [f"{base}/chat/completions"]
        return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        if not provider or not provider.get('base_url') or not provider.get('model'):
            return None

        content, _ = self._generate_with_diagnostics(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=90,
        )
        return content

    def _generate_with_diagnostics(self, provider: Dict, system_prompt: str, user_prompt: str,
                                    temperature: float = 0.2, max_tokens: int = 900,
                                    timeout: int = 90):
        base_url = (provider.get('base_url') or '').rstrip('/')
        api_key = provider.get('api_key') or ''
        model = provider.get('model') or ''
        provider_id = provider.get('id', 0)
        provider_name = provider.get('name', 'unknown')

        endpoints = self._build_candidate_endpoints(base_url)
        diagnostics = {"attempted_endpoints": endpoints}

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exception = None
        for endpoint in endpoints:
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                diagnostics["last_status_code"] = resp.status_code

                if resp.status_code == 200:
                    data = resp.json()
                    content = self._extract_content(data)
                    if content:
                        # Log tokens
                        usage = data.get('usage', {})
                        in_tok = usage.get('prompt_tokens', len(system_prompt + user_prompt) // 4)
                        out_tok = usage.get('completion_tokens', len(content) // 4)
                        try:
                            self.db.log_provider_call(provider_id, model, in_tok, out_tok)
                        except Exception:
                            pass
                        return content, diagnostics
                    diagnostics["http_ok"] = True

                status_code = resp.status_code
                try:
                    err_body = resp.json()
                    err_msg = (
                        err_body.get('error', {}).get('message')
                        or err_body.get('message')
                        or resp.text[:200]
                    ) if isinstance(err_body, dict) else resp.text[:200]
                except Exception:
                    err_msg = resp.text[:200] if resp.text else f"HTTP {status_code}"
                diagnostics["last_error_message"] = err_msg
                if "first_http_error_status" not in diagnostics:
                    diagnostics["first_http_error_status"] = status_code
                self.logger.error(
                    "Custom provider HTTP %s (%s) at %s: %s",
                    status_code, provider_name, endpoint, err_msg,
                )
                if status_code in (401, 403):
                    return None, diagnostics

            except requests.exceptions.Timeout:
                msg = f"Request timed out after {timeout}s at {endpoint}"
                diagnostics["last_error_message"] = msg
                self.logger.error("Custom provider timeout (%s) at %s", provider_name, endpoint)
                last_exception = Exception(msg)
            except Exception as e:
                diagnostics["last_error_message"] = str(e)
                self.logger.error(
                    "Custom provider error (%s) at %s: %s", provider_name, endpoint, e,
                )
                last_exception = e

        return None, diagnostics

    def _extract_content(self, data: dict) -> Optional[str]:
        """Extract text from OpenAI-style response."""
        try:
            choices = data.get('choices', [])
            if choices:
                msg = choices[0].get('message', {})
                return msg.get('content', '').strip() or None
        except (IndexError, KeyError, AttributeError):
            pass
        return None

    def _nvidia_hint(self, provider: Dict, diagnostics: Dict) -> str | None:
        """Return a hint string for NVIDIA NIM errors, or None."""
        base_url = (provider.get('base_url') or '').lower()
        provider_name = (provider.get('name') or '').lower()
        is_nvidia = 'nvidia' in base_url or 'nvidia' in provider_name
        if not is_nvidia:
            return None
        status_code = diagnostics.get('last_status_code')
        if status_code == 404:
            return (
                "NVIDIA 404: Verify the model name in the NVIDIA API Catalog "
                "(https://build.nvidia.com). Base URL should be "
                "https://integrate.api.nvidia.com/v1"
            )
        if status_code in (401, 403):
            return (
                "NVIDIA auth error: Your API key (nvapi-...) may be invalid or expired. "
                "Generate a new key at https://build.nvidia.com."
            )
        return None

    def test_connection(self, provider: Dict) -> Dict:
        if not provider:
            return {"status": "error", "error": "provider_not_found"}

        ping, diagnostics = self._generate_with_diagnostics(
            provider=provider,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly: OK",
            temperature=0,
            max_tokens=12,
            timeout=20,
        )

        if ping:
            return {"status": "ok", "message": "Provider reachable"}

        if diagnostics.get("http_ok"):
            return {"status": "ok", "message": "Provider reachable, but model returned no text."}

        status_code = diagnostics.get("first_http_error_status") or diagnostics.get("last_status_code")
        endpoints = diagnostics.get("attempted_endpoints", [])
        last_error = diagnostics.get("last_error_message", "request_failed")

        nvidia_hint = self._nvidia_hint(provider, diagnostics)

        if status_code in (401, 403):
            return {
                "status": "error", "error": "unauthorized",
                "message": nvidia_hint or "Unauthorized. Verify API key and access.",
                "details": {"attempted_endpoints": endpoints, "last_error": last_error},
            }
        if status_code == 404:
            return {
                "status": "error", "error": "endpoint_not_found",
                "message": nvidia_hint or "404 Not Found. Verify base URL and model.",
                "details": {"attempted_endpoints": endpoints, "last_error": last_error},
            }

        # Timeout or other failure — check if it's a large/slow model
        is_timeout = last_error and "timed out" in last_error.lower()
        if is_timeout:
            timeout_msg = (
                f"Request timed out (20s). Model '{provider.get('model', '')}' may be loading — "
                "large models on NVIDIA NIM can take 60–120s on cold start. "
                "Try again in a moment, or switch to a faster model."
            )
            return {
                "status": "error", "error": "timeout",
                "message": nvidia_hint or timeout_msg,
                "details": {"attempted_endpoints": endpoints, "last_error": last_error},
            }

        return {
            "status": "error", "error": "request_failed",
            "message": nvidia_hint or f"Provider test failed: {last_error}",
            "details": {"attempted_endpoints": endpoints},
        }

    def is_configured(self, provider: Dict) -> bool:
        return bool(provider.get('base_url') and provider.get('model'))
