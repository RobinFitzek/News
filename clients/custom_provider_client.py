"""
Custom OpenAI-compatible provider client.
Supports user-defined APIs for stage roles (news, synthesis, discovery).
"""
import logging
from typing import Optional, Dict, List, Tuple

import requests

from core.database import db


class CustomProviderClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _build_candidate_endpoints(self, base_url: str):
        base = (base_url or '').rstrip('/')
        if not base:
            return []

        if base.endswith('/v1/chat/completions') or base.endswith('/chat/completions'):
            return [base]

        candidates = []

        # If caller provides a root host (e.g. https://integrate.api.nvidia.com),
        # many OpenAI-compatible gateways expect /v1/chat/completions.
        if base.endswith('/v1'):
            candidates.append(f"{base}/chat/completions")
        else:
            candidates.append(f"{base}/v1/chat/completions")
            candidates.append(f"{base}/chat/completions")

        unique_candidates = []
        for endpoint in candidates:
            if endpoint not in unique_candidates:
                unique_candidates.append(endpoint)
        return unique_candidates

    def _provider_specific_hint(self, provider: Dict, diagnostics: Dict) -> Optional[str]:
        base_url = (diagnostics.get("base_url") or "").lower()
        status_code = diagnostics.get("last_status_code")
        last_error = (diagnostics.get("last_error_message") or "").lower()
        provider_name = (provider.get('name') or '').lower()

        is_nvidia = ('integrate.api.nvidia.com' in base_url) or ('nvidia' in provider_name)
        if is_nvidia and status_code == 404:
            return (
                "NVIDIA endpoint returned 404. In Settings use base URL https://integrate.api.nvidia.com/v1 "
                "(or https://integrate.api.nvidia.com) and a valid NVIDIA model id. "
                "If you already use /v1, verify the model name in NVIDIA API Catalog."
            )

        if status_code == 401 or 'unauthorized' in last_error:
            return "Unauthorized request. Verify API key and ensure it is active for this provider."

        return None

    def _extract_error_message(self, response: Optional[requests.Response], fallback: str) -> str:
        if response is None:
            return fallback

        try:
            body = response.json()
            if isinstance(body, dict):
                if isinstance(body.get('error'), dict):
                    msg = body['error'].get('message') or body['error'].get('code')
                    if msg:
                        return str(msg)
                if body.get('error'):
                    return str(body.get('error'))
                if body.get('message'):
                    return str(body.get('message'))
        except Exception:
            pass

        text = (response.text or '').strip()
        if text:
            return text[:300]

        return fallback

    def _extract_content(self, data: Dict) -> str:
        """Extract assistant text across OpenAI-compatible response variants."""
        if not isinstance(data, dict):
            return ""

        choices = data.get('choices')
        if isinstance(choices, list) and choices:
            first = choices[0] or {}

            message = first.get('message')
            if isinstance(message, dict):
                for key in ('content', 'reasoning_content', 'reasoning'):
                    value = message.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

            # Some providers return direct text fields on choice
            for key in ('text', 'content', 'reasoning_content', 'reasoning'):
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # Fallback variants used by some compatible gateways
        output_text = data.get('output_text')
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return ""

    def _generate_with_diagnostics(
        self,
        provider: Dict,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Optional[str], Dict]:
        api_key = provider.get('api_key')
        base_url = (provider.get('base_url') or '').rstrip('/')
        model = provider.get('model')

        diagnostics = {
            "base_url": base_url,
            "model": model,
            "attempted_endpoints": [],
            "last_status_code": None,
            "last_error_message": None,
            "http_ok": False,
            "first_http_error_status": None,
            "first_http_error_message": None,
        }

        if not api_key or not base_url or not model:
            diagnostics["last_error_message"] = "Missing API key, base URL, or model"
            return None, diagnostics

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exception = None
        endpoints = self._build_candidate_endpoints(base_url)
        for endpoint in endpoints:
            diagnostics["attempted_endpoints"].append(endpoint)
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=35)
                response.raise_for_status()
                diagnostics["http_ok"] = True
                data = response.json()
                content = self._extract_content(data)
                if not content:
                    msg = f"Empty content returned at {endpoint}"
                    diagnostics["last_error_message"] = msg
                    self.logger.error(
                        f"Custom provider returned empty content ({provider.get('name', 'unknown')}) at {endpoint}. "
                        f"Body: {str(data)[:300]}"
                    )
                    # Request succeeded (HTTP 2xx), so avoid falling through to alternate
                    # endpoints that can produce misleading 404 noise.
                    return None, diagnostics

                input_tokens = len((system_prompt or "") + (user_prompt or "")) // 4
                output_tokens = len(content) // 4
                db.log_provider_call(
                    provider_id=provider['id'],
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost=0.0,
                )
                diagnostics["last_error_message"] = None
                diagnostics["last_status_code"] = 200
                return content, diagnostics
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                diagnostics["last_status_code"] = status_code
                response_text = self._extract_error_message(e.response, str(e))
                diagnostics["last_error_message"] = response_text
                if diagnostics["first_http_error_status"] is None:
                    diagnostics["first_http_error_status"] = status_code
                    diagnostics["first_http_error_message"] = response_text
                self.logger.error(
                    f"Custom provider HTTP error ({provider.get('name', 'unknown')}) at {endpoint}: "
                    f"{e}. Response: {response_text}",
                )
                last_exception = e
                if status_code in (401, 403):
                    return None, diagnostics
            except Exception as e:
                diagnostics["last_error_message"] = str(e)
                self.logger.error(
                    f"Custom provider call failed ({provider.get('name', 'unknown')}) at {endpoint}: {e}",
                    exc_info=True,
                )
                last_exception = e

        if last_exception:
            self.logger.error(
                f"All endpoint attempts failed for provider {provider.get('name', 'unknown')}. "
                f"Base URL: {base_url}"
            )
        return None, diagnostics

    def get_provider_for_role(self, role: str) -> Optional[Dict]:
        return db.get_api_provider_for_role(role)

    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        if not provider:
            return None
        content, _ = self._generate_with_diagnostics(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content

    def test_connection(self, provider: Dict) -> Dict:
        if not provider:
            return {"status": "error", "error": "provider_not_found"}

        ping, diagnostics = self._generate_with_diagnostics(
            provider=provider,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly: OK",
            temperature=0,
            max_tokens=12,
        )

        if ping:
            return {"status": "ok", "message": "Provider reachable"}

        if diagnostics.get("http_ok"):
            return {
                "status": "ok",
                "message": "Provider reachable, but model returned no displayable text.",
                "details": {
                    "attempted_endpoints": diagnostics.get("attempted_endpoints") or [],
                    "last_error": diagnostics.get("last_error_message") or "empty_content",
                },
            }

        status_code = diagnostics.get("first_http_error_status") or diagnostics.get("last_status_code")
        endpoints: List[str] = diagnostics.get("attempted_endpoints") or []
        last_error = diagnostics.get("first_http_error_message") or diagnostics.get("last_error_message") or "request_failed"

        if status_code in (401, 403):
            hint = self._provider_specific_hint(provider, diagnostics)
            return {
                "status": "error",
                "error": "unauthorized",
                "message": hint or "Unauthorized request. Verify API key and access for this model/provider.",
                "details": {
                    "attempted_endpoints": endpoints,
                    "last_error": last_error,
                },
            }

        if status_code == 404:
            hint = self._provider_specific_hint(provider, diagnostics)
            return {
                "status": "error",
                "error": "endpoint_not_found",
                "message": hint or "Provider returned 404. Endpoint not found. Verify base URL/model for this vendor.",
                "details": {
                    "attempted_endpoints": endpoints,
                    "last_error": last_error,
                },
            }

        hint = self._provider_specific_hint(provider, diagnostics)

        return {
            "status": "error",
            "error": "request_failed",
            "message": hint or f"Provider test failed: {last_error}",
            "details": {
                "attempted_endpoints": endpoints,
            },
        }


custom_provider_client = CustomProviderClient()
