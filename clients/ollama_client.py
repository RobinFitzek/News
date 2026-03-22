"""
Ollama Client — local LLM fallback when Gemini/Perplexity monthly budget is exhausted.

Calls the Ollama REST API on localhost:11434. The user must have Ollama installed
and a model pulled (e.g. `ollama pull llama3`). This client is only activated when:
  - ollama_enabled = True in settings
  - The primary Gemini budget is exhausted for the current billing cycle

No external dependencies beyond the Python standard library (uses urllib).
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3"
OLLAMA_TIMEOUT = 120  # seconds — local models can be slow


class OllamaClient:
    """Thin wrapper around the Ollama REST API for local LLM inference."""

    def __init__(self):
        self._host: Optional[str] = None
        self._model: Optional[str] = None
        self._available: Optional[bool] = None  # cached health check result

    def _get_config(self) -> tuple[str, str, bool]:
        """Load settings from DB (lazy, to avoid circular imports at module load)."""
        try:
            from core.database import db
            host = db.get_setting("ollama_host") or OLLAMA_DEFAULT_HOST
            model = db.get_setting("ollama_model") or OLLAMA_DEFAULT_MODEL
            enabled = str(db.get_setting("ollama_enabled") or "false").lower() == "true"
            return host.rstrip("/"), model, enabled
        except Exception:
            return OLLAMA_DEFAULT_HOST, OLLAMA_DEFAULT_MODEL, False

    def is_enabled(self) -> bool:
        """Return True if Ollama is enabled in settings."""
        _, _, enabled = self._get_config()
        return enabled

    def health_check(self) -> bool:
        """Ping the Ollama server. Returns True if reachable."""
        host, _, _ = self._get_config()
        try:
            req = urllib.request.Request(
                f"{host}/api/tags",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """Return the list of locally available model names."""
        host, _, _ = self._get_config()
        try:
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Send a prompt to the local Ollama model and return the response text.

        Returns an error string (starting with '  ') on failure so callers can
        treat it the same way as Gemini's error returns.
        """
        host, cfg_model, enabled = self._get_config()
        if not enabled:
            return "  Ollama ist deaktiviert. Bitte in den Einstellungen aktivieren."

        active_model = model or cfg_model

        payload = json.dumps({
            "model": active_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                data = json.loads(resp.read())
                text = data.get("response", "").strip()
                if not text:
                    return "  Ollama returned an empty response."
                logger.info(f"Ollama ({active_model}): generated {len(text)} chars")
                return text
        except urllib.error.URLError as e:
            logger.warning(f"Ollama request failed (URLError): {e}")
            return f"  Ollama nicht erreichbar: {e}"
        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")
            return f"  Ollama Fehler: {e}"


# Module-level singleton
ollama_client = OllamaClient()
