"""
Base Adapter Interface for AI Providers.
All provider adapters must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict


class BaseProviderAdapter(ABC):
    """Unified interface for all AI provider types."""

    @abstractmethod
    def generate(self, provider: Dict, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        """Generate text from the AI provider.

        Args:
            provider: Dict with at least {base_url, api_key, model, name, id, extra_config}
            system_prompt: System/instruction prompt
            user_prompt: User query/prompt
            temperature: Sampling temperature
            max_tokens: Max tokens for response

        Returns:
            Generated text or None on failure
        """
        pass

    @abstractmethod
    def test_connection(self, provider: Dict) -> Dict:
        """Test if the provider is reachable and responds.

        Returns:
            Dict with {status: 'ok'|'error', message: str, ...}
        """
        pass

    @abstractmethod
    def is_configured(self, provider: Dict) -> bool:
        """Check if this provider has the minimum config to work."""
        pass

    def get_adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return self.__class__.__name__
