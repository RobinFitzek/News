"""
Provider Registry — Central hub for all AI providers.

This module replaces direct imports of gemini_client/pplx_client/custom_provider_client
in the engine layer. Instead, engine code calls:
    provider_registry.generate_for_stage('stage2_news', system_prompt, user_prompt)

The registry resolves which provider is assigned to the stage, loads the right adapter,
and calls it with the correct config.
"""
import json
import logging
from typing import Optional, Dict, List
from clients.adapters.base import BaseProviderAdapter
from clients.adapters.openai_compatible import OpenAICompatibleAdapter
from clients.adapters.google_gemini import GoogleGeminiAdapter
from clients.adapters.perplexity import PerplexityAdapter

logger = logging.getLogger(__name__)

# Adapter type → Adapter class mapping
ADAPTER_MAP = {
    'openai_compatible': OpenAICompatibleAdapter,
    'google_gemini': GoogleGeminiAdapter,
    'perplexity': PerplexityAdapter,
}

# Shortcut definitions for quick-add UI
PROVIDER_SHORTCUTS = {
    'gemini': {
        'name': 'Gemini',
        'adapter_type': 'google_gemini',
        'provider_type': 'llm',
        'base_url': 'https://generativelanguage.googleapis.com',
        'model': 'gemini-2.5-flash',
        'models': ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-3-flash'],
        'key_placeholder': 'AIzaSy...',
        'key_url': 'https://ai.google.dev',
        'description': 'Google Gemini — fast, budget-friendly synthesis & analysis',
        'icon': 'gemini',
    },
    'perplexity': {
        'name': 'Perplexity',
        'adapter_type': 'perplexity',
        'provider_type': 'search',
        'base_url': 'https://api.perplexity.ai',
        'model': 'sonar',
        'models': ['sonar', 'sonar-pro', 'sonar-reasoning'],
        'key_placeholder': 'pplx-xxx...',
        'key_url': 'https://perplexity.ai/settings',
        'description': 'Perplexity — real-time web search & news intelligence',
        'icon': 'perplexity',
    },
    'openai': {
        'name': 'OpenAI',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
        'models': ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini'],
        'key_placeholder': 'sk-...',
        'key_url': 'https://platform.openai.com/api-keys',
        'description': 'OpenAI — GPT models for analysis & synthesis',
        'icon': 'openai',
    },
    'groq': {
        'name': 'Groq',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'https://api.groq.com/openai/v1',
        'model': 'llama-3.3-70b-versatile',
        'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
        'key_placeholder': 'gsk_...',
        'key_url': 'https://console.groq.com/keys',
        'description': 'Groq — ultra-fast inference, generous free tier',
        'icon': 'groq',
    },
    'together': {
        'name': 'Together AI',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'https://api.together.xyz/v1',
        'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'models': ['meta-llama/Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
                   'Qwen/Qwen2.5-72B-Instruct-Turbo', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
        'key_placeholder': 'tok_...',
        'key_url': 'https://api.together.xyz/settings/api-keys',
        'description': 'Together AI — open-source models, fast inference',
        'icon': 'together',
    },
    'openrouter': {
        'name': 'OpenRouter',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'https://openrouter.ai/api/v1',
        'model': 'google/gemini-2.5-flash',
        'models': ['google/gemini-2.5-flash', 'anthropic/claude-sonnet-4', 'openai/gpt-4o-mini',
                   'meta-llama/llama-3.3-70b-instruct'],
        'key_placeholder': 'sk-or-...',
        'key_url': 'https://openrouter.ai/keys',
        'description': 'OpenRouter — unified gateway to 200+ models',
        'icon': 'openrouter',
    },
    'ollama': {
        'name': 'Ollama (Local)',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'http://localhost:11434/v1',
        'model': 'llama3.2',
        'models': ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5'],
        'key_placeholder': '(not required)',
        'key_url': 'https://ollama.ai',
        'description': 'Ollama — run models locally, zero cost, full privacy',
        'icon': 'ollama',
    },
    'nvidia': {
        'name': 'NVIDIA NIM',
        'adapter_type': 'openai_compatible',
        'provider_type': 'llm',
        'base_url': 'https://integrate.api.nvidia.com/v1',
        'model': 'meta/llama-3.3-70b-instruct',
        'models': [
            'meta/llama-3.3-70b-instruct',
            'nvidia/llama-3.3-nemotron-super-49b-v1',
            'nvidia/llama-3.1-nemotron-70b-instruct',
            'mistralai/mixtral-8x22b-instruct-v0.1',
            'nvidia/nemotron-4-340b-instruct',
        ],
        'key_placeholder': 'nvapi-...',
        'key_url': 'https://build.nvidia.com',
        'description': 'NVIDIA NIM — high-performance inference, 100 free credits/month',
        'icon': 'nvidia',
    },
}

# Stage metadata for the UI (descriptions, tooltips)
STAGE_INFO = {
    'stage2_news': {
        'label': 'Stage 2: News & Market Intel',
        'description': 'Fetches real-time news, analyst ratings, and market context for each stock. Web-search providers (like Perplexity) work best here.',
        'icon': 'news',
    },
    'stage3_synthesis': {
        'label': 'Stage 3: Research Synthesis',
        'description': 'Generates a structured research note combining quant data with news. Any LLM works well — Gemini, GPT, Llama, etc.',
        'icon': 'synthesis',
    },
    'discovery': {
        'label': 'Discovery',
        'description': 'Discovers new stock opportunities using AI-powered trend scanning. Web-search providers recommended.',
        'icon': 'search',
    },
    'insider_context': {
        'label': 'Insider Context',
        'description': 'Adds news context to detected insider trading activity. Web-search providers recommended.',
        'icon': 'user',
    },
    'cycle_news': {
        'label': 'Cycle: News',
        'description': 'News fetching during automated analysis cycles. Same as Stage 2 but for scheduled runs.',
        'icon': 'cycle',
    },
    'cycle_synthesis': {
        'label': 'Cycle: Synthesis',
        'description': 'Research synthesis during automated cycles. Same as Stage 3 but for scheduled runs.',
        'icon': 'bolt',
    },
}


class ProviderRegistry:
    """Central registry that resolves stages to adapters and generates text."""

    def __init__(self):
        self._adapters = {}  # adapter_type -> adapter instance (lazy singleton)
        self._db = None
        self.logger = logging.getLogger(__name__)

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    def _get_adapter(self, adapter_type: str) -> Optional[BaseProviderAdapter]:
        """Get or create an adapter instance for the given type."""
        if adapter_type not in self._adapters:
            adapter_cls = ADAPTER_MAP.get(adapter_type)
            if not adapter_cls:
                self.logger.error(f"Unknown adapter type: {adapter_type}")
                return None
            self._adapters[adapter_type] = adapter_cls()
        return self._adapters[adapter_type]

    def get_provider_for_stage(self, stage_name: str) -> Optional[Dict]:
        """Resolve which provider handles a given stage."""
        return self.db.get_api_provider_for_role(stage_name)

    def generate_for_stage(self, stage_name: str, system_prompt: str, user_prompt: str,
                           temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        """Generate text using whatever provider is assigned to the given stage.

        Returns:
            Tuple of (text, provider_name) or (None, None) if no provider configured.
        """
        provider = self.get_provider_for_stage(stage_name)
        if not provider:
            return None

        adapter_type = provider.get('adapter_type', 'openai_compatible')
        adapter = self._get_adapter(adapter_type)
        if not adapter:
            self.logger.error(f"No adapter for type '{adapter_type}' (stage: {stage_name})")
            return None

        result = adapter.generate(provider, system_prompt, user_prompt, temperature, max_tokens)
        return result

    def generate_for_stage_with_info(self, stage_name: str, system_prompt: str, user_prompt: str,
                                      temperature: float = 0.2, max_tokens: int = 900) -> tuple:
        """Same as generate_for_stage but returns (text, provider_name)."""
        provider = self.get_provider_for_stage(stage_name)
        if not provider:
            return None, None

        adapter_type = provider.get('adapter_type', 'openai_compatible')
        adapter = self._get_adapter(adapter_type)
        if not adapter:
            return None, None

        result = adapter.generate(provider, system_prompt, user_prompt, temperature, max_tokens)
        return result, provider.get('name', 'Unknown')

    def generate_with_provider(self, provider_id: int, system_prompt: str, user_prompt: str,
                                temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
        """Generate text using a specific provider by ID."""
        provider = self.db.get_api_provider(provider_id, include_secret=True)
        if not provider:
            return None

        adapter_type = provider.get('adapter_type', 'openai_compatible')
        adapter = self._get_adapter(adapter_type)
        if not adapter:
            return None

        return adapter.generate(provider, system_prompt, user_prompt, temperature, max_tokens)

    def test_provider(self, provider_id: int) -> Dict:
        """Test a provider's connectivity."""
        provider = self.db.get_api_provider(provider_id, include_secret=True)
        if not provider:
            return {"status": "error", "error": "provider_not_found"}

        adapter_type = provider.get('adapter_type', 'openai_compatible')
        adapter = self._get_adapter(adapter_type)
        if not adapter:
            return {"status": "error", "error": f"unknown_adapter: {adapter_type}"}

        return adapter.test_connection(provider)

    def get_all_providers_with_status(self) -> List[Dict]:
        """Get all providers enriched with usage data (for dashboard/settings)."""
        providers = self.db.get_api_providers(include_secrets=False)
        enriched = []
        for p in providers:
            usage = self.db.get_api_provider_usage(
                provider_id=p['id'],
                monthly_budget_eur=p.get('monthly_budget_eur', 5.0),
            )
            item = dict(p)
            item.update(usage)
            enriched.append(item)
        return enriched

    def get_shortcuts(self) -> Dict:
        """Return shortcut definitions for the UI."""
        return PROVIDER_SHORTCUTS

    def get_stage_info(self) -> Dict:
        """Return stage metadata for the UI."""
        return STAGE_INFO


# Singleton
provider_registry = ProviderRegistry()
