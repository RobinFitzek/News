"""
Gemini API Client with Adaptive Model Selection
Cost-aware: tracks spending and adapts model choice to stay within monthly EUR budget.
"""
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from core.config import GEMINI_API_KEY, GEMINI_MODELS
from datetime import datetime, timedelta
from collections import defaultdict
from core.database import db
import time
import json
import logging


class AdaptiveGeminiClient:
    """Gemini client with cost-aware model selection based on monthly budget"""

    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.client = None
        self.requests = defaultdict(lambda: {'minute': [], 'day': []})
        self.last_reset_date = datetime.now().date()
        self.models = GEMINI_MODELS
        self.logger = logging.getLogger(__name__)
        self._budget_tracker = None

        if self.api_key:
            self._init_client()

    @property
    def budget_tracker(self):
        if self._budget_tracker is None:
            from core.budget_tracker import budget_tracker
            self._budget_tracker = budget_tracker
        return self._budget_tracker

    def _init_client(self):
        """Initialize Gemini client with new SDK"""
        try:
            if not self.api_key or len(self.api_key) < 10:
                self.logger.warning("Invalid or missing Gemini API key")
                return

            self.client = genai.Client(api_key=self.api_key)
            self.logger.info("Gemini client initialized successfully")

        except ValueError as e:
            self.logger.error(f"Invalid API key format: {e}")
            self.client = None

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            self.client = None

    def reload_api_key(self, new_key: str):
        """Reload with new API key"""
        self.api_key = new_key
        if new_key:
            self._init_client()

    def is_configured(self) -> bool:
        """Check if API is configured"""
        return bool(self.api_key and len(self.api_key) > 10 and self.client)

    def get_usage(self) -> dict:
        """Get usage stats from database (persistent across restarts)."""
        self._check_daily_reset()
        status = self.budget_tracker.get_budget_status().get('gemini', {})
        daily_limit = status.get('daily_request_limit', 100)
        today = datetime.now().strftime('%Y-%m-%d')

        # Query actual counts from database, not in-memory
        flash_used = self._get_db_model_count(today, ['flash-8b', 'flash-1.5'])
        pro_used = self._get_db_model_count(today, ['flash-2.5', 'pro'])

        usage = {}
        for tier, config in self.models.items():
            if tier == 'flash':
                continue
            usage[tier] = {
                "model": config['model'],
                "used_today": self._get_db_model_count(today, [tier]),
                "cost_per_1m_input": config.get('cost_per_1m_input', 0),
                "cost_per_1m_output": config.get('cost_per_1m_output', 0),
            }

        # Backward-compatible keys for dashboard template
        flash_limit = max(1, int(daily_limit * 0.7))
        pro_limit = max(1, int(daily_limit * 0.3))
        usage['flash'] = {
            "used_today": flash_used,
            "daily_limit": flash_limit,
            "remaining": max(0, flash_limit - flash_used)
        }
        usage['pro'] = {
            "used_today": pro_used,
            "daily_limit": pro_limit,
            "remaining": max(0, pro_limit - pro_used)
        }

        usage['is_configured'] = self.is_configured()
        usage['monthly_budget_eur'] = status.get('monthly_budget_eur', 5.0)
        usage['spent_eur'] = status.get('spent_eur', 0)
        usage['remaining_eur'] = status.get('remaining_eur', 5.0)
        usage['daily_request_limit'] = daily_limit
        usage['today_requests'] = status.get('today_requests', 0)
        return usage

    def _get_db_model_count(self, date_str: str, tiers: list) -> int:
        """Get request count from api_cost_log for specific model tiers on a date."""
        try:
            from core.database import db
            conn = db._get_conn()
            cursor = conn.cursor()
            # Match by tier key or by model name
            model_names = []
            for tier in tiers:
                model_names.append(tier)
                config = self.models.get(tier)
                if config:
                    model_names.append(config['model'])
            placeholders = ','.join('?' * len(model_names))
            cursor.execute(f"""
                SELECT COUNT(*) as cnt FROM api_cost_log
                WHERE api = 'gemini' AND date = ? AND model IN ({placeholders})
            """, [date_str] + model_names)
            row = cursor.fetchone()
            conn.close()
            return int(row['cnt']) if row else 0
        except Exception:
            return 0

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            for tier in self.requests:
                self.requests[tier]['day'] = []
            self.last_reset_date = today

    def _check_rate_limit(self, tier: str) -> bool:
        """Check if request is within RPM rate limits"""
        self._check_daily_reset()
        now = datetime.now()
        config = self.models.get(tier, {})

        # Cleanup old minute timestamps
        self.requests[tier]['minute'] = [
            t for t in self.requests[tier]['minute']
            if now - t < timedelta(minutes=1)
        ]

        rpm = config.get('rpm', 300)
        return len(self.requests[tier]['minute']) < rpm

    def _wait_if_needed(self, tier: str) -> bool:
        """Wait if RPM limit reached, return False if budget exhausted"""
        wait_count = 0
        while not self._check_rate_limit(tier) and wait_count < 3:
            if not self.budget_tracker.can_afford_request('gemini', tier):
                print(f"  Gemini {tier} budget exhausted")
                return False
            print(f"  Gemini {tier} rate limit, waiting 15s...")
            time.sleep(15)
            wait_count += 1
        return True

    def select_best_model(self, task_type: str = 'analyze') -> str:
        """
        Cost-aware model selection.
        Prefers cheaper models (flash-8b > flash-1.5) when budget is tight.
        Uses flash-2.5 (gemini-3-flash) only when budget allows.
        """
        self._check_daily_reset()

        if not self.budget_tracker.can_afford_request('gemini'):
            self.logger.warning("Gemini daily budget exhausted")
            return None

        # Get remaining daily budget to decide model tier
        daily_budget_usd = self.budget_tracker.get_daily_budget_usd('gemini')
        today_spent = self.budget_tracker.get_today_spending('gemini')
        remaining_today = max(0, daily_budget_usd - today_spent)

        # Find suitable models for this task
        suitable = []
        for tier, config in self.models.items():
            if tier == 'flash':
                continue
            use_for = config.get('use_for', [])
            if task_type in use_for or 'analyze' in use_for:
                est_cost = self.budget_tracker.estimate_request_cost('gemini', tier)
                if est_cost <= remaining_today:
                    suitable.append({
                        'tier': tier,
                        'priority': config.get('priority', 1),
                        'cost': est_cost,
                        'model': config['model']
                    })

        if not suitable:
            # Try flash-8b as last resort (cheapest)
            cheapest_cost = self.budget_tracker.estimate_request_cost('gemini', 'flash-8b')
            if cheapest_cost <= remaining_today:
                return 'flash-8b'
            print("  All Gemini models exhausted for today!")
            return None

        # When budget is tight (< 20% of daily remaining), prefer cheapest
        budget_ratio = remaining_today / daily_budget_usd if daily_budget_usd > 0 else 0

        if budget_ratio < 0.2:
            # Budget-saving mode: pick cheapest
            suitable.sort(key=lambda x: x['cost'])
        else:
            # Normal mode: pick by priority (quality), then cost
            suitable.sort(key=lambda x: (-x['priority'], x['cost']))

        selected = suitable[0]
        self.logger.info(f"Selected {selected['tier']} for '{task_type}' (est. ${selected['cost']:.5f})")
        return selected['tier']

    def generate(self, prompt: str, tier: str = None, task_type: str = 'analyze') -> str:
        """Generate text with cost tracking and adaptive model selection."""
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt provided")
            return "  Ungültiger Prompt"

        if len(prompt) > 100000:
            self.logger.warning("Prompt exceeds length limit")
            prompt = prompt[:100000]

        if not self.is_configured():
            self.logger.warning("Gemini API not configured")
            return "  Gemini API nicht konfiguriert. Bitte API Key in den Einstellungen hinterlegen."

        # Auto-select model if not specified
        if tier is None:
            tier = self.select_best_model(task_type)
            if tier is None:
                return "  Gemini Budget für heute erschöpft."

        if tier not in self.models:
            self.logger.error(f"Invalid tier: {tier}")
            return f"  Ungültiges Modell: {tier}"

        # Check RPM rate limit
        if not self._wait_if_needed(tier):
            fallback = self._get_fallback_tier(tier)
            if fallback:
                self.logger.info(f"Falling back from {tier} to {fallback}")
                tier = fallback
            else:
                return f"  Gemini {tier} Budget erschöpft, keine Fallback-Option."

        # Check cost budget
        if not self.budget_tracker.can_afford_request('gemini', tier):
            fallback = self._get_fallback_tier(tier)
            if fallback and self.budget_tracker.can_afford_request('gemini', fallback):
                tier = fallback
            else:
                return "  Gemini Budget für heute erschöpft."

        config = self.models.get(tier, self.models.get('flash-1.5'))
        model_name = config['model']

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if not response or not hasattr(response, 'text'):
                    self.logger.error("Invalid response structure")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return "  Ungültige Antwort von Gemini API"

                response_text = response.text

                if not response_text:
                    self.logger.warning("Empty response from API")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return "  Leere Antwort von Gemini API"

                # Track RPM
                now = datetime.now()
                self.requests[tier]['minute'].append(now)
                self.requests[tier]['day'].append(now)

                # Extract token counts and log cost
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    um = response.usage_metadata
                    input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(um, 'candidates_token_count', 0) or 0

                # Fallback: estimate tokens if metadata not available
                if input_tokens == 0:
                    input_tokens = len(prompt) // 4
                if output_tokens == 0:
                    output_tokens = len(response_text) // 4

                cost = self.budget_tracker.log_cost(
                    'gemini', tier, input_tokens, output_tokens
                )

                today_count = self.budget_tracker.get_today_request_count('gemini')
                self.logger.info(f"Gemini {tier} OK: ${cost:.5f} ({input_tokens}in/{output_tokens}out)")
                print(f"  Gemini {tier}: req #{today_count} (${cost:.5f}, {model_name})")

                # Clear any auth alert on success
                try:
                    db.clear_system_alert('gemini_auth')
                except Exception:
                    pass

                return response_text

            except google_exceptions.ResourceExhausted:
                self.logger.error(f"Quota exhausted for {tier}")
                fallback = self._get_fallback_tier(tier)
                if fallback:
                    tier = fallback
                    config = self.models.get(tier)
                    model_name = config['model']
                    continue
                return "  Gemini Quota erschöpft"

            except google_exceptions.InvalidArgument as e:
                self.logger.error(f"Invalid argument: {e}")
                return f"  Ungültige Anfrage"

            except google_exceptions.PermissionDenied:
                self.logger.error("Permission denied")
                try:
                    db.raise_system_alert(
                        'gemini_auth',
                        'Gemini API Key Invalid',
                        'The Gemini API key was rejected (Permission Denied). AI analysis is disabled until you update the key.',
                        severity='error', service='gemini',
                        action_url='/settings', action_label='Update API Key')
                except Exception:
                    pass
                return "  Zugriff verweigert. Bitte API Key prüfen."

            except google_exceptions.DeadlineExceeded:
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return "  Anfrage timeout"

            except google_exceptions.ServiceUnavailable:
                self.logger.warning(f"Service unavailable (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return "  Gemini Service vorübergehend nicht verfügbar"

            except google_exceptions.GoogleAPIError as e:
                self.logger.error(f"Google API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return f"  Gemini API Fehler"

            except AttributeError:
                self.logger.error("Client not initialized")
                return "  Gemini Client nicht initialisiert"

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return f"Fehler bei Gemini API: {str(e)}"

        return f"  Gemini Anfrage fehlgeschlagen nach {max_retries} Versuchen"

    def _get_fallback_tier(self, current_tier: str) -> str:
        """Get a cheaper fallback tier when current is exhausted"""
        fallback_chain = {
            'pro': 'flash-2.5',
            'flash-2.5': 'flash-1.5',
            'flash': 'flash-1.5',
            'flash-1.5': 'flash-8b',
            'flash-8b': None
        }

        fallback = fallback_chain.get(current_tier)
        if fallback and self.budget_tracker.can_afford_request('gemini', fallback):
            return fallback

        if fallback:
            return self._get_fallback_tier(fallback)
        return None

    def generate_with_auto_select(self, prompt: str, task_type: str = 'analyze') -> str:
        """Convenience method that always auto-selects the best model"""
        return self.generate(prompt, tier=None, task_type=task_type)


# Singleton
gemini_client = AdaptiveGeminiClient()
