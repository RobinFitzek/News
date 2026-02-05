"""
Gemini API Client with Adaptive Model Selection
Intelligently chooses models based on available quota and task requirements.
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
    """Gemini client with intelligent model selection based on quota and task"""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.client = None
        self.requests = defaultdict(lambda: {'minute': [], 'day': []})
        self.last_reset_date = datetime.now().date()
        self.models = GEMINI_MODELS
        self.logger = logging.getLogger(__name__)

        # Budget tracking from DB
        self._load_budget_settings()

        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client with new SDK and error handling"""
        try:
            if not self.api_key or len(self.api_key) < 10:
                self.logger.warning("Invalid or missing Gemini API key")
                print("⚠️ Gemini API key missing or invalid")
                return

            self.client = genai.Client(api_key=self.api_key)
            self.logger.info("Gemini client initialized successfully")

        except ValueError as e:
            self.logger.error(f"Invalid API key format: {e}")
            print(f"⚠️ Gemini init error: Invalid API key format")
            self.client = None

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            print(f"⚠️ Gemini init error: {e}")
            self.client = None
    
    def _load_budget_settings(self):
        """Load variable budget settings from DB"""
        try:
            # Try to load custom budget from settings
            daily_budget = db.get_setting('gemini_daily_budget')
            if daily_budget:
                self.daily_budget = daily_budget
            else:
                # Default budget allocation
                self.daily_budget = {
                    'flash-8b': 50,   # High volume screening
                    'flash-1.5': 30,  # Standard analysis
                    'flash-2.5': 15,  # Premium when needed
                    'flash': 15,      # Alias
                    'pro': 5          # Rare, final synthesis only
                }
        except:
            self.daily_budget = {'flash-8b': 50, 'flash-1.5': 30, 'flash-2.5': 15, 'flash': 15, 'pro': 5}
    
    def set_daily_budget(self, budget: dict):
        """Set custom daily budget allocation"""
        self.daily_budget = budget
        db.set_setting('gemini_daily_budget', budget)
        print(f"📊 Budget updated: {budget}")
    
    def reload_api_key(self, new_key: str):
        """Reload with new API key"""
        self.api_key = new_key
        if new_key:
            self._init_client()
    
    def is_configured(self) -> bool:
        """Check if API is configured"""
        return bool(self.api_key and len(self.api_key) > 10 and self.client)
    
    def get_usage(self) -> dict:
        """Get usage stats for all model tiers with backward-compatible structure"""
        self._check_daily_reset()
        usage = {}
        
        # Backward compatibility - combine all flash models
        flash_total_used = 0
        flash_total_budget = 0
        pro_total_used = 0
        pro_total_budget = 0
        
        for tier, config in self.models.items():
            day_count = len(self.requests[tier]['day'])
            budget = self.daily_budget.get(tier, config.get('rpd', 100))
            
            usage[tier] = {
                "model": config['model'],
                "used_today": day_count,
                "budget": budget,
                "api_limit": config['rpd'],
                "remaining": min(budget, config['rpd']) - day_count,
                "priority": config.get('priority', 1)
            }
            
            # Aggregate for backward compatibility
            if 'flash' in tier:
                flash_total_used += day_count
                flash_total_budget += budget
            elif 'pro' in tier or tier == 'flash-2.5':  # 2.5 was previously "pro"
                pro_total_used += day_count
                pro_total_budget += budget
        
        # Add backward-compatible keys
        usage['flash'] = {
            "used_today": flash_total_used,
            "daily_limit": flash_total_budget if flash_total_budget > 0 else 1500,
            "remaining": flash_total_budget - flash_total_used
        }
        usage['pro'] = {
            "used_today": pro_total_used,
            "daily_limit": pro_total_budget if pro_total_budget > 0 else 50,
            "remaining": pro_total_budget - pro_total_used
        }
        
        usage['is_configured'] = self.is_configured()
        return usage
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            for tier in self.requests:
                self.requests[tier]['day'] = []
            self.last_reset_date = today
            self._load_budget_settings()  # Reload budget on new day
    
    def _get_remaining(self, tier: str) -> int:
        """Get remaining quota for a tier"""
        self._check_daily_reset()
        config = self.models.get(tier, {})
        day_count = len(self.requests[tier]['day'])
        budget = self.daily_budget.get(tier, config.get('rpd', 100))
        api_limit = config.get('rpd', 100)
        return min(budget, api_limit) - day_count
    
    def _check_rate_limit(self, tier: str) -> bool:
        """Check if request is within rate limits"""
        self._check_daily_reset()
        now = datetime.now()
        config = self.models.get(tier, {})
        
        # Cleanup old minute timestamps
        self.requests[tier]['minute'] = [
            t for t in self.requests[tier]['minute']
            if now - t < timedelta(minutes=1)
        ]
        
        rpm = config.get('rpm', 10)
        rpm_ok = len(self.requests[tier]['minute']) < rpm
        rpd_ok = self._get_remaining(tier) > 0
        
        return rpm_ok and rpd_ok
    
    def _wait_if_needed(self, tier: str) -> bool:
        """Wait if rate limit reached, return False if budget exhausted"""
        wait_count = 0
        while not self._check_rate_limit(tier) and wait_count < 3:
            remaining = self._get_remaining(tier)
            if remaining <= 0:
                print(f"⚠️ Gemini {tier} budget exhausted")
                return False
            print(f"⏳ Gemini {tier} rate limit, waiting 30s...")
            time.sleep(30)
            wait_count += 1
        return True
    
    def select_best_model(self, task_type: str = 'analyze') -> str:
        """
        Adaptively select the best available model for a task.
        
        Task types: 'scan', 'quick_check', 'analyze', 'synthesize', 'final_verdict'
        
        Selection logic:
        1. Find models suitable for the task
        2. Filter by available quota
        3. Choose highest priority model with quota
        4. Fall back to lower priority if premium exhausted
        """
        self._check_daily_reset()
        
        # Find suitable models for this task
        suitable = []
        for tier, config in self.models.items():
            if tier == 'flash':  # Skip alias
                continue
            use_for = config.get('use_for', [])
            if task_type in use_for or 'analyze' in use_for:
                remaining = self._get_remaining(tier)
                if remaining > 0:
                    suitable.append({
                        'tier': tier,
                        'priority': config.get('priority', 1),
                        'remaining': remaining,
                        'model': config['model']
                    })
        
        if not suitable:
            # All models exhausted, try flash-8b as last resort
            if self._get_remaining('flash-8b') > 0:
                return 'flash-8b'
            print("⚠️ All Gemini models exhausted for today!")
            return None
        
        # Sort by priority (higher = better), then by remaining quota
        suitable.sort(key=lambda x: (x['priority'], x['remaining']), reverse=True)
        
        selected = suitable[0]
        print(f"🎯 Selected {selected['tier']} for '{task_type}' (remaining: {selected['remaining']})")
        return selected['tier']
    
    def generate(self, prompt: str, tier: str = None, task_type: str = 'analyze') -> str:
        """
        Generate text with specified or auto-selected model tier with comprehensive error handling.

        Args:
            prompt: The prompt to send
            tier: Specific tier to use (optional, will auto-select if None)
            task_type: Type of task for adaptive selection
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt provided")
            return "⚠️ Ungültiger Prompt"

        if len(prompt) > 100000:  # Reasonable limit
            self.logger.warning("Prompt exceeds length limit")
            prompt = prompt[:100000]

        if not self.is_configured():
            self.logger.warning("Gemini API not configured")
            return "⚠️ Gemini API nicht konfiguriert. Bitte API Key in den Einstellungen hinterlegen."

        # Auto-select model if not specified
        if tier is None:
            tier = self.select_best_model(task_type)
            if tier is None:
                self.logger.warning("All models exhausted")
                return "⚠️ Alle Gemini-Modelle für heute erschöpft."

        # Validate tier
        if tier not in self.models:
            self.logger.error(f"Invalid tier: {tier}")
            return f"⚠️ Ungültiges Modell: {tier}"

        # Check if we can use this tier
        if not self._wait_if_needed(tier):
            # Try to fall back to a cheaper model
            fallback = self._get_fallback_tier(tier)
            if fallback:
                self.logger.info(f"Falling back from {tier} to {fallback}")
                print(f"🔄 Falling back from {tier} to {fallback}")
                tier = fallback
            else:
                self.logger.warning(f"No fallback available for {tier}")
                return f"⚠️ Gemini {tier} Budget erschöpft, keine Fallback-Option verfügbar."

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

                # Validate response
                if not response or not hasattr(response, 'text'):
                    self.logger.error("Invalid response structure")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return "⚠️ Ungültige Antwort von Gemini API"

                response_text = response.text

                if not response_text:
                    self.logger.warning("Empty response from API")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return "⚠️ Leere Antwort von Gemini API"

                # Success - log request
                now = datetime.now()
                self.requests[tier]['minute'].append(now)
                self.requests[tier]['day'].append(now)

                remaining = self._get_remaining(tier)
                budget = self.daily_budget.get(tier, config['rpd'])
                self.logger.info(f"API call successful. Tier: {tier}, Usage: {budget - remaining}/{budget}")
                print(f"✅ Gemini {tier}: {budget - remaining}/{budget} today ({model_name})")

                return response_text

            except google_exceptions.ResourceExhausted as e:
                self.logger.error(f"Quota exhausted for {tier}: {e}")
                print(f"⚠️ Gemini {tier} Quota exhausted")
                # Try fallback immediately
                fallback = self._get_fallback_tier(tier)
                if fallback:
                    self.logger.info(f"Quota exhausted, falling back to {fallback}")
                    tier = fallback
                    config = self.models.get(tier)
                    model_name = config['model']
                    continue
                return f"⚠️ Gemini Quota erschöpft"

            except google_exceptions.InvalidArgument as e:
                self.logger.error(f"Invalid argument: {e}")
                print(f"❌ Gemini: Ungültige Anfrage")
                return f"⚠️ Ungültige Anfrage: {str(e)}"

            except google_exceptions.PermissionDenied as e:
                self.logger.error(f"Permission denied: {e}")
                print(f"❌ Gemini: Zugriff verweigert (API Key prüfen)")
                return "⚠️ Zugriff verweigert. Bitte API Key prüfen."

            except google_exceptions.DeadlineExceeded as e:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(f"❌ Gemini timeout")
                return "⚠️ Anfrage timeout"

            except google_exceptions.ServiceUnavailable as e:
                self.logger.warning(f"Service unavailable (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(f"❌ Gemini service unavailable")
                return "⚠️ Gemini Service vorübergehend nicht verfügbar"

            except google_exceptions.GoogleAPIError as e:
                self.logger.error(f"Google API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(f"❌ Gemini API error: {e}")
                return f"⚠️ Gemini API Fehler: {str(e)}"

            except AttributeError as e:
                self.logger.error(f"Client not initialized: {e}")
                print("❌ Gemini client not initialized")
                return "⚠️ Gemini Client nicht initialisiert"

            except Exception as e:
                self.logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(f"❌ Gemini {tier} Error: {e}")
                return f"Fehler bei Gemini API: {str(e)}"

        # All retries exhausted
        self.logger.error(f"All {max_retries} attempts failed for tier {tier}")
        return f"⚠️ Gemini Anfrage fehlgeschlagen nach {max_retries} Versuchen"
    
    def _get_fallback_tier(self, current_tier: str) -> str:
        """Get a fallback tier when current is exhausted"""
        fallback_chain = {
            'pro': 'flash-2.5',
            'flash-2.5': 'flash-1.5',
            'flash': 'flash-1.5',
            'flash-1.5': 'flash-8b',
            'flash-8b': None
        }
        
        fallback = fallback_chain.get(current_tier)
        if fallback and self._get_remaining(fallback) > 0:
            return fallback
        
        # Try next in chain
        if fallback:
            return self._get_fallback_tier(fallback)
        return None
    
    def generate_with_auto_select(self, prompt: str, task_type: str = 'analyze') -> str:
        """Convenience method that always auto-selects the best model"""
        return self.generate(prompt, tier=None, task_type=task_type)


# Singleton
gemini_client = AdaptiveGeminiClient()
