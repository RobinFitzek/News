"""
Adaptive API Budget Tracker
Calculates daily request limits from monthly EUR budgets.
Tracks estimated costs per request and adapts limits as the month progresses.
"""
import calendar
import logging
from datetime import datetime, date
from typing import Optional
from core.config import (
    GEMINI_MODELS, PERPLEXITY_PRICING, DEFAULT_MONTHLY_BUDGET,
    EUR_TO_USD, PIPELINE_STAGE_SPLIT
)

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks API spending and calculates adaptive daily limits."""

    def __init__(self):
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from core.database import db
            self._db = db
        return self._db

    # --- Monthly budget from settings ---

    def get_monthly_budget(self, api: str) -> float:
        """Get monthly budget in EUR for an API."""
        key = f"{api}_monthly_budget"
        val = self.db.get_setting(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return DEFAULT_MONTHLY_BUDGET.get(api, 5.0)

    def get_monthly_budget_usd(self, api: str) -> float:
        """Monthly budget converted to USD (API pricing is in USD)."""
        return self.get_monthly_budget(api) * EUR_TO_USD

    # --- Cost logging ---

    def log_cost(self, api: str, model: str, input_tokens: int = 0,
                 output_tokens: int = 0, extra_cost: float = 0.0):
        """Log an API request with estimated cost."""
        cost = self._estimate_cost(api, model, input_tokens, output_tokens) + extra_cost
        month = datetime.now().strftime('%Y-%m')
        today = datetime.now().strftime('%Y-%m-%d')
        self.db.log_api_cost(api, model, input_tokens, output_tokens, cost, month, today)
        logger.info(f"Logged cost: {api}/{model} ${cost:.6f} ({input_tokens}in/{output_tokens}out)")
        return cost

    def _estimate_cost(self, api: str, model: str, input_tokens: int,
                       output_tokens: int) -> float:
        """Estimate USD cost for a request based on token counts."""
        if api == 'perplexity':
            pricing = PERPLEXITY_PRICING.get('sonar', {})
            token_cost = (
                input_tokens * pricing.get('cost_per_1m_input', 1.0) / 1_000_000 +
                output_tokens * pricing.get('cost_per_1m_output', 1.0) / 1_000_000
            )
            search_cost = pricing.get('cost_per_1000_searches', 5.0) / 1000
            return token_cost + search_cost

        elif api == 'gemini':
            # Find model config by model name or tier key
            config = GEMINI_MODELS.get(model)
            if not config:
                for tier_config in GEMINI_MODELS.values():
                    if tier_config['model'] == model:
                        config = tier_config
                        break
            if not config:
                config = GEMINI_MODELS.get('flash-1.5', {})

            return (
                input_tokens * config.get('cost_per_1m_input', 0.15) / 1_000_000 +
                output_tokens * config.get('cost_per_1m_output', 0.60) / 1_000_000
            )
        return 0.0

    def estimate_request_cost(self, api: str, model: str) -> float:
        """Estimate cost for a typical request (for budget planning).
        Uses average token counts observed in financial analysis prompts."""
        if api == 'perplexity':
            return self._estimate_cost('perplexity', 'sonar', 800, 400)
        elif api == 'gemini':
            return self._estimate_cost('gemini', model, 1200, 600)
        return 0.01

    # --- Spending queries ---

    def get_month_spending(self, api: str, month: str = None) -> float:
        """Get total USD spending for an API this month."""
        if month is None:
            month = datetime.now().strftime('%Y-%m')
        return self.db.get_api_spending(api, month)

    def get_today_spending(self, api: str) -> float:
        """Get total USD spending for an API today."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.db.get_api_spending_day(api, today)

    def get_today_request_count(self, api: str) -> int:
        """Get number of API requests made today."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.db.get_api_request_count(api, today)

    # --- Adaptive daily limits ---

    def get_daily_budget_usd(self, api: str) -> float:
        """Calculate adaptive daily budget in USD based on remaining monthly budget."""
        monthly_usd = self.get_monthly_budget_usd(api)
        spent_usd = self.get_month_spending(api)
        remaining_usd = max(0, monthly_usd - spent_usd)

        today = date.today()
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        days_left = max(1, days_in_month - today.day + 1)

        daily = remaining_usd / days_left
        return daily

    def get_daily_request_limit(self, api: str, model: str = None) -> int:
        """Calculate how many requests we can afford today for an API/model."""
        daily_usd = self.get_daily_budget_usd(api)
        already_spent = self.get_today_spending(api)
        remaining_today = max(0, daily_usd - already_spent)

        if api == 'perplexity':
            cost_per_req = self.estimate_request_cost('perplexity', 'sonar')
        else:
            model = model or 'flash-1.5'
            cost_per_req = self.estimate_request_cost('gemini', model)

        if cost_per_req <= 0:
            return 999

        return max(0, int(remaining_today / cost_per_req))

    def can_afford_request(self, api: str, model: str = None) -> bool:
        """Check if we can afford one more request today."""
        return self.get_daily_request_limit(api, model) > 0

    # --- Pipeline budget allocation ---

    def get_pipeline_limits(self) -> dict:
        """Calculate adaptive stage limits for the daily pipeline.
        Returns dict with stage1_max, stage2_max, stage3_max, perplexity_max."""
        gemini_daily = self.get_daily_request_limit('gemini', 'flash')
        pplx_daily = self.get_daily_request_limit('perplexity')

        # Stage 1 is free (quant screener, no API calls)
        # Stage 2 uses 1 perplexity call per ticker (news only)
        # Stage 3 uses 1 gemini flash call per ticker (research note)
        stage2 = max(1, int(pplx_daily * PIPELINE_STAGE_SPLIT['stage2'])) if PIPELINE_STAGE_SPLIT['stage2'] > 0 else pplx_daily
        stage3 = max(1, int(gemini_daily * PIPELINE_STAGE_SPLIT['stage3'])) if PIPELINE_STAGE_SPLIT['stage3'] > 0 else gemini_daily

        # Stage 2 limited by perplexity budget
        stage2 = min(stage2, pplx_daily)
        # Stage 3 limited by gemini budget and by stage 2 output
        stage3 = min(stage3, stage2)

        return {
            'stage1_max': 50,  # Free — quant screener, no API cost
            'stage2_max': min(stage2, 25),
            'stage3_max': min(stage3, 10),
            'perplexity_max': pplx_daily,
            'gemini_total': gemini_daily,
        }

    # --- Budget status for UI ---

    def get_budget_status(self) -> dict:
        """Full budget status for dashboard/settings display."""
        month = datetime.now().strftime('%Y-%m')
        today = date.today()
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        days_left = max(1, days_in_month - today.day + 1)

        status = {}
        for api in ['perplexity', 'gemini']:
            monthly_eur = self.get_monthly_budget(api)
            monthly_usd = self.get_monthly_budget_usd(api)
            spent_usd = self.get_month_spending(api)
            spent_eur = spent_usd / EUR_TO_USD if EUR_TO_USD > 0 else spent_usd
            remaining_eur = max(0, monthly_eur - spent_eur)
            daily_usd = self.get_daily_budget_usd(api)
            today_count = self.get_today_request_count(api)
            today_spent_usd = self.get_today_spending(api)

            daily_limit = self.get_daily_request_limit(api)

            status[api] = {
                'monthly_budget_eur': round(monthly_eur, 2),
                'spent_eur': round(spent_eur, 2),
                'remaining_eur': round(remaining_eur, 2),
                'percent_used': round(spent_eur / monthly_eur * 100, 1) if monthly_eur > 0 else 0,
                'daily_budget_usd': round(daily_usd, 4),
                'today_requests': today_count,
                'today_spent_eur': round(today_spent_usd / EUR_TO_USD, 4) if EUR_TO_USD > 0 else 0,
                'daily_request_limit': daily_limit,
                'days_left': days_left,
            }

        # Pipeline capacity
        status['pipeline'] = self.get_pipeline_limits()
        return status


# Singleton
budget_tracker = BudgetTracker()
