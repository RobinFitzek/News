"""
Macro Event Tracker — FOMC, ECB, rate decisions
Hardcoded 2026 calendar + DB storage. Injected into Stage 3 prompt.
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)

# 2026 FOMC meeting dates (rate decision day = 2nd day)
FOMC_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-09",
]

# 2026 ECB Governing Council meeting dates
ECB_2026 = [
    "2026-01-30", "2026-03-05", "2026-04-16",
    "2026-06-04", "2026-07-23", "2026-09-10",
    "2026-10-22", "2026-12-03",
]

# Rate-sensitive sectors/tickers (extend as needed)
RATE_SENSITIVE_SECTORS = ['financials', 'real_estate', 'utilities', 'reits']
RATE_SENSITIVE_KEYWORDS = ['bank', 'reit', 'mortgage', 'insurance', 'utility', 'bond', 'finance']


class MacroTracker:
    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Return rate events in the next N days."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        events = []
        for d in FOMC_2026:
            dt = date.fromisoformat(d)
            if today <= dt <= cutoff:
                days_until = (dt - today).days
                events.append({'type': 'FOMC', 'date': d, 'days_until': days_until,
                                'description': 'Federal Reserve rate decision'})
        for d in ECB_2026:
            dt = date.fromisoformat(d)
            if today <= dt <= cutoff:
                days_until = (dt - today).days
                events.append({'type': 'ECB', 'date': d, 'days_until': days_until,
                                'description': 'ECB Governing Council rate decision'})
        events.sort(key=lambda x: x['date'])
        return events

    def get_next_event(self) -> Optional[Dict]:
        """Return the single nearest upcoming event."""
        events = self.get_upcoming_events(days_ahead=90)
        return events[0] if events else None

    def build_macro_context_block(self) -> str:
        """Return a formatted string for injection into Stage 3 prompt."""
        events = self.get_upcoming_events(days_ahead=14)
        if not events:
            return ""
        lines = ["UPCOMING CENTRAL BANK EVENTS:"]
        for e in events:
            lines.append(f"  - {e['type']} rate decision: {e['date']} (in {e['days_until']} days)")
        lines.append("Consider rate sensitivity in your analysis. Tighten confidence for rate-sensitive sectors if event is within 48h.")
        return "\n".join(lines)

    def is_rate_sensitive(self, ticker: str, name: str = "") -> bool:
        """Heuristic: is this ticker likely rate-sensitive?"""
        text = (ticker + " " + name).lower()
        return any(kw in text for kw in RATE_SENSITIVE_KEYWORDS)

    def check_portfolio_rate_exposure(self) -> List[Dict]:
        """Return portfolio holdings that are rate-sensitive and have upcoming events."""
        events = self.get_upcoming_events(days_ahead=48 // 24 + 1)
        if not events:
            return []
        try:
            from engine.portfolio_manager import portfolio_manager
            holdings = portfolio_manager.get_portfolio_holdings()
            exposed = []
            for h in holdings:
                ticker = h.get('ticker', '')
                name = h.get('name', '')
                if self.is_rate_sensitive(ticker, name):
                    exposed.append({'ticker': ticker, 'name': name,
                                    'events': events})
            return exposed
        except Exception as e:
            logger.warning(f"Could not check portfolio rate exposure: {e}")
            return []


macro_tracker = MacroTracker()
