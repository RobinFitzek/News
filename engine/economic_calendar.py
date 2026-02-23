"""
Economic Calendar Integration
Fetches upcoming market-moving events (FOMC, CPI, Jobs, earnings).
Uses free APIs and web scraping for event data.
"""
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)


# Known FOMC meeting dates for 2024-2026 (hardcoded since they're scheduled in advance)
FOMC_DATES = [
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
    # 2026
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
]

# CPI release approximate dates (usually 2nd week of month, ~8:30 AM ET)
# These are approximate - typically 10th-14th of each month for previous month data
CPI_RELEASE_DAY = 13  # Approximate - usually around 10-14th

# Jobs report (first Friday of month)
def get_jobs_report_date(year: int, month: int) -> str:
    """Get first Friday of a given month."""
    first_day = datetime(year, month, 1)
    # Days until Friday (0=Mon, 4=Fri)
    days_until_friday = (4 - first_day.weekday()) % 7
    if days_until_friday == 0 and first_day.weekday() != 4:
        days_until_friday = 7
    if first_day.weekday() == 4:
        days_until_friday = 0
    return (first_day + timedelta(days=days_until_friday)).strftime('%Y-%m-%d')


class EconomicCalendar:
    """Tracks upcoming market-moving economic events."""
    
    def __init__(self, cache_hours: int = 4):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(hours=cache_hours)
    
    def _is_cache_valid(self) -> bool:
        if not self._cache or not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_duration
    
    def get_upcoming_events(self, days_ahead: int = 14) -> List[Dict]:
        """
        Get upcoming market-moving events in the next N days.
        """
        if self._is_cache_valid() and 'events' in self._cache:
            return self._cache['events']
        
        events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # FOMC meetings
        for date_str in FOMC_DATES:
            event_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if today <= event_date <= end_date:
                days_away = (event_date - today).days
                events.append({
                    'date': date_str,
                    'type': 'FOMC',
                    'name': 'FOMC Meeting',
                    'impact': 'high',
                    'days_away': days_away,
                    'warning': 'Rate decision - high volatility expected' if days_away <= 3 else None,
                })
        
        # CPI releases (approximate - 2nd week each month)
        for offset in range(0, days_ahead + 30, 28):
            check_date = today + timedelta(days=offset)
            cpi_date = datetime(check_date.year, check_date.month, CPI_RELEASE_DAY).date()
            if today <= cpi_date <= end_date:
                days_away = (cpi_date - today).days
                events.append({
                    'date': cpi_date.strftime('%Y-%m-%d'),
                    'type': 'CPI',
                    'name': 'CPI Report',
                    'impact': 'high',
                    'days_away': days_away,
                    'warning': 'Inflation data - expect volatility' if days_away <= 2 else None,
                })
        
        # Jobs report (first Friday of month)
        for offset in range(0, days_ahead + 30, 28):
            check_date = today + timedelta(days=offset)
            jobs_date_str = get_jobs_report_date(check_date.year, check_date.month)
            jobs_date = datetime.strptime(jobs_date_str, '%Y-%m-%d').date()
            if today <= jobs_date <= end_date:
                days_away = (jobs_date - today).days
                events.append({
                    'date': jobs_date_str,
                    'type': 'JOBS',
                    'name': 'Jobs Report',
                    'impact': 'high',
                    'days_away': days_away,
                    'warning': 'Employment data - market-moving' if days_away <= 2 else None,
                })
        
        # Remove duplicates and sort by date
        seen = set()
        unique_events = []
        for e in sorted(events, key=lambda x: x['date']):
            key = (e['date'], e['type'])
            if key not in seen:
                seen.add(key)
                unique_events.append(e)
        
        self._cache['events'] = unique_events
        self._cache_time = datetime.now()
        
        return unique_events
    
    def get_earnings_this_week(self, tickers: List[str] = None) -> List[Dict]:
        """
        Get upcoming earnings for given tickers (or major companies).
        Uses yfinance calendar data.
        """
        if tickers is None:
            # Default to major tech and market movers
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'V']
        
        earnings = []
        today = datetime.now().date()
        week_end = today + timedelta(days=7)
        
        for ticker in tickers[:20]:  # Limit API calls
            try:
                stock = yf.Ticker(ticker)
                cal = stock.calendar
                
                if cal is not None and not cal.empty:
                    earnings_date = None
                    # Handle different calendar formats
                    if isinstance(cal, dict):
                        earnings_date = cal.get('Earnings Date')
                    else:
                        # DataFrame format
                        if 'Earnings Date' in cal.columns:
                            val = cal['Earnings Date'].iloc[0]
                            earnings_date = val
                        elif 'Earnings Date' in cal.index:
                            val = cal.loc['Earnings Date'].iloc[0]
                            earnings_date = val
                    
                    if earnings_date:
                        # Handle various date formats
                        if hasattr(earnings_date, 'date'):
                            ed = earnings_date.date()
                        elif isinstance(earnings_date, str):
                            ed = datetime.strptime(earnings_date[:10], '%Y-%m-%d').date()
                        else:
                            continue
                        
                        if today <= ed <= week_end:
                            earnings.append({
                                'ticker': ticker,
                                'date': ed.strftime('%Y-%m-%d'),
                                'days_away': (ed - today).days,
                            })
            except Exception as e:
                logger.debug(f"Error fetching earnings for {ticker}: {e}")
                continue
        
        return sorted(earnings, key=lambda x: x['date'])
    
    def get_event_risk_for_ticker(self, ticker: str) -> Dict:
        """
        Assess event risk for a specific ticker.
        Returns warnings if major events are imminent.
        """
        events = self.get_upcoming_events(days_ahead=7)
        earnings = self.get_earnings_this_week([ticker])
        
        warnings = []
        risk_level = 'low'
        
        # Check macro events
        for event in events:
            if event['days_away'] <= 2 and event['impact'] == 'high':
                warnings.append(f"{event['name']} in {event['days_away']} day(s)")
                risk_level = 'high'
            elif event['days_away'] <= 5 and event['impact'] == 'high':
                risk_level = 'medium' if risk_level == 'low' else risk_level
        
        # Check earnings
        for earn in earnings:
            if earn['ticker'].upper() == ticker.upper():
                warnings.append(f"Earnings in {earn['days_away']} day(s)")
                risk_level = 'high'
        
        return {
            'ticker': ticker,
            'risk_level': risk_level,
            'warnings': warnings,
            'upcoming_events': len(events),
        }
    
    def get_calendar_summary(self) -> Dict:
        """
        Get a summary for dashboard display.
        """
        events = self.get_upcoming_events(days_ahead=14)
        
        imminent = [e for e in events if e['days_away'] <= 3]
        upcoming = [e for e in events if 3 < e['days_away'] <= 7]
        
        next_event = events[0] if events else None
        
        return {
            'total_events': len(events),
            'imminent_count': len(imminent),
            'imminent_events': imminent,
            'upcoming_events': upcoming,
            'next_event': next_event,
            'risk_alert': len(imminent) > 0,
            'updated': datetime.now().isoformat(),
        }


# Singleton
economic_calendar = EconomicCalendar()
