"""
Earnings Calendar Integration
Tracks upcoming earnings dates and flags stocks with earnings in next 14 days.
Critical for signal accuracy — pre-earnings positions carry high uncertainty.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class EarningsTracker:
    """Track earnings dates and post-earnings drift."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(hours=12)  # Refresh twice daily

    def get_earnings_info(self, ticker: str) -> Optional[Dict]:
        """Get next earnings date and days until earnings."""
        # Check cache first
        if ticker in self._cache:
            entry = self._cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            logger.debug(f"Fetching earnings calendar for {ticker}, type: {type(calendar)}")
            
            if calendar is None:
                logger.debug(f"No earnings calendar for {ticker}")
                return None
            
            # Handle both dict and DataFrame formats from yfinance
            earnings_date = None
            
            if isinstance(calendar, dict):
                # Calendar is a dict
                if 'Earnings Date' in calendar:
                    earnings_date = calendar['Earnings Date']
                    if isinstance(earnings_date, list) and earnings_date:
                        earnings_date = earnings_date[0]  # Take first date if multiple
            elif hasattr(calendar, 'empty') and not calendar.empty:
                # Calendar is a DataFrame
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                elif hasattr(calendar, 'values') and len(calendar.values) > 0:
                    earnings_date = calendar.values[0][0]
            else:
                logger.debug(f"Unexpected calendar format for {ticker}: {type(calendar)}")
                return None

            if earnings_date is None:
                logger.debug(f"Could not extract earnings date from calendar for {ticker}")
                return None

            # Convert to datetime if it's a timestamp
            if hasattr(earnings_date, 'to_pydatetime'):
                earnings_date = earnings_date.to_pydatetime()
            elif not isinstance(earnings_date, datetime):
                earnings_date = datetime.fromisoformat(str(earnings_date))

            days_until = (earnings_date - datetime.now()).days

            data = {
                'ticker': ticker,
                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'is_imminent': 0 <= days_until <= 14,
                'is_within_week': 0 <= days_until <= 7,
                'fetched_at': datetime.now().isoformat(),
            }

            # Cache result
            self._cache[ticker] = {'data': data, 'timestamp': datetime.now()}

            # Store in database
            self._store_earnings_date(data)

            return data

        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return None

    def _store_earnings_date(self, data: Dict):
        """Store earnings date in database."""
        try:
            db.execute("""
                INSERT INTO earnings_calendar (ticker, earnings_date, days_until, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    earnings_date = excluded.earnings_date,
                    days_until = excluded.days_until,
                    fetched_at = excluded.fetched_at
            """, (data['ticker'], data['earnings_date'], data['days_until'], data['fetched_at']))
        except Exception as e:
            logger.warning(f"Could not store earnings date: {e}")

    def check_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Check earnings for multiple tickers. Returns dict of ticker -> earnings_info."""
        results = {}
        for ticker in tickers:
            info = self.get_earnings_info(ticker)
            if info:
                results[ticker] = info
        return results

    def flag_pre_earnings_risk(self, ticker: str, analysis_data: Dict) -> Dict:
        """Add earnings warning flags to analysis data."""
        earnings = self.get_earnings_info(ticker)
        
        if not earnings:
            analysis_data['earnings_risk'] = 'unknown'
            return analysis_data

        if earnings['is_within_week']:
            analysis_data['earnings_risk'] = 'high'
            analysis_data['warnings'] = analysis_data.get('warnings', []) + [
                f"(Earnings) EARNINGS IN {earnings['days_until']} DAYS — High uncertainty, expect 5-15% volatility"
            ]
            # Reduce confidence for signals during earnings week
            if 'confidence' in analysis_data:
                analysis_data['confidence'] = int(analysis_data['confidence'] * 0.6)
                analysis_data['warnings'].append("(Notice) Signal confidence reduced by 40% due to earnings proximity")
        
        elif earnings['is_imminent']:
            analysis_data['earnings_risk'] = 'moderate'
            analysis_data['warnings'] = analysis_data.get('warnings', []) + [
                f"(Earnings) Earnings in {earnings['days_until']} days — Increased volatility expected"
            ]
            if 'confidence' in analysis_data:
                analysis_data['confidence'] = int(analysis_data['confidence'] * 0.85)

        else:
            analysis_data['earnings_risk'] = 'low'

        analysis_data['next_earnings_date'] = earnings['earnings_date']
        analysis_data['days_until_earnings'] = earnings['days_until']

        return analysis_data

    def track_post_earnings_drift(self, ticker: str, earnings_date: datetime) -> Optional[Dict]:
        """Track stock movement in 5 days after earnings (for learning)."""
        try:
            # Get price at earnings date and 5 days later
            end_date = earnings_date + timedelta(days=7)
            hist = yf.Ticker(ticker).history(start=earnings_date, end=end_date)
            
            if hist.empty or len(hist) < 2:
                return None

            open_price = float(hist['Open'].iloc[0])  # Opening price on earnings day
            close_price = float(hist['Close'].iloc[-1])  # Close 5 days later
            
            drift_pct = ((close_price - open_price) / open_price) * 100

            return {
                'ticker': ticker,
                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                'open_price': round(open_price, 2),
                'close_price_5d': round(close_price, 2),
                'drift_pct': round(drift_pct, 2),
                'direction': 'up' if drift_pct > 0 else 'down',
            }

        except Exception as e:
            logger.error(f"Error tracking post-earnings drift for {ticker}: {e}")
            return None

    def get_cached_earnings(self, ticker: str) -> Optional[Dict]:
        """Get earnings info from database cache (no API call)."""
        try:
            result = db.query_one("""
                SELECT ticker, earnings_date, days_until, fetched_at
                FROM earnings_calendar
                WHERE ticker = ?
                AND datetime(fetched_at) > datetime('now', '-12 hours')
            """, (ticker,))
            
            if result:
                return {
                    'ticker': result['ticker'],
                    'earnings_date': result['earnings_date'],
                    'days_until': result['days_until'],
                    'is_imminent': 0 <= result['days_until'] <= 14,
                    'is_within_week': 0 <= result['days_until'] <= 7,
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached earnings: {e}")
            return None


# Singleton
earnings_tracker = EarningsTracker()
