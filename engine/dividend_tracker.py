"""
Dividend & Ex-Date Tracking
Prevents false alarms when stop-loss is triggered by ex-dividend drops.
A 3% drop on ex-dividend day is normal, not a sell signal.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class DividendTracker:
    """Track dividend payments and ex-dividend dates."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(days=1)  # Refresh daily

    def get_dividend_info(self, ticker: str) -> Optional[Dict]:
        """Get next ex-dividend date and dividend amount."""
        # Check cache
        if ticker in self._cache:
            entry = self._cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends

            if dividends is None or dividends.empty:
                # Stock doesn't pay dividends
                return None

            # Get most recent dividend
            last_date = dividends.index[-1]
            last_amount = float(dividends.iloc[-1])

            # Try to estimate next ex-dividend date based on payment frequency
            # Most US stocks pay quarterly
            if len(dividends) >= 4:
                # Calculate average days between dividends
                dates = [d.to_pydatetime() for d in dividends.index[-4:]]
                intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                avg_interval = sum(intervals) / len(intervals)
                
                next_ex_date = last_date.to_pydatetime() + timedelta(days=avg_interval)
            else:
                # Assume quarterly (90 days)
                next_ex_date = last_date.to_pydatetime() + timedelta(days=90)

            days_until = (next_ex_date - datetime.now()).days

            # Calculate expected price drop % on ex-date (dividend yield impact)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            expected_drop_pct = 0
            if current_price > 0:
                expected_drop_pct = (last_amount / current_price) * 100

            data = {
                'ticker': ticker,
                'last_dividend': round(last_amount, 4),
                'last_ex_date': last_date.strftime('%Y-%m-%d'),
                'estimated_next_ex_date': next_ex_date.strftime('%Y-%m-%d'),
                'days_until_ex_date': days_until,
                'expected_drop_pct': round(expected_drop_pct, 2),
                'is_ex_date_soon': 0 <= days_until <= 7,
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'fetched_at': datetime.now().isoformat(),
            }

            # Cache and store
            self._cache[ticker] = {'data': data, 'timestamp': datetime.now()}
            self._store_dividend_info(data)

            return data

        except Exception as e:
            logger.error(f"Error fetching dividend info for {ticker}: {e}")
            return None

    def _store_dividend_info(self, data: Dict):
        """Store dividend info in database."""
        try:
            db.execute("""
                INSERT INTO dividend_calendar (
                    ticker, last_dividend, last_ex_date, estimated_next_ex_date,
                    days_until_ex_date, expected_drop_pct, dividend_yield, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_dividend = excluded.last_dividend,
                    last_ex_date = excluded.last_ex_date,
                    estimated_next_ex_date = excluded.estimated_next_ex_date,
                    days_until_ex_date = excluded.days_until_ex_date,
                    expected_drop_pct = excluded.expected_drop_pct,
                    dividend_yield = excluded.dividend_yield,
                    fetched_at = excluded.fetched_at
            """, (
                data['ticker'],
                data['last_dividend'],
                data['last_ex_date'],
                data['estimated_next_ex_date'],
                data['days_until_ex_date'],
                data['expected_drop_pct'],
                data['dividend_yield'],
                data['fetched_at'],
            ))
        except Exception as e:
            logger.warning(f"Could not store dividend info: {e}")

    def check_stop_loss_false_alarm(self, ticker: str, price_drop_pct: float) -> bool:
        """Check if a stop-loss trigger might be ex-dividend related."""
        div_info = self.get_dividend_info(ticker)
        
        if not div_info:
            return False  # Not a dividend stock

        # Check if we're within 2 days of ex-dividend date
        days_until = div_info['days_until_ex_date']
        expected_drop = div_info['expected_drop_pct']

        # False alarm if:
        # - Within 2 days of ex-date
        # - Drop is roughly equal to expected dividend drop (±1%)
        if -2 <= days_until <= 2 and abs(price_drop_pct + expected_drop) < 1.5:
            return True

        return False

    def adjust_stop_loss_for_dividend(self, ticker: str, stop_loss_pct: float) -> float:
        """Adjust stop-loss threshold for upcoming dividend."""
        div_info = self.get_dividend_info(ticker)
        
        if not div_info or not div_info['is_ex_date_soon']:
            return stop_loss_pct

        # If ex-date is soon, increase stop-loss threshold by dividend amount
        adjusted = stop_loss_pct + div_info['expected_drop_pct']
        logger.info(f"{ticker}: Adjusted stop-loss from {stop_loss_pct}% to {adjusted:.1f}% for upcoming dividend")
        
        return adjusted

    def get_dividend_stocks(self) -> List[Dict]:
        """Get all dividend-paying stocks from portfolio."""
        try:
            results = db.query("""
                SELECT ticker, last_dividend, estimated_next_ex_date, 
                       days_until_ex_date, dividend_yield
                FROM dividend_calendar
                WHERE dividend_yield > 0
                ORDER BY dividend_yield DESC
            """)
            return results
        except Exception as e:
            logger.error(f"Error retrieving dividend stocks: {e}")
            return []


# Singleton
dividend_tracker = DividendTracker()
