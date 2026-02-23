"""
Data Freshness Tracker
Monitors per-ticker yfinance fetch health. Detects when data goes stale
due to API failures, rate-limiting, or silent empty responses.

yfinance fails quietly — this module catches that before bad signals propagate.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db

logger = logging.getLogger(__name__)

# US market holidays (approximate, covers major ones)
US_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (1, 15),  # MLK Day (approx)
    (2, 19),  # Presidents' Day (approx)
    (5, 27),  # Memorial Day (approx)
    (6, 19),  # Juneteenth
    (7, 4),   # Independence Day
    (9, 2),   # Labor Day (approx)
    (11, 28), # Thanksgiving (approx)
    (12, 25), # Christmas
}


class DataFreshnessTracker:
    """Track per-ticker data fetch success/failure to detect stale data."""

    def __init__(self):
        self._init_table()

    def _init_table(self):
        """Create data_freshness table if it doesn't exist."""
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS data_freshness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'yfinance',
                    last_success TEXT,
                    last_failure TEXT,
                    last_error TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    total_successes INTEGER DEFAULT 0,
                    total_failures INTEGER DEFAULT 0,
                    UNIQUE(ticker, source)
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create data_freshness table: {e}")

    def record_success(self, ticker: str, source: str = 'yfinance'):
        """Record a successful data fetch for a ticker."""
        try:
            now = datetime.now().isoformat()
            db.execute("""
                INSERT INTO data_freshness (ticker, source, last_success, consecutive_failures, total_successes)
                VALUES (?, ?, ?, 0, 1)
                ON CONFLICT(ticker, source) DO UPDATE SET
                    last_success = excluded.last_success,
                    consecutive_failures = 0,
                    total_successes = total_successes + 1
            """, (ticker.upper(), source, now))
        except Exception as e:
            logger.debug(f"Error recording fetch success for {ticker}: {e}")

    def record_failure(self, ticker: str, error_msg: str = '', source: str = 'yfinance'):
        """Record a failed data fetch for a ticker."""
        try:
            now = datetime.now().isoformat()
            db.execute("""
                INSERT INTO data_freshness (ticker, source, last_failure, last_error, consecutive_failures, total_failures)
                VALUES (?, ?, ?, ?, 1, 1)
                ON CONFLICT(ticker, source) DO UPDATE SET
                    last_failure = excluded.last_failure,
                    last_error = ?,
                    consecutive_failures = consecutive_failures + 1,
                    total_failures = total_failures + 1
            """, (ticker.upper(), source, now, error_msg[:500], error_msg[:500]))
        except Exception as e:
            logger.debug(f"Error recording fetch failure for {ticker}: {e}")

    def is_market_day(self, dt: datetime = None) -> bool:
        """Check if the given date is a US market trading day (not weekend/holiday)."""
        if dt is None:
            dt = datetime.now()
        # Weekend
        if dt.weekday() >= 5:
            return False
        # Approximate holiday check
        if (dt.month, dt.day) in US_HOLIDAYS:
            return False
        return True

    def get_stale_tickers(self, max_age_hours: int = 24) -> List[Dict]:
        """
        Return tickers whose last successful fetch is older than max_age_hours.
        Only flags staleness on market days — stale over a weekend is expected.
        """
        if not self.is_market_day():
            return []  # Don't flag staleness on non-market days

        try:
            cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
            rows = db.query("""
                SELECT ticker, source, last_success, last_failure, last_error,
                       consecutive_failures, total_successes, total_failures
                FROM data_freshness
                WHERE last_success IS NULL OR last_success < ?
                ORDER BY consecutive_failures DESC, ticker ASC
            """, (cutoff,))

            stale = []
            for r in rows:
                hours_stale = None
                if r['last_success']:
                    try:
                        last = datetime.fromisoformat(r['last_success'])
                        hours_stale = round((datetime.now() - last).total_seconds() / 3600, 1)
                    except Exception:
                        pass

                stale.append({
                    'ticker': r['ticker'],
                    'source': r['source'],
                    'last_success': r['last_success'],
                    'last_failure': r['last_failure'],
                    'last_error': r['last_error'],
                    'consecutive_failures': r['consecutive_failures'],
                    'hours_stale': hours_stale,
                    'never_succeeded': r['last_success'] is None,
                })

            return stale

        except Exception as e:
            logger.error(f"Error getting stale tickers: {e}")
            return []

    def get_freshness_summary(self) -> Dict:
        """
        Overall data health summary.
        Returns total tracked, fresh count, stale count, failure tickers.
        """
        try:
            total_row = db.query_one("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN consecutive_failures > 0 THEN 1 ELSE 0 END) as failing
                FROM data_freshness
            """)
            total = total_row['total'] if total_row else 0
            failing = total_row['failing'] if total_row else 0

            stale_tickers = self.get_stale_tickers()
            stale_count = len(stale_tickers)

            # Tickers with 3+ consecutive failures (problematic)
            problem_rows = db.query("""
                SELECT ticker, consecutive_failures, last_error
                FROM data_freshness
                WHERE consecutive_failures >= 3
                ORDER BY consecutive_failures DESC
                LIMIT 10
            """) or []

            healthy = stale_count == 0 and failing == 0

            return {
                'total_tracked': total,
                'stale_count': stale_count,
                'stale_tickers': [t['ticker'] for t in stale_tickers],
                'failing_count': failing,
                'problem_tickers': [
                    {'ticker': r['ticker'], 'failures': r['consecutive_failures'], 'error': r['last_error']}
                    for r in problem_rows
                ],
                'healthy': healthy,
                'is_market_day': self.is_market_day(),
                'checked_at': datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error building freshness summary: {e}")
            return {
                'total_tracked': 0,
                'stale_count': 0,
                'stale_tickers': [],
                'failing_count': 0,
                'problem_tickers': [],
                'healthy': True,
                'is_market_day': self.is_market_day(),
                'checked_at': datetime.now().isoformat(),
                'error': str(e),
            }

    def get_ticker_freshness(self, ticker: str, source: str = 'yfinance') -> Optional[Dict]:
        """Get freshness data for a single ticker."""
        try:
            row = db.query_one("""
                SELECT ticker, source, last_success, last_failure, last_error,
                       consecutive_failures, total_successes, total_failures
                FROM data_freshness
                WHERE ticker = ? AND source = ?
            """, (ticker.upper(), source))

            if not row:
                return None

            hours_since = None
            if row['last_success']:
                try:
                    last = datetime.fromisoformat(row['last_success'])
                    hours_since = round((datetime.now() - last).total_seconds() / 3600, 1)
                except Exception:
                    pass

            success_rate = 0
            total = (row['total_successes'] or 0) + (row['total_failures'] or 0)
            if total > 0:
                success_rate = round((row['total_successes'] or 0) / total * 100, 1)

            return {
                'ticker': row['ticker'],
                'source': row['source'],
                'last_success': row['last_success'],
                'hours_since_success': hours_since,
                'consecutive_failures': row['consecutive_failures'],
                'success_rate': success_rate,
                'total_fetches': total,
            }

        except Exception as e:
            logger.error(f"Error getting freshness for {ticker}: {e}")
            return None


# Singleton
data_freshness = DataFreshnessTracker()
