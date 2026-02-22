"""
Short Interest Tracker
Monitors short % of float, days to cover, and detects potential squeeze setups
by combining high short interest with rising price momentum.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class ShortInterestTracker:
    """Track short interest metrics and detect squeeze setups."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=30)
        self._init_table()

    def _init_table(self):
        """Create short interest history table if it doesn't exist."""
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS short_interest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    short_pct_float REAL,
                    days_to_cover REAL,
                    short_shares INTEGER,
                    UNIQUE(ticker, date)
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create short_interest_history table: {e}")

    def get_short_data(self, ticker: str) -> Optional[Dict]:
        """Fetch current short interest data from yfinance info."""
        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(cache_key)
            info = stock.info

            if not info or len(info) < 3:
                return None

            short_pct_float = info.get('shortPercentOfFloat')
            if short_pct_float and short_pct_float < 1:
                short_pct_float = short_pct_float * 100  # Convert to percentage

            data = {
                'ticker': cache_key,
                'short_pct_float': round(short_pct_float, 2) if short_pct_float else None,
                'days_to_cover': info.get('shortRatio'),
                'short_shares': info.get('sharesShort'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'fetched_at': datetime.now().isoformat(),
            }

            self._cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data

        except Exception as e:
            logger.error(f"Error fetching short data for {ticker}: {e}")
            return None

    def check_squeeze_setup(self, ticker: str) -> Optional[Dict]:
        """
        Detect short squeeze setups: high short interest combined with rising price.
        A squeeze candidate has SI > 15% of float AND positive recent momentum.
        """
        try:
            short_data = self.get_short_data(ticker)
            if not short_data or short_data.get('short_pct_float') is None:
                return None

            si_pct = short_data['short_pct_float']
            dtc = short_data.get('days_to_cover') or 0

            # Get recent price momentum
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(period="1mo")
            if hist.empty or len(hist) < 5:
                return None

            close = hist['Close']
            price_now = float(close.iloc[-1])
            price_5d_ago = float(close.iloc[-5])
            price_change_5d = ((price_now - price_5d_ago) / price_5d_ago) * 100

            # Squeeze scoring
            squeeze_score = 0
            signals = []

            if si_pct >= 20:
                squeeze_score += 40
                signals.append(f"Very high SI: {si_pct:.1f}%")
            elif si_pct >= 15:
                squeeze_score += 25
                signals.append(f"High SI: {si_pct:.1f}%")
            elif si_pct >= 10:
                squeeze_score += 10
                signals.append(f"Elevated SI: {si_pct:.1f}%")

            if dtc >= 5:
                squeeze_score += 25
                signals.append(f"High days-to-cover: {dtc:.1f}")
            elif dtc >= 3:
                squeeze_score += 15
                signals.append(f"Moderate days-to-cover: {dtc:.1f}")

            if price_change_5d > 5:
                squeeze_score += 25
                signals.append(f"Strong 5d momentum: +{price_change_5d:.1f}%")
            elif price_change_5d > 2:
                squeeze_score += 15
                signals.append(f"Rising 5d momentum: +{price_change_5d:.1f}%")

            is_squeeze_candidate = squeeze_score >= 50

            return {
                'ticker': ticker.upper(),
                'squeeze_score': squeeze_score,
                'is_squeeze_candidate': is_squeeze_candidate,
                'short_pct_float': si_pct,
                'days_to_cover': dtc,
                'price_change_5d': round(price_change_5d, 2),
                'signals': signals,
            }

        except Exception as e:
            logger.error(f"Error checking squeeze setup for {ticker}: {e}")
            return None

    def store_snapshot(self, ticker: str) -> bool:
        """Store current short interest data as a historical snapshot."""
        try:
            data = self.get_short_data(ticker)
            if not data:
                return False

            today = datetime.now().strftime('%Y-%m-%d')
            db.execute("""
                INSERT INTO short_interest_history (ticker, date, short_pct_float, days_to_cover, short_shares)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    short_pct_float = excluded.short_pct_float,
                    days_to_cover = excluded.days_to_cover,
                    short_shares = excluded.short_shares
            """, (
                data['ticker'], today,
                data.get('short_pct_float'),
                data.get('days_to_cover'),
                data.get('short_shares'),
            ))
            logger.info(f"Stored short interest snapshot for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error storing short interest snapshot for {ticker}: {e}")
            return False

    def get_history(self, ticker: str, days: int = 90) -> List[Dict]:
        """Retrieve short interest history for a ticker over the last N days."""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            rows = db.query("""
                SELECT ticker, date, short_pct_float, days_to_cover, short_shares
                FROM short_interest_history
                WHERE ticker = ? AND date >= ?
                ORDER BY date ASC
            """, (ticker.upper(), cutoff))
            return [dict(r) for r in rows] if rows else []

        except Exception as e:
            logger.error(f"Error fetching short interest history for {ticker}: {e}")
            return []


# Singleton
short_interest_tracker = ShortInterestTracker()
