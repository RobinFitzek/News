"""
Discovery Hit Rate Tracker
Measures actual performance of promoted discoveries at 30/60/90 days.
Feeds back into learning optimizer to adjust strategy weights.
"""
import yfinance as yf
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db

logger = logging.getLogger(__name__)


class DiscoveryHitRate:
    """Track and analyze whether promoted discoveries actually performed."""

    def check_outcomes(self) -> Dict:
        """
        For all promoted discoveries, check price performance at 30/60/90 days.
        Stores results in discovery_outcomes table.
        Returns summary of what was updated.
        """
        try:
            # Get all promoted discoveries that have price data
            promoted = db.query("""
                SELECT id, ticker, promoted_at, price as promoted_price
                FROM discovered_stocks
                WHERE status = 'promoted'
                AND promoted_at IS NOT NULL
                AND price IS NOT NULL
                ORDER BY promoted_at DESC
                LIMIT 200
            """)

            updated = 0
            now = datetime.now()

            for disc in promoted:
                disc_id = disc['id']
                ticker = disc['ticker']
                try:
                    promoted_at = datetime.fromisoformat(disc['promoted_at'])
                except (TypeError, ValueError):
                    continue

                promoted_price = disc.get('promoted_price')
                if not promoted_price:
                    continue

                # Check which horizons are due and not yet computed
                existing = db.query_one("""
                    SELECT price_30d, price_60d, price_90d
                    FROM discovery_outcomes
                    WHERE discovery_id = ?
                """, (disc_id,))

                days_elapsed = (now - promoted_at).days
                if days_elapsed < 25:
                    continue  # Too early

                needs_30d = days_elapsed >= 30 and (not existing or existing.get('price_30d') is None)
                needs_60d = days_elapsed >= 60 and (not existing or existing.get('price_60d') is None)
                needs_90d = days_elapsed >= 90 and (not existing or existing.get('price_90d') is None)

                if not (needs_30d or needs_60d or needs_90d):
                    continue

                # Fetch price history
                start = promoted_at - timedelta(days=1)
                end = min(now, promoted_at + timedelta(days=95))
                try:
                    hist = yf.Ticker(ticker).history(start=start, end=end)
                except Exception:
                    continue

                if hist.empty:
                    continue

                def price_at_day(days: int) -> Optional[float]:
                    target = promoted_at + timedelta(days=days)
                    # Find closest trading day
                    nearby = hist[hist.index >= target]
                    if not nearby.empty:
                        return float(nearby['Close'].iloc[0])
                    return None

                p30 = price_at_day(30) if needs_30d else (existing or {}).get('price_30d')
                p60 = price_at_day(60) if needs_60d else (existing or {}).get('price_60d')
                p90 = price_at_day(90) if needs_90d else (existing or {}).get('price_90d')

                def ret(p):
                    if p and promoted_price:
                        return round(((p - promoted_price) / promoted_price) * 100, 2)
                    return None

                # Get strategy for this discovery
                strategy = db.query_one(
                    "SELECT strategy FROM discovered_stocks WHERE id = ?", (disc_id,)
                )
                strat = strategy['strategy'] if strategy else 'unknown'

                if existing:
                    db.execute("""
                        UPDATE discovery_outcomes SET
                            price_30d = COALESCE(?, price_30d),
                            price_60d = COALESCE(?, price_60d),
                            price_90d = COALESCE(?, price_90d),
                            return_30d = COALESCE(?, return_30d),
                            return_60d = COALESCE(?, return_60d),
                            return_90d = COALESCE(?, return_90d),
                            updated_at = ?
                        WHERE discovery_id = ?
                    """, (p30, p60, p90, ret(p30), ret(p60), ret(p90),
                          datetime.now().isoformat(), disc_id))
                else:
                    db.execute("""
                        INSERT OR IGNORE INTO discovery_outcomes
                        (discovery_id, ticker, promoted_at, promoted_price,
                         price_30d, price_60d, price_90d,
                         return_30d, return_60d, return_90d, strategy, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (disc_id, ticker, disc['promoted_at'], promoted_price,
                          p30, p60, p90, ret(p30), ret(p60), ret(p90),
                          strat, datetime.now().isoformat()))
                updated += 1

            logger.info(f"Discovery outcomes updated for {updated} stocks")
            return {'updated': updated}

        except Exception as e:
            logger.error(f"Error checking discovery outcomes: {e}")
            return {'updated': 0, 'error': str(e)}

    def get_strategy_hit_rates(self) -> List[Dict]:
        """
        Group outcomes by strategy, compute hit rate and avg return at each horizon.
        """
        try:
            rows = db.query("""
                SELECT strategy,
                    COUNT(*) as total,
                    SUM(CASE WHEN return_30d > 0 THEN 1 ELSE 0 END) as wins_30d,
                    SUM(CASE WHEN return_60d > 0 THEN 1 ELSE 0 END) as wins_60d,
                    SUM(CASE WHEN return_90d > 0 THEN 1 ELSE 0 END) as wins_90d,
                    AVG(return_30d) as avg_30d,
                    AVG(return_60d) as avg_60d,
                    AVG(return_90d) as avg_90d,
                    COUNT(return_30d) as n_30d,
                    COUNT(return_60d) as n_60d,
                    COUNT(return_90d) as n_90d
                FROM discovery_outcomes
                GROUP BY strategy
                ORDER BY avg_30d DESC
            """)

            result = []
            for row in rows:
                r = dict(row)
                t30 = r.get('n_30d') or 1
                t60 = r.get('n_60d') or 1
                t90 = r.get('n_90d') or 1
                result.append({
                    'strategy': r.get('strategy') or 'unknown',
                    'total': r.get('total', 0),
                    'hit_rate_30d': round(r.get('wins_30d', 0) / t30 * 100, 1) if t30 else None,
                    'hit_rate_60d': round(r.get('wins_60d', 0) / t60 * 100, 1) if t60 else None,
                    'hit_rate_90d': round(r.get('wins_90d', 0) / t90 * 100, 1) if t90 else None,
                    'avg_return_30d': round(r.get('avg_30d') or 0, 2),
                    'avg_return_60d': round(r.get('avg_60d') or 0, 2),
                    'avg_return_90d': round(r.get('avg_90d') or 0, 2),
                })
            return result

        except Exception as e:
            logger.error(f"Error computing strategy hit rates: {e}")
            return []

    def get_overall_hit_rate(self) -> Dict:
        """Overall discovery hit rate across all strategies."""
        try:
            row = db.query_one("""
                SELECT
                    COUNT(*) as total,
                    COUNT(return_30d) as with_30d,
                    SUM(CASE WHEN return_30d > 0 THEN 1 ELSE 0 END) as wins_30d,
                    AVG(return_30d) as avg_30d,
                    COUNT(return_90d) as with_90d,
                    SUM(CASE WHEN return_90d > 0 THEN 1 ELSE 0 END) as wins_90d,
                    AVG(return_90d) as avg_90d
                FROM discovery_outcomes
            """)
            if not row or not row.get('total'):
                return {'available': False}

            n30 = row.get('with_30d') or 1
            n90 = row.get('with_90d') or 1
            return {
                'available': True,
                'total_tracked': row.get('total', 0),
                'hit_rate_30d': round((row.get('wins_30d') or 0) / n30 * 100, 1),
                'avg_return_30d': round(row.get('avg_30d') or 0, 2),
                'hit_rate_90d': round((row.get('wins_90d') or 0) / n90 * 100, 1),
                'avg_return_90d': round(row.get('avg_90d') or 0, 2),
            }
        except Exception as e:
            logger.error(f"Error computing overall hit rate: {e}")
            return {'available': False}

    def get_recent_outcomes(self, limit: int = 20) -> List[Dict]:
        """Get most recent discovery outcomes for display."""
        try:
            return db.query("""
                SELECT ticker, promoted_at, promoted_price,
                       price_30d, price_60d, price_90d,
                       return_30d, return_60d, return_90d, strategy
                FROM discovery_outcomes
                WHERE promoted_at IS NOT NULL
                ORDER BY promoted_at DESC
                LIMIT ?
            """, (limit,))
        except Exception as e:
            logger.error(f"Error fetching recent outcomes: {e}")
            return []


# Singleton
discovery_hit_rate = DiscoveryHitRate()
