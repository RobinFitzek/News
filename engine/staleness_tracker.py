"""
Signal Staleness & Decay Tracking
An analysis from 3 days ago is not the same as one from 30 days ago.
Confidence decays ~5% per week. Stale analyses (>14 days) need refresh.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class StalenessTracker:
    """Track signal age and apply confidence decay."""

    # Decay rate: 5% per week
    DECAY_RATE_PER_DAY = 0.05 / 7  # ~0.7% per day
    STALE_THRESHOLD_DAYS = 14
    VERY_STALE_THRESHOLD_DAYS = 30

    def calculate_age_days(self, timestamp: str) -> int:
        """Calculate how many days old an analysis is."""
        try:
            if isinstance(timestamp, datetime):
                analysis_time = timestamp
            else:
                analysis_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            age = (datetime.now() - analysis_time).days
            return max(0, age)  # Never negative
        except Exception as e:
            logger.error(f"Error calculating age: {e}")
            return 0

    def apply_confidence_decay(self, original_confidence: float, age_days: int) -> float:
        """
        Apply exponential decay to confidence based on age.
        Example: 90% confidence after 7 days -> ~85.5%, after 14 days -> ~81%
        """
        if age_days == 0:
            return original_confidence

        # Exponential decay: confidence = original * (1 - decay_rate)^days
        decay_factor = (1 - self.DECAY_RATE_PER_DAY) ** age_days
        decayed_confidence = original_confidence * decay_factor
        
        return max(0, min(100, decayed_confidence))

    def get_staleness_level(self, age_days: int) -> str:
        """Categorize staleness level."""
        if age_days <= 3:
            return 'fresh'
        elif age_days <= 7:
            return 'recent'
        elif age_days <= self.STALE_THRESHOLD_DAYS:
            return 'aging'
        elif age_days <= self.VERY_STALE_THRESHOLD_DAYS:
            return 'stale'
        else:
            return 'very_stale'

    def get_staleness_icon(self, level: str) -> str:
        """Get icon for staleness level."""
        icons = {
            'fresh': '[Fresh]',
            'recent': '[Recent]',
            'aging': '[Aging]',
            'stale': '[Stale]',
            'very_stale': '[Outdated]',
        }
        return icons.get(level, '[•]')

    def should_refresh(self, age_days: int) -> bool:
        """Determine if analysis should be refreshed."""
        return age_days >= self.STALE_THRESHOLD_DAYS

    def enrich_analysis(self, analysis: Dict) -> Dict:
        """Add staleness metadata to analysis."""
        timestamp = analysis.get('timestamp', analysis.get('created_at', datetime.now().isoformat()))
        age_days = self.calculate_age_days(timestamp)
        
        original_confidence = analysis.get('confidence', 50)
        decayed_confidence = self.apply_confidence_decay(original_confidence, age_days)
        
        staleness_level = self.get_staleness_level(age_days)
        
        analysis['age_days'] = age_days
        analysis['staleness_level'] = staleness_level
        analysis['staleness_icon'] = self.get_staleness_icon(staleness_level)
        analysis['original_confidence'] = original_confidence
        analysis['decayed_confidence'] = round(decayed_confidence, 1)
        analysis['needs_refresh'] = self.should_refresh(age_days)
        
        # Add warning for stale signals
        if staleness_level in ['stale', 'very_stale']:
            if 'warnings' not in analysis:
                analysis['warnings'] = []
            analysis['warnings'].append(
                f"(Stale) Analysis is {age_days} days old — confidence decayed from "
                f"{original_confidence}% to {decayed_confidence:.1f}%"
            )
            if self.should_refresh(age_days):
                analysis['warnings'].append("(Refresh) NEEDS REFRESH — Data may be outdated")

        return analysis

    def get_stale_analyses(self, min_age_days: int = None) -> List[Dict]:
        """Get all analyses that need refreshing."""
        if min_age_days is None:
            min_age_days = self.STALE_THRESHOLD_DAYS

        try:
            cutoff_date = (datetime.now() - timedelta(days=min_age_days)).isoformat()
            
            results = db.query("""
                SELECT id, ticker, signal, confidence, timestamp, recommendation
                FROM analyses
                WHERE timestamp < ?
                ORDER BY timestamp ASC
            """, (cutoff_date,))
            
            stale_list = []
            for r in results:
                age_days = self.calculate_age_days(r['timestamp'])
                stale_list.append({
                    'id': r['id'],
                    'ticker': r['ticker'],
                    'signal': r['signal'],
                    'original_confidence': r['confidence'],
                    'decayed_confidence': round(self.apply_confidence_decay(r['confidence'], age_days), 1),
                    'age_days': age_days,
                    'staleness_level': self.get_staleness_level(age_days),
                })

            return stale_list

        except Exception as e:
            logger.error(f"Error getting stale analyses: {e}")
            return []

    def bulk_update_staleness(self):
        """Update staleness metrics for all analyses (run daily)."""
        try:
            analyses = db.query("""
                SELECT id, timestamp, confidence
                FROM analyses
            """)

            updated = 0
            for analysis in analyses:
                age_days = self.calculate_age_days(analysis['timestamp'])
                decayed_conf = self.apply_confidence_decay(analysis['confidence'], age_days)
                staleness = self.get_staleness_level(age_days)
                needs_refresh = self.should_refresh(age_days)

                db.execute("""
                    UPDATE analyses
                    SET age_days = ?,
                        staleness_level = ?,
                        decayed_confidence = ?,
                        needs_refresh = ?
                    WHERE id = ?
                """, (age_days, staleness, decayed_conf, 1 if needs_refresh else 0, analysis['id']))
                
                updated += 1

            logger.info(f"Updated staleness for {updated} analyses")
            return updated

        except Exception as e:
            logger.error(f"Error in bulk staleness update: {e}")
            return 0

    def sort_by_freshness(self, analyses: List[Dict], primary_sort: str = 'score') -> List[Dict]:
        """
        Sort analyses prioritizing freshness while maintaining signal strength.
        Default: sort by score, but boost fresh signals.
        """
        for analysis in analyses:
            if 'age_days' not in analysis:
                self.enrich_analysis(analysis)

        # Create composite score: base score + freshness bonus
        def sort_key(a):
            score = a.get('composite_score', a.get('score', 50))
            age_days = a.get('age_days', 0)
            
            # Fresh analyses (< 3 days) get +5 points
            # Recent (< 7 days) get +3
            # Aging (< 14 days) get +0
            # Stale get -5
            freshness_bonus = {
                'fresh': 5,
                'recent': 3,
                'aging': 0,
                'stale': -5,
                'very_stale': -10,
            }.get(a.get('staleness_level', 'aging'), 0)

            return score + freshness_bonus

        return sorted(analyses, key=sort_key, reverse=True)


# Singleton
staleness_tracker = StalenessTracker()
