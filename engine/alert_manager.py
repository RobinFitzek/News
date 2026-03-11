"""
Alert Deduplication & Priority Management
Prevents alert fatigue by deduplicating repeated alerts and adding priority ranking.
Uses direction-change + score-delta + geo-event gates to reduce alert fatigue.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import hashlib
import logging

logger = logging.getLogger(__name__)

# Signal direction groups for flip detection
_BUY_SIGNALS = {'STRONG_BUY', 'BUY', 'OPPORTUNITY'}
_SELL_SIGNALS = {'STRONG_SELL', 'SELL', 'CAUTION'}


def _signal_direction(signal: str) -> str:
    """Return 'buy', 'sell', or 'neutral' for a signal string."""
    s = (signal or '').upper()
    if s in _BUY_SIGNALS:
        return 'buy'
    if s in _SELL_SIGNALS:
        return 'sell'
    return 'neutral'


class AlertManager:
    """Manage alerts with deduplication and acknowledgment."""

    def __init__(self):
        self.dedup_window_hours = 24  # Don't re-alert within 24 hours

    def _get_cooldown_hours(self) -> int:
        """Get configured alert cooldown hours (default 24)."""
        try:
            val = db.get_setting('alert_cooldown_hours')
            return int(val) if val is not None else 24
        except Exception:
            return 24

    def _get_watchlist_alert_state(self, ticker: str) -> Dict:
        """Fetch last alert state for a ticker from watchlist table."""
        try:
            row = db.query_one(
                "SELECT last_alert_signal, last_alert_score, last_alert_geo_event_id, last_alerted_at FROM watchlist WHERE ticker = ?",
                (ticker.upper(),)
            )
            return row or {}
        except Exception:
            return {}

    def update_watchlist_alert_state(self, ticker: str, signal: str, score: int, geo_event_id: Optional[int] = None):
        """Update last-alert tracking columns in watchlist after sending an alert."""
        try:
            db.execute(
                """UPDATE watchlist
                   SET last_alert_signal = ?, last_alert_score = ?,
                       last_alert_geo_event_id = ?, last_alerted_at = ?
                   WHERE ticker = ?""",
                (signal, score, geo_event_id, datetime.now().isoformat(), ticker.upper())
            )
        except Exception as e:
            logger.warning(f"Could not update watchlist alert state for {ticker}: {e}")

    def should_send_ticker_alert(self, ticker: str, new_signal: str, new_score: int) -> bool:
        """
        Smart dedup gate for stock signal alerts.

        Allows re-alerting when ANY of:
          1. Direction flipped (buy ↔ sell ↔ neutral)
          2. |risk_score_delta| >= 2
          3. A new geo event has occurred since the last alert
          4. Cooldown window has expired

        Returns True if the alert should be sent.
        """
        cooldown_hours = self._get_cooldown_hours()
        state = self._get_watchlist_alert_state(ticker)

        last_signal = state.get('last_alert_signal') or ''
        last_score = state.get('last_alert_score')
        last_alerted_at = state.get('last_alerted_at')
        last_geo_id = state.get('last_alert_geo_event_id')

        # If never alerted before, allow
        if not last_alerted_at:
            return True

        # Cooldown expired — allow
        try:
            last_dt = datetime.fromisoformat(last_alerted_at)
            elapsed_hours = (datetime.now() - last_dt).total_seconds() / 3600
            if elapsed_hours >= cooldown_hours:
                return True
        except Exception:
            return True

        # Direction flip — allow
        if _signal_direction(new_signal) != _signal_direction(last_signal):
            return True

        # Score delta >= 2 — allow
        if last_score is not None and abs(new_score - int(last_score)) >= 2:
            return True

        # New geo event since last alert — allow
        try:
            latest_geo = db.query_one(
                "SELECT id FROM geopolitical_events ORDER BY id DESC LIMIT 1"
            )
            if latest_geo and (last_geo_id is None or latest_geo['id'] > last_geo_id):
                return True
        except Exception:
            pass

        # None of the gates passed — suppress
        return False

    def generate_alert_hash(self, alert: Dict) -> str:
        """Generate unique hash for alert deduplication."""
        # Hash based on type + ticker + key message components
        key = f"{alert.get('type', '')}:{alert.get('ticker', '')}:{alert.get('severity', '')}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def should_alert(self, alert: Dict) -> bool:
        """Check if we should show this alert (not a duplicate).

        For stock signal alerts, applies direction + score-delta + geo smart dedup.
        """
        ticker = alert.get('ticker', '')
        signal = alert.get('signal', alert.get('type', ''))
        score = alert.get('risk_score', alert.get('score', 5))

        # Smart dedup for stock-signal alerts when ticker is known
        if ticker and signal and signal not in ('GEOPOLITICAL_ALERT', 'GEO', 'SYSTEM'):
            if not self.should_send_ticker_alert(ticker, signal, int(score)):
                alert['is_repeated'] = True
                alert['suppressed_by'] = 'smart_dedup'
                return False

        alert_hash = self.generate_alert_hash(alert)

        try:
            # Check if we've sent this alert recently (legacy hash dedup)
            recent = db.query_one("""
                SELECT id, timestamp, acknowledged
                FROM alerts
                WHERE alert_hash = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (alert_hash, (datetime.now() - timedelta(hours=self.dedup_window_hours)).isoformat()))

            if recent:
                if recent['acknowledged'] == 1:
                    return False
                else:
                    alert['is_repeated'] = True
                    alert['first_seen'] = recent['timestamp']
                    return True

            alert['is_repeated'] = False
            return True

        except Exception as e:
            logger.error(f"Error checking alert deduplication: {e}")
            return True

    def store_alert(self, alert: Dict):
        """Store alert in database with deduplication hash."""
        try:
            alert_hash = self.generate_alert_hash(alert)
            
            db.execute("""
                INSERT INTO alerts (
                    alert_hash, type, ticker, severity, message,
                    timestamp, acknowledged, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                alert_hash,
                alert.get('type', 'UNKNOWN'),
                alert.get('ticker', ''),
                alert.get('severity', 'INFO'),
                alert.get('message', ''),
                datetime.now().isoformat(),
                str(alert),  # Store full alert as JSON-like string
            ))
            
        except Exception as e:
            logger.warning(f"Could not store alert: {e}")

    def acknowledge_alert(self, alert_id: int = None, alert_hash: str = None):
        """Mark alert as acknowledged (user dismissed it)."""
        try:
            if alert_id:
                db.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            elif alert_hash:
                db.execute("UPDATE alerts SET acknowledged = 1 WHERE alert_hash = ?", (alert_hash,))
            logger.info(f"Acknowledged alert: id={alert_id}, hash={alert_hash}")
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")

    def get_active_alerts(self, include_acknowledged: bool = False) -> List[Dict]:
        """Get all active alerts, optionally including acknowledged ones."""
        try:
            query = """
                SELECT id, alert_hash, type, ticker, severity, message, 
                       timestamp, acknowledged
                FROM alerts
                WHERE timestamp > ?
            """
            params = [(datetime.now() - timedelta(days=7)).isoformat()]
            
            if not include_acknowledged:
                query += " AND acknowledged = 0"
            
            query += " ORDER BY timestamp DESC"
            
            alerts = db.query(query, tuple(params))
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []

    def prioritize_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """
        Sort alerts by priority:
        1. New CRITICAL alerts
        2. New WARNING alerts
        3. Repeated CRITICAL alerts
        4. Repeated WARNING alerts
        5. INFO alerts
        """
        def priority_key(alert):
            severity_score = {
                'CRITICAL': 100,
                'WARNING': 50,
                'INFO': 10,
            }.get(alert.get('severity', 'INFO'), 10)
            
            # New alerts get bonus priority
            is_repeated = alert.get('is_repeated', False)
            repeat_penalty = 30 if is_repeated else 0
            
            return severity_score - repeat_penalty

        return sorted(alerts, key=priority_key, reverse=True)

    def get_alert_summary(self) -> Dict:
        """Get summary of current alert state."""
        try:
            active = self.get_active_alerts(include_acknowledged=False)
            
            critical = [a for a in active if a['severity'] == 'CRITICAL']
            warning = [a for a in active if a['severity'] == 'WARNING']
            
            return {
                'total_active': len(active),
                'critical_count': len(critical),
                'warning_count': len(warning),
                'oldest_unacknowledged': min([datetime.fromisoformat(a['timestamp']) 
                                             for a in active]) if active else None,
            }
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {'total_active': 0, 'critical_count': 0, 'warning_count': 0}

    def cleanup_old_alerts(self, days_to_keep: int = 30):
        """Remove alerts older than specified days."""
        try:
            cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            result = db.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,))
            count = result.rowcount if hasattr(result, 'rowcount') else 0
            logger.info(f"Cleaned up {count} old alerts")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
            return 0


# Singleton
alert_manager = AlertManager()
