"""
Alert Deduplication & Priority Management
Prevents alert fatigue by deduplicating repeated alerts and adding priority ranking.
Users can acknowledge alerts to dismiss them without deleting.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import hashlib
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage alerts with deduplication and acknowledgment."""

    def __init__(self):
        self.dedup_window_hours = 24  # Don't re-alert within 24 hours

    def generate_alert_hash(self, alert: Dict) -> str:
        """Generate unique hash for alert deduplication."""
        # Hash based on type + ticker + key message components
        key = f"{alert.get('type', '')}:{alert.get('ticker', '')}:{alert.get('severity', '')}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def should_alert(self, alert: Dict) -> bool:
        """Check if we should show this alert (not a duplicate)."""
        alert_hash = self.generate_alert_hash(alert)
        
        try:
            # Check if we've sent this alert recently
            recent = db.query_one("""
                SELECT id, timestamp, acknowledged
                FROM alerts
                WHERE alert_hash = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (alert_hash, (datetime.now() - timedelta(hours=self.dedup_window_hours)).isoformat()))

            if recent:
                # Alert exists within dedup window
                if recent['acknowledged'] == 1:
                    # User acknowledged it, don't re-alert
                    return False
                else:
                    # Alert exists but not acknowledged - still show but mark as repeated
                    alert['is_repeated'] = True
                    alert['first_seen'] = recent['timestamp']
                    return True
            
            # New alert
            alert['is_repeated'] = False
            return True

        except Exception as e:
            logger.error(f"Error checking alert deduplication: {e}")
            return True  # Default to showing alert if error

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
