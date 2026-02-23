"""
Health Monitor
Self-maintenance module to keep the 24/7 system running smoothly.
Checks disk area, memory footprint, database size and performs periodic cleanups.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import shutil

from core.database import db
from core.config import DB_PATH
from engine.webhook_notifier import webhook_notifier

logger = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self._init_table()

    def _init_table(self):
        """Create system_health table if it does not exist."""
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_type TEXT NOT NULL,
                    value REAL,
                    unit TEXT,
                    status TEXT DEFAULT 'ok',
                    checked_at TEXT DEFAULT (datetime('now'))
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create system_health table: {e}")

    def _log_health(self, check_type: str, value: float, unit: str, status: str):
        try:
            db.execute("""
                INSERT INTO system_health (check_type, value, unit, status)
                VALUES (?, ?, ?, ?)
            """, (check_type, value, unit, status))
        except Exception as e:
            logger.error(f"Error logging health check: {e}")

    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage of the volume where the database resides."""
        try:
            path = str(DB_PATH.parent)
            total, used, free = shutil.disk_usage(path)
            
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            percent = (used / total) * 100 if total > 0 else 0
            
            status = 'ok'
            if percent >= 95:
                status = 'critical'
            elif percent >= 85:
                status = 'warning'
                
            self._log_health('disk', percent, '%', status)
                
            return {
                "path": path,
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "percent": round(percent, 1),
                "status": status
            }
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage, gracefully falling back if psutil is unavailable."""
        try:
            if PSUTIL_AVAILABLE:
                mem = psutil.virtual_memory()
                total_mb = mem.total / (1024**2)
                used_mb = mem.used / (1024**2)
                percent = mem.percent
            else:
                # Read from /proc/meminfo on Linux
                meminfo = {}
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip().split()[0]
                            meminfo[key] = int(val)
                
                total_kb = meminfo.get('MemTotal', 0)
                free_kb = meminfo.get('MemFree', 0)
                cached_kb = meminfo.get('Cached', 0)
                buffers_kb = meminfo.get('Buffers', 0)
                
                used_kb = total_kb - free_kb - cached_kb - buffers_kb
                
                total_mb = total_kb / 1024
                used_mb = used_kb / 1024
                percent = (used_kb / total_kb) * 100 if total_kb > 0 else 0
            
            status = 'ok'
            if percent >= 95:
                status = 'critical'
            elif percent >= 85:
                status = 'warning'
                
            self._log_health('memory', percent, '%', status)
                
            return {
                "total_mb": round(total_mb, 1),
                "used_mb": round(used_mb, 1),
                "percent": round(percent, 1),
                "status": status
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_db_size(self) -> Dict[str, Any]:
        """Measure SQLite database size and largest tables."""
        try:
            # File size
            size_mb = os.path.getsize(str(DB_PATH)) / (1024**2) if DB_PATH.exists() else 0
            
            conn = db._get_conn()
            cursor = conn.cursor()
            
            # Count tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r['name'] for r in cursor.fetchall()]
            table_count = len(tables)
            
            largest_tables = []
            for t in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM {t}")
                    row_count = cursor.fetchone()['cnt']
                    largest_tables.append({"name": t, "rows": row_count, "size_estimate": row_count * 100})  # 100 bytes rough estimate per row for sorting
                except Exception:
                    pass
            
            conn.close()
            
            largest_tables.sort(key=lambda x: x['rows'], reverse=True)
            top_tables = largest_tables[:5]
            
            status = 'ok'
            if size_mb > 1000:  # > 1 GB
                status = 'warning'
            if size_mb > 5000:  # > 5 GB
                status = 'critical'
                
            self._log_health('db_size', size_mb, 'MB', status)
                
            return {
                "size_mb": round(size_mb, 2),
                "table_count": table_count,
                "largest_tables": top_tables,
                "status": status
            }
        except Exception as e:
            logger.error(f"DB size check failed: {e}")
            return {"status": "error", "error": str(e)}

    def cleanup_old_data(self, days: int = 180) -> Dict[str, Any]:
        """Purge stale analysis and shrink scheduler logs."""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            
            conn = db._get_conn()
            cursor = conn.cursor()
            
            # Delete old analysis history
            cursor.execute("DELETE FROM analysis_history WHERE timestamp < ?", (cutoff,))
            deleted_analyses = cursor.rowcount
            
            # Keep only last 1000 scheduler_log entries
            cursor.execute("SELECT id FROM scheduler_log ORDER BY id DESC LIMIT 1 OFFSET 1000")
            row = cursor.fetchone()
            deleted_logs = 0
            if row:
                tenthousandth_id = row['id']
                cursor.execute("DELETE FROM scheduler_log WHERE id <= ?", (tenthousandth_id,))
                deleted_logs = cursor.rowcount
                
            # Delete data_freshness entries for tickers no longer in watchlist
            cursor.execute("""
                DELETE FROM data_freshness 
                WHERE ticker NOT IN (SELECT ticker FROM watchlist)
            """)
            deleted_freshness = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleanup: Removed {deleted_analyses} analyses, {deleted_logs} logs, {deleted_freshness} freshness rows.")
            
            return {
                "deleted_analyses": deleted_analyses,
                "deleted_logs": deleted_logs,
                "deleted_freshness": deleted_freshness
            }
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {}

    def vacuum_database(self) -> Dict[str, Any]:
        """Run VACUUM on SQLite database if overdue."""
        try:
            last_vacuum_str = db.get_setting("last_db_vacuum")
            should_run = False
            if last_vacuum_str:
                last_v = datetime.fromisoformat(last_vacuum_str)
                if (datetime.now() - last_v).days >= 7:
                    should_run = True
            else:
                should_run = True
                
            if should_run:
                # Use a specific connection without transactions since VACUUM requires it
                conn = db._get_conn()
                old_size = os.path.getsize(str(DB_PATH)) / (1024**2)
                
                # Turn off autocommit by setting isolation_level to None and don't use transaction block
                conn.isolation_level = None
                conn.execute("VACUUM")
                conn.close()
                
                new_size = os.path.getsize(str(DB_PATH)) / (1024**2)
                db.set_setting("last_db_vacuum", datetime.now().isoformat())
                logger.info(f"DB Vacuumced: {round(old_size, 2)} MB -> {round(new_size, 2)} MB")
                return {"vacuumed": True, "old_size_mb": old_size, "new_size_mb": new_size}
                
            return {"vacuumed": False, "reason": "Not overdue"}
        except Exception as e:
            logger.error(f"VACUUM failed: {e}")
            return {"vacuumed": False, "error": str(e)}

    def get_error_rate(self, hours: int = 24) -> Dict[str, Any]:
        """Count error entries in scheduler_logs in the last N hours."""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            
            row = db.query_one("""
                SELECT COUNT(*) as total_runs,
                       SUM(CASE WHEN errors IS NOT NULL AND errors != '' THEN 1 ELSE 0 END) as errors
                FROM scheduler_log
                WHERE run_at >= ?
            """, (cutoff,))
            
            total_runs = row['total_runs'] if row and row['total_runs'] else 0
            errors = row['errors'] if row and row['errors'] else 0
            
            percent = (errors / total_runs * 100) if total_runs > 0 else 0
            
            status = 'ok'
            if percent >= 10:
                status = 'warning'
            if percent >= 25:
                status = 'critical'
                
            self._log_health('error_rate', percent, '%', status)
                
            return {
                "total_runs": total_runs,
                "error_count": errors,
                "error_rate_pct": round(percent, 1),
                "status": status
            }
        except Exception as e:
            logger.error(f"Error rate check failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_uptime(self) -> float:
        """Return process uptime in hours."""
        uptime_seconds = time.time() - self.start_time
        return round(uptime_seconds / 3600, 2)

    def get_full_health_report(self) -> Dict[str, Any]:
        """Aggregate all checks into one dictionary."""
        disk = self.check_disk_usage()
        memory = self.check_memory_usage()
        database = self.check_db_size()
        errors = self.get_error_rate(hours=24)
        uptime = self.get_uptime()
        
        # Determine overall status
        statuses = [
            disk.get('status', 'ok'), 
            memory.get('status', 'ok'), 
            database.get('status', 'ok'), 
            errors.get('status', 'ok')
        ]
        
        overall = 'ok'
        if 'critical' in statuses:
            overall = 'critical'
        elif 'warning' in statuses or 'error' in statuses:
            overall = 'warning'
            
        return {
            "overall_status": overall,
            "uptime_hours": uptime,
            "disk": disk,
            "memory": memory,
            "database": database,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }


# Singleton
health_monitor = HealthMonitor()
