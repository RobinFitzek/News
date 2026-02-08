"""
APScheduler-based Scheduler for Investment Monitor
Runs automated scans based on configuration.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, time
import pytz
from core.database import db
from core.notifications import notifications

# Will be imported after scheduler is defined
agents = None

class InvestmentScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        self.is_scanning = False  # Track scanning state
        self._load_settings()
    
    def _load_settings(self):
        """Load scheduler settings from database"""
        self.interval_hours = db.get_setting("scan_interval_hours")
        self.active_start = db.get_setting("active_hours_start")
        self.active_end = db.get_setting("active_hours_end")
        self.timezone = db.get_setting("timezone")
        self.daily_summary_enabled = db.get_setting("daily_summary_enabled")
        self.daily_summary_time = db.get_setting("daily_summary_time")
    
    def reload_settings(self):
        """Reload settings and reschedule jobs"""
        self._load_settings()
        if self.is_running:
            self.stop()
            self.start()
    
    def _is_active_time(self) -> bool:
        """Check if current time is within active hours"""
        try:
            tz = pytz.timezone(self.timezone)
            now = datetime.now(tz).time()
            start = time.fromisoformat(self.active_start)
            end = time.fromisoformat(self.active_end)
            return start <= now <= end
        except Exception:
            return True  # If error, assume always active
    
    def run_scan(self, force=False):
        """Run the Daily Analysis Pipeline"""
        if self.is_scanning:
            print("(!) Scan already in progress")
            return

        if not force and not self._is_active_time():
            print(f"(-) Outside active hours ({self.active_start}-{self.active_end}), skipping scan")
            return

        print(f"\n{'='*50}")
        print(f"      SCHEDULED PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*50}")

        self.is_scanning = True
        try:
            from engine.pipeline import pipeline
            results = pipeline.run_daily_cycle()
            return results
        except ImportError as e:
            error_msg = f"Import Error: {e}. Check if all dependencies are installed."
            print(f"(Error) {error_msg}")
            db.log_scheduler_run(
                tickers_scanned=0,
                alerts_sent=0,
                errors=error_msg,
                duration=0
            )
            # Don't raise in background thread, just log
            # raise Exception(error_msg)
        except AttributeError as e:
            error_msg = f"API Method Error: {e}. Check API client configuration."
            print(f"(Error) {error_msg}")
            db.log_scheduler_run(
                tickers_scanned=0,
                alerts_sent=0,
                errors=error_msg,
                duration=0
            )
        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            print(f"(Error) {error_msg}")
            # Log error
            db.log_scheduler_run(
                tickers_scanned=0,
                alerts_sent=0,
                errors=error_msg,
                duration=0
            )
        finally:
            self.is_scanning = False
    
    def run_daily_summary(self):
        """Send daily summary email"""
        if not self.daily_summary_enabled:
            return
        
        # Get today's analyses
        today_analyses = []
        for analysis in db.get_analysis_history(limit=100):
            if analysis['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d')):
                today_analyses.append(analysis)
        
        if today_analyses:
            notifications.send_daily_summary(today_analyses)
            print("Daily summary sent")
    
    def start(self):
        """Start the scheduler with all cycle jobs"""
        if self.is_running:
            return
        
        # Import cycle processor
        from engine.cycle_processor import cycle_processor
        
        # Daily Quick-Scan (every N hours during active time)
        self.scheduler.add_job(
            self.run_scan,
            IntervalTrigger(hours=self.interval_hours),
            id='main_scan',
            name='Daily Investment Scan',
            replace_existing=True
        )
        
        # Weekly Deep Analysis (Sunday 20:00)
        self.scheduler.add_job(
            lambda: cycle_processor.run_weekly_cycle(),
            CronTrigger(day_of_week='sun', hour=20, minute=0,
                       timezone=self.timezone),
            id='weekly_analysis',
            name='Weekly Deep Analysis',
            replace_existing=True
        )
        
        # Monthly Portfolio Review (28th of each month at 18:00)
        self.scheduler.add_job(
            lambda: cycle_processor.run_monthly_cycle(),
            CronTrigger(day=28, hour=18, minute=0,
                       timezone=self.timezone),
            id='monthly_review',
            name='Monthly Portfolio Review',
            replace_existing=True
        )
        
        # Daily summary job
        if self.daily_summary_enabled:
            summary_hour, summary_minute = map(int, self.daily_summary_time.split(':'))
            self.scheduler.add_job(
                self.run_daily_summary,
                CronTrigger(hour=summary_hour, minute=summary_minute, 
                           timezone=self.timezone),
                id='daily_summary',
                name='Daily Summary Email',
                replace_existing=True
            )
        
        self.scheduler.start()
        self.is_running = True
        print(f"Scheduler started: Daily every {self.interval_hours}h, Weekly Sun 20:00, Monthly 28th 18:00")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        self.scheduler.shutdown()
        self.is_running = False
        print("Scheduler stopped")
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        jobs = []
        if self.is_running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(job.next_run_time) if job.next_run_time else None
                })
        
        return {
            "is_running": self.is_running,
            "is_scanning": self.is_scanning,
            "interval_hours": self.interval_hours,
            "active_hours": f"{self.active_start} - {self.active_end}",
            "timezone": self.timezone,
            "jobs": jobs,
            "last_runs": db.get_scheduler_logs(limit=5)
        }
    
    def trigger_manual_scan(self):
        """Trigger an immediate scan in background"""
        if self.is_scanning:
            print("(!) Scan already in progress")
            return False
            
        print("Manual scan triggered in background")
        self.scheduler.add_job(
            self.run_scan,
            args=[True], # force=True
            trigger='date',
            run_date=datetime.now(),
            id='manual_scan_immediate',
            name='Manual Scan (User Triggered)',
            replace_existing=True
        )
        return True

# Singleton
scheduler = InvestmentScheduler()
