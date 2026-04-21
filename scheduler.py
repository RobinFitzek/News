"""
APScheduler-based Scheduler for Investment Monitor
Runs automated scans based on configuration.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import (
    EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED,
    JobExecutionEvent
)
from contextlib import contextmanager
from datetime import datetime, date, time
import logging
import pytz
from core.database import db
from core.notifications import notifications

logger = logging.getLogger(__name__)


@contextmanager
def _job_lock(job_id: str, ttl_minutes: int = 120):
    """Context manager that skips the wrapped block if job ran recently."""
    acquired = db.try_acquire_job_lock(job_id, ttl_minutes)
    if not acquired:
        logger.warning("Job '%s' skipped — lock held from recent run (TTL=%dm)", job_id, ttl_minutes)
    try:
        yield acquired
    finally:
        if acquired:
            db.release_job_lock(job_id)


def _job_event_listener(event: JobExecutionEvent) -> None:
    """Log job execution outcomes to the scheduler_log table."""
    try:
        job_id = event.job_id
        if event.exception:
            logger.error("Job '%s' raised an exception: %s", job_id, event.exception)
            db.execute(
                "INSERT OR IGNORE INTO scheduler_log (tickers_scanned, alerts_sent, errors, duration_seconds) VALUES (0, 0, ?, 0)",
                (f"[{job_id}] {event.exception}",)
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning("Job '%s' missed its execution window", job_id)
        else:
            logger.debug("Job '%s' completed successfully", job_id)
    except Exception:
        pass  # listener must never raise

# NYSE market holidays for 2026 (format: (month, day))
US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth National Independence Day
    date(2026, 7, 3),   # Independence Day (observed, July 4 falls on Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving Day
    date(2026, 12, 25), # Christmas Day
}

US_MARKET_HOLIDAYS = US_MARKET_HOLIDAYS_2026

# Will be imported after scheduler is defined
agents = None

class InvestmentScheduler:
    # Jobs that are paused during deep sleep for CPU C-state efficiency.
    # The 5/15-min interval jobs are already converted to market-hours CronTriggers
    # so they never fire overnight at all.  The jobs below are the remaining ones
    # that still use IntervalTrigger and would otherwise prevent C6/C7/C8.
    _SLEEP_SENSITIVE_JOBS = ['main_scan', 'geopolitical_scan']

    def __init__(self):
        from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor
        self.scheduler = BackgroundScheduler(
            executors={'default': APSThreadPoolExecutor(max_workers=4)},
            job_defaults={
                'coalesce': True,          # merge missed firings into one catch-up run
                'misfire_grace_time': 600, # 10 min grace after resume from system suspend
                'max_instances': 1,        # prevent overlapping runs of the same job
            }
        )
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
        # Discovery settings
        self.discovery_enabled = db.get_setting("discovery_enabled")
        self.discovery_daily_time = db.get_setting("discovery_daily_time") or "06:00"
        self.discovery_weekly_day = db.get_setting("discovery_weekly_day") or "wed"
        self.discovery_weekly_time = db.get_setting("discovery_weekly_time") or "12:00"
        # Holiday skip
        self.holiday_skip_enabled = db.get_setting("holiday_skip_enabled") if db.get_setting("holiday_skip_enabled") is not None else True
    
    def reload_settings(self):
        """Reload settings and reschedule jobs"""
        self._load_settings()
        if self.is_running:
            self.stop()
            self.start()
    
    def _is_market_holiday(self) -> bool:
        """Check if today is a US market holiday."""
        if not self.holiday_skip_enabled:
            return False
        try:
            tz = pytz.timezone(self.timezone)
            today = datetime.now(tz).date()
            return today in US_MARKET_HOLIDAYS
        except Exception:
            return False

    def _is_active_time(self) -> bool:
        """Check if current time is within active hours and on a weekday"""
        try:
            tz = pytz.timezone(self.timezone)
            now = datetime.now(tz)
            # Skip weekends
            if now.weekday() >= 5:
                return False
            current_time = now.time()
            start = time.fromisoformat(self.active_start)
            end = time.fromisoformat(self.active_end)
            return start <= current_time <= end
        except Exception:
            return True  # If error, assume always active

    def is_deep_sleep_active(self) -> bool:
        """Check if Deep Sleep mode is currently active (BREATHE-5b)"""
        enabled = db.get_setting("deep_sleep_enabled")
        if not enabled or str(enabled).lower() == "false":
            return False
            
        try:
            tz = pytz.timezone(self.timezone)
            now = datetime.now(tz)
            
            # Check weekend full day sleep
            if now.weekday() >= 5 and db.get_setting("deep_sleep_full_weekends"):
                return True
                
            sleep_start_str = db.get_setting("deep_sleep_start") or "22:00"
            sleep_end_str = db.get_setting("deep_sleep_end") or "07:00"
            
            current_time = now.time()
            start = time.fromisoformat(sleep_start_str)
            end = time.fromisoformat(sleep_end_str)
            
            if start > end: # crosses midnight
                return current_time >= start or current_time <= end
            else:
                return start <= current_time <= end
        except Exception:
            return False

    def is_market_open(self) -> bool:
        """Check if US markets are currently open (Mon-Fri 9:30-16:00 ET)"""
        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            if now.weekday() >= 5:
                return False
            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= now.time() <= market_close
        except Exception:
            return True
    
    def run_scan(self, force=False):
        """Run the Daily Analysis Pipeline"""
        if self.is_scanning:
            logger.warning("Scan already in progress, skipping")
            return

        if not force and self.is_deep_sleep_active():
            intensity = db.get_setting("deep_sleep_intensity") or "deep"
            if intensity == "hibernate":
                logger.info("Deep Sleep (Hibernate) active, skipping scan")
                return
            logger.info("Deep Sleep active, reducing scan frequency (skipped this run)")
            return

        if not force and not self._is_active_time():
            logger.info("Outside active hours (%s-%s), skipping scan", self.active_start, self.active_end)
            return

        if not force and self._is_market_holiday():
            logger.info("Market closed today (US holiday) — skipping scan")
            return

        logger.info("SCHEDULED PIPELINE - %s", datetime.now().strftime('%Y-%m-%d %H:%M'))

        self.is_scanning = True
        try:
            from engine.pipeline import pipeline
            results = pipeline.run_daily_cycle()
            return results
        except ImportError as e:
            error_msg = f"Import Error: {e}. Check if all dependencies are installed."
            logger.error(error_msg)
            db.log_scheduler_run(tickers_scanned=0, alerts_sent=0, errors=error_msg, duration=0)
        except AttributeError as e:
            error_msg = f"API Method Error: {e}. Check API client configuration."
            logger.error(error_msg, exc_info=True)
            db.log_scheduler_run(tickers_scanned=0, alerts_sent=0, errors=error_msg, duration=0)
        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            db.log_scheduler_run(tickers_scanned=0, alerts_sent=0, errors=error_msg, duration=0)
        finally:
            self.is_scanning = False
    
    def run_macro_event_check(self):
        """Check for upcoming rate events and alert on rate-sensitive portfolio holdings."""
        try:
            from engine.macro_tracker import macro_tracker
            events = macro_tracker.get_upcoming_events(days_ahead=2)
            for event in events:
                if event['days_until'] <= 1:
                    exposed = macro_tracker.check_portfolio_rate_exposure()
                    if exposed:
                        tickers = [h['ticker'] for h in exposed]
                        msg = (
                            f"Rate Event Tomorrow: {event['type']} decision on {event['date']}. "
                            f"Rate-sensitive holdings: {', '.join(tickers)}"
                        )
                        try:
                            from engine.webhook_notifier import webhook_notifier
                            webhook_notifier.reload()
                            webhook_notifier.send_custom(title="Macro Rate Alert", message=msg, level="warning")
                        except Exception:
                            pass
                        logger.info("[MACRO] %s", msg)
        except Exception as e:
            logger.error("Macro event check failed: %s", e, exc_info=True)

    def run_geopolitical_scan(self):
        """Run global geopolitical scan and alert on high-severity events"""
        import re
        from clients.perplexity_client import pplx_client
        logger.info("GEO SCAN - %s", datetime.now().strftime('%Y-%m-%d %H:%M'))

        raw_summary = pplx_client.get_geopolitical_scan()
        if not raw_summary:
            logger.warning("Geopolitical scan returned no data")
            return

        scan_id = db.save_geopolitical_scan(raw_summary)
        logger.info("Geopolitical scan saved (id=%s)", scan_id)

        # Alert + priority re-analysis on high-severity events (severity >= 8)
        scores = [int(m) for m in re.findall(r'SCHWEREGRAD[:\s/]+(\d+)', raw_summary)]
        max_severity = max(scores) if scores else 0
        if max_severity >= 8:
            notifications.send_geopolitical_alert(raw_summary, max_severity)
            logger.info("High-severity alert sent (max severity: %d)", max_severity)
            self._trigger_priority_reanalysis()

    def _trigger_priority_reanalysis(self):
        """Trigger a full watchlist re-analysis after a high-severity geo event.
        Cooldown: skips if the last scan finished less than 2 hours ago."""
        try:
            logs = db.get_scheduler_logs(limit=1)
            if logs:
                last_run_str = logs[0].get('run_at', '')
                if last_run_str:
                    from datetime import timezone
                    last_run = datetime.fromisoformat(last_run_str)
                    # Make naive datetime timezone-aware for comparison
                    if last_run.tzinfo is None:
                        last_run = last_run.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    elapsed_hours = (now_utc - last_run).total_seconds() / 3600
                    if elapsed_hours < 2:
                        logger.info("Priority re-analysis skipped — last scan was %.1fh ago (cooldown: 2h)", elapsed_hours)
                        return

            logger.info("Triggering priority watchlist re-analysis due to high-severity geo event")
            import threading
            t = threading.Thread(target=self.run_scan, kwargs={'force': True}, daemon=True)
            t.start()
        except Exception as e:
            logger.error("Priority re-analysis trigger failed: %s", e, exc_info=True)

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
            logger.info("Daily summary sent")

            # Fire per-signal alerts for strong signals (enables plugin notifiers)
            for analysis in today_analyses:
                signal = analysis.get('signal') or ''
                if signal in ('STRONG_BUY', 'STRONG_SELL'):
                    try:
                        notifications.send_alert(
                            ticker=analysis.get('ticker', ''),
                            signal=signal,
                            recommendation=analysis.get('recommendation', ''),
                            confidence=analysis.get('confidence') or 0,
                            risk_score=analysis.get('risk_score') or 5,
                        )
                    except Exception as e:
                        logger.error("send_alert error for %s: %s", analysis.get('ticker'), e)
    
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
        
        # Daily Auto-Discovery (free, runs before first scan)
        if self.discovery_enabled:
            try:
                disc_hour, disc_minute = map(int, self.discovery_daily_time.split(':'))
                self.scheduler.add_job(
                    self.run_discovery,
                    CronTrigger(hour=disc_hour, minute=disc_minute,
                               timezone=self.timezone),
                    id='daily_discovery',
                    name='Daily Auto-Discovery',
                    replace_existing=True
                )

                # Weekly AI Discovery
                weekly_hour, weekly_minute = map(int, self.discovery_weekly_time.split(':'))
                self.scheduler.add_job(
                    self.run_ai_discovery,
                    CronTrigger(day_of_week=self.discovery_weekly_day,
                               hour=weekly_hour, minute=weekly_minute,
                               timezone=self.timezone),
                    id='weekly_ai_discovery',
                    name='Weekly AI Discovery',
                    replace_existing=True
                )
            except Exception as e:
                logger.error("Error scheduling discovery jobs: %s", e, exc_info=True)

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

        # Weekly report (Sunday evening at 18:00)
        self.scheduler.add_job(
            self.run_weekly_report,
            CronTrigger(day_of_week='sun', hour=18, minute=0,
                       timezone=self.timezone),
            id='weekly_report',
            name='Weekly Report',
            replace_existing=True
        )

        # Weekly AI letter (Sunday 19:00)
        self.scheduler.add_job(
            self.run_weekly_letter,
            CronTrigger(day_of_week='sun', hour=19, minute=0,
                       timezone=self.timezone),
            id='weekly_letter',
            name='Weekly AI Letter',
            replace_existing=True
        )

        # NLP sentiment scoring — every hour during market hours (#38/#57)
        self.scheduler.add_job(
            self.run_nlp_sentiment_scoring,
            CronTrigger(hour='9-17', minute=5, timezone=self.timezone),
            id='nlp_sentiment',
            name='NLP Sentiment Scoring',
            replace_existing=True
        )

        # Weekly pairs-trading cointegration scan (#40) — Saturday 08:00
        self.scheduler.add_job(
            self.run_pairs_scan,
            CronTrigger(day_of_week='sat', hour=8, minute=0, timezone=self.timezone),
            id='pairs_scan',
            name='Pairs Trading Cointegration Scan',
            replace_existing=True
        )

        # Quarterly supply chain refresh (#44/#58) — 1st of Jan/Apr/Jul/Oct at 07:00
        self.scheduler.add_job(
            self.run_supply_chain_refresh,
            CronTrigger(month='1,4,7,10', day=1, hour=7, minute=0, timezone=self.timezone),
            id='supply_chain_refresh',
            name='Supply Chain Quarterly Refresh',
            replace_existing=True
        )

        # Daily dark pool / volume anomaly scan (#52) — weekdays at 18:00 (after market close)
        self.scheduler.add_job(
            self.run_dark_pool_scan,
            CronTrigger(day_of_week='mon-fri', hour=18, minute=0, timezone=self.timezone),
            id='dark_pool_scan',
            name='Dark Pool Volume Anomaly Scan',
            replace_existing=True
        )

        # Weekly 13F institutional holdings refresh (#25) — Friday 18:00
        # EDGAR 13F filings are quarterly; refreshing weekly ensures we catch new filings
        # shortly after the Feb/May/Aug/Nov 15 EDGAR deadline.
        self.scheduler.add_job(
            self.refresh_13f_holdings,
            CronTrigger(day_of_week='fri', hour=18, minute=0, timezone=self.timezone),
            id='refresh_13f',
            name='13F Institutional Holdings Refresh',
            replace_existing=True
        )

        # Daily macro snapshot — weekdays at 17:00 (after US market close)
        self.scheduler.add_job(
            self.capture_macro_snapshot,
            CronTrigger(day_of_week='mon-fri', hour=17, minute=5, timezone=self.timezone),
            id='macro_snapshot',
            name='Daily Macro Snapshot',
            replace_existing=True
        )

        # Discovery hit rate check (daily at 21:00)
        self.scheduler.add_job(
            self.check_hit_rates,
            CronTrigger(hour=21, minute=0, timezone=self.timezone),
            id='hit_rate_check',
            name='Hit Rate Check',
            replace_existing=True
        )

        # Daily health check (run at 03:00)
        self.scheduler.add_job(
            self.run_health_check,
            CronTrigger(hour=3, minute=0, timezone=self.timezone),
            id='health_check',
            name='System Health Check',
            replace_existing=True
        )

        # Signal Grader (daily at 22:00)
        self.scheduler.add_job(
            self.grade_signals,
            CronTrigger(hour=22, minute=0, timezone=self.timezone),
            id='grade_signals',
            name='Grade Signals',
            replace_existing=True
        )

        # Auto Paper Trading Entry (Mon-Fri 09:35 AM NY time)
        self.scheduler.add_job(
            self.run_auto_paper_entry,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=35, timezone=self.timezone),
            id='auto_paper_entry',
            name='Auto Paper Entry',
            replace_existing=True
        )

        # Auto Paper Trading Exit (Mon-Fri 15:50 PM NY time)
        self.scheduler.add_job(
            self.run_auto_paper_exit,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=50, timezone=self.timezone),
            id='auto_paper_exit',
            name='Auto Paper Exit',
            replace_existing=True
        )

        # Broker Position Sync — Mon–Fri 09:00–15:55, every 5 min (market hours only).
        # Handler already guards is_market_open(); CronTrigger ensures zero overnight wakeups
        # so the CPU can reach deep C-states between 16:05 and 09:00 the next morning.
        self.scheduler.add_job(
            self.run_broker_sync,
            CronTrigger(day_of_week='mon-fri', hour='9-15', minute='*/5',
                        timezone=self.timezone),
            id='broker_sync',
            name='Broker Position Sync',
            replace_existing=True
        )

        # Broker P&L Snapshot — Mon–Fri 09:00–15:45, every 15 min
        self.scheduler.add_job(
            self.run_broker_pnl_snapshot,
            CronTrigger(day_of_week='mon-fri', hour='9-15', minute='*/15',
                        timezone=self.timezone),
            id='broker_pnl_snapshot',
            name='Broker P&L Snapshot',
            replace_existing=True
        )

        # Price alert check — Mon–Fri 09:00–15:45, every 15 min
        self.scheduler.add_job(
            self.check_price_alerts,
            CronTrigger(day_of_week='mon-fri', hour='9-15', minute='*/15',
                        timezone=self.timezone),
            id='price_alert_check',
            name='Price Alert Check',
            replace_existing=True
        )

        # Meta-Labeler Retrain (Sunday 22:30, after signal grading at 22:00)
        self.scheduler.add_job(
            self.retrain_meta_labeler,
            CronTrigger(day_of_week='sun', hour=22, minute=30, timezone=self.timezone),
            id='meta_labeler_retrain',
            name='Meta-Labeler Retrain',
            replace_existing=True
        )

        # MCPT Strategy Validation (Sunday 23:00)
        self.scheduler.add_job(
            self.run_mcpt_validation,
            CronTrigger(day_of_week='sun', hour=23, minute=0, timezone=self.timezone),
            id='mcpt_validation',
            name='MCPT Strategy Validation',
            replace_existing=True
        )

        # Geopolitical scan — 08:00, 14:00, 20:00 daily (3× per day at fixed times).
        # Previously IntervalTrigger(hours=6) fired at unpredictable overnight hours (e.g., 04:00).
        # Fixed cron times keep the scan cadence equivalent while eliminating the 04:00 wakeup
        # that prevented deep C-states.  Deep sleep job suspension (below) pauses this during
        # the overnight window if deep_sleep_enabled is set.
        self.scheduler.add_job(
            self.run_geopolitical_scan,
            CronTrigger(hour='8,14,20', timezone=self.timezone),
            id='geopolitical_scan',
            name='Geopolitical Scan',
            replace_existing=True
        )

        # Macro event check (daily at 08:00 — alerts on rate decisions within 48h)
        self.scheduler.add_job(
            self.run_macro_event_check,
            CronTrigger(hour=8, minute=0, timezone=self.timezone),
            id='macro_event_check',
            name='Macro Event Check',
            replace_existing=True
        )

        # RSS geo trigger — 07:00–22:45, every 15 min.
        # Skipping 22:45–07:00 lets the CPU reach C8 overnight while still catching
        # breaking geopolitical news during all waking hours.
        self.scheduler.add_job(
            self.check_rss_geo_trigger,
            CronTrigger(hour='7-22', minute='*/15', timezone=self.timezone),
            id='rss_geo_trigger',
            name='RSS Geo Trigger',
            replace_existing=True
        )

        # Daily DB backup (03:30, after health check at 03:00)
        self.scheduler.add_job(
            self.run_db_backup,
            CronTrigger(hour=3, minute=30, timezone=self.timezone),
            id='db_backup',
            name='DB Backup',
            replace_existing=True
        )

        # Weekly corporate actions refresh (Saturday 04:00) (#43)
        self.scheduler.add_job(
            self.refresh_corporate_actions,
            CronTrigger(day_of_week='sat', hour=4, minute=0, timezone=self.timezone),
            id='corporate_actions_refresh',
            name='Corporate Actions Refresh',
            replace_existing=True
        )

        # Daily Graham screen (07:30 on weekdays)
        self.scheduler.add_job(
            self.run_graham_screen,
            CronTrigger(day_of_week='mon-fri', hour=7, minute=30,
                        timezone=self.timezone),
            id='graham_screen',
            name='Graham Value Screen',
            replace_existing=True
        )

        # Daily Fear & Greed + VIX snapshot (08:00)
        self.scheduler.add_job(
            self.run_fear_greed_snapshot,
            CronTrigger(day_of_week='mon-fri', hour=8, minute=0,
                        timezone=self.timezone),
            id='fear_greed_snapshot',
            name='Fear & Greed Snapshot',
            replace_existing=True
        )

        # Daily Senate trades refresh (06:00)
        self.scheduler.add_job(
            self.run_politician_refresh,
            CronTrigger(day_of_week='mon-fri', hour=6, minute=0,
                        timezone=self.timezone),
            id='politician_refresh',
            name='Senate Trade Data Refresh',
            replace_existing=True
        )

        # ── CPU C-state overnight mode ────────────────────────────────────────
        # Schedule explicit transitions at deep_sleep boundary times so APScheduler
        # pauses the remaining IntervalTrigger jobs (main_scan, geopolitical_scan)
        # during the overnight window — allowing the CPU to stay in C6/C7/C8.
        deep_sleep_on = db.get_setting('deep_sleep_enabled')
        if deep_sleep_on:
            try:
                sleep_start = db.get_setting('deep_sleep_start') or '22:00'
                sleep_end   = db.get_setting('deep_sleep_end')   or '07:00'
                sh, sm = map(int, sleep_start.split(':'))
                eh, em = map(int, sleep_end.split(':'))
                self.scheduler.add_job(
                    self._enter_deep_sleep_mode,
                    CronTrigger(hour=sh, minute=sm, timezone=self.timezone),
                    id='deep_sleep_enter',
                    name='Enter Deep Sleep (C-state)',
                    replace_existing=True
                )
                self.scheduler.add_job(
                    self._exit_deep_sleep_mode,
                    CronTrigger(hour=eh, minute=em, timezone=self.timezone),
                    id='deep_sleep_exit',
                    name='Exit Deep Sleep (C-state)',
                    replace_existing=True
                )
            except Exception as e:
                logger.error("Could not schedule deep sleep transitions: %s", e, exc_info=True)

        self.scheduler.add_listener(
            _job_event_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        self.scheduler.start()
        self.is_running = True

        # If the scheduler starts while deep sleep is already active (e.g., after reboot),
        # immediately pause the sleep-sensitive jobs.
        if deep_sleep_on and self.is_deep_sleep_active():
            self._enter_deep_sleep_mode()

        logger.info("Scheduler started: Daily every %sh, Weekly Sun 20:00, Monthly 28th 18:00", self.interval_hours)

        # Start two-way Telegram bot (if enabled)
        try:
            from clients.telegram_bot import telegram_bot
            telegram_bot.start()
        except Exception as e:
            logger.warning("Telegram bot failed to start: %s", e)

    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        self.scheduler.shutdown(wait=False)
        # Create a fresh scheduler instance — the old executor's thread pool
        # is permanently destroyed after shutdown() and cannot accept new jobs.
        self.scheduler = BackgroundScheduler(
            job_defaults={'coalesce': True, 'misfire_grace_time': 600}
        )
        self.is_running = False
        try:
            from clients.telegram_bot import telegram_bot
            telegram_bot.stop()
        except Exception:
            pass
        logger.info("Scheduler stopped")
    
    @staticmethod
    def _relative_time(dt) -> str:
        """Return human-readable countdown like '2h 15m' or '45m'"""
        from datetime import datetime, timezone
        if dt is None:
            return None
        now = datetime.now(timezone.utc)
        diff = (dt.astimezone(timezone.utc) - now).total_seconds()
        if diff < 0:
            return 'now'
        days = int(diff // 86400)
        hours = int((diff % 86400) // 3600)
        minutes = int((diff % 3600) // 60)
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def get_status(self) -> dict:
        """Get scheduler status"""
        from datetime import datetime, timezone
        jobs = []
        if self.is_running:
            for job in self.scheduler.get_jobs():
                nrt = job.next_run_time
                # compute seconds until next run for sorting and 'soon' flag
                if nrt is not None:
                    now = datetime.now(timezone.utc)
                    secs = (nrt.astimezone(timezone.utc) - now).total_seconds()
                else:
                    secs = float('inf')
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(nrt) if nrt else None,
                    "next_run_relative": self._relative_time(nrt),
                    "next_run_formatted": nrt.strftime("%d.%m  %H:%M") if nrt else None,
                    "soon": secs < 3600,
                    "_secs": secs,
                })
        # sort soonest first
        jobs.sort(key=lambda j: j["_secs"])
        for j in jobs:
            del j["_secs"]

        return {
            "is_running": self.is_running,
            "is_scanning": self.is_scanning,
            "is_sleeping": self.is_deep_sleep_active(),
            "is_market_open": self.is_market_open(),
            "interval_hours": self.interval_hours,
            "active_hours": f"{self.active_start} - {self.active_end}",
            "timezone": self.timezone,
            "jobs": jobs,
            "last_runs": db.get_scheduler_logs(limit=5)
        }
    
    def run_discovery(self):
        """Run daily auto-discovery (free strategies)"""
        with _job_lock('run_discovery', ttl_minutes=60) as locked:
            if not locked:
                return
            try:
                from engine.auto_discovery import auto_discovery
                result = auto_discovery.run_daily_discovery()
                logger.info("Discovery completed: %d found, %d promoted",
                            result.get('discoveries', 0), len(result.get('promoted', [])))
            except Exception as e:
                logger.error("Discovery failed: %s", e, exc_info=True)

    def run_ai_discovery(self):
        """Run weekly AI discovery (Perplexity)"""
        with _job_lock('run_ai_discovery', ttl_minutes=360) as locked:
            if not locked:
                return
            try:
                from engine.auto_discovery import auto_discovery
                result = auto_discovery.run_weekly_ai_discovery()
                logger.info("AI Discovery completed: %d found", result.get('discoveries', 0))
            except Exception as e:
                logger.error("AI Discovery failed: %s", e, exc_info=True)

    def run_weekly_report(self):
        """Generate and send weekly portfolio report."""
        with _job_lock('run_weekly_report', ttl_minutes=360) as locked:
            if not locked:
                return
            try:
                from engine.report_generator import report_generator
                html = report_generator.generate_weekly_report()
                if html:
                    report_generator.send_report(html)
                    logger.info("Weekly report generated and sent")
            except Exception as e:
                logger.error("Weekly report failed: %s", e, exc_info=True)

    def run_weekly_letter(self):
        """Generate and send AI weekly investment letter."""
        with _job_lock('run_weekly_letter', ttl_minutes=360) as locked:
            if not locked:
                return
            try:
                enabled = db.get_setting('weekly_letter_enabled')
                if not enabled:
                    return
                from engine.weekly_letter import weekly_letter_generator
                weekly_letter_generator.generate_and_send()
            except Exception as e:
                logger.error("Weekly letter failed: %s", e, exc_info=True)

    def run_nlp_sentiment_scoring(self):
        """Score watchlist ticker sentiment from RSS headlines (#38/#57)."""
        with _job_lock('run_nlp_sentiment', ttl_minutes=45) as locked:
            if not locked:
                return
            try:
                from engine.nlp_scorer import run_hourly_scoring, ensure_schema
                ensure_schema()
                results = run_hourly_scoring()
                if results:
                    logger.info("NLP sentiment: scored %d tickers", len(results))
            except Exception as e:
                logger.error("NLP sentiment scoring failed: %s", e, exc_info=True)

    def run_pairs_scan(self):
        """Weekly cointegration scan for pairs trading (#40)."""
        try:
            from engine.pairs_trader import run_weekly_scan
            pairs = run_weekly_scan()
            logger.info("Pairs trading scan: %d cointegrated pairs found", len(pairs))
        except Exception as e:
            logger.error("Pairs trading scan failed: %s", e, exc_info=True)

    def run_supply_chain_refresh(self):
        """Quarterly refresh of supply chain data for stale watchlist tickers (#44/#58)."""
        try:
            from engine.supply_chain import refresh_stale_tickers
            refreshed = refresh_stale_tickers()
            logger.info("Supply chain refresh: %d tickers updated", refreshed)
        except Exception as e:
            logger.error("Supply chain refresh failed: %s", e, exc_info=True)

    def run_dark_pool_scan(self):
        """Daily dark pool / volume anomaly scan for watchlist tickers (#52)."""
        try:
            from engine.dark_pool_tracker import scan_watchlist
            count = scan_watchlist()
            logger.info("Dark pool scan: %d signals detected", count)
        except Exception as e:
            logger.error("Dark pool scan failed: %s", e, exc_info=True)

    def check_hit_rates(self):
        """Check discovery hit rates and log outcomes. Also flags stale data."""
        try:
            from engine.discovery_hit_rate import discovery_hit_rate
            result = discovery_hit_rate.check_outcomes()
            checked = result.get('checked', 0) if result else 0
            logger.info("Hit rate check: %d outcomes evaluated", checked)
        except Exception as e:
            logger.error("Hit rate check failed: %s", e, exc_info=True)

        # Flag stale data using DataFreshnessTracker
        try:
            from engine.data_freshness import data_freshness
            stale_tickers = data_freshness.get_stale_tickers()
            if stale_tickers:
                tickers_str = ', '.join(t['ticker'] for t in stale_tickers)
                msg = f"Stale data ({len(stale_tickers)} tickers): {tickers_str}"
                logger.warning(msg)
                try:
                    from engine.webhook_notifier import webhook_notifier
                    webhook_notifier.send_custom(
                        title="Data Staleness Alert",
                        message=msg,
                        level="warning"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error("Staleness check failed: %s", e, exc_info=True)

    def check_price_alerts(self):
        """Check all active price alerts and fire notifications when triggered."""
        try:
            tz = pytz.timezone(self.timezone)
            now = datetime.now(tz)
            # Only run Monday–Friday, 9:30–16:05 ET
            if now.weekday() >= 5:
                return
            if self._is_market_holiday():
                return
            market_open = time(9, 30)
            market_close = time(16, 5)
            if not (market_open <= now.time() <= market_close):
                return

            alerts = db.query(
                "SELECT * FROM price_alerts WHERE active = 1 ORDER BY created_at DESC"
            ) or []
            if not alerts:
                return

            import yfinance as yf

            # Group by ticker to avoid duplicate fetches
            tickers_needed = list({a['ticker'] for a in alerts})
            prices = {}
            for ticker in tickers_needed:
                try:
                    info = yf.Ticker(ticker).fast_info
                    price = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
                    if price:
                        prices[ticker] = float(price)
                except Exception:
                    pass

            triggered = 0
            for alert in alerts:
                ticker = alert['ticker']
                current_price = prices.get(ticker)
                if current_price is None:
                    continue

                threshold = float(alert['threshold'])
                direction = alert.get('direction', 'above')
                hit = (direction == 'above' and current_price >= threshold) or \
                      (direction == 'below' and current_price <= threshold)

                if hit:
                    msg = (
                        f"🔔 *Price Alert: {ticker}*\n"
                        f"Current: ${current_price:.2f} — "
                        f"triggered {direction} ${threshold:.2f}"
                    )
                    try:
                        from engine.webhook_notifier import webhook_notifier
                        webhook_notifier.reload()
                        webhook_notifier.send_custom(msg)
                    except Exception:
                        pass
                    db.execute(
                        "UPDATE price_alerts SET active = 0, triggered_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), alert['id'])
                    )
                    triggered += 1
                    logger.info("Price alert triggered: %s", msg)

            if triggered:
                logger.info("Price alerts: %d triggered out of %d active", triggered, len(alerts))

            # Also scan watchlist for intraday breakouts → auto-analysis
            trigger_pct = float(db.get_setting('intraday_trigger_pct') or 3.0)
            self._check_intraday_breakouts(threshold_pct=trigger_pct)

            # Check open auto-paper-trade positions for TP/SL/time exits on every 15-min tick
            try:
                from engine.auto_paper_trader import auto_paper_trader
                auto_paper_trader.check_open_positions()
            except Exception as e:
                logger.warning("Auto-paper-trader position check failed: %s", e)

            # Portfolio-level anomaly detection (#46/#55) — run on every 15-min tick
            try:
                from engine.portfolio_anomaly import run_anomaly_checks
                run_anomaly_checks()
            except Exception as e:
                logger.warning("Portfolio anomaly check failed: %s", e)
        except Exception as e:
            logger.error("Price alert check failed: %s", e, exc_info=True)

    def refresh_13f_holdings(self):
        """Weekly job: refresh top-20 institutional 13F holdings (#25)."""
        try:
            from engine.institutional_tracker import institutional_tracker
            results = institutional_tracker.refresh_top_filer_holdings()
            total = sum(results.values())
            logger.info("13F refresh complete: %d filers, %d holdings stored", len(results), total)
        except Exception as e:
            logger.error("13F holdings refresh failed: %s", e, exc_info=True)

    def capture_macro_snapshot(self):
        """Daily job: fetch macro market data and store snapshot (#22)."""
        try:
            from engine.macro_tracker import macro_tracker
            snap = macro_tracker.fetch_and_store_snapshot()
            if snap:
                logger.info("Macro snapshot stored: regime=%s, vix=%s, spread=%s",
                            snap.get("regime_label"), snap.get("vix"), snap.get("spread_2y10y"))
        except Exception as e:
            logger.error("Macro snapshot failed: %s", e, exc_info=True)

        # Evaluate composite cross-asset signals after each snapshot (#47)
        try:
            from engine.composite_signals import evaluate_composite_signals
            triggered = evaluate_composite_signals()
            if triggered:
                names = ", ".join(s["pattern_name"] for s in triggered)
                logger.info("Composite signals triggered: %s", names)
        except Exception as e:
            logger.warning("Composite signal evaluation failed: %s", e)

    def run_health_check(self):
        """Run daily health checks and weekly cleanups."""
        try:
            from engine.health_monitor import health_monitor
            report = health_monitor.get_full_health_report()
            
            # Weekly cleanup on Sunday
            try:
                tz = pytz.timezone(self.timezone)
                now = datetime.now(tz)
                if now.weekday() == 6:  # Sunday
                    health_monitor.cleanup_old_data()
                    health_monitor.vacuum_database()
            except Exception as e:
                logger.error("Health cleanup failed: %s", e, exc_info=True)
                
            # Alert on critical
            if report.get('overall_status') == 'critical':
                try:
                    from engine.webhook_notifier import webhook_notifier
                    msgs_critical = []
                    if report.get('disk', {}).get('status') == 'critical':
                        msgs_critical.append(f"Disk at {report['disk'].get('percent', 0)}%")
                    if report.get('memory', {}).get('status') == 'critical':
                        msgs_critical.append(f"Memory at {report['memory'].get('percent', 0)}%")
                    if report.get('database', {}).get('status') == 'critical':
                        msgs_critical.append(f"DB size {report['database'].get('size_mb', 0)}MB")
                    if report.get('errors', {}).get('status') == 'critical':
                        msgs_critical.append(f"Error rate {report['errors'].get('error_rate_pct', 0)}%")
                    
                    if msgs_critical:
                        msg = "🔴 *CRITICAL SYSTEM HEALTH*\n" + "\n".join(msgs_critical)
                        webhook_notifier.send_custom(msg)
                except Exception as e:
                    logger.error("Health alert failed: %s", e, exc_info=True)
                    
            logger.info("Health check completed. Status: %s", report.get("overall_status", "unknown"))
        except Exception as e:
            logger.error("Health check overall failed: %s", e, exc_info=True)

    def grade_signals(self):
        """Grade past signals and auto-tune weights."""
        try:
            from engine.signal_grader import signal_grader
            graded = signal_grader.grade_pending_signals()
            logger.info("Signal Grader: Graded %d pending signals.", graded)
            
            # Auto-tune weights if enough data
            try:
                tune_result = signal_grader.auto_tune_weights()
                if tune_result.get('tuned'):
                    try:
                        from engine.webhook_notifier import webhook_notifier
                        webhook_notifier.send_custom(f"🤖 *Auto-Tuned Quant Weights*\n{tune_result.get('message')}")
                    except Exception:
                        pass
            except Exception as e:
                logger.error("Auto tuning quant weights failed: %s", e, exc_info=True)
                
        except Exception as e:
            logger.error("Signal grading failed: %s", e, exc_info=True)

    def run_auto_paper_entry(self):
        """Enter automated paper trades based on recent strong signals."""
        try:
            from engine.auto_paper_trader import auto_paper_trader
            entered = auto_paper_trader.process_new_signals()
            logger.info("Auto Paper Trader: Entered %d new positions.", entered)
        except Exception as e:
            logger.error("Auto paper entry failed: %s", e, exc_info=True)

    def run_auto_paper_exit(self):
        """Check open paper trades for exit conditions and expire stale pending confirmations."""
        try:
            from engine.auto_paper_trader import auto_paper_trader
            auto_paper_trader.expire_pending()
            exited = auto_paper_trader.check_open_positions()
            logger.info("Auto Paper Trader: Exited %d positions.", exited)
        except Exception as e:
            logger.error("Auto paper exit failed: %s", e, exc_info=True)

    def run_broker_sync(self):
        """Sync positions from real broker (market hours + non-paper mode only). Phase 6."""
        mode = (db.get_setting("auto_trade_mode") or "paper").lower()
        if mode == "paper":
            return  # Nothing to sync in paper mode
        if not self.is_market_open():
            return
        try:
            from engine.order_manager import order_manager
            synced = order_manager.sync_broker_positions()
            if synced:
                logger.info("Broker Sync: %d positions synced from %s", synced, mode.upper())
        except Exception as e:
            logger.error("Broker sync failed: %s", e, exc_info=True)

    def run_broker_pnl_snapshot(self):
        """Store broker account equity in today's paper_snapshot row (Phase 6)."""
        mode = (db.get_setting("auto_trade_mode") or "paper").lower()
        if mode == "paper":
            return
        if not self.is_market_open():
            return
        try:
            from clients.broker_client import get_broker_client
            account = get_broker_client().get_account()
            broker_equity = account.get("equity", 0)
            today = datetime.now().strftime('%Y-%m-%d')
            db.execute(
                "UPDATE paper_snapshots SET broker_value = ? WHERE snapshot_date = ?",
                (broker_equity, today)
            )
            logger.info("Broker P&L Snapshot: equity $%.2f stored for %s", broker_equity, today)
        except Exception as e:
            logger.error("Broker P&L snapshot failed: %s", e, exc_info=True)

    def retrain_meta_labeler(self):
        """Retrain the Random Forest meta-labeler on graded signal outcomes."""
        try:
            from engine.meta_labeler import meta_labeler
            result = meta_labeler.train()
            if result.get('trained'):
                logger.info("Meta-labeler retrained: v%s, CV accuracy=%s, samples=%s",
                        result.get("model_version"), result.get("cv_accuracy"), result.get("training_samples"))
            else:
                logger.info("Meta-labeler not retrained: %s", result.get("reason", "unknown"))
        except Exception as e:
            logger.error("Meta-labeler retrain failed: %s", e, exc_info=True)

    def run_mcpt_validation(self):
        """Run Monte Carlo Permutation Test to validate strategy significance."""
        try:
            from engine.mcpt_validator import mcpt_validator
            result = mcpt_validator.run_validation()
            if result.get('status') == 'completed':
                logger.info("MCPT Validation: p=%s, PF=%s, significant=%s",
                        result.get("p_value"), result.get("actual_pf"), result.get("significant"))
                if not result.get('significant', True):
                    try:
                        from engine.webhook_notifier import webhook_notifier
                        webhook_notifier.send_custom(
                            f"MCPT Warning: p-value={result['p_value']:.3f}, "
                            f"strategy may not be statistically significant."
                        )
                    except Exception:
                        pass
            else:
                logger.info("MCPT Validation: %s (%d/%d signals)",
                        result.get("status", "unknown"), result.get("n_signals", 0), result.get("min_required", 30))
        except Exception as e:
            logger.error("MCPT validation failed: %s", e, exc_info=True)

    def check_rss_geo_trigger(self):
        """Scan RSS feeds for geo keywords; fire an immediate geo scan on hit (60-min cooldown)."""
        try:
            from clients.rss_client import rss_geo_scanner
            hits = rss_geo_scanner.scan()
            if rss_geo_scanner.should_trigger(hits):
                logger.info("[RSS GEO] %d keyword hit(s) — firing immediate geo scan", len(hits))
                for h in hits[:3]:
                    logger.debug("  · %s", h)
                rss_geo_scanner.mark_triggered()
                self.run_geopolitical_scan()
        except Exception as e:
            logger.error("RSS geo trigger failed: %s", e, exc_info=True)

    def _check_intraday_breakouts(self, threshold_pct: float = 3.0):
        """Check all watchlist tickers for ±threshold_pct% intraday move and queue analysis."""
        import threading
        try:
            import yfinance as yf
            watchlist = db.get_watchlist()
            if not watchlist:
                return
            triggered = []
            for item in watchlist:
                ticker = item['ticker']
                try:
                    info = yf.Ticker(ticker).fast_info
                    current = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
                    prev_close = getattr(info, 'previous_close', None)
                    if current and prev_close and prev_close > 0:
                        pct = (current - prev_close) / prev_close * 100
                        if abs(pct) >= threshold_pct:
                            triggered.append((ticker, pct))
                except Exception:
                    pass

            for ticker, pct in triggered:
                sign = "+" if pct > 0 else ""
                logger.info("Breakout: %s %s%.1f%% — queuing analysis", ticker, sign, pct)
                try:
                    from engine.webhook_notifier import webhook_notifier
                    webhook_notifier.reload()
                    webhook_notifier.send_custom(
                        f"⚡ *Intraday Breakout: {ticker}*\n"
                        f"Move: {sign}{pct:.1f}% — triggering re-analysis"
                    )
                except Exception:
                    pass

                def _analyze(t=ticker, p=pct):
                    try:
                        from engine.agents import InvestmentSwarm
                        swarm = InvestmentSwarm()
                        result = swarm.analyze_single_stock(t)
                        if result and result.get('recommendation'):
                            result.setdefault('anomaly', f'Intraday breakout {p:+.1f}%')
                            db.save_analysis(t, result)
                            logger.info("Breakout analysis saved for %s", t)
                    except Exception as e:
                        logger.error("Breakout analysis failed for %s: %s", t, e, exc_info=True)

                threading.Thread(target=_analyze, daemon=True).start()
        except Exception as e:
            logger.error("Intraday breakout check failed: %s", e, exc_info=True)

    def refresh_corporate_actions(self):
        """Weekly refresh: fetch splits + dividends for all watchlist tickers (#43)."""
        try:
            from engine.corporate_actions import corporate_actions_tracker
            result = corporate_actions_tracker.refresh_all()
            # Apply any pending split adjustments to portfolio_trades
            adjusted = corporate_actions_tracker.apply_splits_to_portfolio()
            # Credit any pending dividends to paper portfolio
            credited = corporate_actions_tracker.credit_dividends_to_paper_portfolio()
            logger.info("Corporate actions: %s tickers, %s splits, %s dividends | %d trades adjusted, %d dividends credited",
                         result["tickers"], result["splits"], result["dividends"], adjusted, credited)
        except Exception as e:
            logger.error("Corporate actions refresh failed: %s", e, exc_info=True)

    def run_db_backup(self):
        """Create a daily DB backup with rotation (7 daily + 4 weekly)."""
        try:
            result = db.backup_db()
            msg = f"DB backup: {result['file']} ({result['size_mb']} MB)"
            logger.info(msg)
            db.log_scheduler_run(tickers_scanned=0, alerts_sent=0, errors="", duration=0)
        except Exception as e:
            logger.error("DB backup failed: %s", e, exc_info=True)

    def _enter_deep_sleep_mode(self):
        """Pause sleep-sensitive interval jobs so the CPU can reach deep C-states (C6/C7/C8).

        Called automatically at deep_sleep_start time (default 22:00) and once at startup
        if the scheduler boots during the sleep window.  The five market-hours jobs
        (broker_sync, broker_pnl_snapshot, price_alert_check, rss_geo_trigger, and the
        fixed-time geo scan) already use CronTriggers that don't fire overnight, so only
        the remaining IntervalTrigger jobs need explicit suspension.
        """
        if not self.is_running:
            return
        cstate_ok = db.get_setting('cstate_overnight_mode')
        if cstate_ok is False:  # explicitly disabled
            return
        paused = []
        for job_id in self._SLEEP_SENSITIVE_JOBS:
            try:
                job = self.scheduler.get_job(job_id)
                if job and job.next_run_time is not None:  # not already paused
                    self.scheduler.pause_job(job_id)
                    paused.append(job_id)
            except Exception:
                pass
        if paused:
            logger.info("[C-state] Deep sleep → paused jobs: %s", ", ".join(paused))
        else:
            logger.info("[C-state] Deep sleep entered (no interval jobs to pause)")

    def _exit_deep_sleep_mode(self):
        """Resume jobs that were paused by _enter_deep_sleep_mode."""
        if not self.is_running:
            return
        resumed = []
        for job_id in self._SLEEP_SENSITIVE_JOBS:
            try:
                job = self.scheduler.get_job(job_id)
                if job and job.next_run_time is None:  # currently paused
                    self.scheduler.resume_job(job_id)
                    resumed.append(job_id)
            except Exception:
                pass
        if resumed:
            logger.info("[C-state] Deep sleep ended → resumed jobs: %s", ", ".join(resumed))
        else:
            logger.info("[C-state] Deep sleep ended (nothing to resume)")

    def run_graham_screen(self):
        """Daily Graham intrinsic value screen across watchlist."""
        try:
            from engine.graham_screener import graham_screener
            tickers = [row["ticker"] for row in db.get_watchlist()]
            if not tickers:
                return
            result = graham_screener.screen_watchlist(tickers, discount_factor=0.2)
            buy_count = result.get("buy_candidates", 0)
            logger.info("Graham screen: %s/%d IV-calculable, %d buy candidates (AAA yield: %s%%)",
                     result["iv_calculable"], len(tickers), buy_count, result["aaa_yield"])
        except Exception as e:
            logger.error("Graham screen failed: %s", e, exc_info=True)

    def run_fear_greed_snapshot(self):
        """Daily Fear & Greed + VIX snapshot."""
        try:
            from engine.fear_greed_tracker import fear_greed_tracker
            value = fear_greed_tracker.get_current_fear_greed()
            vix = fear_greed_tracker.get_latest_vix_features()
            label = fear_greed_tracker.get_fg_label(value) if value is not None else "N/A"
            logger.info("Fear & Greed: %.1f (%s) | VIX=%.1f MA10=%.1f",
                     value, label, vix.get("vix"), vix.get("vix_ma10"))
        except Exception as e:
            logger.error("Fear & Greed snapshot failed: %s", e, exc_info=True)

    def run_politician_refresh(self):
        """Daily Senate trade data refresh."""
        try:
            from engine.politician_tracker import politician_tracker
            # Force cache refresh
            politician_tracker._raw_cache = None
            politician_tracker._cache_time = None
            politician_tracker._by_ticker = None
            trades = politician_tracker.fetch_senate_trades()
            top = politician_tracker.get_top_traded_tickers(days=30, top_n=5)
            ticker_str = ", ".join(t["ticker"] for t in top[:5])
            logger.info("Senate trades: %d records loaded. Top 5 (30d): %s", len(trades), ticker_str)
        except Exception as e:
            logger.error("Politician refresh failed: %s", e, exc_info=True)

    def trigger_manual_scan(self):
        """Trigger an immediate scan in background"""
        if self.is_scanning:
            logger.warning("Scan already in progress, skipping")
            return False
            
        logger.info("Manual scan triggered in background")
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
