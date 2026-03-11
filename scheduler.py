"""
APScheduler-based Scheduler for Investment Monitor
Runs automated scans based on configuration.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, date, time
import pytz
from core.database import db
from core.notifications import notifications

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
            print("(!) Scan already in progress")
            return

        if not force and not self._is_active_time():
            print(f"(-) Outside active hours ({self.active_start}-{self.active_end}), skipping scan")
            return

        if not force and self._is_market_holiday():
            print(f"(-) Market closed today (US holiday) — skipping scan")
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
                        print(f"[MACRO] {msg}")
        except Exception as e:
            print(f"(Error) Macro event check failed: {e}")

    def run_geopolitical_scan(self):
        """Run global geopolitical scan and alert on high-severity events"""
        import re
        from clients.perplexity_client import pplx_client
        print(f"\n[GEO SCAN] {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        raw_summary = pplx_client.get_geopolitical_scan()
        if not raw_summary:
            print("  Geopolitical scan returned no data")
            return

        scan_id = db.save_geopolitical_scan(raw_summary)
        print(f"  Geopolitical scan saved (id={scan_id})")

        # Alert + priority re-analysis on high-severity events (severity >= 8)
        scores = [int(m) for m in re.findall(r'SCHWEREGRAD[:\s/]+(\d+)', raw_summary)]
        max_severity = max(scores) if scores else 0
        if max_severity >= 8:
            notifications.send_geopolitical_alert(raw_summary, max_severity)
            print(f"  High-severity alert sent (max severity: {max_severity})")
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
                        print(f"  Priority re-analysis skipped — last scan was {elapsed_hours:.1f}h ago (cooldown: 2h)")
                        return

            print("  Triggering priority watchlist re-analysis due to high-severity geo event...")
            import threading
            t = threading.Thread(target=self.run_scan, kwargs={'force': True}, daemon=True)
            t.start()
        except Exception as e:
            print(f"  Priority re-analysis trigger failed: {e}")

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
                print(f"(!) Error scheduling discovery jobs: {e}")

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

        # Price alert check (every 15 min during market hours Mon–Fri)
        self.scheduler.add_job(
            self.check_price_alerts,
            IntervalTrigger(minutes=15),
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

        # Geopolitical scan (every 6 hours, independent of market hours)
        self.scheduler.add_job(
            self.run_geopolitical_scan,
            IntervalTrigger(hours=6),
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

        # RSS geo trigger (every 15 min — fires immediate scan on keyword hit)
        self.scheduler.add_job(
            self.check_rss_geo_trigger,
            IntervalTrigger(minutes=15),
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

        self.scheduler.start()
        self.is_running = True
        print(f"Scheduler started: Daily every {self.interval_hours}h, Weekly Sun 20:00, Monthly 28th 18:00")

        # Start two-way Telegram bot (if enabled)
        try:
            from clients.telegram_bot import telegram_bot
            telegram_bot.start()
        except Exception as e:
            print(f"(Warning) Telegram bot failed to start: {e}")

    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        self.scheduler.shutdown()
        self.is_running = False
        try:
            from clients.telegram_bot import telegram_bot
            telegram_bot.stop()
        except Exception:
            pass
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
            "is_market_open": self.is_market_open(),
            "interval_hours": self.interval_hours,
            "active_hours": f"{self.active_start} - {self.active_end}",
            "timezone": self.timezone,
            "jobs": jobs,
            "last_runs": db.get_scheduler_logs(limit=5)
        }
    
    def run_discovery(self):
        """Run daily auto-discovery (free strategies)"""
        try:
            from engine.auto_discovery import auto_discovery
            result = auto_discovery.run_daily_discovery()
            print(f"Discovery completed: {result.get('discoveries', 0)} found, {len(result.get('promoted', []))} promoted")
        except Exception as e:
            print(f"(Error) Discovery failed: {e}")

    def run_ai_discovery(self):
        """Run weekly AI discovery (Perplexity)"""
        try:
            from engine.auto_discovery import auto_discovery
            result = auto_discovery.run_weekly_ai_discovery()
            print(f"AI Discovery completed: {result.get('discoveries', 0)} found")
        except Exception as e:
            print(f"(Error) AI Discovery failed: {e}")

    def run_weekly_report(self):
        """Generate and send weekly portfolio report."""
        try:
            from engine.report_generator import report_generator
            html = report_generator.generate_weekly_report()
            if html:
                report_generator.send_report(html)
                print("Weekly report generated and sent")
        except Exception as e:
            print(f"(Error) Weekly report failed: {e}")

    def run_weekly_letter(self):
        """Generate and send AI weekly investment letter."""
        try:
            enabled = db.get_setting('weekly_letter_enabled')
            if not enabled:
                return
            from engine.weekly_letter import weekly_letter_generator
            weekly_letter_generator.generate_and_send()
        except Exception as e:
            print(f"(Error) Weekly letter failed: {e}")

    def check_hit_rates(self):
        """Check discovery hit rates and log outcomes. Also flags stale data."""
        try:
            from engine.discovery_hit_rate import discovery_hit_rate
            result = discovery_hit_rate.check_outcomes()
            checked = result.get('checked', 0) if result else 0
            print(f"Hit rate check: {checked} outcomes evaluated")
        except Exception as e:
            print(f"(Error) Hit rate check failed: {e}")

        # Flag stale data using DataFreshnessTracker
        try:
            from engine.data_freshness import data_freshness
            stale_tickers = data_freshness.get_stale_tickers()
            if stale_tickers:
                tickers_str = ', '.join(t['ticker'] for t in stale_tickers)
                msg = f"Stale data ({len(stale_tickers)} tickers): {tickers_str}"
                print(f"(!) {msg}")
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
            print(f"(Error) Staleness check failed: {e}")

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
                    print(f"Price alert triggered: {msg}")

            if triggered:
                print(f"Price alerts: {triggered} triggered out of {len(alerts)} active")

            # Also scan watchlist for intraday breakouts → auto-analysis
            trigger_pct = float(db.get_setting('intraday_trigger_pct') or 3.0)
            self._check_intraday_breakouts(threshold_pct=trigger_pct)
        except Exception as e:
            print(f"(Error) Price alert check failed: {e}")

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
                print(f"(Error) Health cleanup failed: {e}")
                
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
                    print(f"(Error) Health alert failed: {e}")
                    
            print(f"Health check completed. Status: {report.get('overall_status', 'unknown')}")
        except Exception as e:
            print(f"(Error) Health check overall failed: {e}")

    def grade_signals(self):
        """Grade past signals and auto-tune weights."""
        try:
            from engine.signal_grader import signal_grader
            graded = signal_grader.grade_pending_signals()
            print(f"Signal Grader: Graded {graded} pending signals.")
            
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
                print(f"(Error) Auto tuning quant weights failed: {e}")
                
        except Exception as e:
            print(f"(Error) Signal grading failed: {e}")

    def run_auto_paper_entry(self):
        """Enter automated paper trades based on recent strong signals."""
        try:
            from engine.auto_paper_trader import auto_paper_trader
            entered = auto_paper_trader.process_new_signals()
            print(f"Auto Paper Trader: Entered {entered} new positions.")
        except Exception as e:
            print(f"(Error) Auto paper entry failed: {e}")

    def run_auto_paper_exit(self):
        """Check open paper trades for exit conditions."""
        try:
            from engine.auto_paper_trader import auto_paper_trader
            exited = auto_paper_trader.check_open_positions()
            print(f"Auto Paper Trader: Exited {exited} positions.")
        except Exception as e:
            print(f"(Error) Auto paper exit failed: {e}")

    def retrain_meta_labeler(self):
        """Retrain the Random Forest meta-labeler on graded signal outcomes."""
        try:
            from engine.meta_labeler import meta_labeler
            result = meta_labeler.train()
            if result.get('trained'):
                print(f"Meta-labeler retrained: v{result.get('model_version')}, "
                      f"CV accuracy={result.get('cv_accuracy')}, "
                      f"samples={result.get('training_samples')}")
            else:
                print(f"Meta-labeler not retrained: {result.get('reason', 'unknown')}")
        except Exception as e:
            print(f"(Error) Meta-labeler retrain failed: {e}")

    def run_mcpt_validation(self):
        """Run Monte Carlo Permutation Test to validate strategy significance."""
        try:
            from engine.mcpt_validator import mcpt_validator
            result = mcpt_validator.run_validation()
            if result.get('status') == 'completed':
                print(f"MCPT Validation: p={result.get('p_value')}, "
                      f"PF={result.get('actual_pf')}, "
                      f"significant={result.get('significant')}")
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
                print(f"MCPT Validation: {result.get('status', 'unknown')} "
                      f"({result.get('n_signals', 0)}/{result.get('min_required', 30)} signals)")
        except Exception as e:
            print(f"(Error) MCPT validation failed: {e}")

    def check_rss_geo_trigger(self):
        """Scan RSS feeds for geo keywords; fire an immediate geo scan on hit (60-min cooldown)."""
        try:
            from clients.rss_client import rss_geo_scanner
            hits = rss_geo_scanner.scan()
            if rss_geo_scanner.should_trigger(hits):
                print(f"[RSS GEO] {len(hits)} keyword hit(s) — firing immediate geo scan")
                for h in hits[:3]:
                    print(f"  · {h}")
                rss_geo_scanner.mark_triggered()
                self.run_geopolitical_scan()
        except Exception as e:
            print(f"(Error) RSS geo trigger failed: {e}")

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
                print(f"  Breakout: {ticker} {sign}{pct:.1f}% — queuing analysis")
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
                            print(f"  Breakout analysis saved for {t}")
                    except Exception as e:
                        print(f"  Breakout analysis failed for {t}: {e}")

                threading.Thread(target=_analyze, daemon=True).start()
        except Exception as e:
            print(f"(Error) Intraday breakout check failed: {e}")

    def refresh_corporate_actions(self):
        """Weekly refresh: fetch splits + dividends for all watchlist tickers (#43)."""
        try:
            from engine.corporate_actions import corporate_actions_tracker
            result = corporate_actions_tracker.refresh_all()
            # Apply any pending split adjustments to portfolio_trades
            adjusted = corporate_actions_tracker.apply_splits_to_portfolio()
            # Credit any pending dividends to paper portfolio
            credited = corporate_actions_tracker.credit_dividends_to_paper_portfolio()
            print(f"Corporate actions: {result['tickers']} tickers, {result['splits']} splits, {result['dividends']} dividends | {adjusted} trades adjusted, {credited} dividends credited")
        except Exception as e:
            print(f"(Error) Corporate actions refresh failed: {e}")

    def run_db_backup(self):
        """Create a daily DB backup with rotation (7 daily + 4 weekly)."""
        try:
            result = db.backup_db()
            msg = f"DB backup: {result['file']} ({result['size_mb']} MB)"
            print(msg)
            db.log_scheduler_run(tickers_scanned=0, alerts_sent=0, errors="", duration=0)
        except Exception as e:
            print(f"(Error) DB backup failed: {e}")

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
