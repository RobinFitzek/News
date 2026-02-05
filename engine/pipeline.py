"""
Daily Analysis Pipeline
Orchestrates the tiered analysis funnel:
1. Stage 1: Flash-8b Scan (All Tickers) -> Filter Top 30 (Limit)
2. Stage 2: Flash Deep Dive (Top 15)
3. Stage 3: Pro Synthesis (Top 5 Analysis)
"""
from core.database import db
from engine.agents import swarm
import time
import logging
from typing import Dict, List

class DailyPipeline:
    def __init__(self):
        self.limits = {
            'stage1_max': 30,  # Max tickers to scan (Flash-8b)
            'stage2_max': 15,  # Max tickers to analyze deep (Flash)
            'stage3_max': 5    # Max tickers to synthesize (Pro)
        }
        self.logger = logging.getLogger(__name__)
        
    def run_daily_cycle(self) -> Dict:
        """Run the full daily analysis cycle with comprehensive error handling"""
        self.logger.info("Starting daily analysis cycle")
        print(f"🚀 Starting Daily Analysis Cycle (50 Request Budget)...")

        start_time = time.time()
        errors = []

        try:
            # 0. Get Configuration with error handling
            try:
                settings = db.get_all_settings()
                variant = settings.get('analysis_variant', 'balanced')
            except Exception as e:
                self.logger.error(f"Failed to load settings: {e}")
                variant = 'balanced'  # Fallback to default

            try:
                watchlist = db.get_watchlist(active_only=True)
                watchlist_tickers = [item['ticker'] for item in watchlist if item.get('ticker')]
            except Exception as e:
                self.logger.error(f"Failed to load watchlist: {e}")
                raise Exception(f"Cannot load watchlist: {str(e)}")

            if not watchlist_tickers:
                error_msg = "Watchlist is empty. Please add stocks to your watchlist first."
                self.logger.warning(error_msg)
                print(f"⚠️ {error_msg}")
                raise Exception(error_msg)
        
        # 1. Ticker Selection based on Variant
        scan_list = []
        if variant == "conservative":
            # Conservative: Focus only on watchlist, but pick the most stable/interesting ones if many
            scan_list = watchlist_tickers[:self.limits['stage1_max']]
        elif variant == "aggressive":
            # Aggressive: Watchlist + "Discovery" (adding some related tickers for context)
            # In a real scenario, this would call an API, here we just expand or use Flash-8b to suggest
            scan_list = watchlist_tickers[:20] # Take first 20 from watchlist
            # For the other 10, we could potentially find peers (simplified here)
            print("🔥 Aggressive Mode: Watchlist + Discovery Context")
        else: # balanced
            # Balanced: Primarily watchlist, maybe some top priority ones
            scan_list = watchlist_tickers[:self.limits['stage1_max']]

            print(f"📋 Variant: {variant} | Scanning: {len(scan_list)} tickers")

            # === STAGE 1: Quick Scan (Flash-8b) - Budget: 30 ===
            try:
                candidates = swarm.stage1_scan(scan_list, variant)
                self.logger.info(f"Stage 1 complete: {len(candidates)} candidates")
            except Exception as e:
                self.logger.error(f"Stage 1 failed: {e}", exc_info=True)
                errors.append(f"Stage 1 error: {str(e)}")
                raise Exception(f"Stage 1 failed: {str(e)}")

            # Filter for Stage 2 (Top 15)
            stage2_candidates = [c for c in candidates if c.get('score', 0) >= 30][:self.limits['stage2_max']]
            print(f"\n📉 Promoting {len(stage2_candidates)} candidates to Stage 2 (Flash Budget: 15)")

            full_results = []

            # === STAGE 2: Deep Dive (Flash) - Budget: 15 ===
            for candidate in stage2_candidates:
                try:
                    result = swarm.stage2_analyze(candidate, variant)
                    candidate.update(result)
                    full_results.append(candidate)
                except Exception as e:
                    ticker = candidate.get('ticker', 'Unknown')
                    self.logger.error(f"Stage 2 failed for {ticker}: {e}")
                    errors.append(f"Stage 2 error for {ticker}: {str(e)}")
                    # Continue with other candidates

            self.logger.info(f"Stage 2 complete: {len(full_results)} analyzed")

            # === STAGE 3: Final Synthesis (Pro) - Budget: 5 ===
            stage3_candidates = full_results[:self.limits['stage3_max']]
            print(f"\n💎 Promoting {len(stage3_candidates)} finalists to Stage 3 (Pro Budget: 5)")

            final_reports = []
            for candidate in stage3_candidates:
                try:
                    final = swarm.stage3_synthesize(candidate, variant)
                    final_reports.append(final)

                    # Save to DB with error handling
                    try:
                        db.save_analysis(final['ticker'], final)
                    except Exception as e:
                        self.logger.error(f"Failed to save analysis for {final['ticker']}: {e}")
                        errors.append(f"DB save error for {final['ticker']}: {str(e)}")

                except Exception as e:
                    ticker = candidate.get('ticker', 'Unknown')
                    self.logger.error(f"Stage 3 failed for {ticker}: {e}")
                    errors.append(f"Stage 3 error for {ticker}: {str(e)}")
                    # Continue with other candidates

            duration = time.time() - start_time
            print(f"\n✅ Daily Cycle Complete in {duration:.1f}s")
            print(f"📊 Processed: {len(scan_list)} (8b) -> {len(stage2_candidates)} (Flash) -> {len(final_reports)} (Pro)")

            if errors:
                print(f"⚠️ {len(errors)} errors encountered during cycle")
                self.logger.warning(f"Cycle completed with errors: {errors}")

            # Log run
            try:
                error_summary = '; '.join(errors[:5]) if errors else ""  # Limit error message length
                db.log_scheduler_run(
                    tickers_scanned=len(scan_list),
                    alerts_sent=0,
                    duration=duration,
                    errors=error_summary
                )
            except Exception as e:
                self.logger.error(f"Failed to log scheduler run: {e}")

            return final_reports

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Daily cycle failed: {e}", exc_info=True)

            # Try to log the failure
            try:
                db.log_scheduler_run(
                    tickers_scanned=0,
                    alerts_sent=0,
                    duration=duration,
                    errors=str(e)
                )
            except Exception as log_error:
                self.logger.error(f"Failed to log error: {log_error}")

            raise

# Singleton
pipeline = DailyPipeline()
