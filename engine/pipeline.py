"""
Daily Analysis Pipeline
Orchestrates the tiered analysis funnel with adaptive budget limits:
1. Stage 1: Flash-8b Scan (All Tickers) -> Filter Top candidates
2. Stage 2: Flash Deep Dive (Top N based on budget)
3. Stage 3: Synthesis (Top N finalists)

Stage limits are calculated dynamically from the monthly EUR budget.
"""
from core.database import db
from core.budget_tracker import budget_tracker
from engine.agents import swarm
from engine.learning_optimizer import learning_optimizer
import time
import logging
from typing import Dict, List

class DailyPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_daily_cycle(self) -> Dict:
        """Run the full daily analysis cycle with adaptive budget limits"""
        # Calculate today's limits from budget
        limits = budget_tracker.get_pipeline_limits()
        self.logger.info(f"Pipeline limits: {limits}")
        print(f"  Starting Daily Analysis Cycle")
        print(f"  Budget: Stage1={limits['stage1_max']}, Stage2={limits['stage2_max']}, "
              f"Stage3={limits['stage3_max']}, Perplexity={limits['perplexity_max']}")

        start_time = time.time()
        errors = []

        try:
            # 0. Get Configuration
            try:
                settings = db.get_all_settings()
                variant = settings.get('analysis_variant', 'balanced')
            except Exception as e:
                self.logger.error(f"Failed to load settings: {e}")
                variant = 'balanced'

            try:
                watchlist = db.get_watchlist(active_only=True)
                watchlist_tickers = [item['ticker'] for item in watchlist if item.get('ticker')]
            except Exception as e:
                self.logger.error(f"Failed to load watchlist: {e}")
                raise Exception(f"Cannot load watchlist: {str(e)}")

            if not watchlist_tickers:
                error_msg = "Watchlist is empty. Please add stocks to your watchlist first."
                self.logger.warning(error_msg)
                raise Exception(error_msg)

            # 1. Ticker Selection based on Variant
            scan_list = []
            if variant == "conservative":
                scan_list = watchlist_tickers[:limits['stage1_max']]
            elif variant == "aggressive":
                scan_list = watchlist_tickers[:limits['stage1_max']]
                print("  Aggressive Mode: Watchlist + Discovery Context")
            else:
                scan_list = watchlist_tickers[:limits['stage1_max']]

            print(f"  Variant: {variant} | Scanning: {len(scan_list)} tickers")

            # === STAGE 1: Quant Screening (zero API cost) ===
            try:
                candidates = swarm.stage1_scan(scan_list, variant)
                self.logger.info(f"Stage 1 (quant screen) complete: {len(candidates)} candidates")
            except Exception as e:
                self.logger.error(f"Stage 1 failed: {e}", exc_info=True)
                errors.append(f"Stage 1 error: {str(e)}")
                raise Exception(f"Stage 1 failed: {str(e)}")

            # Filter for Stage 2 (adaptive limit) — higher threshold since scores are math-based
            stage2_candidates = [c for c in candidates if c.get('composite_score', c.get('score', 0)) >= 40][:limits['stage2_max']]
            print(f"\n  Promoting {len(stage2_candidates)} candidates to Stage 2 (limit: {limits['stage2_max']})")

            full_results = []

            # === STAGE 2: Deep Dive (Flash) ===
            for candidate in stage2_candidates:
                # Check budget before each request
                if not budget_tracker.can_afford_request('gemini'):
                    self.logger.warning("Gemini budget exhausted mid-stage2")
                    break

                try:
                    result = swarm.stage2_analyze(candidate, variant)
                    candidate.update(result)
                    full_results.append(candidate)
                except Exception as e:
                    ticker = candidate.get('ticker', 'Unknown')
                    self.logger.error(f"Stage 2 failed for {ticker}: {e}")
                    errors.append(f"Stage 2 error for {ticker}: {str(e)}")

            self.logger.info(f"Stage 2 complete: {len(full_results)} analyzed")

            # === STAGE 3: Final Synthesis ===
            stage3_candidates = full_results[:limits['stage3_max']]
            print(f"\n  Promoting {len(stage3_candidates)} finalists to Stage 3 (limit: {limits['stage3_max']})")

            final_reports = []
            for candidate in stage3_candidates:
                if not budget_tracker.can_afford_request('gemini'):
                    self.logger.warning("Gemini budget exhausted mid-stage3")
                    break

                try:
                    final = swarm.stage3_synthesize(candidate, variant)
                    final_reports.append(final)

                    try:
                        db.save_analysis(final['ticker'], final)
                    except Exception as e:
                        self.logger.error(f"Failed to save analysis for {final['ticker']}: {e}")
                        errors.append(f"DB save error for {final['ticker']}: {str(e)}")

                    # Record prediction for learning system verification
                    try:
                        signal = final.get('signal', 'Neutral')
                        qm = final.get('quant_metrics', {})
                        confidence = qm.get('composite_score', 50)
                        learning_optimizer.record_and_learn(final['ticker'], signal, confidence)
                    except Exception as e:
                        self.logger.error(f"Failed to record prediction for {final['ticker']}: {e}")

                except Exception as e:
                    ticker = candidate.get('ticker', 'Unknown')
                    self.logger.error(f"Stage 3 failed for {ticker}: {e}")
                    errors.append(f"Stage 3 error for {ticker}: {str(e)}")

            duration = time.time() - start_time
            print(f"\n  Daily Cycle Complete in {duration:.1f}s")
            print(f"  Processed: {len(scan_list)} -> {len(stage2_candidates)} -> {len(final_reports)}")

            if errors:
                print(f"  {len(errors)} errors encountered during cycle")

            # Log run
            try:
                error_summary = '; '.join(errors[:5]) if errors else ""
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
