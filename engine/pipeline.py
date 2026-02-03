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

class DailyPipeline:
    def __init__(self):
        self.limits = {
            'stage1_max': 30,  # Max tickers to scan (Flash-8b)
            'stage2_max': 15,  # Max tickers to analyze deep (Flash)
            'stage3_max': 5    # Max tickers to synthesize (Pro)
        }
        
    def run_daily_cycle(self):
        """Run the full daily analysis cycle (30 Flash-8b, 15 Flash, 5 Pro)"""
        print(f"🚀 Starting Daily Analysis Cycle (50 Request Budget)...")
        
        # 0. Get Configuration
        settings = db.get_all_settings()
        variant = settings.get('analysis_variant', 'balanced')
        watchlist = db.get_watchlist(active_only=True)
        watchlist_tickers = [item['ticker'] for item in watchlist]
        
        if not watchlist_tickers:
            print("⚠️ Watchlist empty. No analysis possible.")
            return []
        
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
        start_time = time.time()
        candidates = swarm.stage1_scan(scan_list, variant)
        
        # Filter for Stage 2 (Top 15)
        stage2_candidates = [c for c in candidates if c['score'] >= 30][:self.limits['stage2_max']]
        print(f"\n📉 Promoting {len(stage2_candidates)} candidates to Stage 2 (Flash Budget: 15)")
        
        full_results = []
        
        # === STAGE 2: Deep Dive (Flash) - Budget: 15 ===
        for candidate in stage2_candidates:
            result = swarm.stage2_analyze(candidate, variant)
            candidate.update(result)
            full_results.append(candidate)
            
        # === STAGE 3: Final Synthesis (Pro) - Budget: 5 ===
        # Take Top 5 from the Deep Dive results for the final verdict
        stage3_candidates = full_results[:self.limits['stage3_max']]
        print(f"\n💎 Promoting {len(stage3_candidates)} finalists to Stage 3 (Pro Budget: 5)")
        
        final_reports = []
        for candidate in stage3_candidates:
            final = swarm.stage3_synthesize(candidate, variant)
            final_reports.append(final)
            
            # Save to DB
            db.save_analysis(final['ticker'], final)
            
        duration = time.time() - start_time
        print(f"\n✅ Daily Cycle Complete in {duration:.1f}s")
        print(f"📊 Processed: {len(scan_list)} (8b) -> {len(stage2_candidates)} (Flash) -> {len(final_reports)} (Pro)")
        
        # Log run
        db.log_scheduler_run(
            tickers_scanned=len(scan_list),
            alerts_sent=0,
            duration=duration
        )
        
        return final_reports

# Singleton
pipeline = DailyPipeline()
