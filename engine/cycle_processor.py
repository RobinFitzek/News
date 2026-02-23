"""
Cycle Processor for Investment Algorithm
Handles daily, weekly, and monthly analysis cycles - runs like clockwork on homeserver.
Now with Self-Learning capabilities!
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from core.config import PIPELINE_STAGE_SPLIT
from core.database import db
from engine.strategy_engine import strategy_manager, risk_classifier, prompt_builder
from engine.quant_screener import quant_screener
from engine.ai_crosscheck import ai_crosscheck


class CycleProcessor:
    """Processes investment analysis cycles with self-learning optimization"""

    def __init__(self):
        self.agents = None  # Lazy load to avoid circular imports
        self.optimizer = None  # Lazy load learning optimizer
        self._budget_tracker = None

    @property
    def budget_tracker(self):
        if self._budget_tracker is None:
            from core.budget_tracker import budget_tracker
            self._budget_tracker = budget_tracker
        return self._budget_tracker

    def _get_current_budget(self) -> Dict:
        """Get the current API budget from adaptive budget tracker"""
        limits = self.budget_tracker.get_pipeline_limits()
        return {
            'flash-8b': limits['stage1_max'],
            'flash': limits['stage2_max'],
            'flash-1.5': limits['stage2_max'],
            'pro': limits['stage3_max'],
            'perplexity': limits['perplexity_max'],
        }
    
    def _get_agents(self):
        """Lazy load agents to avoid circular imports"""
        if self.agents is None:
            from agents import swarm
            self.agents = swarm
        return self.agents
    
    def _get_optimizer(self):
        """Lazy load learning optimizer"""
        if self.optimizer is None:
            from learning_optimizer import learning_optimizer
            self.optimizer = learning_optimizer
        return self.optimizer
    
    def run_cycle(self, cycle_type: str) -> Dict:
        """Run a specific analysis cycle"""
        if cycle_type == 'daily':
            return self.run_daily_cycle()
        elif cycle_type == 'weekly':
            return self.run_weekly_cycle()
        elif cycle_type == 'monthly':
            return self.run_monthly_cycle()
        else:
            print(f"(!) Unknown cycle type: {cycle_type}")
            return {'error': f'Unknown cycle: {cycle_type}'}
    
    def run_daily_cycle(self) -> Dict:
        """
        Daily Quick-Scan: Fast screening with self-learning optimization
        - Prioritizes tickers based on historical accuracy
        - Records predictions for later verification
        - Uses smart caching to reduce API calls
        """
        print(f"\n{'='*60}")
        print(f"DAILY CYCLE (SELF-LEARNING) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        strategy = strategy_manager.active_strategy
        budget = self._get_current_budget()  # Use selected tier budget
        optimizer = self._get_optimizer()
        
        # Get watchlist and optimize order based on learning
        watchlist = db.get_watchlist(active_only=True)
        tickers = [item['ticker'] for item in watchlist]
        
        if not tickers:
            print("(!) Watchlist leer - kein Scan möglich")
            return {'status': 'empty_watchlist', 'tips': []}
        
        # 🧠 LEARNING: Optimize ticker order based on historical accuracy
        optimized_tickers = optimizer.optimize_daily_cycle(tickers)
        print(f"[Strategy] {strategy['name']} | Watchlist: {len(tickers)} Ticker (optimiert)")
        print(f"[Budget] Flash-8b:{budget.get('flash-8b',30)}, Flash-1.5:{budget.get('flash-1.5',15)}, Pro:{budget.get('pro',5)}")
        
        # LEARNING: Verify old predictions (uses configurable window, default 90 days)
        optimizer.feedback.verify_predictions()
        
        # Classify all tickers (with caching)
        classified = self._classify_tickers_cached(optimized_tickers)
        
        # Stage 1: Quick Scan (Flash-8b) - Max budget items
        agents = self._get_agents()
        scan_results = []
        
        for ticker, cat_info in classified.items():
            if len(scan_results) >= budget['flash-8b']:
                break
            
            try:
                prompt = prompt_builder.build_scan_prompt(
                    ticker, 
                    strategy['name'],
                    cat_info['category']
                )
                result = self._quick_scan(ticker, prompt, cat_info)
                scan_results.append(result)
                print(f"  [OK] {ticker}: Score {result['score']} ({cat_info['category']})")
            except Exception as e:
                print(f"  [ERROR] {ticker}: {e}")
        
        # Sort by score
        scan_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Stage 2: Top candidates get deeper analysis (Flash)
        top_candidates = scan_results[:budget['flash']]
        analyzed = []
        
        for candidate in top_candidates:
            if candidate['score'] < 30:
                continue
            result = self._analyze_candidate(candidate, strategy['name'])
            analyzed.append(result)
        
        # Stage 3: Final synthesis for best candidates (Pro)
        final_tips = []
        for candidate in analyzed[:budget['pro']]:
            tip = self._synthesize_tip(candidate, strategy['name'])
            final_tips.append(tip)
            
            # Save to DB
            analysis_id, _, _ = db.save_analysis(tip['ticker'], tip)

            # Cross-check AI claims against market data
            try:
                text = ' '.join(filter(None, [
                    tip.get('fundamental', ''),
                    tip.get('recommendation', ''),
                    tip.get('technical', ''),
                ]))
                if text.strip():
                    cc = ai_crosscheck.check_analysis(tip['ticker'], text)
                    db.save_crosscheck(tip['ticker'], analysis_id, cc)
            except Exception:
                pass

        duration = time.time() - start_time
        
        # Log cycle
        db.save_cycle_result(
            'daily',
            1,  # Default strategy ID
            json.dumps(tickers[:budget['flash-8b']]),
            json.dumps([t['ticker'] for t in final_tips])
        )
        
        # Generate broker-style summary
        summary = self._generate_broker_summary(final_tips, 'daily')
        
        print(f"\n[DONE] Daily Cycle Complete in {duration:.1f}s")
        print(f"[Stats] {len(tickers)} gescannt → {len(analyzed)} analysiert → {len(final_tips)} Tipps")
        
        return {
            'status': 'completed',
            'cycle': 'daily',
            'duration': duration,
            'scanned': len(scan_results),
            'analyzed': len(analyzed),
            'tips': final_tips,
            'summary': summary
        }
    
    def run_weekly_cycle(self) -> Dict:
        """
        Weekly Deep Analysis: Thorough analysis of top performers and underperformers
        Budget: 30 Flash-8b, 30 Flash, 10 Pro
        """
        print(f"\n{'='*60}")
        print(f"WEEKLY CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        strategy = strategy_manager.active_strategy
        budget = self._get_current_budget()  # Use selected tier budget
        
        # Get watchlist
        watchlist = db.get_watchlist(active_only=True)
        tickers = [item['ticker'] for item in watchlist]
        
        # Also get discovery suggestions
        from discovery_engine import discovery_engine
        discoveries = discovery_engine.discover_trending(limit=5)
        all_tickers = list(set(tickers + discoveries))
        
        print(f"[Strategy] {strategy['name']}")
        print(f"[Stats] Watchlist: {len(tickers)} + {len(discoveries)} Discovery = {len(all_tickers)} Ticker")
        
        # More thorough analysis with higher budget
        classified = self._classify_tickers(all_tickers)
        
        # Full 3-stage pipeline with weekly budget
        scan_results = self._batch_scan(classified, budget['flash-8b'], strategy['name'])
        analyzed = self._batch_analyze(scan_results[:budget['flash']], strategy['name'])
        final_tips = self._batch_synthesize(analyzed[:budget['pro']], strategy['name'])
        
        duration = time.time() - start_time
        summary = self._generate_broker_summary(final_tips, 'weekly')
        
        db.save_cycle_result('weekly', 1, json.dumps(all_tickers), json.dumps([t['ticker'] for t in final_tips]))
        
        print(f"\n[DONE] Weekly Cycle Complete in {duration:.1f}s")
        
        return {
            'status': 'completed',
            'cycle': 'weekly',
            'duration': duration,
            'tips': final_tips,
            'discoveries': discoveries,
            'summary': summary
        }
    
    def run_monthly_cycle(self) -> Dict:
        """
        Monthly Portfolio Review: Full portfolio analysis with rebalancing recommendations
        Budget: 50 Flash-8b, 40 Flash, 20 Pro
        """
        print(f"\n{'='*60}")
        print(f"MONTHLY PORTFOLIO REVIEW - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        strategy = strategy_manager.active_strategy
        
        # Full portfolio scan
        watchlist = db.get_watchlist(active_only=True)
        
        # Get historical analysis for trend comparison
        portfolio_analysis = []
        for item in watchlist:
            ticker = item['ticker']
            history = db.get_analysis_history(ticker, limit=4)  # Last month
            category = risk_classifier.classify_ticker(ticker)
            
            portfolio_analysis.append({
                'ticker': ticker,
                'category': category,
                'history': history,
                'trend': self._calculate_trend(history)
            })
        
        # Generate rebalancing recommendations
        rebalancing = self._generate_rebalancing(portfolio_analysis, strategy)
        
        duration = time.time() - start_time
        summary = self._generate_broker_summary(portfolio_analysis, 'monthly', rebalancing)
        
        db.save_cycle_result('monthly', 1, json.dumps([item['ticker'] for item in watchlist]), 
                            json.dumps(rebalancing))
        
        print(f"\n[DONE] Monthly Review Complete in {duration:.1f}s")
        
        return {
            'status': 'completed',
            'cycle': 'monthly',
            'duration': duration,
            'portfolio_analysis': portfolio_analysis,
            'rebalancing': rebalancing,
            'summary': summary
        }
    
    def _classify_tickers(self, tickers: List[str]) -> Dict:
        """Classify all tickers by category"""
        return {t: risk_classifier.classify_ticker(t) for t in tickers}
    
    def _classify_tickers_cached(self, tickers: List[str]) -> Dict:
        """Classify tickers with smart caching to avoid redundant API calls"""
        optimizer = self._get_optimizer()
        result = {}
        
        for ticker in tickers:
            # Try cache first
            cached_data = optimizer.cache.get_stock_data(ticker)
            if cached_data:
                # Quick classification from cached data
                result[ticker] = risk_classifier.classify_ticker(ticker)
            else:
                # Full classification (slower, fetches data)
                result[ticker] = risk_classifier.classify_ticker(ticker)
        
        return result
    
    def _record_prediction(self, tip: Dict):
        """Record prediction for later verification (learning feedback)"""
        try:
            optimizer = self._get_optimizer()
            confidence = tip.get('risk_score', 5) * 10  # Convert risk to confidence
            optimizer.record_and_learn(
                ticker=tip['ticker'],
                signal=tip.get('signal', 'Hold'),
                confidence=100 - confidence  # Lower risk = higher confidence
            )
        except Exception as e:
            print(f"(!) Could not record prediction: {e}")
    
    def _quick_scan(self, ticker: str, prompt: str, category_info: Dict) -> Dict:
        """Quick scan using quantitative screener (zero API cost)"""
        result = quant_screener.screen_ticker(ticker)

        score = result.get('composite_score', result.get('score', 0)) if result and 'error' not in result else 0
        scan_result = result.get('initial_reason', '') if result else ''

        return {
            'ticker': ticker,
            'score': score,
            'composite_score': score,
            'signal': result.get('signal', 'Neutral') if result else 'Neutral',
            'category': category_info['category'],
            'risk_level': category_info['risk_level'],
            'scan_result': scan_result,
            'quant_data': result,
        }
    
    def _analyze_candidate(self, candidate: Dict, strategy: str) -> Dict:
        """Deep analysis: Perplexity news only (quant data already computed in Stage 1)"""
        ticker = candidate['ticker']

        # Perplexity: Get real-time news (if budget allows)
        try:
            from clients.perplexity_client import pplx_client
            if pplx_client.is_configured() and pplx_client.get_usage()['remaining'] > 0:
                intel = pplx_client.quick_scan(ticker)
                candidate['perplexity_intel'] = intel.get('raw', '') if intel else ''
                candidate['news'] = candidate['perplexity_intel'] or 'No recent news'
                print(f"    {ticker}: News added")
            else:
                candidate['news'] = 'Perplexity not available'
        except Exception as e:
            print(f"    Perplexity skipped for {ticker}: {e}")
            candidate['news'] = 'News unavailable'

        # Quant data from Stage 1 serves as the analysis
        qd = candidate.get('quant_data', {})
        candidate['analysis'] = candidate.get('scan_result', '')
        candidate['quant_metrics'] = {
            'valuation': qd.get('valuation', {}),
            'technicals': qd.get('technicals', {}),
            'momentum': qd.get('momentum', {}),
            'quality': qd.get('quality', {}),
            'anomalies': qd.get('anomalies', []),
            'composite_score': qd.get('composite_score', 0),
            'signal': qd.get('signal', 'Neutral'),
        }
        return candidate
    
    def _synthesize_tip(self, candidate: Dict, strategy: str) -> Dict:
        """Final synthesis: research note with risks and catalysts (no buy/sell)"""
        agents = self._get_agents()
        ticker = candidate['ticker']

        qm = candidate.get('quant_metrics', {})
        anomaly_text = ""
        for a in qm.get('anomalies', []):
            anomaly_text += f"  - {a.get('description', '')}\n"
        if not anomaly_text:
            anomaly_text = "  Keine Anomalien."

        prompt = f"""Du bist ein Research-Analyst. Schreibe eine kurze Research-Notiz fuer {ticker}.

Quant Signal: {qm.get('signal', 'Neutral')} (Score: {qm.get('composite_score', 'N/A')}/100)
Anomalien:
{anomaly_text}
News: {candidate.get('news', 'Keine Nachrichten')}

Aufgabe:
1. Fasse die wichtigsten Risiken zusammen (2-3 Saetze)
2. Nenne potenzielle Katalysatoren (2-3 Saetze)
3. Was sollte ein Investor beobachten? (1-2 Saetze)

KEINE Kauf/Verkauf-Empfehlung. Nur Fakten und Kontext.

Format:
Risk Score: [1-10]
Risiken: [Text]
Katalysatoren: [Text]
Beobachten: [Text]
"""

        response = agents.gemini.generate(prompt, tier='flash')

        import re
        risk_match = re.search(r"Risk Score:\s*(\d+)", response)

        return {
            'ticker': ticker,
            'category': candidate.get('category', 'growth'),
            'recommendation': response,
            'risk_score': int(risk_match.group(1)) if risk_match else 5,
            'signal': qm.get('signal', 'Neutral'),
            'composite_score': qm.get('composite_score', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _batch_scan(self, classified: Dict, limit: int, strategy: str) -> List[Dict]:
        """Batch scan tickers"""
        results = []
        for ticker, cat_info in list(classified.items())[:limit]:
            prompt = prompt_builder.build_scan_prompt(ticker, strategy, cat_info['category'])
            result = self._quick_scan(ticker, prompt, cat_info)
            results.append(result)
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _batch_analyze(self, candidates: List[Dict], strategy: str) -> List[Dict]:
        """Batch analyze candidates"""
        return [self._analyze_candidate(c, strategy) for c in candidates if c['score'] >= 30]
    
    def _batch_synthesize(self, candidates: List[Dict], strategy: str) -> List[Dict]:
        """Batch synthesize tips with learning feedback"""
        tips = []
        for c in candidates:
            tip = self._synthesize_tip(c, strategy)
            tips.append(tip)
            analysis_id, _, _ = db.save_analysis(tip['ticker'], tip)
            # Cross-check AI claims against market data
            try:
                text = ' '.join(filter(None, [
                    tip.get('fundamental', ''),
                    tip.get('recommendation', ''),
                    tip.get('technical', ''),
                ]))
                if text.strip():
                    cc = ai_crosscheck.check_analysis(tip['ticker'], text)
                    db.save_crosscheck(tip['ticker'], analysis_id, cc)
            except Exception:
                pass
            # 🧠 LEARNING: Record prediction for later verification
            self._record_prediction(tip)
        return tips
    
    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate trend from analysis history"""
        if not history or len(history) < 2:
            return 'neutral'
        
        # Compare risk scores
        latest = history[0].get('risk_score', 5)
        oldest = history[-1].get('risk_score', 5)
        
        if latest < oldest:
            return 'improving'
        elif latest > oldest:
            return 'worsening'
        return 'stable'
    
    def _generate_rebalancing(self, portfolio: List[Dict], strategy: Dict) -> Dict:
        """Generate rebalancing recommendations"""
        asset_mix = strategy.get('asset_mix', {})
        current_mix = {}
        
        for item in portfolio:
            cat = item['category'].get('category', 'growth')
            current_mix[cat] = current_mix.get(cat, 0) + 1
        
        # Calculate percentages
        total = len(portfolio) or 1
        current_pct = {k: (v/total)*100 for k, v in current_mix.items()}
        
        recommendations = []
        for cat, target in asset_mix.items():
            current = current_pct.get(cat, 0)
            diff = target - current
            
            if abs(diff) > 5:
                action = 'increase' if diff > 0 else 'decrease'
                recommendations.append({
                    'category': cat,
                    'current': round(current, 1),
                    'target': target,
                    'action': action,
                    'change': round(abs(diff), 1)
                })
        
        return {
            'current_allocation': current_pct,
            'target_allocation': asset_mix,
            'recommendations': recommendations
        }
    
    def _generate_broker_summary(self, tips: List, cycle: str, rebalancing: Dict = None) -> str:
        """Generate a broker-style summary for the user"""
        if not tips:
            return "📊 Keine relevanten Tipps in diesem Zyklus."
        
        if cycle == 'daily':
            lines = [f"**Daily Research Update** - {datetime.now().strftime('%d.%m.%Y')}"]
            for tip in tips[:5]:
                signal = tip.get('signal', 'Neutral')
                ticker = tip.get('ticker', 'N/A')
                risk = tip.get('risk_score', 5)
                score = tip.get('composite_score', tip.get('score', 0))
                icon = '+' if signal == 'Opportunity' else '!' if signal == 'Caution' else '~'
                lines.append(f"[{icon}] **{ticker}**: {signal} (Score: {score}/100, Risiko: {risk}/10)")
            return '\n'.join(lines)
        
        elif cycle == 'weekly':
            lines = [f"**Weekly Research Summary** - KW {datetime.now().isocalendar()[1]}"]
            for tip in tips[:10]:
                lines.append(f"  {tip.get('ticker')}: {tip.get('signal', 'Neutral')} (Score: {tip.get('composite_score', 'N/A')})")
            return '\n'.join(lines)
        
        elif cycle == 'monthly':
            lines = [f"📆 **Monthly Portfolio Review** - {datetime.now().strftime('%B %Y')}"]
            if rebalancing and rebalancing.get('recommendations'):
                lines.append("\n**Rebalancing Empfehlungen:**")
                for rec in rebalancing['recommendations']:
                    action_emoji = '⬆️' if rec['action'] == 'increase' else '⬇️'
                    lines.append(f"{action_emoji} {rec['category'].upper()}: {rec['current']}% → {rec['target']}%")
            return '\n'.join(lines)
        
        return "📊 Analyse abgeschlossen."


# Singleton
cycle_processor = CycleProcessor()
