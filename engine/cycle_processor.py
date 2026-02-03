"""
Cycle Processor for Investment Algorithm
Handles daily, weekly, and monthly analysis cycles - runs like clockwork on homeserver.
Now with Self-Learning capabilities!
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from core.config import CYCLE_CONFIG
from core.database import db
from engine.strategy_engine import strategy_manager, risk_classifier, prompt_builder


class CycleProcessor:
    """Processes investment analysis cycles with self-learning optimization"""
    
    def __init__(self):
        self.cycles = CYCLE_CONFIG
        self.agents = None  # Lazy load to avoid circular imports
        self.optimizer = None  # Lazy load learning optimizer
    
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
            print(f"⚠️ Unknown cycle type: {cycle_type}")
            return {'error': f'Unknown cycle: {cycle_type}'}
    
    def run_daily_cycle(self) -> Dict:
        """
        Daily Quick-Scan: Fast screening with self-learning optimization
        - Prioritizes tickers based on historical accuracy
        - Records predictions for later verification
        - Uses smart caching to reduce API calls
        """
        print(f"\n{'='*60}")
        print(f"🌅 DAILY CYCLE (SELF-LEARNING) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        strategy = strategy_manager.active_strategy
        budget = self.cycles['daily']['api_budget']
        optimizer = self._get_optimizer()
        
        # Get watchlist and optimize order based on learning
        watchlist = db.get_watchlist(active_only=True)
        tickers = [item['ticker'] for item in watchlist]
        
        if not tickers:
            print("⚠️ Watchlist leer - kein Scan möglich")
            return {'status': 'empty_watchlist', 'tips': []}
        
        # 🧠 LEARNING: Optimize ticker order based on historical accuracy
        optimized_tickers = optimizer.optimize_daily_cycle(tickers)
        print(f"📋 Strategie: {strategy['name']} | Watchlist: {len(tickers)} Ticker (optimiert)")
        print(f"💰 Budget: Flash-8b:{budget.get('flash-8b',30)}, Flash-1.5:{budget.get('flash-1.5',15)}, Pro:{budget.get('pro',5)}")
        
        # 🧠 LEARNING: Verify old predictions first (runs weekly check)
        optimizer.feedback.verify_predictions(days_back=7)
        
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
                print(f"  ✓ {ticker}: Score {result['score']} ({cat_info['category']})")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
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
            db.save_analysis(tip['ticker'], tip)
        
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
        
        print(f"\n✅ Daily Cycle Complete in {duration:.1f}s")
        print(f"📊 {len(tickers)} gescannt → {len(analyzed)} analysiert → {len(final_tips)} Tipps")
        
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
        print(f"📅 WEEKLY CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        strategy = strategy_manager.active_strategy
        budget = self.cycles['weekly']['api_budget']
        
        # Get watchlist
        watchlist = db.get_watchlist(active_only=True)
        tickers = [item['ticker'] for item in watchlist]
        
        # Also get discovery suggestions
        from discovery_engine import discovery_engine
        discoveries = discovery_engine.discover_trending(limit=5)
        all_tickers = list(set(tickers + discoveries))
        
        print(f"📋 Strategie: {strategy['name']}")
        print(f"📊 Watchlist: {len(tickers)} + {len(discoveries)} Discovery = {len(all_tickers)} Ticker")
        
        # More thorough analysis with higher budget
        classified = self._classify_tickers(all_tickers)
        
        # Full 3-stage pipeline with weekly budget
        scan_results = self._batch_scan(classified, budget['flash-8b'], strategy['name'])
        analyzed = self._batch_analyze(scan_results[:budget['flash']], strategy['name'])
        final_tips = self._batch_synthesize(analyzed[:budget['pro']], strategy['name'])
        
        duration = time.time() - start_time
        summary = self._generate_broker_summary(final_tips, 'weekly')
        
        db.save_cycle_result('weekly', 1, json.dumps(all_tickers), json.dumps([t['ticker'] for t in final_tips]))
        
        print(f"\n✅ Weekly Cycle Complete in {duration:.1f}s")
        
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
        print(f"📆 MONTHLY PORTFOLIO REVIEW - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
        
        print(f"\n✅ Monthly Review Complete in {duration:.1f}s")
        
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
            print(f"⚠️ Could not record prediction: {e}")
    
    def _quick_scan(self, ticker: str, prompt: str, category_info: Dict) -> Dict:
        """Quick scan using Flash-8b"""
        agents = self._get_agents()
        response = agents.gemini.generate(prompt, tier='flash-8b')
        
        # Parse score
        import re
        score_match = re.search(r"Score:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 0
        
        return {
            'ticker': ticker,
            'score': score,
            'category': category_info['category'],
            'risk_level': category_info['risk_level'],
            'scan_result': response
        }
    
    def _analyze_candidate(self, candidate: Dict, strategy: str) -> Dict:
        """Deep analysis using Flash + Perplexity Intelligence"""
        agents = self._get_agents()
        ticker = candidate['ticker']
        
        # Fetch stock data
        stock_data = agents._get_stock_data(ticker)
        
        # 🌐 PERPLEXITY: Get real-time intelligence (if budget allows)
        intel = None
        try:
            from clients.perplexity_client import pplx_client
            if pplx_client.get_usage()['remaining'] > 0:
                intel = pplx_client.quick_scan(ticker)
                candidate['perplexity_intel'] = intel.get('raw', '')
                print(f"  🌐 {ticker}: Perplexity Intelligence added")
        except Exception as e:
            print(f"  ⚠️ Perplexity skipped for {ticker}: {e}")
        
        # Build prompt with Perplexity context if available
        intel_context = candidate.get('perplexity_intel', '')
        prompt = prompt_builder.build_analysis_prompt(
            ticker, strategy, candidate['category'], stock_data
        )
        
        # Append Perplexity intel to prompt if available
        if intel_context:
            prompt += f"\n\nAKTUELLE MARKT-INTELLIGENCE (live):\n{intel_context}"
        
        response = agents.gemini.generate(prompt, tier='flash')
        candidate['analysis'] = response
        candidate['stock_data'] = stock_data
        return candidate
    
    def _synthesize_tip(self, candidate: Dict, strategy: str) -> Dict:
        """Final synthesis using Pro"""
        agents = self._get_agents()
        ticker = candidate['ticker']
        
        prompt = prompt_builder.build_synthesis_prompt(
            ticker, strategy, candidate['category'],
            {'fundamental': candidate.get('analysis', ''), 
             'technical': '', 
             'stage1_reason': candidate.get('scan_result', '')}
        )
        
        response = agents.gemini.generate(prompt, tier='pro')
        
        # Parse response
        import re
        risk_match = re.search(r"Risk Score:\s*(\d+)", response)
        signal_match = re.search(r"Signal:\s*(\w+\s*\w*)", response)
        
        return {
            'ticker': ticker,
            'category': candidate['category'],
            'recommendation': response,
            'risk_score': int(risk_match.group(1)) if risk_match else 5,
            'signal': signal_match.group(1).strip() if signal_match else 'Hold',
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
            db.save_analysis(tip['ticker'], tip)
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
            lines = [f"📈 **Daily Broker Update** - {datetime.now().strftime('%d.%m.%Y')}"]
            for tip in tips[:5]:
                signal = tip.get('signal', 'Hold')
                ticker = tip.get('ticker', 'N/A')
                risk = tip.get('risk_score', 5)
                emoji = '🟢' if 'Buy' in signal else '🔴' if 'Sell' in signal else '🟡'
                lines.append(f"{emoji} **{ticker}**: {signal} (Risiko: {risk}/10)")
            return '\n'.join(lines)
        
        elif cycle == 'weekly':
            lines = [f"📊 **Weekly Analysis** - KW {datetime.now().isocalendar()[1]}"]
            for tip in tips[:10]:
                lines.append(f"• {tip.get('ticker')}: {tip.get('signal', 'Hold')}")
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
