"""
Self-Learning Module for Investment Algorithm
Tracks prediction accuracy and learns from historical performance.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db
import yfinance as yf


VERIFICATION_WINDOWS = {
    'momentum': {'days': 20, 'label': '20 days (momentum signals decay fast)'},
    'value': {'days': 180, 'label': '6 months (value takes time to materialize)'},
    'default': {'days': 60, 'label': '60 days (balanced default)'},
}


def classify_signal_type(confidence: int = 50, signal: str = '',
                         momentum_score: float = None,
                         valuation_score: float = None) -> str:
    """Determine signal type from prediction context for verification window selection."""
    if momentum_score is not None and momentum_score > 70:
        return 'momentum'
    if valuation_score is not None and valuation_score > 70:
        return 'value'
    return 'default'


class FeedbackTracker:
    """Tracks prediction accuracy and provides learning feedback"""

    def __init__(self):
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure feedback tracking tables exist"""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Prediction outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TIMESTAMP NOT NULL,
                signal TEXT NOT NULL,
                predicted_direction TEXT,
                confidence INTEGER,
                actual_price_at_prediction REAL,
                actual_price_after REAL,
                actual_direction TEXT,
                accuracy_score REAL,
                days_elapsed INTEGER,
                verified_at TIMESTAMP,
                signal_type TEXT,
                verification_window_days INTEGER
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_tier TEXT NOT NULL,
                task_type TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.5,
                avg_confidence REAL DEFAULT 50,
                last_updated TIMESTAMP
            )
        """)
        
        # Prompt effectiveness table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT UNIQUE,
                category TEXT,
                strategy TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.5,
                avg_confidence REAL DEFAULT 50,
                last_used TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_prediction(self, ticker: str, signal: str, confidence: int,
                          current_price: float,
                          momentum_score: float = None,
                          valuation_score: float = None) -> int:
        """Record a new prediction for later verification"""
        conn = db._get_conn()
        cursor = conn.cursor()

        # Determine predicted direction from signal
        # Support both old (Buy/Sell) and new (Opportunity/Caution) signal types
        if signal in ('Opportunity',) or 'Buy' in signal:
            direction = 'up'
        elif signal in ('Caution',) or 'Sell' in signal:
            direction = 'down'
        else:
            direction = 'neutral'

        # Classify signal type for adaptive verification window
        sig_type = classify_signal_type(
            confidence=confidence, signal=signal,
            momentum_score=momentum_score, valuation_score=valuation_score
        )
        window_days = VERIFICATION_WINDOWS[sig_type]['days']

        cursor.execute("""
            INSERT INTO prediction_outcomes
            (ticker, prediction_date, signal, predicted_direction, confidence,
             actual_price_at_prediction, signal_type, verification_window_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, datetime.now(), signal, direction, confidence, current_price,
              sig_type, window_days))

        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()

        return prediction_id
    
    def verify_predictions(self, days_back: int = None) -> List[Dict]:
        """Verify predictions using per-row adaptive windows.
        Each prediction has its own verification_window_days based on signal type.
        Falls back to global setting if no per-row window is stored."""
        if days_back is None:
            try:
                days_back = int(db.get_setting('learning_verification_days') or 90)
            except (ValueError, TypeError):
                days_back = 90
        conn = db._get_conn()
        cursor = conn.cursor()

        # Get unverified predictions that have passed their verification window
        # Use per-row window if available, otherwise fall back to days_back
        cursor.execute("""
            SELECT * FROM prediction_outcomes
            WHERE verified_at IS NULL
            AND prediction_date < datetime('now', '-' ||
                COALESCE(verification_window_days, ?) || ' days')
        """, (days_back,))

        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        verified = []
        for pred in predictions:
            window = pred.get('verification_window_days') or days_back
            result = self._verify_single_prediction(pred, window)
            if result:
                result['signal_type'] = pred.get('signal_type', 'default')
                result['verification_window_days'] = window
                verified.append(result)

        return verified
    
    def _verify_single_prediction(self, prediction: Dict, days: int) -> Optional[Dict]:
        """Verify a single prediction against actual price movement + SPY benchmark."""
        try:
            ticker = prediction['ticker']
            period = f'{min(days + 10, 400)}d'

            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])
            original_price = prediction['actual_price_at_prediction']

            if not original_price or original_price == 0:
                return None

            # Calculate actual direction
            price_change = ((current_price - original_price) / original_price) * 100

            if price_change > 2:
                actual_direction = 'up'
            elif price_change < -2:
                actual_direction = 'down'
            else:
                actual_direction = 'neutral'

            # Calculate accuracy score
            predicted = prediction['predicted_direction']
            if predicted == actual_direction:
                accuracy_score = 1.0
            elif predicted == 'neutral' or actual_direction == 'neutral':
                accuracy_score = 0.5
            else:
                accuracy_score = 0.0

            # SPY benchmark comparison
            benchmark_return = None
            beat_benchmark = None
            try:
                spy_hist = yf.Ticker('SPY').history(period=period)
                if not spy_hist.empty and len(spy_hist) >= 2:
                    spy_start = float(spy_hist['Close'].iloc[0])
                    spy_end = float(spy_hist['Close'].iloc[-1])
                    if spy_start > 0:
                        benchmark_return = round(((spy_end - spy_start) / spy_start) * 100, 2)
                        beat_benchmark = price_change > benchmark_return
            except Exception:
                pass

            # Update database
            conn = db._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prediction_outcomes SET
                    actual_price_after = ?,
                    actual_direction = ?,
                    accuracy_score = ?,
                    days_elapsed = ?,
                    verified_at = ?
                WHERE id = ?
            """, (current_price, actual_direction, accuracy_score, days,
                  datetime.now(), prediction['id']))
            conn.commit()
            conn.close()

            return {
                'ticker': ticker,
                'predicted': predicted,
                'actual': actual_direction,
                'accuracy': accuracy_score,
                'price_change': round(price_change, 2),
                'benchmark_return': benchmark_return,
                'beat_benchmark': beat_benchmark,
            }

        except Exception as e:
            print(f"Verification error for {prediction['ticker']}: {e}")
            return None
    
    def get_accuracy_stats(self) -> Dict:
        """Get overall prediction accuracy statistics"""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(accuracy_score) as avg_accuracy,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN accuracy_score = 1.0 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN accuracy_score = 0.0 THEN 1 ELSE 0 END) as wrong
            FROM prediction_outcomes
            WHERE verified_at IS NOT NULL
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row['total'] == 0:
            return {
                'total_verified': 0,
                'avg_accuracy': 0.5,
                'avg_confidence': 50.0,
                'correct_predictions': 0,
                'wrong_predictions': 0,
                'hit_rate': 0.0,
                'message': 'Keine verifizierten Vorhersagen'
            }
        
        stats = {
            'total_verified': row['total'],
            'avg_accuracy': round(row['avg_accuracy'] or 0.5, 3),
            'avg_confidence': round(row['avg_confidence'] or 50, 1),
            'correct_predictions': row['correct'] or 0,
            'wrong_predictions': row['wrong'] or 0,
            'hit_rate': round((row['correct'] or 0) / row['total'] * 100, 1)
        }

        # Health warning: if accuracy is below random chance over meaningful sample
        if stats['total_verified'] >= 20 and stats['avg_accuracy'] < 0.55:
            stats['health_warning'] = (
                f"System accuracy ({stats['avg_accuracy']:.0%}) is near or below random chance "
                f"over {stats['total_verified']} predictions. Consider buying index ETFs instead."
            )

        return stats
    
    def get_ticker_accuracy(self, ticker: str) -> Dict:
        """Get accuracy stats for a specific ticker"""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(accuracy_score) as accuracy,
                AVG(confidence) as confidence
            FROM prediction_outcomes
            WHERE ticker = ? AND verified_at IS NOT NULL
        """, (ticker.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row['total'] == 0:
            return {'ticker': ticker, 'accuracy': 0.5, 'predictions': 0}
        
        return {
            'ticker': ticker,
            'total_predictions': row['total'],
            'accuracy': round(row['accuracy'] or 0.5, 3),
            'avg_confidence': round(row['confidence'] or 50, 1)
        }


class SmartCache:
    """Intelligent caching to avoid redundant API calls"""
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get(self, key: str) -> Optional[any]:
        """Get cached value if still valid"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.cache_duration:
                print(f"📦 Cache hit: {key}")
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: any):
        """Cache a value"""
        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Get cached stock data or fetch if expired"""
        cache_key = f"stock_{ticker}"
        cached = self.get(cache_key)
        
        if cached:
            return cached
        
        # Fetch fresh data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")
            
            data = {
                'name': info.get('longName', ticker),
                'current_price': info.get('currentPrice'),
                'pe_ratio': info.get('trailingPE'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                '30d_change': self._calc_change(hist),
                'fetched_at': datetime.now().isoformat()
            }
            
            self.set(cache_key, data)
            return data
            
        except Exception as e:
            print(f"⚠️ Stock data error for {ticker}: {e}")
            return None
    
    def _calc_change(self, hist) -> float:
        """Calculate price change percentage"""
        if hist.empty:
            return 0.0
        try:
            latest = hist['Close'].iloc[-1]
            oldest = hist['Close'].iloc[0]
            return round(((latest - oldest) / oldest) * 100, 2)
        except:
            return 0.0
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        now = datetime.now()
        expired = [k for k, v in self.cache.items() 
                   if now - v['timestamp'] > self.cache_duration]
        for k in expired:
            del self.cache[k]
        return len(expired)


class AdaptivePrioritizer:
    """Prioritizes tickers based on historical performance and momentum"""
    
    def __init__(self, feedback_tracker: FeedbackTracker):
        self.tracker = feedback_tracker
    
    def prioritize_watchlist(self, tickers: List[str]) -> List[Dict]:
        """
        Prioritize tickers based on:
        1. Recent prediction accuracy (reward accurate predictions)
        2. Price momentum (prioritize movers)
        3. Time since last analysis
        """
        prioritized = []
        
        for ticker in tickers:
            score = self._calculate_priority_score(ticker)
            prioritized.append({
                'ticker': ticker,
                'priority_score': score['total'],
                'components': score
            })
        
        # Sort by priority score descending
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
        return prioritized
    
    def _calculate_priority_score(self, ticker: str) -> Dict:
        """Calculate priority score for a ticker"""
        scores = {
            'accuracy_bonus': 0,
            'momentum_bonus': 0,
            'staleness_bonus': 0,
            'total': 50  # Base score
        }
        
        # 1. Accuracy bonus: prioritize tickers where we're accurate
        accuracy = self.tracker.get_ticker_accuracy(ticker)
        if accuracy['accuracy'] > 0.6:
            scores['accuracy_bonus'] = 20
        elif accuracy['accuracy'] < 0.4:
            scores['accuracy_bonus'] = -10  # Deprioritize inaccurate predictions
        
        # 2. Staleness bonus: prioritize tickers not analyzed recently
        latest = db.get_latest_analysis(ticker)
        if latest:
            try:
                last_date = datetime.fromisoformat(latest['timestamp'].replace(' ', 'T'))
                days_since = (datetime.now() - last_date).days
                scores['staleness_bonus'] = min(days_since * 5, 30)  # Max 30
            except:
                scores['staleness_bonus'] = 15
        else:
            scores['staleness_bonus'] = 30  # Never analyzed = high priority
        
        scores['total'] = 50 + scores['accuracy_bonus'] + scores['staleness_bonus'] + scores['momentum_bonus']
        return scores


class LearningOptimizer:
    """Main optimizer that combines all learning components"""
    
    def __init__(self):
        self.feedback = FeedbackTracker()
        self.cache = SmartCache(cache_duration_minutes=30)
        self.prioritizer = AdaptivePrioritizer(self.feedback)
    
    def optimize_daily_cycle(self, tickers: List[str]) -> List[str]:
        """
        Optimize ticker list for daily cycle:
        1. Prioritize based on learning
        2. Filter out recently analyzed (unless high priority)
        3. Return optimized order
        """
        prioritized = self.prioritizer.prioritize_watchlist(tickers)
        return [p['ticker'] for p in prioritized]
    
    def record_and_learn(self, ticker: str, signal: str, confidence: int):
        """Record prediction and trigger learning if enough data"""
        # Get current price for later verification
        stock_data = self.cache.get_stock_data(ticker)
        current_price = stock_data.get('current_price', 0) if stock_data else 0

        # Record the prediction
        self.feedback.record_prediction(ticker, signal, confidence, current_price)

        # Periodically verify old predictions (uses configurable window)
        verified = self.feedback.verify_predictions()
        if verified:
            print(f"  Verified {len(verified)} predictions")
            for v in verified[:3]:
                bench_text = ""
                if v.get('benchmark_return') is not None:
                    bench_text = f", SPY: {v['benchmark_return']:+.1f}%"
                    bench_text += " (beat)" if v.get('beat_benchmark') else " (missed)"
                status = 'OK' if v['accuracy'] == 1.0 else 'WRONG' if v['accuracy'] == 0.0 else 'NEUTRAL'
                print(f"    [{status}] {v['ticker']}: predicted {v['predicted']}, was {v['actual']} ({v['price_change']:+.1f}%{bench_text})")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        stats = self.feedback.get_accuracy_stats()
        stats['cache_size'] = len(self.cache.cache)
        return stats
    
    def calculate_optimal_weights(self) -> Dict:
        """Analyse last 100 verified predictions and suggest weight adjustments.

        Groups by signal_type (momentum vs value), calculates per-factor accuracy,
        and suggests weight changes. Requires 20+ verified predictions minimum.
        """
        conn = db._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM prediction_outcomes
            WHERE verified_at IS NOT NULL
            ORDER BY verified_at DESC LIMIT 100
        """)
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if len(predictions) < 20:
            return {
                'sufficient_data': False,
                'prediction_count': len(predictions),
                'message': f'Need 20+ verified predictions, have {len(predictions)}',
            }

        # Group by signal_type
        type_stats = {}
        for pred in predictions:
            sig_type = pred.get('signal_type', 'default')
            if sig_type not in type_stats:
                type_stats[sig_type] = {'total': 0, 'correct': 0}
            type_stats[sig_type]['total'] += 1
            if pred.get('accuracy_score', 0) >= 0.5:
                type_stats[sig_type]['correct'] += 1

        for st in type_stats.values():
            st['accuracy'] = round(st['correct'] / st['total'], 3) if st['total'] > 0 else 0.5

        # Current weights
        from engine.quant_screener import quant_screener
        current_weights = dict(quant_screener.config['composite_weights'])

        # Calculate suggested weights based on factor performance
        momentum_acc = type_stats.get('momentum', {}).get('accuracy', 0.5)
        value_acc = type_stats.get('value', {}).get('accuracy', 0.5)
        default_acc = type_stats.get('default', {}).get('accuracy', 0.5)

        # Regime context
        regime = None
        try:
            from engine.market_regime import market_regime
            regime = market_regime.get_current_regime().get('regime')
        except Exception:
            pass

        # Adjust: if momentum predictions are more accurate, boost technical + momentum weights
        suggested = dict(current_weights)

        if momentum_acc > default_acc + 0.1:
            suggested['momentum'] = min(0.45, current_weights['momentum'] + 0.05)
            suggested['technical'] = min(0.45, current_weights['technical'] + 0.05)
            # Reduce valuation/quality proportionally
            excess = (suggested['momentum'] - current_weights['momentum']) + \
                     (suggested['technical'] - current_weights['technical'])
            suggested['valuation'] = max(0.10, current_weights['valuation'] - excess / 2)
            suggested['quality'] = max(0.10, current_weights['quality'] - excess / 2)

        if value_acc > default_acc + 0.1:
            suggested['valuation'] = min(0.45, current_weights['valuation'] + 0.05)
            suggested['quality'] = min(0.45, current_weights['quality'] + 0.03)
            excess = (suggested['valuation'] - current_weights['valuation']) + \
                     (suggested['quality'] - current_weights['quality'])
            suggested['momentum'] = max(0.10, current_weights['momentum'] - excess / 2)
            suggested['technical'] = max(0.10, current_weights['technical'] - excess / 2)

        # Bear market: boost quality + valuation
        if regime == 'bear':
            suggested['quality'] = min(0.40, suggested['quality'] + 0.05)
            suggested['valuation'] = min(0.40, suggested['valuation'] + 0.05)
            suggested['momentum'] = max(0.10, suggested['momentum'] - 0.05)
            suggested['technical'] = max(0.10, suggested['technical'] - 0.05)

        # Normalize to sum = 1.0
        total = sum(suggested.values())
        if total > 0:
            suggested = {k: round(v / total, 4) for k, v in suggested.items()}

        return {
            'sufficient_data': True,
            'prediction_count': len(predictions),
            'current_weights': current_weights,
            'suggested_weights': suggested,
            'factor_accuracy': type_stats,
            'regime': regime,
            'changes': {k: round(suggested[k] - current_weights.get(k, 0), 4) for k in suggested},
        }

    def auto_adjust_weights(self, dry_run: bool = True) -> Dict:
        """Apply suggested weight adjustments.

        Args:
            dry_run: if True, only returns what would change without applying
        """
        suggestion = self.calculate_optimal_weights()
        if not suggestion.get('sufficient_data'):
            return suggestion

        result = {
            'current': suggestion['current_weights'],
            'suggested': suggestion['suggested_weights'],
            'applied': not dry_run,
        }

        if not dry_run:
            # Convert 4-factor weights to 2-factor format for the override system
            tech_w = suggestion['suggested_weights']['technical'] + suggestion['suggested_weights']['momentum']
            mom_w = suggestion['suggested_weights']['momentum']

            db.set_setting('quant_weights_override', {
                'tech_weight': round(tech_w, 4),
                'momentum_weight': round(mom_w, 4),
            })

            from engine.quant_screener import quant_screener
            quant_screener.reload_weights()
            result['active_weights'] = dict(quant_screener.config['composite_weights'])

        return result

    def should_trust_prediction(self, ticker: str, confidence: int) -> Tuple[bool, str]:
        """
        Determine if a prediction should be trusted based on historical accuracy.
        Returns (should_trust, reason)
        """
        ticker_stats = self.feedback.get_ticker_accuracy(ticker)
        
        # If we have history and accuracy is poor, be skeptical
        if ticker_stats['total_predictions'] >= 5:
            if ticker_stats['accuracy'] < 0.3:
                return False, f"Historically inaccurate for {ticker} ({ticker_stats['accuracy']:.0%})"
            elif ticker_stats['accuracy'] > 0.7:
                return True, f"Historically accurate for {ticker} ({ticker_stats['accuracy']:.0%})"
        
        # Default: trust high confidence predictions
        if confidence >= 70:
            return True, f"High confidence ({confidence}%)"
        elif confidence >= 50:
            return True, f"Medium confidence ({confidence}%)"
        else:
            return False, f"Low confidence ({confidence}%)"


# Singletons
feedback_tracker = FeedbackTracker()
smart_cache = SmartCache()
learning_optimizer = LearningOptimizer()
