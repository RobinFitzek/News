"""
Self-Learning Module for Investment Algorithm
Tracks prediction accuracy and learns from historical performance.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db
import yfinance as yf


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
                verified_at TIMESTAMP
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
                          current_price: float) -> int:
        """Record a new prediction for later verification"""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Determine predicted direction from signal
        direction = 'up' if 'Buy' in signal else 'down' if 'Sell' in signal else 'neutral'
        
        cursor.execute("""
            INSERT INTO prediction_outcomes 
            (ticker, prediction_date, signal, predicted_direction, confidence, 
             actual_price_at_prediction)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, datetime.now(), signal, direction, confidence, current_price))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return prediction_id
    
    def verify_predictions(self, days_back: int = 7) -> List[Dict]:
        """Verify predictions from N days ago against actual results"""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Get unverified predictions older than specified days
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cursor.execute("""
            SELECT * FROM prediction_outcomes 
            WHERE verified_at IS NULL 
            AND prediction_date < ?
        """, (cutoff_date,))
        
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        verified = []
        for pred in predictions:
            result = self._verify_single_prediction(pred, days_back)
            if result:
                verified.append(result)
        
        return verified
    
    def _verify_single_prediction(self, prediction: Dict, days: int) -> Optional[Dict]:
        """Verify a single prediction against actual price movement"""
        try:
            ticker = prediction['ticker']
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{days + 5}d')
            
            if hist.empty:
                return None
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
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
                'price_change': price_change
            }
            
        except Exception as e:
            print(f"⚠️ Verification error for {prediction['ticker']}: {e}")
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
        
        return {
            'total_verified': row['total'],
            'avg_accuracy': round(row['avg_accuracy'] or 0.5, 3),
            'avg_confidence': round(row['avg_confidence'] or 50, 1),
            'correct_predictions': row['correct'] or 0,
            'wrong_predictions': row['wrong'] or 0,
            'hit_rate': round((row['correct'] or 0) / row['total'] * 100, 1)
        }
    
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
        
        # Periodically verify old predictions
        verified = self.feedback.verify_predictions(days_back=7)
        if verified:
            print(f"📊 Verified {len(verified)} predictions")
            for v in verified[:3]:
                emoji = '✅' if v['accuracy'] == 1.0 else '❌' if v['accuracy'] == 0.0 else '➖'
                print(f"  {emoji} {v['ticker']}: predicted {v['predicted']}, was {v['actual']} ({v['price_change']:+.1f}%)")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        stats = self.feedback.get_accuracy_stats()
        stats['cache_size'] = len(self.cache.cache)
        return stats
    
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
