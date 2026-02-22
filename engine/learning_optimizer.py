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
        
        # Migrations: add columns that may not exist yet
        migrations = [
            ('has_ai', 'INTEGER DEFAULT 0'),
            ('benchmark_return', 'REAL'),
            ('beat_benchmark', 'INTEGER'),
        ]
        for col, col_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE prediction_outcomes ADD COLUMN {col} {col_type}")
                conn.commit()
            except Exception:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def record_prediction(self, ticker: str, signal: str, confidence: int,
                          current_price: float,
                          momentum_score: float = None,
                          valuation_score: float = None,
                          has_ai: bool = False) -> int:
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
             actual_price_at_prediction, signal_type, verification_window_days, has_ai)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, datetime.now(), signal, direction, confidence, current_price,
              sig_type, window_days, 1 if has_ai else 0))

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
                # Ticker may be delisted — add to graveyard and count as loss
                try:
                    conn = db._get_conn()
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO ticker_graveyard (ticker, last_seen, reason)
                        VALUES (?, ?, ?)
                    """, (ticker, prediction['prediction_date'][:10], 'No data from yfinance (possible delisting)'))
                    # Mark prediction as failed
                    cursor.execute("""
                        UPDATE prediction_outcomes SET
                            accuracy_score = 0.0,
                            actual_direction = 'unknown',
                            days_elapsed = ?,
                            verified_at = ?
                        WHERE id = ?
                    """, (days, datetime.now(), prediction['id']))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
                return {
                    'ticker': ticker,
                    'predicted': prediction['predicted_direction'],
                    'actual': 'unknown',
                    'accuracy': 0.0,
                    'price_change': 0,
                    'benchmark_return': None,
                    'beat_benchmark': False,
                    'graveyard': True,
                }

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

            # Update database (including benchmark data)
            conn = db._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prediction_outcomes SET
                    actual_price_after = ?,
                    actual_direction = ?,
                    accuracy_score = ?,
                    days_elapsed = ?,
                    verified_at = ?,
                    benchmark_return = ?,
                    beat_benchmark = ?
                WHERE id = ?
            """, (current_price, actual_direction, accuracy_score, days,
                  datetime.now(), benchmark_return,
                  1 if beat_benchmark else (0 if beat_benchmark is not None else None),
                  prediction['id']))
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
                SUM(CASE WHEN accuracy_score = 0.0 THEN 1 ELSE 0 END) as wrong,
                AVG(benchmark_return) as avg_benchmark_return,
                SUM(CASE WHEN beat_benchmark = 1 THEN 1 ELSE 0 END) as beat_benchmark_count,
                SUM(CASE WHEN beat_benchmark IS NOT NULL THEN 1 ELSE 0 END) as benchmark_total
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
                'benchmark_beat_rate': 0.0,
                'avg_benchmark_return': 0.0,
                'benchmark_total': 0,
                'message': 'Keine verifizierten Vorhersagen'
            }

        benchmark_total = row['benchmark_total'] or 0
        beat_count = row['beat_benchmark_count'] or 0

        stats = {
            'total_verified': row['total'],
            'avg_accuracy': round(row['avg_accuracy'] or 0.5, 3),
            'avg_confidence': round(row['avg_confidence'] or 50, 1),
            'correct_predictions': row['correct'] or 0,
            'wrong_predictions': row['wrong'] or 0,
            'hit_rate': round((row['correct'] or 0) / row['total'] * 100, 1),
            'avg_benchmark_return': round(row['avg_benchmark_return'] or 0, 2),
            'benchmark_beat_rate': round(beat_count / benchmark_total * 100, 1) if benchmark_total > 0 else 0.0,
            'benchmark_total': benchmark_total,
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
    
    def calculate_signal_ev(self) -> Dict:
        """
        Calculate expected value (average return) per signal type and confidence bucket.
        
        Returns data like:
        - "If you followed all BUY signals with >70% confidence, avg return = +4.2%"
        - Per confidence bucket: 50-60, 60-70, 70-80, 80-90, 90-100
        """
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Get all verified predictions with price data
        cursor.execute("""
            SELECT 
                signal,
                confidence,
                actual_price_at_prediction as entry_price,
                actual_price_after as exit_price,
                accuracy_score,
                predicted_direction,
                actual_direction
            FROM prediction_outcomes
            WHERE verified_at IS NOT NULL 
              AND actual_price_at_prediction > 0 
              AND actual_price_after > 0
        """)
        
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if len(predictions) < 5:
            return {
                'sufficient_data': False,
                'total_predictions': len(predictions),
                'message': f'Need at least 5 verified predictions, have {len(predictions)}'
            }
        
        # Classify signals
        def classify_signal(signal: str) -> str:
            signal_upper = signal.upper() if signal else ''
            if 'BUY' in signal_upper or signal_upper in ('OPPORTUNITY',):
                return 'BUY'
            elif 'SELL' in signal_upper or signal_upper in ('CAUTION',):
                return 'SELL'
            return 'HOLD'
        
        def get_confidence_bucket(conf: int) -> str:
            if conf >= 90: return '90-100'
            if conf >= 80: return '80-90'
            if conf >= 70: return '70-80'
            if conf >= 60: return '60-70'
            return '50-60'
        
        # Calculate returns and group by signal type
        signal_returns = {'BUY': [], 'SELL': [], 'HOLD': []}
        confidence_returns = {'50-60': [], '60-70': [], '70-80': [], '80-90': [], '90-100': []}
        
        for pred in predictions:
            entry = pred['entry_price']
            exit_price = pred['exit_price']
            pct_return = ((exit_price - entry) / entry) * 100
            
            signal_type = classify_signal(pred['signal'])
            signal_returns[signal_type].append(pct_return)
            
            bucket = get_confidence_bucket(pred['confidence'] or 50)
            confidence_returns[bucket].append(pct_return)
        
        # Compute stats per signal type
        def compute_stats(returns: list) -> Dict:
            if not returns:
                return {'count': 0, 'avg_return': 0, 'win_rate': 0, 'ev_per_trade': 0}
            
            wins = sum(1 for r in returns if r > 0)
            losses = len(returns) - wins
            avg_return = sum(returns) / len(returns)
            win_rate = (wins / len(returns)) * 100 if returns else 0
            avg_win = sum(r for r in returns if r > 0) / wins if wins else 0
            avg_loss = sum(r for r in returns if r <= 0) / losses if losses else 0
            
            return {
                'count': len(returns),
                'avg_return': round(avg_return, 2),
                'win_rate': round(win_rate, 1),
                'wins': wins,
                'losses': losses,
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'ev_per_trade': round(avg_return, 2),  # Expected value = avg return
            }
        
        by_signal = {k: compute_stats(v) for k, v in signal_returns.items()}
        by_confidence = {k: compute_stats(v) for k, v in confidence_returns.items()}
        
        # Find best performing bucket
        best_bucket = max(by_confidence.items(), key=lambda x: x[1]['avg_return'] if x[1]['count'] >= 3 else -999)
        best_signal = max(by_signal.items(), key=lambda x: x[1]['avg_return'] if x[1]['count'] >= 3 else -999)
        
        # Overall EV
        all_returns = [r for lst in signal_returns.values() for r in lst]
        overall_stats = compute_stats(all_returns)
        
        # Verdict
        verdict = "NEUTRAL"
        verdict_text = "Insufficient data or mixed results"
        
        if overall_stats['count'] >= 10:
            if overall_stats['avg_return'] > 2:
                verdict = "PROFITABLE"
                verdict_text = f"System generates +{overall_stats['avg_return']:.1f}% avg return per signal"
            elif overall_stats['avg_return'] < -2:
                verdict = "UNPROFITABLE"
                verdict_text = f"System loses {overall_stats['avg_return']:.1f}% avg per signal. Consider buying index ETFs."
            else:
                verdict = "BREAK-EVEN"
                verdict_text = f"System returns {overall_stats['avg_return']:+.1f}% avg, roughly break-even."
        
        return {
            'sufficient_data': True,
            'total_predictions': len(predictions),
            'by_signal': by_signal,
            'by_confidence': by_confidence,
            'overall': overall_stats,
            'best_signal_type': {'type': best_signal[0], **best_signal[1]} if best_signal[1]['count'] >= 3 else None,
            'best_confidence_bucket': {'bucket': best_bucket[0], **best_bucket[1]} if best_bucket[1]['count'] >= 3 else None,
            'verdict': verdict,
            'verdict_text': verdict_text,
        }
    
    def calculate_significance(self) -> Dict:
        """
        Calculate statistical significance of prediction accuracy.
        
        Answers: "Is this system actually better than random guessing?"
        Uses binomial test against null hypothesis of 50% accuracy.
        """
        import math
        
        conn = db._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN accuracy_score >= 0.5 THEN 1 ELSE 0 END) as successes
            FROM prediction_outcomes
            WHERE verified_at IS NOT NULL
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        n = row['total'] or 0
        k = row['successes'] or 0
        
        if n < 5:
            return {
                'sufficient_data': False,
                'total_predictions': n,
                'message': f'Need at least 5 verified predictions, have {n}',
            }
        
        observed_rate = k / n
        
        # Binomial test: P(getting k or more successes out of n if true rate = 0.5)
        # Using normal approximation for large n
        p0 = 0.5  # Null hypothesis: random guessing
        
        if n >= 30:
            # Normal approximation
            mean = n * p0
            std = math.sqrt(n * p0 * (1 - p0))
            z_score = (k - mean) / std if std > 0 else 0
            
            # One-tailed p-value (testing if better than random)
            # Using approximation: P(Z > z) ≈ 0.5 * erfc(z / sqrt(2))
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2))
        else:
            # Exact binomial for small samples
            # P(X >= k) for X ~ Binomial(n, 0.5)
            p_value = self._binomial_tail(n, k, p0)
        
        # 95% confidence interval for observed rate (Wilson score)
        ci_lower, ci_upper = self._wilson_ci(k, n, 0.95)
        
        # Interpretation
        if p_value < 0.01 and observed_rate > 0.5:
            significance = 'highly_significant'
            interpretation = f'Strong evidence system beats random (p={p_value:.4f})'
            confidence_level = 'high'
        elif p_value < 0.05 and observed_rate > 0.5:
            significance = 'significant'
            interpretation = f'Moderate evidence system beats random (p={p_value:.4f})'
            confidence_level = 'medium'
        elif p_value < 0.10 and observed_rate > 0.5:
            significance = 'marginally_significant'
            interpretation = f'Weak evidence system beats random (p={p_value:.4f})'
            confidence_level = 'low'
        elif observed_rate < 0.5:
            significance = 'worse_than_random'
            interpretation = f'System appears WORSE than random guessing ({observed_rate:.1%} vs 50%)'
            confidence_level = 'concerning'
        else:
            significance = 'not_significant'
            interpretation = f'Cannot distinguish from random (need more data or better edge)'
            confidence_level = 'uncertain'
        
        # Sample size needed for significance
        needed_for_sig = self._sample_size_needed(observed_rate, p0, 0.05)
        
        return {
            'sufficient_data': True,
            'total_predictions': n,
            'successes': k,
            'observed_rate': round(observed_rate, 4),
            'p_value': round(p_value, 4),
            'z_score': round(z_score if n >= 30 else 0, 2),
            
            'confidence_interval': {
                'lower': round(ci_lower, 4),
                'upper': round(ci_upper, 4),
                'level': '95%',
            },
            
            'significance': significance,
            'interpretation': interpretation,
            'confidence_level': confidence_level,
            
            'sample_size_analysis': {
                'current': n,
                'needed_for_significance': needed_for_sig,
                'progress_pct': round(min(100, n / needed_for_sig * 100), 1) if needed_for_sig > 0 else 100,
            },
            
            'verdict': self._get_significance_verdict(significance, observed_rate, n),
        }
    
    def _binomial_tail(self, n: int, k: int, p: float) -> float:
        """Calculate P(X >= k) for X ~ Binomial(n, p)."""
        import math
        
        def binomial_coef(n, k):
            if k < 0 or k > n:
                return 0
            return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        
        tail_prob = 0
        for i in range(k, n + 1):
            tail_prob += binomial_coef(n, i) * (p ** i) * ((1 - p) ** (n - i))
        
        return tail_prob
    
    def _wilson_ci(self, k: int, n: int, confidence: float = 0.95) -> tuple:
        """Wilson score confidence interval for proportions."""
        import math
        
        if n == 0:
            return (0, 1)
        
        p_hat = k / n
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
        
        return (max(0, center - spread), min(1, center + spread))
    
    def _sample_size_needed(self, observed_rate: float, null_rate: float, alpha: float = 0.05) -> int:
        """Estimate sample size needed to achieve significance at alpha level."""
        import math
        
        if observed_rate <= null_rate:
            return 1000  # Large number if we're not beating the null
        
        effect_size = observed_rate - null_rate
        z_alpha = 1.645  # One-tailed for alpha=0.05
        z_beta = 0.84    # 80% power
        
        # Sample size formula for proportions
        p_avg = (observed_rate + null_rate) / 2
        n = ((z_alpha + z_beta) ** 2 * 2 * p_avg * (1 - p_avg)) / (effect_size ** 2)
        
        return max(int(math.ceil(n)), 20)
    
    def _get_significance_verdict(self, significance: str, rate: float, n: int) -> str:
        """Get human-readable verdict for dashboard display."""
        if significance == 'highly_significant':
            return f"✅ System has proven edge ({rate:.0%} accuracy over {n} trades)"
        elif significance == 'significant':
            return f"✅ System likely has edge ({rate:.0%} accuracy, keep validating)"
        elif significance == 'marginally_significant':
            return f"⚠️ Possible edge but needs more data ({rate:.0%} over {n} trades)"
        elif significance == 'worse_than_random':
            return f"🛑 System underperforms random guessing ({rate:.0%}) — review strategy"
        else:
            return f"⏳ Insufficient evidence yet ({n} trades, need more data)"


    def calculate_calibration(self) -> Dict:
        """Calculate calibration curve data: predicted confidence vs actual hit rate.

        Returns bucket data for calibration chart + Brier score.
        A well-calibrated system has predicted confidence ≈ actual accuracy.
        """
        conn = db._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT confidence, accuracy_score
            FROM prediction_outcomes
            WHERE verified_at IS NOT NULL
              AND confidence IS NOT NULL
              AND accuracy_score IS NOT NULL
        """)
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if len(predictions) < 5:
            return {
                'sufficient_data': False,
                'total': len(predictions),
                'message': f'Need at least 5 verified predictions, have {len(predictions)}'
            }

        # Define buckets
        bucket_ranges = [
            (0, 40, '0-40'),
            (40, 50, '40-50'),
            (50, 60, '50-60'),
            (60, 70, '60-70'),
            (70, 80, '70-80'),
            (80, 90, '80-90'),
            (90, 101, '90-100'),
        ]

        buckets = []
        brier_sum = 0.0

        for low, high, label in bucket_ranges:
            in_bucket = [p for p in predictions if low <= (p['confidence'] or 0) < high]
            if not in_bucket:
                continue

            count = len(in_bucket)
            midpoint = (low + min(high, 100)) / 2
            predicted_accuracy = midpoint / 100  # e.g., 0.75 for 70-80 bucket
            actual_accuracy = sum(p['accuracy_score'] for p in in_bucket) / count
            hit_rate = sum(1 for p in in_bucket if p['accuracy_score'] >= 0.5) / count

            buckets.append({
                'range': label,
                'count': count,
                'predicted_accuracy': round(predicted_accuracy, 3),
                'actual_accuracy': round(actual_accuracy, 3),
                'hit_rate': round(hit_rate * 100, 1),
                'gap': round((actual_accuracy - predicted_accuracy) * 100, 1),
            })

        # Brier Score: mean of (forecast_probability - outcome)^2
        # Lower is better. 0 = perfect, 0.25 = random coin flip
        for p in predictions:
            forecast = (p['confidence'] or 50) / 100
            outcome = p['accuracy_score']
            brier_sum += (forecast - outcome) ** 2

        brier_score = round(brier_sum / len(predictions), 4)

        # Mean Absolute Calibration Error (average gap across buckets)
        if buckets:
            calibration_error = round(
                sum(abs(b['actual_accuracy'] - b['predicted_accuracy']) for b in buckets) / len(buckets) * 100, 1
            )
        else:
            calibration_error = 0

        # Interpretation
        if brier_score < 0.15:
            interpretation = 'Well-calibrated. Confidence scores are meaningful.'
        elif brier_score < 0.25:
            interpretation = 'Moderately calibrated. Some overconfidence or underconfidence.'
        else:
            interpretation = 'Poorly calibrated. Confidence scores are unreliable.'

        return {
            'sufficient_data': True,
            'total': len(predictions),
            'buckets': buckets,
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'interpretation': interpretation,
        }

    def calculate_signal_decay(self) -> Dict:
        """Calculate signal accuracy at multiple time horizons.

        Shows whether signals are best acted on immediately or have lasting
        predictive power. Checks returns at 1d, 3d, 7d, 14d, 30d windows.
        """
        conn = db._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker, prediction_date, predicted_direction, confidence,
                   actual_price_at_prediction
            FROM prediction_outcomes
            WHERE actual_price_at_prediction IS NOT NULL
              AND actual_price_at_prediction > 0
              AND prediction_date > datetime('now', '-180 days')
            ORDER BY ticker, prediction_date
        """)
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if len(predictions) < 5:
            return {
                'sufficient_data': False,
                'total': len(predictions),
                'message': f'Need at least 5 predictions with price data, have {len(predictions)}'
            }

        # Group by ticker for batch fetching
        by_ticker = {}
        for p in predictions:
            by_ticker.setdefault(p['ticker'], []).append(p)

        windows = [1, 3, 7, 14, 30]
        window_results = {w: {'hits': 0, 'total': 0, 'returns': []} for w in windows}

        for ticker, preds in by_ticker.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="8mo")
                if hist.empty or len(hist) < 5:
                    continue

                # Build date->price index
                price_map = {}
                for idx, row in hist.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    price_map[date_str] = float(row['Close'])

                # Sorted dates for lookups
                sorted_dates = sorted(price_map.keys())

                for pred in preds:
                    pred_date = pred['prediction_date'][:10]
                    entry_price = pred['actual_price_at_prediction']
                    direction = pred['predicted_direction']

                    if entry_price <= 0:
                        continue

                    for w in windows:
                        # Find the trading day closest to pred_date + w days
                        from datetime import datetime as dt, timedelta as td
                        target = dt.strptime(pred_date, '%Y-%m-%d') + td(days=w)
                        target_str = target.strftime('%Y-%m-%d')

                        # Find nearest available date
                        future_price = None
                        for d in sorted_dates:
                            if d >= target_str:
                                future_price = price_map[d]
                                break

                        if future_price is None:
                            continue

                        pct_return = ((future_price - entry_price) / entry_price) * 100

                        # Check if prediction was correct
                        if direction == 'up':
                            hit = 1 if pct_return > 2 else (0.5 if abs(pct_return) <= 2 else 0)
                        elif direction == 'down':
                            hit = 1 if pct_return < -2 else (0.5 if abs(pct_return) <= 2 else 0)
                        else:
                            hit = 0.5 if abs(pct_return) <= 2 else 0

                        window_results[w]['hits'] += hit
                        window_results[w]['total'] += 1
                        window_results[w]['returns'].append(pct_return)
            except Exception:
                continue

        # Build output
        decay_data = []
        for w in windows:
            wr = window_results[w]
            if wr['total'] == 0:
                continue
            accuracy = round(wr['hits'] / wr['total'], 3)
            avg_return = round(sum(wr['returns']) / len(wr['returns']), 2) if wr['returns'] else 0
            decay_data.append({
                'window_days': w,
                'label': f'{w}d',
                'accuracy': accuracy,
                'accuracy_pct': round(accuracy * 100, 1),
                'avg_return': avg_return,
                'count': wr['total'],
            })

        if not decay_data:
            return {
                'sufficient_data': False,
                'total': len(predictions),
                'message': 'Could not compute returns for any prediction window'
            }

        # Find peak accuracy window
        peak = max(decay_data, key=lambda d: d['accuracy'])
        worst = min(decay_data, key=lambda d: d['accuracy'])

        # Detect if there's decay
        if len(decay_data) >= 3:
            early_acc = decay_data[0]['accuracy'] if decay_data else 0
            late_acc = decay_data[-1]['accuracy'] if decay_data else 0
            if early_acc > late_acc + 0.05:
                pattern = 'decaying'
                interpretation = f'Signals are most accurate at {peak["label"]} and lose edge over time. Act fast.'
            elif late_acc > early_acc + 0.05:
                pattern = 'improving'
                interpretation = f'Signals improve with time — best at {peak["label"]}. These are slow-burn picks.'
            else:
                pattern = 'stable'
                interpretation = f'Signal accuracy is stable across timeframes. Peak at {peak["label"]}.'
        else:
            pattern = 'insufficient'
            interpretation = 'Not enough data points to determine decay pattern.'

        return {
            'sufficient_data': True,
            'total_predictions': len(predictions),
            'windows': decay_data,
            'peak_window': peak['label'],
            'peak_accuracy': peak['accuracy_pct'],
            'pattern': pattern,
            'interpretation': interpretation,
        }

    def calculate_ab_comparison(self) -> Dict:
        """Compare accuracy of quant-only vs quant+AI predictions.

        Answers: Does the AI layer actually improve signal accuracy?
        """
        conn = db._get_conn()
        cursor = conn.cursor()

        # Check if has_ai column exists
        try:
            cursor.execute("""
                SELECT
                    has_ai,
                    COUNT(*) as count,
                    AVG(accuracy_score) as avg_accuracy,
                    SUM(CASE WHEN accuracy_score >= 0.5 THEN 1 ELSE 0 END) as wins
                FROM prediction_outcomes
                WHERE verified_at IS NOT NULL AND has_ai IS NOT NULL
                GROUP BY has_ai
            """)
            rows = [dict(row) for row in cursor.fetchall()]
        except Exception:
            conn.close()
            return {
                'sufficient_data': False,
                'message': 'A/B tracking not yet active. New predictions will be tagged automatically.'
            }

        conn.close()

        quant_only = next((r for r in rows if r['has_ai'] == 0), None)
        quant_ai = next((r for r in rows if r['has_ai'] == 1), None)

        def format_group(row):
            if not row or row['count'] == 0:
                return {'count': 0, 'accuracy': 0, 'win_rate': 0}
            return {
                'count': row['count'],
                'accuracy': round(row['avg_accuracy'], 3),
                'win_rate': round(row['wins'] / row['count'] * 100, 1),
            }

        qo = format_group(quant_only)
        qa = format_group(quant_ai)

        total = qo['count'] + qa['count']
        if total < 10:
            return {
                'sufficient_data': False,
                'quant_only': qo,
                'quant_ai': qa,
                'total': total,
                'message': f'Need at least 10 verified A/B predictions, have {total}'
            }

        # Determine verdict
        diff = qa['accuracy'] - qo['accuracy']
        if qo['count'] < 5 or qa['count'] < 5:
            verdict = 'insufficient_data'
            verdict_text = 'Need more data in both groups for comparison'
        elif diff > 0.05:
            verdict = 'ai_adds_value'
            verdict_text = f'AI improves accuracy by +{diff:.0%}'
        elif diff < -0.05:
            verdict = 'ai_adds_noise'
            verdict_text = f'AI reduces accuracy by {diff:.0%}. Consider quant-only mode.'
        else:
            verdict = 'no_difference'
            verdict_text = 'No meaningful difference between quant-only and quant+AI'

        return {
            'sufficient_data': True,
            'quant_only': qo,
            'quant_ai': qa,
            'total': total,
            'difference': round(diff, 3),
            'verdict': verdict,
            'verdict_text': verdict_text,
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
        self._ensure_weight_tables()
    
    def _ensure_weight_tables(self):
        """Create weight versioning table."""
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weight_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                old_weights TEXT NOT NULL,
                new_weights TEXT NOT NULL,
                reason TEXT,
                trigger TEXT NOT NULL,
                accuracy_before REAL,
                accuracy_after REAL,
                backtest_run_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def _log_weight_change(self, old_weights: Dict, new_weights: Dict,
                           trigger: str, reason: str = None,
                           accuracy_before: float = None,
                           accuracy_after: float = None,
                           backtest_run_id: int = None):
        """Log a weight change to the audit trail."""
        db.execute("""
            INSERT INTO weight_versions
            (old_weights, new_weights, trigger, reason, accuracy_before, accuracy_after, backtest_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (json.dumps(old_weights), json.dumps(new_weights), trigger, reason,
              accuracy_before, accuracy_after, backtest_run_id))

    def get_weight_history(self, limit: int = 20) -> List[Dict]:
        """Get weight change audit trail."""
        rows = db.query("""
            SELECT * FROM weight_versions
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        for row in rows:
            try:
                row['old_weights'] = json.loads(row['old_weights']) if isinstance(row['old_weights'], str) else row['old_weights']
                row['new_weights'] = json.loads(row['new_weights']) if isinstance(row['new_weights'], str) else row['new_weights']
            except (json.JSONDecodeError, TypeError):
                pass
        return rows

    def rollback_weights(self, version_id: int) -> Dict:
        """Rollback to a previous weight version."""
        row = db.query_one("SELECT * FROM weight_versions WHERE id = ?", (version_id,))
        if not row:
            return {'success': False, 'message': f'Version {version_id} not found'}

        try:
            target_weights = json.loads(row['old_weights']) if isinstance(row['old_weights'], str) else row['old_weights']
        except (json.JSONDecodeError, TypeError):
            return {'success': False, 'message': 'Could not parse weights from version'}

        # Get current weights for logging
        from engine.quant_screener import quant_screener
        current_weights = dict(quant_screener.config['composite_weights'])

        # Apply the old weights as a 2-factor override
        tech_w = target_weights.get('technical', 0.25) + target_weights.get('momentum', 0.25)
        mom_w = target_weights.get('momentum', 0.25)

        db.set_setting('quant_weights_override', {
            'tech_weight': round(tech_w, 4),
            'momentum_weight': round(mom_w, 4),
        })
        quant_screener.reload_weights()

        # Log the rollback
        self._log_weight_change(
            current_weights, target_weights,
            trigger='rollback',
            reason=f'Rollback to version #{version_id}'
        )

        return {
            'success': True,
            'message': f'Rolled back to version #{version_id}',
            'restored_weights': target_weights,
        }

    def optimize_daily_cycle(self, tickers: List[str]) -> List[str]:
        """
        Optimize ticker list for daily cycle:
        1. Prioritize based on learning
        2. Filter out recently analyzed (unless high priority)
        3. Return optimized order
        """
        prioritized = self.prioritizer.prioritize_watchlist(tickers)
        return [p['ticker'] for p in prioritized]
    
    def record_and_learn(self, ticker: str, signal: str, confidence: int, has_ai: bool = False):
        """Record prediction and trigger learning if enough data"""
        # Get current price for later verification
        stock_data = self.cache.get_stock_data(ticker)
        current_price = stock_data.get('current_price', 0) if stock_data else 0

        # Record the prediction
        self.feedback.record_prediction(ticker, signal, confidence, current_price, has_ai=has_ai)

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
            # Log the weight change before applying
            self._log_weight_change(
                suggestion['current_weights'],
                suggestion['suggested_weights'],
                trigger='auto_adjust',
                reason='Learning optimizer auto-adjustment based on prediction accuracy',
            )

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
