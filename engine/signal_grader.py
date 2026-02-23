"""
Signal Performance Feedback Loop
Grades past AI/Quant signals based on actual 30/60/90 day forward returns.
Provides self-tuning weight adjustments based on accuracy.
"""
import yfinance as yf
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any

from core.database import db

logger = logging.getLogger(__name__)


class SignalGrader:
    def __init__(self):
        self._init_table()

    def _init_table(self):
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS signal_grades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    ticker TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence INTEGER,
                    signal_date TEXT NOT NULL,
                    price_at_signal REAL,
                    price_30d REAL,
                    price_60d REAL,
                    price_90d REAL,
                    return_30d REAL,
                    return_60d REAL,
                    return_90d REAL,
                    grade TEXT,
                    graded_at TEXT,
                    UNIQUE(analysis_id)
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create signal_grades table: {e}")

    def _sync_new_signals(self):
        """Import new analyses that are not yet tracked."""
        try:
            db.execute("""
                INSERT OR IGNORE INTO signal_grades (analysis_id, ticker, signal, confidence, signal_date, grade)
                SELECT id, ticker, signal, confidence, timestamp, 'pending'
                FROM analysis_history
                WHERE signal IN ('STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL')
                  AND confidence IS NOT NULL
            """)
        except Exception as e:
            logger.error(f"Failed to sync signals for grading: {e}")

    def grade_pending_signals(self) -> int:
        """Find pending signals older than 30 days and grade them."""
        self._sync_new_signals()
        
        # We process 'pending' signals older than 30 days, 
        # or 'partially_correct' ones that might get a 60/90d update.
        cutoff_30d = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        
        records = db.query("""
            SELECT * FROM signal_grades
            WHERE signal_date <= ? AND (grade = 'pending' OR grade = 'incorrect' OR grade = 'partially_correct' OR price_90d IS NULL)
            LIMIT 50
        """, (cutoff_30d,))
        
        if not records:
            return 0
            
        count = 0
        
        for rec in records:
            ticker = rec['ticker']
            signal_date_str = rec['signal_date']
            try:
                sig_dt = datetime.fromisoformat(signal_date_str)
            except ValueError:
                sig_dt = datetime.strptime(signal_date_str, '%Y-%m-%d %H:%M:%S')
                
            try:
                # Need at least 6 months of data to cover 90d checks plus some padding
                hist = yf.Ticker(ticker).history(period="6mo")
                if hist.empty:
                    continue
                    
                # Ensure index is timezone-naive for comparison
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_convert(None)
                    
                # Helper to find close price near a target date
                def get_price_near(target_dt, max_days_diff=5):
                    # Filter history to dates >= target_dt
                    after = hist[hist.index >= target_dt]
                    if not after.empty:
                        # Check if the closest date is within max_days_diff
                        if (after.index[0] - target_dt).days <= max_days_diff:
                            return float(after.iloc[0]['Close'])
                    return None

                price_at_signal = rec['price_at_signal'] or get_price_near(sig_dt)
                if not price_at_signal:
                    continue
                    
                dt_30d = sig_dt + timedelta(days=30)
                dt_60d = sig_dt + timedelta(days=60)
                dt_90d = sig_dt + timedelta(days=90)
                
                price_30d = rec['price_30d'] or get_price_near(dt_30d)
                price_60d = rec['price_60d'] or get_price_near(dt_60d)
                price_90d = rec['price_90d'] or get_price_near(dt_90d)
                
                ret_30d = ((price_30d - price_at_signal) / price_at_signal) if price_30d else None
                ret_60d = ((price_60d - price_at_signal) / price_at_signal) if price_60d else None
                ret_90d = ((price_90d - price_at_signal) / price_at_signal) if price_90d else None
                
                # Grading Logic
                grade = 'pending'
                is_bullish = rec['signal'] in ['STRONG_BUY', 'BUY']
                is_bearish = rec['signal'] in ['STRONG_SELL', 'SELL']
                
                if ret_30d is not None:
                    if is_bullish:
                        if ret_30d > 0.02: # +2%
                            grade = 'correct'
                        elif ret_60d and ret_60d > 0.04:
                            grade = 'partially_correct'
                        elif ret_90d and ret_90d > 0.05:
                            grade = 'partially_correct'
                        else:
                            grade = 'incorrect'
                    elif is_bearish:
                        if ret_30d < -0.02:
                            grade = 'correct'
                        elif ret_60d and ret_60d < -0.04:
                            grade = 'partially_correct'
                        elif ret_90d and ret_90d < -0.05:
                            grade = 'partially_correct'
                        else:
                            grade = 'incorrect'
                    else:
                        # HOLD
                        if abs(ret_30d) <= 0.05:
                            grade = 'correct'
                        else:
                            grade = 'incorrect'

                db.execute("""
                    UPDATE signal_grades SET
                        price_at_signal = ?, price_30d = ?, price_60d = ?, price_90d = ?,
                        return_30d = ?, return_60d = ?, return_90d = ?,
                        grade = ?, graded_at = ?
                    WHERE id = ?
                """, (
                    price_at_signal, price_30d, price_60d, price_90d,
                    ret_30d, ret_60d, ret_90d,
                    grade, datetime.now().isoformat(),
                    rec['id']
                ))
                
                if grade != rec['grade']:
                    count += 1
            except Exception as e:
                logger.debug(f"Error grading signal {rec['id']} for {ticker}: {e}")
                
        return count

    def get_accuracy_by_signal(self) -> Dict[str, Any]:
        """Return accuracy breakdown by signal type."""
        rows = db.query("""
            SELECT signal, 
                   COUNT(*) as total,
                   SUM(CASE WHEN grade = 'correct' THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN grade = 'partially_correct' THEN 1 ELSE 0 END) as partial
            FROM signal_grades
            WHERE grade IN ('correct', 'partially_correct', 'incorrect')
            GROUP BY signal
        """)
        
        result = {}
        for r in rows:
            tot = r['total']
            cor = r['correct']
            par = r['partial']
            # Partial correct counts as 0.5
            weighted_correct = cor + (par * 0.5)
            acc = (weighted_correct / tot * 100) if tot > 0 else 0
            
            result[r['signal']] = {
                "total": tot,
                "correct": cor,
                "partial": par,
                "accuracy_pct": round(acc, 1)
            }
        
        return result

    def get_accuracy_by_month(self) -> List[Dict[str, Any]]:
        """Monthly accuracy trend for the last 12 months."""
        rows = db.query("""
            SELECT strftime('%Y-%m', signal_date) as month,
                   COUNT(*) as total,
                   SUM(CASE WHEN grade = 'correct' THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN grade = 'partially_correct' THEN 1 ELSE 0 END) as partial
            FROM signal_grades
            WHERE grade IN ('correct', 'partially_correct', 'incorrect')
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """)
        
        trend = []
        for r in reversed(rows):
            tot = r['total']
            weighted_correct = r['correct'] + (r['partial'] * 0.5)
            acc = (weighted_correct / tot * 100) if tot > 0 else 0
            trend.append({
                "month": r['month'],
                "total": tot,
                "accuracy_pct": round(acc, 1)
            })
            
        return trend

    def get_weight_recommendations(self) -> Dict[str, Any]:
        """Suggest weight adjustments based on accuracy."""
        # A rudimentary analysis: in a full scale app, we'd join fundamental_snapshots and see which factors correlate 
        # with 'correct' vs 'incorrect' outcomes. Here we produce safe, conservative adjustments based on global accuracy.
        acc = self.get_accuracy_by_signal()
        strong_buy_acc = acc.get('STRONG_BUY', {}).get('accuracy_pct', 50)
        
        recommendations = {
            "valuation": "neutral",
            "technical": "neutral",
            "momentum": "neutral",
            "quality": "neutral"
        }
        
        # If accuracy is poor, suggest relying more on valuation and quality than technicals
        if strong_buy_acc < 50:
            recommendations["valuation"] = "increase"
            recommendations["quality"] = "increase"
            recommendations["technical"] = "decrease"
        elif strong_buy_acc > 65:
            # If accuracy is great, we might want to scale up momentum riding
            recommendations["momentum"] = "increase"
            
        return recommendations

    def auto_tune_weights(self) -> Dict[str, Any]:
        """Automatically update quant_weights_override in DB settings if data supports it."""
        try:
            total_graded = db.query_one("SELECT COUNT(*) as c FROM signal_grades WHERE grade IN ('correct', 'incorrect')")['c']
            if total_graded < 50:
                return {"tuned": False, "reason": "Insufficient graded signals (<50)"}
                
            recs = self.get_weight_recommendations()
            
            # Fetch current overrides or defaults
            current_overrides = db.get_setting('quant_weights_override') or {}
            tech_w = current_overrides.get('tech_weight', 0.5)
            mom_w = current_overrides.get('momentum_weight', 0.5)
            
            changed = False
            # Conservative ±5% adjustments (0.05) per cycle
            if recs["technical"] == "increase" and tech_w < 0.8:
                tech_w += 0.05
                changed = True
            elif recs["technical"] == "decrease" and tech_w > 0.2:
                tech_w -= 0.05
                changed = True
                
            if recs["momentum"] == "increase" and mom_w < 0.8:
                mom_w += 0.05
                changed = True
            elif recs["momentum"] == "decrease" and mom_w > 0.2:
                mom_w -= 0.05
                changed = True
                
            if changed:
                new_overrides = {
                    "tech_weight": round(tech_w, 2),
                    "momentum_weight": round(mom_w, 2)
                }
                db.set_setting('quant_weights_override', new_overrides)
                
                msg = f"Tuned factor weights automatically. Tech={new_overrides['tech_weight']}, Mom={new_overrides['momentum_weight']}"
                logger.info(msg)
                
                try:
                    from engine.quant_screener import quant_screener
                    quant_screener.reload_weights()
                except Exception:
                    pass
                
                return {"tuned": True, "weights": new_overrides, "message": msg}
                
            return {"tuned": False, "reason": "No meaningful adjustments required"}
        except Exception as e:
            logger.error(f"Auto tune failed: {e}")
            return {"tuned": False, "error": str(e)}

    def get_monthly_self_report(self) -> Dict[str, Any]:
        """Return summary dict: signals this month, accuracy, best/worst performing signal, weight changes applied."""
        current_month = datetime.now().strftime('%Y-%m')
        
        month_rows = db.query("""
            SELECT signal, grade, return_30d
            FROM signal_grades
            WHERE strftime('%Y-%m', signal_date) = ? AND grade IN ('correct', 'incorrect', 'partially_correct')
        """, (current_month,))
        
        signals_graded = len(month_rows)
        correct = sum(1 for r in month_rows if r['grade'] == 'correct')
        acc = (correct / signals_graded * 100) if signals_graded > 0 else 0
        
        best_sig = None
        worst_sig = None
        
        if signals_graded > 0:
            returns = [r for r in month_rows if r['return_30d'] is not None]
            if returns:
                returns.sort(key=lambda x: x['return_30d'])
                worst_sig = returns[0]['return_30d']
                best_sig = returns[-1]['return_30d']
                
        return {
            "month": current_month,
            "signals_graded": signals_graded,
            "accuracy_pct": round(acc, 1),
            "best_return_30d_pct": round(best_sig * 100, 2) if best_sig else None,
            "worst_return_30d_pct": round(worst_sig * 100, 2) if worst_sig else None
        }


# Singleton
signal_grader = SignalGrader()
