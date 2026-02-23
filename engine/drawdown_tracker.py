"""
Drawdown Tracker Module
Tracks drawdowns, recovery periods, and worst-case scenarios.
The goal: Force confrontation with reality before risking real money.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from core.database import db

logger = logging.getLogger(__name__)


class DrawdownTracker:
    """
    Tracks and analyzes drawdowns for both paper trading and backtests.
    
    Key metrics:
    - Max drawdown (worst peak-to-trough decline)
    - Current drawdown (are we in a drawdown right now?)
    - Recovery time (how long to recover from worst drawdown)
    - Underwater periods (how much time spent below previous highs)
    """
    
    def __init__(self):
        pass
    
    def analyze_equity_curve(self, snapshots: List[Dict]) -> Dict:
        """
        Analyze an equity curve for drawdown metrics.
        
        Args:
            snapshots: List of {'date': str, 'value': float}
        
        Returns:
            Comprehensive drawdown analysis
        """
        if not snapshots or len(snapshots) < 2:
            return {
                'sufficient_data': False,
                'message': 'Need at least 2 data points',
            }
        
        # Sort by date
        sorted_snaps = sorted(snapshots, key=lambda x: x['date'])
        values = [s['value'] for s in sorted_snaps]
        dates = [s['date'] for s in sorted_snaps]
        
        # Calculate running maximum (peak)
        running_max = []
        peak = values[0]
        for v in values:
            peak = max(peak, v)
            running_max.append(peak)
        
        # Calculate drawdowns
        drawdowns = []
        for i, (value, peak) in enumerate(zip(values, running_max)):
            dd = ((peak - value) / peak) * 100 if peak > 0 else 0
            drawdowns.append({
                'date': dates[i],
                'value': value,
                'peak': peak,
                'drawdown_pct': round(dd, 2),
            })
        
        # Find max drawdown
        max_dd = max(d['drawdown_pct'] for d in drawdowns)
        max_dd_idx = next(i for i, d in enumerate(drawdowns) if d['drawdown_pct'] == max_dd)
        max_dd_date = drawdowns[max_dd_idx]['date']
        
        # Find when max drawdown started (last peak before)
        dd_start_idx = max_dd_idx
        for i in range(max_dd_idx, -1, -1):
            if drawdowns[i]['drawdown_pct'] == 0:
                dd_start_idx = i
                break
        
        # Find recovery (next time drawdown hit 0 after max)
        recovery_idx = None
        for i in range(max_dd_idx, len(drawdowns)):
            if drawdowns[i]['drawdown_pct'] == 0:
                recovery_idx = i
                break
        
        # Current drawdown
        current_dd = drawdowns[-1]['drawdown_pct']
        
        # Underwater analysis
        underwater_days = sum(1 for d in drawdowns if d['drawdown_pct'] > 0)
        underwater_pct = (underwater_days / len(drawdowns)) * 100
        
        # Worst drawdown periods (top 3)
        dd_periods = self._find_drawdown_periods(drawdowns)
        
        return {
            'sufficient_data': True,
            'data_points': len(values),
            'start_date': dates[0],
            'end_date': dates[-1],
            
            'max_drawdown': {
                'pct': round(max_dd, 2),
                'date': max_dd_date,
                'peak_value': drawdowns[dd_start_idx]['peak'] if dd_start_idx < len(drawdowns) else values[0],
                'trough_value': drawdowns[max_dd_idx]['value'],
                'recovered': recovery_idx is not None,
                'recovery_days': recovery_idx - max_dd_idx if recovery_idx else None,
            },
            
            'current_drawdown': {
                'pct': round(current_dd, 2),
                'in_drawdown': current_dd > 1.0,  # Consider >1% as "in drawdown"
                'peak_value': running_max[-1],
                'current_value': values[-1],
            },
            
            'underwater_analysis': {
                'total_days': len(drawdowns),
                'underwater_days': underwater_days,
                'underwater_pct': round(underwater_pct, 1),
                'longest_underwater': self._longest_underwater(drawdowns),
            },
            
            'worst_periods': dd_periods[:3],
            
            'risk_metrics': {
                'max_dd': round(max_dd, 2),
                'avg_drawdown': round(sum(d['drawdown_pct'] for d in drawdowns) / len(drawdowns), 2),
                'current_dd': round(current_dd, 2),
            },
            
            'reality_check': self._get_reality_check(max_dd, current_dd, underwater_pct),
        }
    
    def _find_drawdown_periods(self, drawdowns: List[Dict]) -> List[Dict]:
        """Find distinct drawdown periods and their characteristics."""
        periods = []
        in_drawdown = False
        period_start = None
        period_max = 0
        
        for i, d in enumerate(drawdowns):
            if d['drawdown_pct'] > 0 and not in_drawdown:
                # Starting new drawdown
                in_drawdown = True
                period_start = i
                period_max = d['drawdown_pct']
            elif d['drawdown_pct'] > 0 and in_drawdown:
                # Continuing drawdown
                period_max = max(period_max, d['drawdown_pct'])
            elif d['drawdown_pct'] == 0 and in_drawdown:
                # Ending drawdown
                in_drawdown = False
                periods.append({
                    'start_date': drawdowns[period_start]['date'],
                    'end_date': drawdowns[i]['date'],
                    'max_dd_pct': round(period_max, 2),
                    'duration_days': i - period_start,
                })
        
        # If still in drawdown at end
        if in_drawdown:
            periods.append({
                'start_date': drawdowns[period_start]['date'],
                'end_date': drawdowns[-1]['date'],
                'max_dd_pct': round(period_max, 2),
                'duration_days': len(drawdowns) - period_start,
                'ongoing': True,
            })
        
        # Sort by severity
        return sorted(periods, key=lambda x: x['max_dd_pct'], reverse=True)
    
    def _longest_underwater(self, drawdowns: List[Dict]) -> Dict:
        """Find the longest underwater period."""
        max_streak = 0
        current_streak = 0
        streak_start = None
        max_streak_start = None
        max_streak_end = None
        
        for i, d in enumerate(drawdowns):
            if d['drawdown_pct'] > 0:
                if current_streak == 0:
                    streak_start = i
                current_streak += 1
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_start = streak_start
                    max_streak_end = i - 1
                current_streak = 0
        
        # Check if we ended in a streak
        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_start = streak_start
            max_streak_end = len(drawdowns) - 1
        
        if max_streak == 0:
            return {'days': 0, 'message': 'Never underwater'}
        
        return {
            'days': max_streak,
            'start_date': drawdowns[max_streak_start]['date'] if max_streak_start else None,
            'end_date': drawdowns[max_streak_end]['date'] if max_streak_end else None,
        }
    
    def _get_reality_check(self, max_dd: float, current_dd: float, underwater_pct: float) -> Dict:
        """Generate reality check messages based on metrics."""
        warnings = []
        severity = 'ok'
        
        if max_dd >= 30:
            warnings.append(f"🛑 Max drawdown of {max_dd:.1f}% would test any investor's resolve")
            severity = 'severe'
        elif max_dd >= 20:
            warnings.append(f"⚠️ Max drawdown of {max_dd:.1f}% is significant — could you hold through this?")
            severity = 'concerning'
        elif max_dd >= 10:
            warnings.append(f"📉 Max drawdown of {max_dd:.1f}% is moderate but noticeable")
            severity = 'moderate'
        
        if current_dd >= 10:
            warnings.append(f"📍 Currently in {current_dd:.1f}% drawdown — recovery needed")
            severity = max(severity, 'concerning') if severity != 'severe' else severity
        
        if underwater_pct >= 50:
            warnings.append(f"⏱️ Spent {underwater_pct:.0f}% of time below previous highs")
        
        if not warnings:
            warnings.append("✅ Drawdown metrics look healthy")
        
        return {
            'severity': severity,
            'warnings': warnings,
            'would_you_hold': self._would_you_hold_message(max_dd),
        }
    
    def _would_you_hold_message(self, max_dd: float) -> str:
        """Generate message about holding through drawdowns."""
        if max_dd >= 40:
            return "A 40%+ drawdown means watching $100K become $60K. Most people sell at the bottom."
        elif max_dd >= 30:
            return "A 30% drawdown takes a 43% gain to recover. Can you wait that long?"
        elif max_dd >= 20:
            return "A 20% drawdown means every $5 invested became $4. This tests your conviction."
        elif max_dd >= 10:
            return "A 10% drawdown is common but still uncomfortable. Know your limits."
        else:
            return "Drawdowns under 10% are typical. Stay disciplined."
    
    def get_paper_trading_drawdown(self) -> Dict:
        """Get drawdown analysis for paper trading portfolio."""
        try:
            conn = db._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT snapshot_date, portfolio_value 
                FROM paper_snapshots 
                ORDER BY snapshot_date
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {'sufficient_data': False, 'message': 'No paper trading snapshots'}
            
            snapshots = [
                {'date': row['snapshot_date'], 'value': row['portfolio_value']}
                for row in rows
            ]
            
            return self.analyze_equity_curve(snapshots)
            
        except Exception as e:
            logger.warning(f"Error getting paper trading drawdown: {e}")
            return {'sufficient_data': False, 'error': str(e)}
    
    def get_benchmark_comparison(self, period_days: int = 365) -> Dict:
        """Compare drawdowns to SPY benchmark over same period."""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period=f"{period_days}d")
            
            if hist.empty:
                return {'error': 'Could not fetch SPY data'}
            
            snapshots = [
                {'date': date.strftime('%Y-%m-%d'), 'value': float(row['Close'])}
                for date, row in hist.iterrows()
            ]
            
            analysis = self.analyze_equity_curve(snapshots)
            analysis['ticker'] = 'SPY'
            analysis['period'] = f"{period_days} days"
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error getting SPY benchmark: {e}")
            return {'error': str(e)}
    
    def get_reality_dashboard(self) -> Dict:
        """
        Get comprehensive reality check for dashboard display.
        Combines paper trading and benchmark drawdowns.
        """
        paper = self.get_paper_trading_drawdown()
        spy = self.get_benchmark_comparison(90)  # 90-day comparison
        
        result = {
            'paper_trading': paper if paper.get('sufficient_data') else None,
            'spy_benchmark': spy if not spy.get('error') else None,
            'generated_at': datetime.now().isoformat(),
        }
        
        # Overall reality check
        if paper.get('sufficient_data'):
            paper_max_dd = paper.get('max_drawdown', {}).get('pct', 0)
            spy_max_dd = spy.get('max_drawdown', {}).get('pct', 0) if spy.get('sufficient_data') else 0
            
            if paper_max_dd < spy_max_dd * 0.8:
                result['comparison'] = f"📈 Your drawdown ({paper_max_dd:.1f}%) is less severe than SPY ({spy_max_dd:.1f}%)"
            elif paper_max_dd > spy_max_dd * 1.5:
                result['comparison'] = f"📉 Your drawdown ({paper_max_dd:.1f}%) is worse than SPY ({spy_max_dd:.1f}%)"
            else:
                result['comparison'] = f"↔️ Your drawdown ({paper_max_dd:.1f}%) is similar to SPY ({spy_max_dd:.1f}%)"
        
        return result


# Singleton
drawdown_tracker = DrawdownTracker()
