"""
Multi-Timeframe Confirmation Module
Adds weekly/monthly trend overlay to validate daily signals.
Higher timeframe confirmation increases signal reliability.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiTimeframe:
    """
    Analyzes multiple timeframes to confirm or reject signals.
    
    Philosophy: A daily buy signal is more reliable when weekly and monthly
    trends are also bullish (aligned). Divergence = caution.
    """
    
    def __init__(self, cache_minutes: int = 15):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=cache_minutes)
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        entry = self._cache[key]
        return datetime.now() - entry['timestamp'] < self._cache_duration
    
    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Analyze a ticker across daily, weekly, and monthly timeframes.
        Returns trend for each timeframe and alignment status.
        """
        cache_key = f"mtf_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data']
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get enough history for monthly analysis
            hist = stock.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                return {
                    'ticker': ticker,
                    'error': 'Insufficient data',
                    'daily': None,
                    'weekly': None,
                    'monthly': None,
                    'alignment': 'unknown',
                    'confidence_modifier': 0,
                }
            
            # Analyze each timeframe
            daily = self._analyze_daily(hist)
            weekly = self._analyze_weekly(hist)
            monthly = self._analyze_monthly(hist)
            
            # Calculate alignment
            alignment = self._calculate_alignment(daily, weekly, monthly)
            
            result = {
                'ticker': ticker,
                'daily': daily,
                'weekly': weekly,
                'monthly': monthly,
                'alignment': alignment['status'],
                'alignment_score': alignment['score'],
                'confidence_modifier': alignment['confidence_modifier'],
                'summary': alignment['summary'],
                'analyzed_at': datetime.now().isoformat(),
            }
            
            # Cache result
            self._cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now(),
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Multi-timeframe analysis error for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'alignment': 'unknown',
                'confidence_modifier': 0,
            }
    
    def _analyze_daily(self, hist) -> Dict:
        """Analyze daily trend using last 20 days."""
        if len(hist) < 20:
            return {'trend': 'neutral', 'strength': 0}
        
        recent = hist.tail(20)
        
        # SMA trend
        sma5 = recent['Close'].tail(5).mean()
        sma20 = recent['Close'].mean()
        
        # RSI
        delta = recent['Close'].diff()
        gains = delta.where(delta > 0, 0).mean()
        losses = (-delta.where(delta < 0, 0)).mean()
        rs = gains / losses if losses != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Determine trend
        if sma5 > sma20 * 1.02 and rsi > 50:
            trend = 'bullish'
            strength = min(100, int((sma5 / sma20 - 1) * 500 + (rsi - 50)))
        elif sma5 < sma20 * 0.98 and rsi < 50:
            trend = 'bearish'
            strength = min(100, int((1 - sma5 / sma20) * 500 + (50 - rsi)))
        else:
            trend = 'neutral'
            strength = abs(50 - rsi)
        
        # Recent momentum (5-day return)
        ret_5d = ((recent['Close'].iloc[-1] / recent['Close'].iloc[-5]) - 1) * 100
        
        return {
            'trend': trend,
            'strength': round(strength),
            'rsi': round(rsi, 1),
            'sma5_vs_sma20': round((sma5 / sma20 - 1) * 100, 2),
            'return_5d': round(ret_5d, 2),
        }
    
    def _analyze_weekly(self, hist) -> Dict:
        """Analyze weekly trend using weekly resampled data."""
        if len(hist) < 30:
            return {'trend': 'neutral', 'strength': 0}
        
        # Resample to weekly
        weekly = hist.resample('W')['Close'].last().dropna()
        
        if len(weekly) < 8:
            return {'trend': 'neutral', 'strength': 0}
        
        # SMA4 vs SMA8 (4-week vs 8-week)
        sma4 = weekly.tail(4).mean()
        sma8 = weekly.tail(8).mean()
        
        # Weekly RSI
        delta = weekly.diff()
        gains = delta.where(delta > 0, 0).tail(14).mean()
        losses = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gains / losses if losses != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Determine trend
        if sma4 > sma8 * 1.01 and rsi > 50:
            trend = 'bullish'
            strength = min(100, int((sma4 / sma8 - 1) * 300 + (rsi - 50)))
        elif sma4 < sma8 * 0.99 and rsi < 50:
            trend = 'bearish'
            strength = min(100, int((1 - sma4 / sma8) * 300 + (50 - rsi)))
        else:
            trend = 'neutral'
            strength = abs(50 - rsi)
        
        # 4-week return
        ret_4w = ((weekly.iloc[-1] / weekly.iloc[-4]) - 1) * 100 if len(weekly) >= 4 else 0
        
        return {
            'trend': trend,
            'strength': round(strength),
            'rsi': round(rsi, 1),
            'sma4_vs_sma8': round((sma4 / sma8 - 1) * 100, 2),
            'return_4w': round(ret_4w, 2),
        }
    
    def _analyze_monthly(self, hist) -> Dict:
        """Analyze monthly trend using monthly resampled data."""
        if len(hist) < 90:
            return {'trend': 'neutral', 'strength': 0}
        
        # Resample to monthly
        monthly = hist.resample('ME')['Close'].last().dropna()
        
        if len(monthly) < 3:
            return {'trend': 'neutral', 'strength': 0}
        
        # 3-month SMA trend
        current = monthly.iloc[-1]
        avg_3m = monthly.tail(3).mean()
        
        # 3-month return
        ret_3m = ((current / monthly.iloc[-3]) - 1) * 100 if len(monthly) >= 3 else 0
        
        # Determine trend based on position relative to 3M average
        if current > avg_3m * 1.02:
            trend = 'bullish'
            strength = min(100, int((current / avg_3m - 1) * 200))
        elif current < avg_3m * 0.98:
            trend = 'bearish'
            strength = min(100, int((1 - current / avg_3m) * 200))
        else:
            trend = 'neutral'
            strength = 30
        
        return {
            'trend': trend,
            'strength': round(strength),
            'return_3m': round(ret_3m, 2),
            'vs_3m_avg': round((current / avg_3m - 1) * 100, 2),
        }
    
    def _calculate_alignment(self, daily: Dict, weekly: Dict, monthly: Dict) -> Dict:
        """
        Calculate alignment between timeframes.
        
        Full alignment (all same trend) = high confidence boost
        Partial alignment (2/3 same) = moderate boost
        No alignment (all different) = confidence penalty
        """
        trends = [
            daily.get('trend', 'neutral'),
            weekly.get('trend', 'neutral'),
            monthly.get('trend', 'neutral'),
        ]
        
        # Count bullish/bearish/neutral
        bullish_count = trends.count('bullish')
        bearish_count = trends.count('bearish')
        
        if bullish_count == 3:
            return {
                'status': 'aligned_bullish',
                'score': 100,
                'confidence_modifier': 15,
                'summary': 'All timeframes bullish — high confidence buy',
            }
        elif bearish_count == 3:
            return {
                'status': 'aligned_bearish',
                'score': 100,
                'confidence_modifier': 15,
                'summary': 'All timeframes bearish — high confidence sell',
            }
        elif bullish_count == 2:
            return {
                'status': 'mostly_bullish',
                'score': 66,
                'confidence_modifier': 8,
                'summary': '2/3 timeframes bullish — moderate buy confidence',
            }
        elif bearish_count == 2:
            return {
                'status': 'mostly_bearish',
                'score': 66,
                'confidence_modifier': 8,
                'summary': '2/3 timeframes bearish — moderate sell confidence',
            }
        elif bullish_count == 1 and bearish_count == 1:
            return {
                'status': 'conflicting',
                'score': 25,
                'confidence_modifier': -10,
                'summary': 'Timeframes conflicting — reduced reliability',
            }
        else:
            return {
                'status': 'neutral',
                'score': 50,
                'confidence_modifier': 0,
                'summary': 'No strong trend across timeframes',
            }
    
    def get_confirmation_for_signal(self, ticker: str, signal: str, confidence: int) -> Dict:
        """
        Get confirmation status for a trading signal.
        Returns adjusted confidence based on multi-timeframe alignment.
        """
        analysis = self.analyze_ticker(ticker)
        
        if analysis.get('error'):
            return {
                'ticker': ticker,
                'original_confidence': confidence,
                'adjusted_confidence': confidence,
                'confirmed': None,
                'reason': 'Unable to analyze timeframes',
            }
        
        alignment = analysis.get('alignment', 'unknown')
        modifier = analysis.get('confidence_modifier', 0)
        
        # Check if signal aligns with multi-timeframe trend
        signal_type = 'buy' if signal.upper() in ('BUY', 'OPPORTUNITY', 'STRONG BUY') else \
                      'sell' if signal.upper() in ('SELL', 'CAUTION', 'STRONG SELL') else 'hold'
        
        confirmed = None
        if signal_type == 'buy' and alignment in ('aligned_bullish', 'mostly_bullish'):
            confirmed = True
            reason = f"Buy signal confirmed by {alignment.replace('_', ' ')}"
        elif signal_type == 'sell' and alignment in ('aligned_bearish', 'mostly_bearish'):
            confirmed = True
            reason = f"Sell signal confirmed by {alignment.replace('_', ' ')}"
        elif signal_type == 'buy' and alignment in ('aligned_bearish', 'mostly_bearish'):
            confirmed = False
            modifier = -15  # Extra penalty for conflicting signal
            reason = "Buy signal conflicts with bearish trend"
        elif signal_type == 'sell' and alignment in ('aligned_bullish', 'mostly_bullish'):
            confirmed = False
            modifier = -15
            reason = "Sell signal conflicts with bullish trend"
        else:
            reason = analysis.get('summary', 'Mixed signals')
        
        adjusted = max(20, min(95, confidence + modifier))
        
        return {
            'ticker': ticker,
            'signal': signal,
            'original_confidence': confidence,
            'adjusted_confidence': adjusted,
            'confidence_change': adjusted - confidence,
            'confirmed': confirmed,
            'alignment': alignment,
            'daily_trend': analysis.get('daily', {}).get('trend'),
            'weekly_trend': analysis.get('weekly', {}).get('trend'),
            'monthly_trend': analysis.get('monthly', {}).get('trend'),
            'reason': reason,
        }


# Singleton
multi_timeframe = MultiTimeframe()

# Backward-compatible alias used by older routes/imports
multi_timeframe_analyzer = multi_timeframe
