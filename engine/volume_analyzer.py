"""
Volume Confirmation Analysis
RSI 28 on normal volume is different from RSI 28 on 3x volume (capitulation).
Adds volume-weighted signals to identify institutional accumulation/distribution.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """Analyze volume patterns for signal confirmation."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=30)

    def get_volume_metrics(self, ticker: str, hist=None) -> Dict:
        """Calculate volume-based metrics for signal confirmation."""
        # Check cache
        if ticker in self._cache:
            entry = self._cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            if hist is None:
                hist = yf.Ticker(ticker).history(period="3mo")

            if hist.empty or len(hist) < 20:
                return self._empty_metrics()

            volume = hist['Volume']
            close = hist['Close']

            # 20-day average volume
            avg_volume_20d = float(volume.rolling(20).mean().iloc[-1])
            current_volume = float(volume.iloc[-1])
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0

            # VWAP (Volume Weighted Average Price) for last 20 days
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            current_vwap = float(vwap.iloc[-1]) if not vwap.empty else float(close.iloc[-1])
            current_price = float(close.iloc[-1])
            
            # VWAP deviation
            vwap_deviation_pct = ((current_price - current_vwap) / current_vwap) * 100 if current_vwap > 0 else 0

            # Volume trend (increasing or decreasing over 20 days)
            recent_avg_vol = float(volume.iloc[-5:].mean())
            older_avg_vol = float(volume.iloc[-20:-10].mean())
            volume_trend = 'increasing' if recent_avg_vol > older_avg_vol * 1.2 else \
                          'decreasing' if recent_avg_vol < older_avg_vol * 0.8 else 'stable'

            # Accumulation/Distribution indicator (simplified)
            # High volume on up days = accumulation, high volume on down days = distribution
            last_10_days = hist.iloc[-10:]
            up_days_volume = last_10_days[last_10_days['Close'] > last_10_days['Open']]['Volume'].sum()
            down_days_volume = last_10_days[last_10_days['Close'] < last_10_days['Open']]['Volume'].sum()
            total_vol = up_days_volume + down_days_volume
            
            if total_vol > 0:
                acc_dist_ratio = (up_days_volume - down_days_volume) / total_vol
                if acc_dist_ratio > 0.3:
                    acc_dist = 'accumulation'
                elif acc_dist_ratio < -0.3:
                    acc_dist = 'distribution'
                else:
                    acc_dist = 'neutral'
            else:
                acc_dist = 'neutral'

            # High volume anomaly detection
            high_volume_anomaly = volume_ratio >= 3.0  # 3x average = unusual activity

            metrics = {
                'ticker': ticker,
                'avg_volume_20d': int(avg_volume_20d),
                'current_volume': int(current_volume),
                'volume_ratio': round(volume_ratio, 2),
                'vwap': round(current_vwap, 2),
                'vwap_deviation_pct': round(vwap_deviation_pct, 2),
                'volume_trend': volume_trend,
                'accumulation_distribution': acc_dist,
                'high_volume_anomaly': high_volume_anomaly,
                'volume_confirmation': self._get_volume_confirmation(volume_ratio, acc_dist),
            }

            # Cache and store
            self._cache[ticker] = {'data': metrics, 'timestamp': datetime.now()}
            self._store_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating volume metrics for {ticker}: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'avg_volume_20d': 0,
            'current_volume': 0,
            'volume_ratio': 1.0,
            'vwap': 0,
            'vwap_deviation_pct': 0,
            'volume_trend': 'unknown',
            'accumulation_distribution': 'neutral',
            'high_volume_anomaly': False,
            'volume_confirmation': 'weak',
        }

    def _get_volume_confirmation(self, volume_ratio: float, acc_dist: str) -> str:
        """Determine signal confirmation strength based on volume."""
        if volume_ratio >= 2.0 and acc_dist == 'accumulation':
            return 'strong_bullish'
        elif volume_ratio >= 2.0 and acc_dist == 'distribution':
            return 'strong_bearish'
        elif volume_ratio >= 1.5:
            return 'moderate'
        elif volume_ratio < 0.8:
            return 'weak'
        else:
            return 'neutral'

    def _store_metrics(self, metrics: Dict):
        """Store volume metrics in database."""
        try:
            db.execute("""
                INSERT INTO volume_metrics (
                    ticker, avg_volume_20d, current_volume, volume_ratio,
                    vwap, vwap_deviation_pct, volume_trend, accumulation_distribution,
                    high_volume_anomaly, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    avg_volume_20d = excluded.avg_volume_20d,
                    current_volume = excluded.current_volume,
                    volume_ratio = excluded.volume_ratio,
                    vwap = excluded.vwap,
                    vwap_deviation_pct = excluded.vwap_deviation_pct,
                    volume_trend = excluded.volume_trend,
                    accumulation_distribution = excluded.accumulation_distribution,
                    high_volume_anomaly = excluded.high_volume_anomaly,
                    calculated_at = excluded.calculated_at
            """, (
                metrics['ticker'],
                metrics['avg_volume_20d'],
                metrics['current_volume'],
                metrics['volume_ratio'],
                metrics['vwap'],
                metrics['vwap_deviation_pct'],
                metrics['volume_trend'],
                metrics['accumulation_distribution'],
                1 if metrics['high_volume_anomaly'] else 0,
                datetime.now().isoformat(),
            ))
        except Exception as e:
            logger.warning(f"Could not store volume metrics: {e}")

    def enhance_signal(self, ticker: str, signal: str, metrics: Dict = None) -> Dict:
        """Enhance trading signal with volume confirmation."""
        if metrics is None:
            metrics = self.get_volume_metrics(ticker)

        volume_conf = metrics.get('volume_confirmation', 'neutral')
        volume_ratio = metrics.get('volume_ratio', 1.0)
        
        enhancement = {
            'original_signal': signal,
            'volume_ratio': volume_ratio,
            'volume_confirmation': volume_conf,
        }

        # Adjust signal strength based on volume
        if signal in ['BUY', 'STRONG_BUY']:
            if volume_conf == 'strong_bullish':
                enhancement['enhanced_signal'] = 'STRONG_BUY'
                enhancement['note'] = f"[High Vol] High volume ({volume_ratio:.1f}x) + accumulation confirms bullish signal"
            elif volume_conf == 'weak':
                enhancement['enhanced_signal'] = 'BUY'
                enhancement['note'] = f"[Low Vol] Low volume ({volume_ratio:.1f}x) weakens signal — wait for confirmation"
            else:
                enhancement['enhanced_signal'] = signal
                enhancement['note'] = f"Volume: {volume_ratio:.1f}x average"

        elif signal in ['SELL', 'STRONG_SELL']:
            if volume_conf == 'strong_bearish':
                enhancement['enhanced_signal'] = 'STRONG_SELL'
                enhancement['note'] = f"[High Vol] High volume ({volume_ratio:.1f}x) + distribution confirms bearish signal"
            elif volume_conf == 'weak':
                enhancement['enhanced_signal'] = 'SELL'
                enhancement['note'] = f"[Low Vol] Low volume ({volume_ratio:.1f}x) weakens signal"
            else:
                enhancement['enhanced_signal'] = signal
                enhancement['note'] = f"Volume: {volume_ratio:.1f}x average"
        else:
            enhancement['enhanced_signal'] = signal
            enhancement['note'] = f"Volume: {volume_ratio:.1f}x average"

        return enhancement


# Singleton
volume_analyzer = VolumeAnalyzer()
