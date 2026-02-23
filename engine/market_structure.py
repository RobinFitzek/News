"""
Hierarchical Market Structure — Multi-Scale Turning Points

Uses the Directional Change (DC) algorithm to identify market turning points
at multiple levels of significance. Provides trend regime labels and structure
scores for integration into the quant screener's technical scoring.

Based on: neurotrader888/market-structure
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TurningPoint:
    """A confirmed market turning point at a specific hierarchy level."""
    index: int
    price: float
    type: str  # 'high' or 'low'
    confirmed_index: int = 0
    level: int = 0


class DirectionalChange:
    """
    ATR-based Directional Change algorithm.

    Detects turning points by tracking pending highs/lows and confirming
    when price reverses by at least 1×ATR from the extreme.
    """

    @staticmethod
    def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 14) -> np.ndarray:
        """Compute Average True Range."""
        n = len(closes)
        if n < 2:
            return np.ones(n) * 0.01

        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        atr = np.zeros(n)
        atr[:period] = np.mean(tr[:period]) if n >= period else np.mean(tr)
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Backfill initial values
        if n >= period:
            atr[:period] = atr[period]

        return atr

    @staticmethod
    def detect(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               atr_period: int = 14, sigma: float = 1.0) -> List[TurningPoint]:
        """
        Detect level-0 turning points using directional change.

        A top is confirmed when price drops below (highest_since_bottom - sigma*ATR).
        A bottom is confirmed when price rises above (lowest_since_top + sigma*ATR).
        """
        n = len(closes)
        if n < atr_period + 2:
            return []

        atr = DirectionalChange.compute_atr(highs, lows, closes, atr_period)
        turning_points: List[TurningPoint] = []

        # State tracking
        looking_for = 'high'  # Start looking for a high
        pending_high_price = highs[0]
        pending_high_idx = 0
        pending_low_price = lows[0]
        pending_low_idx = 0

        for i in range(1, n):
            threshold = sigma * atr[i]

            if looking_for == 'high':
                # Track highest point
                if highs[i] > pending_high_price:
                    pending_high_price = highs[i]
                    pending_high_idx = i

                # Confirm top if price drops enough
                if pending_high_price - lows[i] >= threshold:
                    turning_points.append(TurningPoint(
                        index=pending_high_idx,
                        price=pending_high_price,
                        type='high',
                        confirmed_index=i,
                        level=0
                    ))
                    looking_for = 'low'
                    pending_low_price = lows[i]
                    pending_low_idx = i

            else:  # looking_for == 'low'
                # Track lowest point
                if lows[i] < pending_low_price:
                    pending_low_price = lows[i]
                    pending_low_idx = i

                # Confirm bottom if price rises enough
                if highs[i] - pending_low_price >= threshold:
                    turning_points.append(TurningPoint(
                        index=pending_low_idx,
                        price=pending_low_price,
                        type='low',
                        confirmed_index=i,
                        level=0
                    ))
                    looking_for = 'high'
                    pending_high_price = highs[i]
                    pending_high_idx = i

        return turning_points


class HierarchicalExtremes:
    """
    Builds multi-level turning points from level-0 DC extremes.

    Level N+1 top: a level-N top that is higher than both adjacent level-N tops.
    Level N+1 bottom: a level-N bottom that is lower than both adjacent level-N bottoms.
    """

    @staticmethod
    def build(turning_points: List[TurningPoint], max_levels: int = 4) -> Dict[int, List[TurningPoint]]:
        """Build hierarchy of turning points up to max_levels."""
        hierarchy: Dict[int, List[TurningPoint]] = {0: list(turning_points)}

        for level in range(1, max_levels + 1):
            prev_points = hierarchy[level - 1]
            if len(prev_points) < 3:
                break

            upgraded: List[TurningPoint] = []

            # Separate highs and lows at previous level
            highs = [p for p in prev_points if p.type == 'high']
            lows = [p for p in prev_points if p.type == 'low']

            # Upgrade highs: a high higher than both neighbors
            for i in range(1, len(highs) - 1):
                if highs[i].price > highs[i - 1].price and highs[i].price > highs[i + 1].price:
                    upgraded.append(TurningPoint(
                        index=highs[i].index,
                        price=highs[i].price,
                        type='high',
                        confirmed_index=highs[i].confirmed_index,
                        level=level
                    ))

            # Upgrade lows: a low lower than both neighbors
            for i in range(1, len(lows) - 1):
                if lows[i].price < lows[i - 1].price and lows[i].price < lows[i + 1].price:
                    upgraded.append(TurningPoint(
                        index=lows[i].index,
                        price=lows[i].price,
                        type='low',
                        confirmed_index=lows[i].confirmed_index,
                        level=level
                    ))

            if not upgraded:
                break

            # Sort by index
            upgraded.sort(key=lambda p: p.index)

            # Auto-insert missing alternation: if two tops in a row,
            # insert the lowest low between them (and vice versa)
            fixed: List[TurningPoint] = [upgraded[0]]
            for i in range(1, len(upgraded)):
                if upgraded[i].type == fixed[-1].type:
                    # Same type consecutive — insert opposite between them
                    between_start = fixed[-1].index
                    between_end = upgraded[i].index
                    candidates = [p for p in prev_points
                                  if between_start < p.index < between_end
                                  and p.type != upgraded[i].type]
                    if candidates:
                        if upgraded[i].type == 'high':
                            # Need a low between two highs
                            best = min(candidates, key=lambda p: p.price)
                        else:
                            # Need a high between two lows
                            best = max(candidates, key=lambda p: p.price)
                        fixed.append(TurningPoint(
                            index=best.index, price=best.price,
                            type=best.type, confirmed_index=best.confirmed_index,
                            level=level
                        ))
                fixed.append(upgraded[i])

            hierarchy[level] = fixed

        return hierarchy


class MarketStructureAnalyzer:
    """
    Analyzes market structure using hierarchical turning points.
    Provides trend regime labels and structure scores for the quant screener.
    """

    def __init__(self, atr_period: int = 14, max_levels: int = 4,
                 trend_lookback: int = 20, cache_minutes: int = 30):
        self.atr_period = atr_period
        self.max_levels = max_levels
        self.trend_lookback = trend_lookback
        self._cache: Dict[str, Dict] = {}
        self._cache_duration = timedelta(minutes=cache_minutes)

        # Load config if available
        try:
            from core.config import MARKET_STRUCTURE_CONFIG
            self.atr_period = MARKET_STRUCTURE_CONFIG.get('atr_period', atr_period)
            self.max_levels = MARKET_STRUCTURE_CONFIG.get('max_hierarchy_levels', max_levels)
            self.trend_lookback = MARKET_STRUCTURE_CONFIG.get('trend_lookback_bars', trend_lookback)
        except (ImportError, AttributeError):
            pass

    def analyze(self, ticker: str, hist) -> Dict:
        """
        Full market structure analysis for a ticker.
        Returns dict with trend_regime, structure_score, hierarchy summary.
        """
        # Check cache
        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            if hist is None or hist.empty or len(hist) < self.atr_period + 5:
                return {}

            highs = hist['High'].values.astype(float)
            lows = hist['Low'].values.astype(float)
            closes = hist['Close'].values.astype(float)

            # Detect level-0 turning points
            tp = DirectionalChange.detect(highs, lows, closes, self.atr_period)
            if len(tp) < 4:
                return {'trend_regime': 'insufficient_data', 'structure_score': 50}

            # Build hierarchy
            hierarchy = HierarchicalExtremes.build(tp, self.max_levels)

            # Determine trend regime from hierarchy
            trend_regime = self._get_trend_regime(hierarchy, len(closes))
            structure_score = self._regime_to_score(trend_regime)

            # Get dynamic support/resistance from level-1+ extremes
            support, resistance = self._get_sr_levels(hierarchy, closes[-1])

            # Count turning points per level
            level_counts = {lvl: len(pts) for lvl, pts in hierarchy.items()}

            result = {
                'trend_regime': trend_regime,
                'structure_score': structure_score,
                'support': support,
                'resistance': resistance,
                'level_counts': level_counts,
                'total_turning_points': len(tp),
            }

            self._cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
            return result

        except Exception as e:
            logger.warning(f"Market structure analysis failed for {ticker}: {e}")
            return {}

    def _get_trend_regime(self, hierarchy: Dict[int, List[TurningPoint]],
                          n_bars: int) -> str:
        """
        Determine trend regime from hierarchical turning points.

        Checks higher levels first for stronger signals.
        """
        # Use the highest available level with enough points
        for level in sorted(hierarchy.keys(), reverse=True):
            points = hierarchy[level]
            # Filter to recent points (within lookback window)
            recent = [p for p in points if p.index >= n_bars - self.trend_lookback * 5]
            if len(recent) < 3:
                continue

            recent_highs = [p for p in recent if p.type == 'high']
            recent_lows = [p for p in recent if p.type == 'low']

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                hh = recent_highs[-1].price > recent_highs[-2].price  # Higher high
                hl = recent_lows[-1].price > recent_lows[-2].price    # Higher low
                lh = recent_highs[-1].price < recent_highs[-2].price  # Lower high
                ll = recent_lows[-1].price < recent_lows[-2].price    # Lower low

                if level >= 2:
                    if hh and hl:
                        return 'strong_uptrend'
                    if lh and ll:
                        return 'strong_downtrend'
                if hh and hl:
                    return 'uptrend'
                if lh and ll:
                    return 'downtrend'
                if hh and ll:
                    return 'ranging'  # Expanding range
                if lh and hl:
                    return 'ranging'  # Contracting range

        return 'ranging'

    @staticmethod
    def _regime_to_score(regime: str) -> int:
        """Map trend regime to a 0-100 score for technical scoring."""
        scores = {
            'strong_uptrend': 90,
            'uptrend': 75,
            'ranging': 50,
            'downtrend': 25,
            'strong_downtrend': 10,
            'insufficient_data': 50,
        }
        return scores.get(regime, 50)

    def _get_sr_levels(self, hierarchy: Dict[int, List[TurningPoint]],
                       current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Get nearest support and resistance from hierarchy."""
        all_highs = []
        all_lows = []

        for level, points in hierarchy.items():
            weight = level + 1  # Higher level = more significant
            for p in points:
                if p.type == 'high':
                    all_highs.append((p.price, weight))
                else:
                    all_lows.append((p.price, weight))

        # Nearest resistance above current price
        resistance = None
        above = [(p, w) for p, w in all_highs if p > current_price]
        if above:
            # Prefer higher-level resistance
            above.sort(key=lambda x: (x[0] - current_price) / x[1])
            resistance = round(above[0][0], 2)

        # Nearest support below current price
        support = None
        below = [(p, w) for p, w in all_lows if p < current_price]
        if below:
            below.sort(key=lambda x: (current_price - x[0]) / x[1])
            support = round(below[0][0], 2)

        return support, resistance

    def get_dc_turning_points(self, hist, sigma: float = 1.0) -> List[TurningPoint]:
        """
        Public access to raw DC turning points.
        Used by harmonic pattern detector (Phase 2).
        """
        if hist is None or hist.empty or len(hist) < self.atr_period + 2:
            return []

        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        closes = hist['Close'].values.astype(float)

        return DirectionalChange.detect(highs, lows, closes, self.atr_period, sigma)


# Singleton instance
market_structure_analyzer = MarketStructureAnalyzer()
