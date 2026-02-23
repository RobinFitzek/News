"""
XABCD Harmonic Pattern Recognition

Identifies Gartley, Bat, Butterfly, Crab, Deep Crab, Cypher, and Shark patterns
using Directional Change pivot points and Fibonacci ratio matching.

Multi-sigma scanning for robustness. Shark pattern weighted highest (best performer).

Based on: neurotrader888/TechnicalAnalysisAutomation
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternDef:
    """
    XABCD pattern ratio definitions.
    Each ratio is (min, max) range or None to skip.
    XB = AB/XA, AC = BC/AB, BD = CD/BC, XD = AD/XA
    """
    name: str
    xb: Optional[Tuple[float, float]]  # AB / XA
    ac: Optional[Tuple[float, float]]  # BC / AB
    bd: Optional[Tuple[float, float]]  # CD / BC
    xd: Optional[Tuple[float, float]]  # AD / XA
    weight: float = 1.0  # Pattern reliability weight


# Pattern definitions — ranges based on standard Fibonacci ratios
PATTERNS = [
    PatternDef('gartley',    (0.580, 0.658), (0.382, 0.886), (1.130, 1.618), (0.746, 0.826), weight=1.0),
    PatternDef('bat',        (0.382, 0.500), (0.382, 0.886), (1.618, 2.618), (0.846, 0.926), weight=1.0),
    PatternDef('butterfly',  (0.746, 0.826), (0.382, 0.886), (1.618, 2.618), (1.232, 1.658), weight=0.9),
    PatternDef('crab',       (0.382, 0.618), (0.382, 0.886), (2.618, 3.618), (1.578, 1.658), weight=0.9),
    PatternDef('deep_crab',  (0.846, 0.926), (0.382, 0.886), (2.000, 3.618), (1.578, 1.658), weight=0.8),
    PatternDef('cypher',     (0.382, 0.618), (1.130, 1.414), None,           (0.746, 0.826), weight=0.9),
    PatternDef('shark',      None,           (1.130, 1.618), (1.618, 2.236), (0.846, 1.170), weight=1.3),  # Best performer
]


class HarmonicDetector:
    """
    Detects XABCD harmonic patterns from price data using DC turning points.
    """

    def __init__(self, error_threshold: float = 0.5, min_confidence: int = 40,
                 sigma_levels: List[float] = None, cache_minutes: int = 30):
        self.error_threshold = error_threshold
        self.min_confidence = min_confidence
        self.sigma_levels = sigma_levels or [0.01, 0.02, 0.03, 0.04]
        self._cache: Dict[str, Dict] = {}
        self._cache_duration = timedelta(minutes=cache_minutes)

        # Load config if available
        try:
            from core.config import HARMONIC_CONFIG
            self.error_threshold = HARMONIC_CONFIG.get('error_threshold', error_threshold)
            self.min_confidence = HARMONIC_CONFIG.get('min_confidence', min_confidence)
            self.sigma_levels = HARMONIC_CONFIG.get('sigma_levels', self.sigma_levels)
        except (ImportError, AttributeError):
            pass

    def detect(self, ticker: str, hist) -> List[Dict]:
        """
        Detect harmonic patterns for a ticker.
        Uses multi-sigma DC scanning for robustness.
        Returns list of detected patterns sorted by confidence.
        """
        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            if hist is None or hist.empty or len(hist) < 30:
                return []

            from engine.market_structure import market_structure_analyzer

            all_patterns = []

            for sigma_pct in self.sigma_levels:
                # Get DC turning points at this sigma level
                # Convert percentage to ATR multiplier
                sigma_mult = max(0.5, sigma_pct * 100)  # 0.01 -> 0.5 min, 0.04 -> 4.0

                tp = market_structure_analyzer.get_dc_turning_points(hist, sigma=sigma_mult)
                if len(tp) < 5:
                    continue

                # Extract XABCD candidates from consecutive turning points
                patterns = self._find_patterns(tp, hist, sigma_pct)
                all_patterns.extend(patterns)

            # Deduplicate: keep best confidence per pattern type
            best = {}
            for p in all_patterns:
                key = (p['pattern_name'], p['direction'])
                if key not in best or p['confidence'] > best[key]['confidence']:
                    best[key] = p

            result = sorted(best.values(), key=lambda x: x['confidence'], reverse=True)
            result = [p for p in result if p['confidence'] >= self.min_confidence]

            self._cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
            return result

        except Exception as e:
            logger.warning(f"Harmonic pattern detection failed for {ticker}: {e}")
            return []

    def _find_patterns(self, turning_points, hist, sigma: float) -> List[Dict]:
        """Find XABCD patterns from a sequence of turning points."""
        patterns = []
        n = len(turning_points)

        closes = hist['Close'].values.astype(float)
        current_price = closes[-1]

        # Need at least 5 points for XABCD
        # Slide a window of 5 consecutive turning points
        for i in range(n - 4):
            x = turning_points[i]
            a = turning_points[i + 1]
            b = turning_points[i + 2]
            c = turning_points[i + 3]
            d = turning_points[i + 4]

            # Determine direction
            if x.type == 'low' and a.type == 'high':
                direction = 'bullish'
            elif x.type == 'high' and a.type == 'low':
                direction = 'bearish'
            else:
                continue

            # Compute legs
            xa = abs(a.price - x.price)
            ab = abs(b.price - a.price)
            bc = abs(c.price - b.price)
            cd = abs(d.price - c.price)
            ad = abs(d.price - a.price) if direction == 'bullish' else abs(a.price - d.price)

            if xa < 1e-8 or ab < 1e-8 or bc < 1e-8:
                continue

            # Compute ratios
            xb_ratio = ab / xa
            ac_ratio = bc / ab
            bd_ratio = cd / bc if bc > 1e-8 else 0
            xd_ratio = ad / xa

            # Try to match each pattern
            for pdef in PATTERNS:
                error = self._compute_error(pdef, xb_ratio, ac_ratio, bd_ratio, xd_ratio)
                if error is not None and error < self.error_threshold:
                    # Confidence: inverse of error, scaled by pattern weight
                    confidence = int(max(0, min(100,
                        (1.0 - error / self.error_threshold) * 80 * pdef.weight + 20
                    )))

                    # Only report patterns near completion (D point close to current price)
                    d_recency = len(closes) - d.index
                    if d_recency > 20:
                        confidence = int(confidence * 0.5)  # Older patterns get lower confidence

                    # Entry/exit levels
                    if direction == 'bullish':
                        entry_zone = (d.price, d.price * 1.005)
                        stop_loss = d.price * 0.97
                        target1 = d.price + 0.382 * cd
                        target2 = d.price + 0.618 * cd
                    else:
                        entry_zone = (d.price * 0.995, d.price)
                        stop_loss = d.price * 1.03
                        target1 = d.price - 0.382 * cd
                        target2 = d.price - 0.618 * cd

                    patterns.append({
                        'pattern_name': pdef.name,
                        'direction': direction,
                        'confidence': confidence,
                        'error_score': round(error, 4),
                        'sigma': sigma,
                        'x_price': round(x.price, 2),
                        'a_price': round(a.price, 2),
                        'b_price': round(b.price, 2),
                        'c_price': round(c.price, 2),
                        'd_price': round(d.price, 2),
                        'd_index': d.index,
                        'd_recency': d_recency,
                        'entry_zone': (round(entry_zone[0], 2), round(entry_zone[1], 2)),
                        'stop_loss': round(stop_loss, 2),
                        'targets': [round(target1, 2), round(target2, 2)],
                        'ratios': {
                            'XB': round(xb_ratio, 3),
                            'AC': round(ac_ratio, 3),
                            'BD': round(bd_ratio, 3),
                            'XD': round(xd_ratio, 3),
                        }
                    })

        return patterns

    @staticmethod
    def _compute_error(pdef: PatternDef, xb: float, ac: float,
                       bd: float, xd: float) -> Optional[float]:
        """
        Compute total log-ratio error between actual and expected ratios.
        Returns None if any non-optional ratio is completely outside bounds.
        """
        total_error = 0.0
        n_ratios = 0

        def ratio_error(actual: float, expected_range: Optional[Tuple[float, float]]) -> Optional[float]:
            if expected_range is None:
                return 0.0  # This ratio is not checked for this pattern
            lo, hi = expected_range
            if actual <= 0:
                return None
            mid = (lo + hi) / 2
            return abs(np.log(actual / mid))

        for actual, expected in [(xb, pdef.xb), (ac, pdef.ac),
                                  (bd, pdef.bd), (xd, pdef.xd)]:
            err = ratio_error(actual, expected)
            if err is None:
                return None
            total_error += err
            if expected is not None:
                n_ratios += 1

        return total_error / max(n_ratios, 1)


# Singleton instance
harmonic_detector = HarmonicDetector()
