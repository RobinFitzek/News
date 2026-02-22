"""
Technical Pattern Recognition Engine
Detects classic chart patterns from price history using pivot detection
and slope analysis. Pure math-based — zero API cost beyond yfinance data.

SUPPORTED PATTERNS:
- Double Bottom (bullish reversal)
- Cup and Handle (bullish continuation)
- Ascending Triangle (bullish breakout)
- Head and Shoulders (bearish reversal)
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db
import logging

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """Detect classic chart patterns from price history."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=30)

    def detect_patterns(self, ticker: str) -> List[Dict]:
        """
        Scan a ticker for all supported chart patterns.
        Returns list of detected patterns sorted by confidence.
        """
        # Check cache
        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')

            if hist.empty or len(hist) < 60:
                logger.debug(f"{ticker}: insufficient price history for pattern detection ({len(hist) if not hist.empty else 0} bars)")
                return []

            prices = hist['Close'].values
            volumes = hist['Volume'].values if 'Volume' in hist.columns else None

            # Find pivots at multiple windows for robustness
            pivots_narrow = self._find_pivots(prices, window=5)
            pivots_wide = self._find_pivots(prices, window=10)

            patterns = []

            # Run each pattern detector
            for detector, pivot_set in [
                (self._detect_double_bottom, pivots_narrow),
                (self._detect_cup_and_handle, pivots_wide),
                (self._detect_ascending_triangle, pivots_narrow),
                (self._detect_head_and_shoulders, pivots_wide),
            ]:
                try:
                    result = detector(pivot_set, prices, volumes)
                    if result and result['confidence'] >= 40:
                        patterns.append(result)
                except Exception as e:
                    logger.debug(f"{ticker}: {detector.__name__} error: {e}")

            # Sort by confidence descending
            patterns.sort(key=lambda x: x['confidence'], reverse=True)

            self._cache[cache_key] = {
                'data': patterns,
                'timestamp': datetime.now(),
            }

            if patterns:
                logger.info(f"{ticker}: detected {len(patterns)} pattern(s) — {', '.join(p['pattern'] for p in patterns)}")

            return patterns

        except Exception as e:
            logger.warning(f"Pattern detection failed for {ticker}: {e}")
            return []

    def _find_pivots(self, prices: np.ndarray, window: int = 5) -> Dict:
        """
        Find local highs and lows in price series using a rolling window.
        Returns dict with 'highs' and 'lows', each a list of (index, price).
        """
        highs = []
        lows = []
        n = len(prices)

        for i in range(window, n - window):
            segment = prices[i - window:i + window + 1]

            # Local high: center is max of window
            if prices[i] == np.max(segment):
                highs.append((i, float(prices[i])))

            # Local low: center is min of window
            if prices[i] == np.min(segment):
                lows.append((i, float(prices[i])))

        return {'highs': highs, 'lows': lows}

    def _detect_double_bottom(self, pivots: Dict, prices: np.ndarray,
                               volumes: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Double bottom: two lows at similar levels with a peak between.
        Bullish reversal pattern.
        """
        lows = pivots['lows']
        if len(lows) < 2:
            return None

        n = len(prices)
        best_confidence = 0
        best_result = None

        # Check pairs of lows (prefer recent)
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                idx1, price1 = lows[i]
                idx2, price2 = lows[j]

                # Lows should be separated by at least 15 bars
                if idx2 - idx1 < 15:
                    continue

                # Lows should be within 3% of each other
                avg_low = (price1 + price2) / 2
                if avg_low == 0:
                    continue
                spread_pct = abs(price1 - price2) / avg_low * 100
                if spread_pct > 3.0:
                    continue

                # There should be a peak between the two lows
                between = prices[idx1:idx2 + 1]
                peak_val = np.max(between)
                neckline_rise = (peak_val - avg_low) / avg_low * 100

                if neckline_rise < 3.0:
                    continue  # Peak not prominent enough

                # Second low should be recent (within last 30% of data)
                recency = idx2 / n
                if recency < 0.5:
                    continue

                # Calculate confidence
                confidence = 50
                confidence += max(0, 10 - spread_pct * 5)  # Tighter lows = higher
                confidence += min(15, neckline_rise)         # Higher neckline = better
                confidence += (recency - 0.5) * 20           # More recent = better

                # Volume confirmation: second low on lower volume is textbook
                if volumes is not None and idx1 < len(volumes) and idx2 < len(volumes):
                    if volumes[idx2] < volumes[idx1] * 0.9:
                        confidence += 10

                # Current price above neckline = breakout confirmation
                current = prices[-1]
                if current > peak_val:
                    confidence += 10

                confidence = min(95, max(0, int(confidence)))

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {
                        'pattern': 'Double Bottom',
                        'confidence': confidence,
                        'direction': 'bullish',
                        'description': (
                            f"Two lows at ~${avg_low:.2f} (spread {spread_pct:.1f}%), "
                            f"neckline at ${peak_val:.2f} ({neckline_rise:.1f}% above lows). "
                            f"{'Breakout confirmed.' if current > peak_val else 'Awaiting neckline breakout.'}"
                        ),
                    }

        return best_result

    def _detect_cup_and_handle(self, pivots: Dict, prices: np.ndarray,
                                volumes: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Cup and handle: U-shaped base followed by a small pullback (handle).
        Bullish continuation pattern.
        """
        n = len(prices)
        if n < 60:
            return None

        # Look for a U-shape in the last 60-120 bars
        search_len = min(120, n)
        segment = prices[-search_len:]
        seg_len = len(segment)

        # Cup: first third should trend down, middle is flat/bottom, last third trends up
        third = seg_len // 3
        left_rim = np.mean(segment[:10])
        cup_bottom = np.min(segment[third:2 * third])
        right_rim = np.mean(segment[2 * third:2 * third + 10])

        if left_rim == 0 or right_rim == 0:
            return None

        # Cup depth should be 12-35% from rim
        avg_rim = (left_rim + right_rim) / 2
        cup_depth_pct = (avg_rim - cup_bottom) / avg_rim * 100

        if cup_depth_pct < 12 or cup_depth_pct > 35:
            return None

        # Rims should be roughly level (within 5%)
        rim_diff_pct = abs(left_rim - right_rim) / avg_rim * 100
        if rim_diff_pct > 5:
            return None

        # Handle: small pullback in last 15-25% of segment
        handle_start = int(seg_len * 0.75)
        handle_segment = segment[handle_start:]

        if len(handle_segment) < 5:
            return None

        handle_high = np.max(handle_segment[:5])
        handle_low = np.min(handle_segment)
        handle_depth_pct = (handle_high - handle_low) / handle_high * 100 if handle_high > 0 else 0

        # Handle should be shallow (less than half of cup depth)
        has_handle = 1 < handle_depth_pct < (cup_depth_pct * 0.5)

        # Calculate confidence
        confidence = 45

        # Cup symmetry
        left_slope = (segment[third] - segment[0]) / third if third > 0 else 0
        right_slope = (segment[2 * third] - segment[third]) / third if third > 0 else 0
        if left_slope < 0 and right_slope > 0:
            confidence += 15

        # Rim alignment
        confidence += max(0, 10 - rim_diff_pct * 3)

        # Cup depth in ideal range
        if 15 <= cup_depth_pct <= 30:
            confidence += 10

        # Handle present
        if has_handle:
            confidence += 15

        # Volume: ideally decreasing during cup, increasing on right side
        if volumes is not None:
            vol_seg = volumes[-search_len:]
            vol_left = np.mean(vol_seg[:third]) if third > 0 else 0
            vol_right = np.mean(vol_seg[2 * third:]) if third > 0 else 0
            if vol_right > vol_left * 1.1:
                confidence += 5

        confidence = min(95, max(0, int(confidence)))

        if confidence < 40:
            return None

        return {
            'pattern': 'Cup and Handle',
            'confidence': confidence,
            'direction': 'bullish',
            'description': (
                f"U-shaped base with {cup_depth_pct:.1f}% depth, "
                f"rims at ~${avg_rim:.2f} (diff {rim_diff_pct:.1f}%). "
                f"{'Handle pullback of {:.1f}% detected.'.format(handle_depth_pct) if has_handle else 'No clear handle yet.'}"
            ),
        }

    def _detect_ascending_triangle(self, pivots: Dict, prices: np.ndarray,
                                    volumes: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Ascending triangle: flat resistance with rising support (higher lows).
        Bullish breakout pattern.
        """
        highs = pivots['highs']
        lows = pivots['lows']

        if len(highs) < 3 or len(lows) < 3:
            return None

        n = len(prices)

        # Use recent pivots (last 60% of data)
        cutoff = int(n * 0.4)
        recent_highs = [(i, p) for i, p in highs if i >= cutoff]
        recent_lows = [(i, p) for i, p in lows if i >= cutoff]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        # Flat resistance: highs should be within 2% of each other
        high_prices = [p for _, p in recent_highs]
        resistance_level = np.mean(high_prices)
        if resistance_level == 0:
            return None
        high_spread_pct = (np.max(high_prices) - np.min(high_prices)) / resistance_level * 100

        if high_spread_pct > 2.5:
            return None

        # Rising support: lows should be increasing
        low_indices = [i for i, _ in recent_lows]
        low_prices_arr = [p for _, p in recent_lows]

        if len(low_indices) < 2:
            return None

        # Fit a line through the lows
        slope, intercept = np.polyfit(low_indices, low_prices_arr, 1)

        # Slope must be positive (rising support)
        if slope <= 0:
            return None

        # Slope significance: support should rise at least 0.5% over the period
        support_rise_pct = (slope * (low_indices[-1] - low_indices[0])) / low_prices_arr[0] * 100 if low_prices_arr[0] > 0 else 0

        if support_rise_pct < 0.5:
            return None

        # Calculate confidence
        confidence = 50
        confidence += max(0, 10 - high_spread_pct * 5)    # Tighter resistance = better
        confidence += min(15, support_rise_pct * 3)         # Steeper support = stronger
        confidence += min(10, (len(recent_highs) - 2) * 5)  # More touches = better

        # Volume: ideally decreasing as triangle tightens
        if volumes is not None:
            vol_first_half = np.mean(volumes[cutoff:cutoff + (n - cutoff) // 2])
            vol_second_half = np.mean(volumes[cutoff + (n - cutoff) // 2:])
            if vol_first_half > 0 and vol_second_half < vol_first_half * 0.9:
                confidence += 5

        # Current price near apex
        current = prices[-1]
        current_support = slope * (n - 1) + intercept
        squeeze_pct = (resistance_level - current_support) / resistance_level * 100 if resistance_level > 0 else 0
        if squeeze_pct < 3:
            confidence += 10  # Near apex = imminent breakout

        confidence = min(95, max(0, int(confidence)))

        if confidence < 40:
            return None

        return {
            'pattern': 'Ascending Triangle',
            'confidence': confidence,
            'direction': 'bullish',
            'description': (
                f"Flat resistance at ~${resistance_level:.2f} (spread {high_spread_pct:.1f}%), "
                f"rising support with {support_rise_pct:.1f}% slope. "
                f"{len(recent_highs)} resistance touches, {len(recent_lows)} support touches. "
                f"{'Near apex — breakout imminent.' if squeeze_pct < 3 else 'Triangle still forming.'}"
            ),
        }

    def _detect_head_and_shoulders(self, pivots: Dict, prices: np.ndarray,
                                    volumes: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Head and shoulders: three peaks with the middle (head) being highest.
        Bearish reversal pattern.
        """
        highs = pivots['highs']
        lows = pivots['lows']

        if len(highs) < 3 or len(lows) < 2:
            return None

        n = len(prices)
        best_confidence = 0
        best_result = None

        # Try all triplets of highs (prefer recent ones)
        recent_highs = [(i, p) for i, p in highs if i > n * 0.2]

        for a in range(len(recent_highs) - 2):
            for b in range(a + 1, len(recent_highs) - 1):
                for c in range(b + 1, len(recent_highs)):
                    idx_l, price_l = recent_highs[a]   # Left shoulder
                    idx_h, price_h = recent_highs[b]   # Head
                    idx_r, price_r = recent_highs[c]   # Right shoulder

                    # Head must be highest
                    if price_h <= price_l or price_h <= price_r:
                        continue

                    # Shoulders should be similar height (within 5%)
                    avg_shoulder = (price_l + price_r) / 2
                    if avg_shoulder == 0:
                        continue
                    shoulder_diff_pct = abs(price_l - price_r) / avg_shoulder * 100
                    if shoulder_diff_pct > 5:
                        continue

                    # Head should be meaningfully higher than shoulders (>3%)
                    head_prominence = (price_h - avg_shoulder) / avg_shoulder * 100
                    if head_prominence < 3:
                        continue

                    # Spacing: peaks should be roughly evenly spaced
                    gap_left = idx_h - idx_l
                    gap_right = idx_r - idx_h
                    if gap_left < 10 or gap_right < 10:
                        continue
                    symmetry = min(gap_left, gap_right) / max(gap_left, gap_right)

                    # Neckline: find lows between shoulders and head
                    neckline_lows = [p for i, p in lows if idx_l < i < idx_r]
                    if len(neckline_lows) < 1:
                        continue
                    neckline = np.mean(neckline_lows)

                    # Right shoulder should be recent
                    recency = idx_r / n
                    if recency < 0.6:
                        continue

                    # Calculate confidence
                    confidence = 45
                    confidence += max(0, 10 - shoulder_diff_pct * 3)  # Symmetrical shoulders
                    confidence += min(15, head_prominence * 2)         # Prominent head
                    confidence += symmetry * 10                        # Even spacing
                    confidence += (recency - 0.6) * 25                 # Recency

                    # Volume: ideally declining from left shoulder to head to right shoulder
                    if volumes is not None:
                        vol_l = np.mean(volumes[max(0, idx_l - 3):idx_l + 3])
                        vol_h = np.mean(volumes[max(0, idx_h - 3):idx_h + 3])
                        vol_r = np.mean(volumes[max(0, idx_r - 3):idx_r + 3])
                        if vol_h < vol_l and vol_r < vol_h:
                            confidence += 10

                    # Current price below neckline = breakdown confirmed
                    current = prices[-1]
                    if current < neckline:
                        confidence += 10

                    confidence = min(95, max(0, int(confidence)))

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            'pattern': 'Head and Shoulders',
                            'confidence': confidence,
                            'direction': 'bearish',
                            'description': (
                                f"Head at ${price_h:.2f}, shoulders at ~${avg_shoulder:.2f} "
                                f"(diff {shoulder_diff_pct:.1f}%), neckline at ${neckline:.2f}. "
                                f"Symmetry {symmetry:.0%}. "
                                f"{'Breakdown confirmed!' if current < neckline else 'Watching for neckline break.'}"
                            ),
                        }

        return best_result


# Singleton
pattern_recognizer = PatternRecognizer()
