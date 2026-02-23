"""
Time Series Visibility Graph Indicator

Converts price series to natural visibility graphs and computes
Average Shortest Path Length (ASPL) as a topological indicator.
Positive ASPL > Negative ASPL = bullish structure.

Based on: neurotrader888/TimeSeriesVisibilityGraphs
"""
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class VisibilityGraphAnalyzer:
    """
    Computes rolling visibility graph metrics on price data.
    Uses ts2vg for graph construction and networkx for path analysis.
    """

    def __init__(self, lookback: int = 12, cache_minutes: int = 30):
        self.lookback = lookback
        self._cache: Dict[str, Dict] = {}
        self._cache_duration = timedelta(minutes=cache_minutes)
        self._libs_available = None

        # Load config
        try:
            from core.config import VISIBILITY_GRAPH_CONFIG
            self.lookback = VISIBILITY_GRAPH_CONFIG.get('lookback_bars', lookback)
            cache_min = VISIBILITY_GRAPH_CONFIG.get('cache_minutes', cache_minutes)
            self._cache_duration = timedelta(minutes=cache_min)
        except (ImportError, AttributeError):
            pass

    def _check_libs(self) -> bool:
        """Check if ts2vg and networkx are available."""
        if self._libs_available is not None:
            return self._libs_available
        try:
            import ts2vg  # noqa: F401
            import networkx  # noqa: F401
            self._libs_available = True
        except ImportError:
            logger.warning("ts2vg or networkx not installed. Visibility graph analysis disabled.")
            self._libs_available = False
        return self._libs_available

    def analyze(self, ticker: str, hist, lookback: int = None) -> Dict:
        """
        Compute visibility graph metrics for a ticker.
        Returns ASPL ratio, VG signal, and VG score (0-100).
        """
        if not self._check_libs():
            return {}

        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            if hist is None or hist.empty:
                return {}

            lb = lookback or self.lookback
            closes = hist['Close'].values.astype(float)

            if len(closes) < lb + 5:
                return {}

            # Use the most recent window
            window = closes[-(lb + 5):]

            pos_aspl, neg_aspl = self._compute_aspl_pair(window, lb)

            if pos_aspl is None or neg_aspl is None:
                return {}

            # Compute ratio
            aspl_ratio = pos_aspl / neg_aspl if neg_aspl > 0 else 1.0

            # Determine signal
            vg_signal = 'bullish' if pos_aspl > neg_aspl else 'bearish'

            # Map to score (0-100)
            vg_score = self._ratio_to_score(aspl_ratio)

            result = {
                'pos_aspl': round(pos_aspl, 4),
                'neg_aspl': round(neg_aspl, 4),
                'aspl_ratio': round(aspl_ratio, 4),
                'vg_signal': vg_signal,
                'vg_score': vg_score,
                'lookback': lb,
            }

            self._cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
            return result

        except Exception as e:
            logger.warning(f"Visibility graph analysis failed for {ticker}: {e}")
            return {}

    def _compute_aspl_pair(self, prices: np.ndarray, lookback: int):
        """
        Compute positive and negative ASPL over a rolling window.
        Returns the average of the last few windows for stability.
        """
        import ts2vg
        import networkx as nx

        pos_aspls = []
        neg_aspls = []

        # Compute over last 3-5 windows for smoothing
        n_windows = min(5, len(prices) - lookback + 1)

        for i in range(n_windows):
            start = len(prices) - lookback - (n_windows - 1 - i)
            end = start + lookback
            if start < 0:
                continue

            window = prices[start:end]

            try:
                # Positive graph (normal prices)
                vg_pos = ts2vg.NaturalVG()
                vg_pos.build(window)
                G_pos = vg_pos.as_networkx()
                if nx.is_connected(G_pos) and G_pos.number_of_nodes() > 1:
                    pos_aspls.append(nx.average_shortest_path_length(G_pos))

                # Negative graph (inverted prices — visibility from below)
                vg_neg = ts2vg.NaturalVG()
                vg_neg.build(-window)
                G_neg = vg_neg.as_networkx()
                if nx.is_connected(G_neg) and G_neg.number_of_nodes() > 1:
                    neg_aspls.append(nx.average_shortest_path_length(G_neg))
            except Exception:
                continue

        if not pos_aspls or not neg_aspls:
            return None, None

        return float(np.mean(pos_aspls)), float(np.mean(neg_aspls))

    @staticmethod
    def _ratio_to_score(ratio: float) -> int:
        """Map ASPL ratio to 0-100 score."""
        if ratio > 1.3:
            return 85
        elif ratio > 1.15:
            return 75
        elif ratio > 1.05:
            return 65
        elif ratio > 0.95:
            return 50
        elif ratio > 0.85:
            return 35
        elif ratio > 0.7:
            return 25
        else:
            return 15


# Singleton instance
vg_analyzer = VisibilityGraphAnalyzer()
