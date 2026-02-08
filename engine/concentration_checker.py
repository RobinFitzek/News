"""
Correlation & Concentration Risk Checker
Flags sector over-concentration, high pairwise correlations, and position-size risks.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConcentrationChecker:
    """Detects concentration and correlation risks in recommendations and portfolio."""

    def __init__(self):
        self._sector_cache: Dict = {}
        self._sector_cache_time: Optional[datetime] = None
        self._sector_cache_duration = timedelta(hours=24)

    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker, with 24h cache."""
        if self._sector_cache_time and datetime.now() - self._sector_cache_time > self._sector_cache_duration:
            self._sector_cache.clear()
            self._sector_cache_time = None

        if ticker in self._sector_cache:
            return self._sector_cache[ticker]

        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
        except Exception:
            sector = 'Unknown'

        self._sector_cache[ticker] = sector
        if not self._sector_cache_time:
            self._sector_cache_time = datetime.now()
        return sector

    def _get_correlation_matrix(self, tickers: List[str], days: int = 60) -> Optional[Dict]:
        """Calculate pairwise correlations from 60-day close prices."""
        if len(tickers) < 2:
            return None

        try:
            end = datetime.now()
            start = end - timedelta(days=days + 10)
            data = yf.download(tickers, start=start, end=end, progress=False)

            if data.empty:
                return None

            close = data['Close'] if 'Close' in data.columns else data
            if hasattr(close, 'columns') and len(close.columns) >= 2:
                corr = close.corr()
                return corr.to_dict()
        except Exception as e:
            logger.warning(f"Correlation calc failed: {e}")
        return None

    def check_concentration(self, recommendations: List[Dict]) -> Dict:
        """Check concentration risk in a set of recommendations.

        Args:
            recommendations: list of dicts with at least 'ticker' key

        Returns:
            dict with warnings, sector_breakdown, correlation_pairs, diversification_score
        """
        if not recommendations:
            return {'warnings': [], 'sector_breakdown': {}, 'diversification_score': 100}

        tickers = [r.get('ticker', '') for r in recommendations if r.get('ticker')]
        if not tickers:
            return {'warnings': [], 'sector_breakdown': {}, 'diversification_score': 100}

        warnings = []

        # Sector concentration
        sector_counts: Dict[str, List[str]] = {}
        for t in tickers:
            sector = self._get_sector(t)
            sector_counts.setdefault(sector, []).append(t)

        total = len(tickers)
        sector_breakdown = {}
        for sector, sector_tickers in sector_counts.items():
            pct = round(len(sector_tickers) / total * 100, 1)
            sector_breakdown[sector] = {
                'count': len(sector_tickers),
                'percentage': pct,
                'tickers': sector_tickers,
            }
            if pct > 40:
                warnings.append({
                    'type': 'sector_concentration',
                    'severity': 'HIGH',
                    'message': f"{sector} represents {pct}% of recommendations ({', '.join(sector_tickers)})",
                })

        # Pairwise correlations
        high_correlation_pairs = []
        if len(tickers) >= 2:
            corr_matrix = self._get_correlation_matrix(tickers[:20])  # cap at 20 tickers
            if corr_matrix:
                checked = set()
                for t1 in tickers:
                    for t2 in tickers:
                        if t1 >= t2 or (t1, t2) in checked:
                            continue
                        checked.add((t1, t2))
                        try:
                            corr_val = corr_matrix.get(t1, {}).get(t2)
                            if corr_val is not None and abs(corr_val) > 0.8:
                                high_correlation_pairs.append({
                                    'ticker1': t1,
                                    'ticker2': t2,
                                    'correlation': round(corr_val, 3),
                                })
                                warnings.append({
                                    'type': 'high_correlation',
                                    'severity': 'MEDIUM',
                                    'message': f"{t1} and {t2} are highly correlated ({corr_val:.2f})",
                                })
                        except (KeyError, TypeError):
                            continue

        # Diversification score: 100 = perfect, penalise for concentration and correlations
        score = 100
        n_sectors = len([s for s in sector_breakdown if s != 'Unknown'])
        if n_sectors <= 1 and total > 1:
            score -= 40
        elif n_sectors <= 2 and total > 3:
            score -= 20

        for sb in sector_breakdown.values():
            if sb['percentage'] > 50:
                score -= 25
            elif sb['percentage'] > 40:
                score -= 15

        score -= len(high_correlation_pairs) * 10
        score = max(0, min(100, score))

        return {
            'warnings': warnings,
            'sector_breakdown': sector_breakdown,
            'correlation_pairs': high_correlation_pairs,
            'diversification_score': score,
        }

    def check_portfolio_concentration(self, holdings: List[Dict]) -> Dict:
        """Check concentration risk in current portfolio holdings.

        Args:
            holdings: list of dicts with 'ticker', 'shares', 'total_cost' etc.

        Returns:
            dict with warnings, sector_breakdown, position_sizes, diversification_score
        """
        active = [h for h in holdings if h.get('shares', 0) > 0]
        if not active:
            return {'warnings': [], 'sector_breakdown': {}, 'diversification_score': 100, 'position_sizes': []}

        warnings = []
        total_value = sum(h.get('total_cost', 0) for h in active)

        # Position size checks
        position_sizes = []
        for h in active:
            cost = h.get('total_cost', 0)
            pct = round(cost / total_value * 100, 1) if total_value > 0 else 0
            position_sizes.append({
                'ticker': h['ticker'],
                'value': round(cost, 2),
                'percentage': pct,
            })
            if pct > 25:
                warnings.append({
                    'type': 'position_size',
                    'severity': 'HIGH',
                    'message': f"{h['ticker']} is {pct}% of portfolio (recommend < 25%)",
                })

        # Sector concentration
        sector_values: Dict[str, float] = {}
        sector_tickers: Dict[str, List[str]] = {}
        for h in active:
            sector = self._get_sector(h['ticker'])
            sector_values[sector] = sector_values.get(sector, 0) + h.get('total_cost', 0)
            sector_tickers.setdefault(sector, []).append(h['ticker'])

        sector_breakdown = {}
        for sector, value in sector_values.items():
            pct = round(value / total_value * 100, 1) if total_value > 0 else 0
            sector_breakdown[sector] = {
                'value': round(value, 2),
                'percentage': pct,
                'tickers': sector_tickers.get(sector, []),
            }
            if pct > 40:
                warnings.append({
                    'type': 'sector_concentration',
                    'severity': 'HIGH',
                    'message': f"{sector} sector is {pct}% of portfolio",
                })

        # Diversification score
        score = 100
        n_sectors = len([s for s in sector_breakdown if s != 'Unknown'])
        if n_sectors <= 1 and len(active) > 1:
            score -= 40
        for sb in sector_breakdown.values():
            if sb['percentage'] > 50:
                score -= 25
            elif sb['percentage'] > 40:
                score -= 15
        for ps in position_sizes:
            if ps['percentage'] > 30:
                score -= 15
            elif ps['percentage'] > 25:
                score -= 10
        score = max(0, min(100, score))

        return {
            'warnings': warnings,
            'sector_breakdown': sector_breakdown,
            'position_sizes': position_sizes,
            'diversification_score': score,
        }


# Singleton
concentration_checker = ConcentrationChecker()
