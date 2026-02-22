"""
Value at Risk (VaR) Calculator
Historical simulation VaR from portfolio price history.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from core.database import db

logger = logging.getLogger(__name__)


class VaRCalculator:
    """
    Calculates portfolio Value at Risk using historical simulation.

    VaR answers: "What is the worst expected loss over a given period
    at a given confidence level?"
    """

    def __init__(self, cache_minutes: int = 30):
        self._cache = {}
        self._cache_duration = timedelta(minutes=cache_minutes)

    def calculate_var(
        self,
        tickers: List[str],
        weights: Optional[List[float]] = None,
        confidence: float = 0.95,
        days: int = 1,
        lookback_days: int = 252,
    ) -> Dict:
        """
        Calculate historical simulation VaR for a weighted portfolio.

        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (must sum to 1). Equal weight if None.
            confidence: Confidence level (e.g. 0.95 for 95%)
            days: Holding period in days
            lookback_days: Number of historical trading days to use

        Returns:
            Dict with VaR metrics and portfolio statistics
        """
        if not tickers:
            return {'error': 'No tickers provided'}

        # Equal weight if none specified
        if weights is None:
            weights = [1.0 / len(tickers)] * len(tickers)

        weights = np.array(weights)

        # Fetch price history and compute daily returns per ticker
        all_returns = []
        failed_tickers = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=f"{lookback_days + 30}d")

                if hist.empty or len(hist) < 20:
                    failed_tickers.append(ticker)
                    continue

                # Trim to lookback window
                hist = hist.tail(lookback_days + 1)
                returns = hist['Close'].pct_change().dropna().values
                all_returns.append(returns)
            except Exception as e:
                logger.warning(f"VaR fetch error for {ticker}: {e}")
                failed_tickers.append(ticker)

        if not all_returns:
            return {'error': 'Could not fetch data for any tickers'}

        # Align return series to the shortest length
        min_len = min(len(r) for r in all_returns)
        aligned = np.column_stack([r[-min_len:] for r in all_returns])

        # Adjust weights if some tickers failed (redistribute proportionally)
        if failed_tickers:
            valid_mask = [t not in failed_tickers for t in tickers]
            weights = weights[valid_mask]
            weights = weights / weights.sum()

        # Portfolio returns as weighted sum
        portfolio_returns = aligned @ weights

        # VaR at the given confidence level
        var_daily = float(np.percentile(portfolio_returns, (1 - confidence) * 100))
        var_weekly = var_daily * np.sqrt(5)
        var_monthly = var_daily * np.sqrt(21)

        return {
            'var_daily': round(var_daily, 6),
            'var_weekly': round(var_weekly, 6),
            'var_monthly': round(var_monthly, 6),
            'confidence': confidence,
            'worst_day': round(float(np.min(portfolio_returns)), 6),
            'best_day': round(float(np.max(portfolio_returns)), 6),
            'avg_return': round(float(np.mean(portfolio_returns)), 6),
            'volatility': round(float(np.std(portfolio_returns)), 6),
            'lookback_days': min_len,
            'tickers_used': [t for t in tickers if t not in failed_tickers],
            'tickers_failed': failed_tickers,
            'calculated_at': datetime.now().isoformat(),
        }

    def calculate_portfolio_var(self, confidence: float = 0.95, lookback_days: int = 252) -> Dict:
        """
        Calculate VaR for current portfolio holdings from DB.
        Weights are derived from position sizes (shares * avg_price).
        """
        try:
            holdings = db.get_portfolio_holdings()
            active = [h for h in holdings if h['shares'] > 0]

            if not active:
                return {'error': 'No active portfolio holdings found'}

            tickers = [h['ticker'] for h in active]
            position_values = [h['shares'] * h['avg_price'] for h in active]
            total_value = sum(position_values)

            if total_value <= 0:
                return {'error': 'Portfolio has no positive value'}

            weights = [v / total_value for v in position_values]

            result = self.calculate_var(
                tickers=tickers,
                weights=weights,
                confidence=confidence,
                lookback_days=lookback_days,
            )
            result['portfolio_value'] = round(total_value, 2)
            result['holdings_count'] = len(active)

            return result

        except Exception as e:
            logger.error(f"Portfolio VaR calculation error: {e}")
            return {'error': str(e)}


# Singleton
var_calculator = VaRCalculator()
