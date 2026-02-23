"""
Risk Profiling Engine
Categorizes stocks into Volatility Buckets and Risk Profiles using
Beta, Max Drawdown, and Debt-to-Equity metrics.
"""
import yfinance as yf
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RiskProfiler:
    def __init__(self):
        self._benchmark_cache = None

    def _get_spy_history(self) -> np.ndarray:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        try:
            spy = yf.Ticker("SPY").history(period="1y")
            if not spy.empty:
                pct_change = spy['Close'].pct_change().dropna().values
                self._benchmark_cache = pct_change
                return pct_change
        except Exception as e:
            logger.warning(f"Could not fetch SPY history for Beta calc: {e}")
        return np.array([])

    def calculate_risk_profile(self, ticker: str) -> str:
        """Fetch data and determine the risk profile badge."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                return "Unknown"

            # 1. Beta calculation
            beta = info.get('beta')
            if beta is None:
                # Calculate manually if missing
                spy_returns = self._get_spy_history()
                stock_returns = hist['Close'].pct_change().dropna().values
                min_len = min(len(spy_returns), len(stock_returns))
                if min_len > 20:
                    cov = np.cov(stock_returns[-min_len:], spy_returns[-min_len:])[0][1]
                    var = np.var(spy_returns[-min_len:])
                    beta = cov / var if var > 0 else 1.0
                else:
                    beta = 1.0

            # 2. Max Drawdown
            cumulative = (1 + hist['Close'].pct_change().fillna(0)).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min() * 100  # Percentage (negative)

            # 3. Debt load
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity is None:
                debt_to_equity = 0

            # 4. Market Cap
            market_cap = info.get('marketCap', 0)

            # Categorization Logic
            # Blue Chip: High cap, low beta, low dd
            if market_cap > 50e9 and beta < 1.15 and max_dd > -30:
                return "Blue Chip"
            
            # Value / Distressed: High debt or bad drawdown but historically stable, or very low P/E
            pe_ratio = info.get('trailingPE', 0)
            if (debt_to_equity > 150 or max_dd < -45) and (pe_ratio and pe_ratio < 15):
                return "Deep Value / Distressed"
            elif pe_ratio and pe_ratio < 15 and beta < 1.2:
                return "Value"

            # Speculative: High Beta, High DD, small cap
            if beta > 1.8 or max_dd < -60 or (market_cap < 2e9 and beta > 1.5):
                return "Speculative"
            
            # High Growth: High Beta, Medium/Large cap
            if beta > 1.3:
                return "High Growth"

            return "Moderate Risk"

        except Exception as e:
            logger.error(f"Failed to calculate risk profile for {ticker}: {e}")
            return "Unknown"

# Singleton
risk_profiler = RiskProfiler()
