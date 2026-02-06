"""
Correlation Matrix & Portfolio Concentration
NVDA and AMD both in "Technology" at 10% each looks fine, but they're 0.85+ correlated.
This means you effectively have a 20% position in a single risk factor.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between portfolio holdings."""

    def __init__(self):
        self._correlation_cache = {}
        self._price_history_cache = {}
        self._cache_duration = timedelta(hours=6)
        self.lookback_days = 90  # 90-day rolling correlation

    def get_correlation_matrix(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for list of tickers."""
        if len(tickers) < 2:
            return None

        cache_key = '|'.join(sorted(tickers))
        if cache_key in self._correlation_cache:
            entry = self._correlation_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            # Get price history for all tickers
            price_data = {}
            for ticker in tickers:
                hist = self._get_price_history(ticker)
                if hist is not None and not hist.empty:
                    price_data[ticker] = hist['Close']

            if len(price_data) < 2:
                return None

            # Create DataFrame with aligned dates
            df = pd.DataFrame(price_data)
            
            # Calculate daily returns
            returns = df.pct_change().dropna()
            
            if returns.empty or len(returns) < 20:
                return None

            # Calculate correlation matrix
            corr_matrix = returns.corr()

            # Cache result
            self._correlation_cache[cache_key] = {
                'data': corr_matrix,
                'timestamp': datetime.now()
            }

            # Store in database
            self._store_correlation_matrix(corr_matrix)

            return corr_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None

    def _get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get price history with caching."""
        if ticker in self._price_history_cache:
            entry = self._price_history_cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            hist = yf.Ticker(ticker).history(period=f"{self.lookback_days}d")
            self._price_history_cache[ticker] = {
                'data': hist,
                'timestamp': datetime.now()
            }
            return hist
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return None

    def _store_correlation_matrix(self, corr_matrix: pd.DataFrame):
        """Store correlation pairs in database."""
        try:
            tickers = corr_matrix.columns.tolist()
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:  # Only upper triangle
                    correlation = float(corr_matrix.loc[ticker1, ticker2])
                    db.execute("""
                        INSERT INTO correlation_matrix (ticker1, ticker2, correlation, calculated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(ticker1, ticker2) DO UPDATE SET
                            correlation = excluded.correlation,
                            calculated_at = excluded.calculated_at
                    """, (ticker1, ticker2, correlation, datetime.now().isoformat()))
        except Exception as e:
            logger.warning(f"Could not store correlation matrix: {e}")

    def find_high_correlations(self, holdings: List[Dict], threshold: float = 0.75) -> List[Dict]:
        """Find pairs of stocks with high correlation that create concentration risk."""
        if len(holdings) < 2:
            return []

        tickers = [h['ticker'] for h in holdings]
        corr_matrix = self.get_correlation_matrix(tickers)

        if corr_matrix is None:
            return []

        high_corr_pairs = []
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                try:
                    correlation = float(corr_matrix.loc[ticker1, ticker2])
                    
                    if abs(correlation) >= threshold:
                        # Get position sizes
                        h1 = next(h for h in holdings if h['ticker'] == ticker1)
                        h2 = next(h for h in holdings if h['ticker'] == ticker2)
                        
                        combined_pct = h1.get('position_pct', 0) + h2.get('position_pct', 0)
                        
                        high_corr_pairs.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': round(correlation, 3),
                            'position1_pct': h1.get('position_pct', 0),
                            'position2_pct': h2.get('position_pct', 0),
                            'combined_pct': round(combined_pct, 1),
                            'effective_risk': round(combined_pct * abs(correlation), 1),
                            'type': 'positive' if correlation > 0 else 'negative',
                        })
                except (KeyError, StopIteration):
                    continue

        # Sort by effective risk
        high_corr_pairs.sort(key=lambda x: x['effective_risk'], reverse=True)
        
        return high_corr_pairs

    def calculate_effective_position_size(self, ticker: str, holdings: List[Dict]) -> float:
        """
        Calculate effective position size accounting for correlations.
        A 10% position that's 0.8 correlated with another 10% position
        effectively creates 18% risk exposure (10 + 10*0.8).
        """
        if len(holdings) < 2:
            target_holding = next((h for h in holdings if h['ticker'] == ticker), None)
            return target_holding.get('position_pct', 0) if target_holding else 0

        tickers = [h['ticker'] for h in holdings]
        corr_matrix = self.get_correlation_matrix(tickers)

        if corr_matrix is None:
            target_holding = next((h for h in holdings if h['ticker'] == ticker), None)
            return target_holding.get('position_pct', 0) if target_holding else 0

        try:
            target_holding = next(h for h in holdings if h['ticker'] == ticker)
            base_pct = target_holding.get('position_pct', 0)
            
            # Add correlated exposure from other holdings
            correlated_exposure = base_pct
            
            for other_ticker in tickers:
                if other_ticker == ticker:
                    continue
                
                correlation = abs(float(corr_matrix.loc[ticker, other_ticker]))
                other_holding = next(h for h in holdings if h['ticker'] == other_ticker)
                other_pct = other_holding.get('position_pct', 0)
                
                # Add portion of other position that correlates
                correlated_exposure += other_pct * correlation

            return round(correlated_exposure, 1)

        except (KeyError, StopIteration) as e:
            logger.warning(f"Error calculating effective position for {ticker}: {e}")
            target_holding = next((h for h in holdings if h['ticker'] == ticker), None)
            return target_holding.get('position_pct', 0) if target_holding else 0

    def generate_correlation_alerts(self, holdings: List[Dict], 
                                   threshold: float = 0.75,
                                   combined_limit: float = 15.0) -> List[Dict]:
        """Generate alerts for high-correlation pairs that exceed combined limit."""
        high_corr = self.find_high_correlations(holdings, threshold)
        
        alerts = []
        for pair in high_corr:
            if pair['combined_pct'] > combined_limit and pair['type'] == 'positive':
                alerts.append({
                    'type': 'CORRELATION_RISK',
                    'severity': 'WARNING' if pair['combined_pct'] < combined_limit * 1.5 else 'CRITICAL',
                    'ticker1': pair['ticker1'],
                    'ticker2': pair['ticker2'],
                    'correlation': pair['correlation'],
                    'combined_pct': pair['combined_pct'],
                    'effective_risk': pair['effective_risk'],
                    'message': f"{pair['ticker1']} and {pair['ticker2']} are {pair['correlation']:.1%} correlated "
                               f"with combined position of {pair['combined_pct']:.1f}% "
                               f"(effective risk: {pair['effective_risk']:.1f}%)"
                })

        return alerts

    def get_diversification_score(self, holdings: List[Dict]) -> Optional[int]:
        """
        Calculate portfolio diversification score (0-100).
        Lower correlations = higher score.
        """
        if len(holdings) < 2:
            return 0

        tickers = [h['ticker'] for h in holdings]
        corr_matrix = self.get_correlation_matrix(tickers)

        if corr_matrix is None:
            return None

        try:
            # Get all correlation values (upper triangle)
            correlations = []
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:
                    corr = abs(float(corr_matrix.loc[ticker1, ticker2]))
                    correlations.append(corr)

            if not correlations:
                return None

            # Average absolute correlation
            avg_corr = np.mean(correlations)
            
            # Convert to score: 0 correlation = 100 score, 1 correlation = 0 score
            score = int((1 - avg_corr) * 100)
            
            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return None


# Singleton
correlation_analyzer = CorrelationAnalyzer()
