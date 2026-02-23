"""
Position Sizing Module
Calculates optimal position sizes using Kelly Criterion and volatility-adjusted methods.
The goal: Never bet too much on any single position.
"""
import math
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates position sizes based on:
    1. Kelly Criterion (optimal betting based on edge and odds)
    2. Volatility adjustment (reduce size for volatile stocks)
    3. Portfolio concentration limits (max % in any single stock)
    """
    
    def __init__(self, cache_minutes: int = 30):
        self._vol_cache = {}
        self._cache_duration = timedelta(minutes=cache_minutes)
        
        # Default config
        self.config = {
            'max_position_pct': 10.0,      # Never more than 10% in one stock
            'min_position_pct': 1.0,       # Minimum meaningful position
            'kelly_fraction': 0.25,         # Use quarter-Kelly (conservative)
            'target_volatility': 20.0,      # Target annualized vol (%)
            'max_sector_pct': 30.0,         # Max in one sector
        }
    
    def calculate_position_size(
        self,
        ticker: str,
        signal_confidence: int,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        portfolio_value: float = 100000,
    ) -> Dict:
        """
        Calculate recommended position size for a trade.
        
        Args:
            ticker: Stock ticker
            signal_confidence: Signal confidence 0-100
            win_rate: Historical win rate (0-1), uses backtest data if None
            avg_win: Average winning trade % (e.g., 5.0 for 5%)
            avg_loss: Average losing trade % (e.g., -3.0 for -3%)
            portfolio_value: Total portfolio value
        
        Returns:
            Dict with position size recommendations
        """
        # Get volatility for the stock
        vol_data = self._get_volatility(ticker)
        annual_vol = vol_data.get('annual_vol', 30.0)  # Default 30% if unknown
        
        # Default win/loss stats from signal confidence
        if win_rate is None:
            # Estimate win rate from confidence
            # Higher confidence = higher assumed win rate
            win_rate = 0.45 + (signal_confidence / 200)  # 45% to 95%
        
        if avg_win is None:
            avg_win = 5.0  # Default 5% winner
        if avg_loss is None:
            avg_loss = -3.0  # Default -3% loser
        
        # Calculate Kelly position
        kelly_pct = self._kelly_criterion(win_rate, avg_win, abs(avg_loss))
        
        # Apply fractional Kelly (more conservative)
        fractional_kelly = kelly_pct * self.config['kelly_fraction']
        
        # Volatility adjustment
        vol_multiplier = self.config['target_volatility'] / annual_vol if annual_vol > 0 else 1.0
        vol_adjusted = fractional_kelly * vol_multiplier
        
        # Confidence adjustment
        conf_multiplier = signal_confidence / 70  # Baseline is 70% confidence
        conf_adjusted = vol_adjusted * min(conf_multiplier, 1.5)  # Cap at 1.5x
        
        # Apply limits
        final_pct = max(
            self.config['min_position_pct'],
            min(self.config['max_position_pct'], conf_adjusted)
        )
        
        # Round to sensible precision
        final_pct = round(final_pct, 1)
        
        # Calculate dollar amounts
        position_value = portfolio_value * (final_pct / 100)
        current_price = vol_data.get('current_price', 0)
        shares = int(position_value / current_price) if current_price > 0 else 0
        
        # Risk assessment
        risk_level = 'low' if final_pct <= 3 else 'medium' if final_pct <= 6 else 'high'
        
        return {
            'ticker': ticker,
            'recommended_pct': final_pct,
            'position_value': round(position_value, 2),
            'shares': shares,
            'current_price': current_price,
            'risk_level': risk_level,
            
            # Breakdown
            'kelly_raw': round(kelly_pct, 2),
            'kelly_fractional': round(fractional_kelly, 2),
            'volatility_adj': round(vol_adjusted, 2),
            'confidence_adj': round(conf_adjusted, 2),
            
            # Inputs used
            'inputs': {
                'signal_confidence': signal_confidence,
                'win_rate': round(win_rate, 3),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'annual_volatility': round(annual_vol, 1),
            },
            
            # Limits applied
            'limits': {
                'max_position': self.config['max_position_pct'],
                'min_position': self.config['min_position_pct'],
                'kelly_fraction': self.config['kelly_fraction'],
            },
        }
    
    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion optimal bet size.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = win probability
            R = win/loss ratio (average win / average loss)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        
        # Ensure win_rate is bounded
        win_rate = max(0.01, min(0.99, win_rate))
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Kelly can be negative (don't bet) or very high (risky)
        kelly = max(0, min(kelly * 100, 50))  # Cap at 50%
        
        return kelly
    
    def _get_volatility(self, ticker: str) -> Dict:
        """Get annualized volatility for a stock."""
        cache_key = f"vol_{ticker}"
        
        # Check cache
        if cache_key in self._vol_cache:
            entry = self._vol_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty or len(hist) < 20:
                return {'annual_vol': 30.0, 'current_price': 0, 'error': 'Insufficient data'}
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Annualized volatility (daily vol * sqrt(252))
            daily_vol = returns.std()
            annual_vol = daily_vol * math.sqrt(252) * 100
            
            current_price = float(hist['Close'].iloc[-1])
            
            result = {
                'annual_vol': annual_vol,
                'daily_vol': daily_vol * 100,
                'current_price': current_price,
            }
            
            # Cache it
            self._vol_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now(),
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Volatility calc error for {ticker}: {e}")
            return {'annual_vol': 30.0, 'current_price': 0, 'error': str(e)}
    
    def get_portfolio_allocation(
        self,
        signals: list,
        portfolio_value: float = 100000,
        existing_positions: dict = None,
    ) -> Dict:
        """
        Calculate allocation for multiple signals considering portfolio limits.
        
        Args:
            signals: List of {ticker, confidence, signal_type}
            portfolio_value: Total portfolio value
            existing_positions: Dict of {ticker: current_pct}
        """
        if existing_positions is None:
            existing_positions = {}
        
        allocations = []
        total_allocated = sum(existing_positions.values())
        remaining = 100 - total_allocated
        
        for sig in signals:
            ticker = sig.get('ticker')
            confidence = sig.get('confidence', 50)
            
            # Skip if already have position
            if ticker in existing_positions:
                continue
            
            # Get recommended size
            sizing = self.calculate_position_size(
                ticker=ticker,
                signal_confidence=confidence,
                portfolio_value=portfolio_value,
            )
            
            # Adjust for remaining capacity
            adjusted_pct = min(sizing['recommended_pct'], remaining * 0.3)  # Max 30% of remaining
            
            if adjusted_pct >= self.config['min_position_pct']:
                allocations.append({
                    'ticker': ticker,
                    'recommended_pct': round(adjusted_pct, 1),
                    'position_value': round(portfolio_value * adjusted_pct / 100, 2),
                    'shares': sizing['shares'],
                    'confidence': confidence,
                    'risk_level': sizing['risk_level'],
                })
                remaining -= adjusted_pct
        
        return {
            'allocations': allocations,
            'total_new_allocation': round(100 - remaining - total_allocated, 1),
            'remaining_cash_pct': round(remaining, 1),
            'existing_positions_pct': round(total_allocated, 1),
        }


# Singleton
position_sizer = PositionSizer()
