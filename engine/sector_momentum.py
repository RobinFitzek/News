"""
Sector Momentum Overlay
Tracks sector ETF performance to provide market context for signals.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# SPDR Sector ETFs covering 11 GICS sectors
SECTOR_ETFS = {
    'XLK': {'name': 'Technology', 'color': '#3b82f6'},
    'XLF': {'name': 'Financials', 'color': '#22c55e'},
    'XLE': {'name': 'Energy', 'color': '#f97316'},
    'XLV': {'name': 'Healthcare', 'color': '#ec4899'},
    'XLI': {'name': 'Industrials', 'color': '#a855f7'},
    'XLC': {'name': 'Communication', 'color': '#14b8a6'},
    'XLY': {'name': 'Consumer Disc.', 'color': '#eab308'},
    'XLP': {'name': 'Consumer Staples', 'color': '#84cc16'},
    'XLU': {'name': 'Utilities', 'color': '#06b6d4'},
    'XLRE': {'name': 'Real Estate', 'color': '#f43f5e'},
    'XLB': {'name': 'Materials', 'color': '#8b5cf6'},
}

# Ticker to sector mapping (common stocks)
TICKER_SECTOR_MAP = {
    # Technology
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'GOOG': 'XLK', 'META': 'XLK',
    'NVDA': 'XLK', 'AVGO': 'XLK', 'CSCO': 'XLK', 'ADBE': 'XLK', 'CRM': 'XLK',
    'ORCL': 'XLK', 'ACN': 'XLK', 'IBM': 'XLK', 'INTC': 'XLK', 'AMD': 'XLK',
    # Financials
    'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
    'C': 'XLF', 'SCHW': 'XLF', 'BLK': 'XLF', 'AXP': 'XLF', 'V': 'XLF', 'MA': 'XLF',
    # Energy
    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'SLB': 'XLE', 'EOG': 'XLE',
    # Healthcare
    'UNH': 'XLV', 'JNJ': 'XLV', 'PFE': 'XLV', 'ABBV': 'XLV', 'LLY': 'XLV',
    'MRK': 'XLV', 'TMO': 'XLV', 'ABT': 'XLV', 'DHR': 'XLV', 'BMY': 'XLV',
    # Consumer Discretionary
    'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY',
    'SBUX': 'XLY', 'TGT': 'XLY', 'LOW': 'XLY',
    # Consumer Staples
    'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'COST': 'XLP', 'WMT': 'XLP',
    # Communication
    'NFLX': 'XLC', 'DIS': 'XLC', 'T': 'XLC', 'VZ': 'XLC', 'CMCSA': 'XLC',
    # Industrials
    'CAT': 'XLI', 'BA': 'XLI', 'HON': 'XLI', 'UPS': 'XLI', 'GE': 'XLI',
    'RTX': 'XLI', 'DE': 'XLI', 'MMM': 'XLI', 'LMT': 'XLI',
    # Utilities
    'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'D': 'XLU',
    # Real Estate
    'AMT': 'XLRE', 'PLD': 'XLRE', 'CCI': 'XLRE', 'EQIX': 'XLRE',
    # Materials
    'LIN': 'XLB', 'APD': 'XLB', 'SHW': 'XLB', 'FCX': 'XLB', 'NEM': 'XLB',
}


class SectorMomentum:
    """Tracks sector ETF momentum for market context."""
    
    def __init__(self, cache_minutes: int = 15):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=cache_minutes)
    
    def _is_cache_valid(self) -> bool:
        if not self._cache or not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_duration
    
    def get_sector_rankings(self) -> List[Dict]:
        """
        Get all sectors ranked by momentum.
        Returns list sorted by 1-month return, best first.
        """
        if self._is_cache_valid() and 'rankings' in self._cache:
            return self._cache['rankings']
        
        rankings = []
        spy_return = self._get_return('SPY', '1mo')
        
        for etf, info in SECTOR_ETFS.items():
            try:
                return_1mo = self._get_return(etf, '1mo')
                return_1wk = self._get_return(etf, '5d')
                return_3mo = self._get_return(etf, '3mo')
                
                relative_strength = (return_1mo - spy_return) if spy_return else 0
                
                rankings.append({
                    'etf': etf,
                    'name': info['name'],
                    'color': info['color'],
                    'return_1wk': round(return_1wk, 2) if return_1wk else 0,
                    'return_1mo': round(return_1mo, 2) if return_1mo else 0,
                    'return_3mo': round(return_3mo, 2) if return_3mo else 0,
                    'relative_strength': round(relative_strength, 2),
                    'momentum': 'hot' if return_1mo > 3 else 'cold' if return_1mo < -3 else 'neutral',
                })
            except Exception as e:
                logger.warning(f"Error fetching {etf}: {e}")
                continue
        
        # Sort by 1-month return
        rankings.sort(key=lambda x: x['return_1mo'], reverse=True)
        
        # Assign rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1
        
        self._cache['rankings'] = rankings
        self._cache_time = datetime.now()
        
        return rankings
    
    def _get_return(self, ticker: str, period: str) -> Optional[float]:
        """Get return for a ticker over a period."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty or len(hist) < 2:
                return None
            start = hist['Close'].iloc[0]
            end = hist['Close'].iloc[-1]
            if start <= 0:
                return None
            return ((end - start) / start) * 100
        except Exception:
            return None
    
    def get_rotation_signals(self) -> Dict:
        """
        Identify sector rotation signals.
        Returns sectors gaining and losing momentum.
        """
        rankings = self.get_sector_rankings()
        
        if not rankings:
            return {'gaining': [], 'losing': [], 'message': 'No data available'}
        
        # Compare 1-week vs 1-month momentum
        gaining = []
        losing = []
        
        for r in rankings:
            # If recent (1wk) is stronger than trend (1mo), sector is gaining
            weekly_excess = r['return_1wk'] - (r['return_1mo'] / 4)  # Weekly pace vs monthly pace
            
            if weekly_excess > 1.5:
                gaining.append({
                    'etf': r['etf'],
                    'name': r['name'],
                    'signal': 'Money flowing in',
                    'weekly': r['return_1wk'],
                    'monthly': r['return_1mo'],
                })
            elif weekly_excess < -1.5:
                losing.append({
                    'etf': r['etf'],
                    'name': r['name'],
                    'signal': 'Money flowing out',
                    'weekly': r['return_1wk'],
                    'monthly': r['return_1mo'],
                })
        
        return {
            'gaining': gaining[:3],  # Top 3 gaining
            'losing': losing[:3],    # Top 3 losing
            'updated': datetime.now().isoformat(),
        }
    
    def get_sector_for_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Get sector info for a specific ticker.
        Returns sector ETF, name, and current momentum.
        """
        ticker = ticker.upper()
        
        # Check static map first
        etf = TICKER_SECTOR_MAP.get(ticker)
        
        if not etf:
            # Try to get from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', '')
                
                # Map sector name to ETF
                sector_lower = sector.lower()
                if 'technology' in sector_lower or 'tech' in sector_lower:
                    etf = 'XLK'
                elif 'financial' in sector_lower:
                    etf = 'XLF'
                elif 'energy' in sector_lower:
                    etf = 'XLE'
                elif 'health' in sector_lower:
                    etf = 'XLV'
                elif 'industrial' in sector_lower:
                    etf = 'XLI'
                elif 'communication' in sector_lower:
                    etf = 'XLC'
                elif 'consumer discretionary' in sector_lower:
                    etf = 'XLY'
                elif 'consumer staples' in sector_lower:
                    etf = 'XLP'
                elif 'utilities' in sector_lower:
                    etf = 'XLU'
                elif 'real estate' in sector_lower:
                    etf = 'XLRE'
                elif 'materials' in sector_lower or 'basic' in sector_lower:
                    etf = 'XLB'
            except Exception:
                pass
        
        if not etf:
            return None
        
        # Get sector data from rankings
        rankings = self.get_sector_rankings()
        for r in rankings:
            if r['etf'] == etf:
                return {
                    'ticker': ticker,
                    'sector_etf': etf,
                    'sector_name': r['name'],
                    'sector_return_1mo': r['return_1mo'],
                    'sector_momentum': r['momentum'],
                    'sector_rank': r['rank'],
                    'relative_strength': r['relative_strength'],
                }
        
        return None
    
    def get_heat_map_data(self) -> Dict:
        """
        Get data formatted for a sector heat map visualization.
        """
        rankings = self.get_sector_rankings()
        
        if not rankings:
            return {'sectors': [], 'spy_return': 0}
        
        spy_return = self._get_return('SPY', '1mo') or 0
        
        return {
            'sectors': rankings,
            'spy_return': round(spy_return, 2),
            'best_sector': rankings[0] if rankings else None,
            'worst_sector': rankings[-1] if rankings else None,
            'updated': datetime.now().isoformat(),
        }


# Singleton
sector_momentum = SectorMomentum()
