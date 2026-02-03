"""
Discovery Engine for Investment Algorithm
Finds new investment opportunities based on existing portfolio and market trends.
Lightweight for N100 homeserver deployment.
"""
import yfinance as yf
from typing import List, Dict, Optional
from core.database import db


class DiscoveryEngine:
    """Discovers new investment opportunities efficiently"""
    
    def __init__(self):
        # Predefined sector ETFs for discovery
        self.sector_etfs = {
            'technology': ['QQQ', 'VGT', 'XLK'],
            'healthcare': ['XLV', 'VHT', 'IBB'],
            'finance': ['XLF', 'VFH', 'KBE'],
            'consumer': ['XLY', 'VCR', 'XLP'],
            'energy': ['XLE', 'VDE', 'OIH'],
            'industrial': ['XLI', 'VIS', 'IYJ'],
        }
        
        # Popular growth stocks for discovery
        self.growth_universe = [
            'NVDA', 'AMD', 'AVGO', 'PLTR', 'SNOW', 'CRM', 'NOW', 'DDOG',
            'NET', 'CRWD', 'ZS', 'MDB', 'U', 'SHOP', 'SQ', 'COIN'
        ]
        
        # Dividend stocks for conservative discovery
        self.dividend_universe = [
            'JNJ', 'PG', 'KO', 'PEP', 'VZ', 'T', 'XOM', 'CVX',
            'ABBV', 'MRK', 'BMY', 'CSCO', 'INTC', 'IBM'
        ]
    
    def discover_similar(self, ticker: str, limit: int = 5) -> List[str]:
        """Find similar stocks based on sector and industry"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            
            if not sector:
                return []
            
            # Get peer companies
            similar = []
            
            # Check growth universe for tech
            if sector.lower() in ['technology', 'communication services']:
                candidates = [t for t in self.growth_universe if t != ticker.upper()]
                similar.extend(candidates[:limit])
            
            # Check dividend universe for defensive sectors
            elif sector.lower() in ['healthcare', 'consumer defensive', 'utilities']:
                candidates = [t for t in self.dividend_universe if t != ticker.upper()]
                similar.extend(candidates[:limit])
            
            # Filter out tickers already in watchlist
            watchlist = {item['ticker'] for item in db.get_watchlist()}
            return [t for t in similar if t not in watchlist][:limit]
            
        except Exception as e:
            print(f"⚠️ Discovery error for {ticker}: {e}")
            return []
    
    def discover_trending(self, limit: int = 5) -> List[str]:
        """Discover trending stocks not in watchlist"""
        watchlist = {item['ticker'] for item in db.get_watchlist()}
        
        # Combine universes
        candidates = list(set(self.growth_universe + self.dividend_universe))
        
        # Filter out existing watchlist items
        new_candidates = [t for t in candidates if t not in watchlist]
        
        # Simple momentum check - get top movers
        trending = []
        for ticker in new_candidates[:limit * 2]:  # Check more than needed
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    trending.append((ticker, change))
            except:
                continue
        
        # Sort by performance and return top
        trending.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in trending[:limit]]
    
    def discover_by_criteria(self, 
                              min_market_cap: int = None,
                              max_pe_ratio: float = None,
                              min_dividend_yield: float = None,
                              sector: str = None,
                              limit: int = 10) -> List[Dict]:
        """
        Discover stocks by specific criteria
        Note: Lightweight implementation - uses predefined universes
        """
        results = []
        universe = self.growth_universe + self.dividend_universe
        
        watchlist = {item['ticker'] for item in db.get_watchlist()}
        candidates = [t for t in universe if t not in watchlist]
        
        for ticker in candidates:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Apply filters
                if min_market_cap and info.get('marketCap', 0) < min_market_cap:
                    continue
                
                if max_pe_ratio and info.get('trailingPE', 999) > max_pe_ratio:
                    continue
                
                if min_dividend_yield and info.get('dividendYield', 0) < min_dividend_yield:
                    continue
                
                if sector and info.get('sector', '').lower() != sector.lower():
                    continue
                
                results.append({
                    'ticker': ticker,
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield')
                })
                
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                continue
        
        return results
    
    def discover_sector_leaders(self, sector: str, limit: int = 3) -> List[str]:
        """Get top ETFs for a specific sector"""
        sector_key = sector.lower()
        
        if sector_key in self.sector_etfs:
            return self.sector_etfs[sector_key][:limit]
        
        # Default to broad market
        return ['SPY', 'QQQ', 'DIA'][:limit]
    
    def get_discovery_suggestions(self) -> Dict:
        """Get a summary of discovery suggestions for the dashboard"""
        return {
            'trending': self.discover_trending(limit=3),
            'sectors': {
                'tech': self.sector_etfs.get('technology', [])[:2],
                'health': self.sector_etfs.get('healthcare', [])[:2],
                'energy': self.sector_etfs.get('energy', [])[:2]
            },
            'categories': {
                'growth': self.growth_universe[:5],
                'dividend': self.dividend_universe[:5]
            }
        }


# Singleton
discovery_engine = DiscoveryEngine()
