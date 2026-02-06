"""
Discovery Engine for Investment Algorithm
Finds new investment opportunities using Perplexity AI and dynamic yfinance screening.
No hardcoded stock lists — all discovery is data-driven.
"""
import yfinance as yf
from typing import List, Dict, Optional
from core.database import db
from clients.perplexity_client import pplx_client
import logging

logger = logging.getLogger(__name__)

# Broad sector ETFs used for sector-relative screening (not stock picks)
SECTOR_ETFS = {
    'technology': ['QQQ', 'VGT', 'XLK'],
    'healthcare': ['XLV', 'VHT', 'IBB'],
    'finance': ['XLF', 'VFH', 'KBE'],
    'consumer': ['XLY', 'VCR', 'XLP'],
    'energy': ['XLE', 'VDE', 'OIH'],
    'industrial': ['XLI', 'VIS', 'IYJ'],
}


class DiscoveryEngine:
    """Discovers new investment opportunities using Perplexity + dynamic screening."""

    def __init__(self):
        self.sector_etfs = SECTOR_ETFS

    def discover_similar(self, ticker: str, limit: int = 5) -> List[str]:
        """Find similar stocks by querying yfinance for sector peers."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')

            if not sector:
                return []

            # Use the quant screener's sector peers for discovery
            from engine.quant_screener import quant_screener
            peers = quant_screener.sector_cache.SECTOR_PEERS.get(sector, [])

            watchlist = {item['ticker'] for item in db.get_watchlist()}
            similar = [t for t in peers if t != ticker.upper() and t not in watchlist]
            return similar[:limit]

        except Exception as e:
            logger.warning(f"Discovery error for {ticker}: {e}")
            return []

    def discover_trending(self, limit: int = 5) -> List[str]:
        """Discover trending stocks using Perplexity (primary) or S&P 500 momentum (fallback).
        No hardcoded stock lists."""
        watchlist = {item['ticker'] for item in db.get_watchlist()}

        # Primary: Perplexity AI discovery
        if pplx_client.is_configured():
            usage = pplx_client.get_usage()
            if usage['remaining'] > 0:
                try:
                    discovery_result = pplx_client.discover_trending_stocks(
                        sector=None, focus="balanced", limit=limit
                    )
                    if discovery_result['success'] and discovery_result['stocks']:
                        discovered = [
                            stock['ticker']
                            for stock in discovery_result['stocks']
                            if stock['ticker'] not in watchlist
                        ]
                        if discovered:
                            return discovered[:limit]
                except Exception as e:
                    logger.warning(f"Perplexity discovery failed: {e}")

        # Fallback: Scan sector ETF top holdings for momentum
        return self._momentum_discovery(watchlist, limit)

    def _momentum_discovery(self, watchlist: set, limit: int) -> List[str]:
        """Dynamic momentum discovery using major index component screening."""
        # Use S&P 500 proxy tickers from sector ETFs and quant screener peers
        from engine.quant_screener import quant_screener
        all_peers = set()
        for peers in quant_screener.sector_cache.SECTOR_PEERS.values():
            all_peers.update(peers)

        candidates = [t for t in all_peers if t not in watchlist]

        trending = []
        for ticker in candidates:
            try:
                hist = yf.Ticker(ticker).history(period='5d')
                if not hist.empty and len(hist) >= 2:
                    change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    trending.append((ticker, change))
            except Exception:
                continue

        trending.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in trending[:limit]]

    def discover_by_criteria(self,
                              min_market_cap: int = None,
                              max_pe_ratio: float = None,
                              min_dividend_yield: float = None,
                              sector: str = None,
                              limit: int = 10) -> List[Dict]:
        """Discover stocks by criteria using sector peers from quant screener."""
        from engine.quant_screener import quant_screener
        all_peers = set()
        for peers in quant_screener.sector_cache.SECTOR_PEERS.values():
            all_peers.update(peers)

        watchlist = {item['ticker'] for item in db.get_watchlist()}
        candidates = [t for t in all_peers if t not in watchlist]

        results = []
        for ticker in candidates:
            try:
                info = yf.Ticker(ticker).info

                if min_market_cap and (info.get('marketCap', 0) or 0) < min_market_cap:
                    continue
                if max_pe_ratio and (info.get('trailingPE') or 999) > max_pe_ratio:
                    continue
                if min_dividend_yield and (info.get('dividendYield') or 0) < min_dividend_yield:
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
            except Exception:
                continue

        return results

    def discover_sector_leaders(self, sector: str, limit: int = 3) -> List[str]:
        """Get top ETFs for a specific sector."""
        sector_key = sector.lower()
        if sector_key in self.sector_etfs:
            return self.sector_etfs[sector_key][:limit]
        return ['SPY', 'QQQ', 'DIA'][:limit]

    def discover_with_perplexity(self, sector: str = None, focus: str = "balanced", limit: int = 5) -> Dict:
        """Advanced discovery using Perplexity AI with full details."""
        if not pplx_client.is_configured():
            return {'success': False, 'error': 'Perplexity API not configured', 'stocks': []}

        usage = pplx_client.get_usage()
        if usage['remaining'] <= 0:
            return {'success': False, 'error': f"Daily limit reached ({usage['daily_limit']} calls)", 'stocks': []}

        try:
            result = pplx_client.discover_trending_stocks(
                sector=sector, focus=focus, limit=limit
            )

            if result['success']:
                watchlist = {item['ticker'] for item in db.get_watchlist()}
                filtered_stocks = [s for s in result['stocks'] if s['ticker'] not in watchlist]
                result['stocks'] = filtered_stocks
                result['filtered_count'] = len(filtered_stocks)
                return result

            return result

        except Exception as e:
            logger.error(f"Perplexity discovery failed: {e}")
            return {'success': False, 'error': str(e), 'stocks': []}

    def get_discovery_suggestions(self) -> Dict:
        """Get discovery suggestions for the dashboard."""
        suggestions = {
            'trending': self.discover_trending(limit=3),
            'sectors': {k: v[:2] for k, v in self.sector_etfs.items()},
        }

        if pplx_client.is_configured():
            usage = pplx_client.get_usage()
            if usage['remaining'] > 0:
                try:
                    pplx_discovery = self.discover_with_perplexity(limit=3, focus="balanced")
                    if pplx_discovery['success']:
                        suggestions['perplexity_picks'] = [
                            {
                                'ticker': s['ticker'],
                                'score': s['score'],
                                'reason': s['reason'][:50] + '...' if len(s['reason']) > 50 else s['reason']
                            }
                            for s in pplx_discovery['stocks'][:3]
                        ]
                except Exception as e:
                    logger.warning(f"Could not get Perplexity suggestions: {e}")

        return suggestions


# Singleton
discovery_engine = DiscoveryEngine()
