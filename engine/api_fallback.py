"""
API Fallback Chain
Wraps yfinance with fallback data sources. If yfinance fails after retries,
attempts Yahoo Finance v8 JSON endpoint (no API key) before giving up.

Integrates with DataFreshnessTracker to record success/failure.
"""
import atexit
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import yfinance as yf

logger = logging.getLogger(__name__)


class APIFallbackChain:
    """Fetch stock data with automatic fallback between sources."""

    # Priority order of data sources
    SOURCES = ['yfinance', 'yahoo_json']

    _CACHE_FILE = Path(__file__).parent.parent / 'core' / 'data' / 'api_fallback_cache.json'

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=15)
        self._load_from_disk()
        atexit.register(self._save_to_disk)

    def get_stock_info(self, ticker: str, retries: int = 2) -> Tuple[Dict, str]:
        """
        Fetch stock info using the fallback chain.
        Returns (info_dict, source_used). Returns ({}, 'none') on total failure.
        """
        # Check cache first
        cache_key = f"info:{ticker}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['ts'] < self._cache_duration:
                return entry['data'], entry['source']

        try:
            from engine.data_freshness import data_freshness
        except Exception:
            data_freshness = None

        # Attempt 1: yfinance
        info = self._try_yfinance(ticker, retries)
        if info:
            if data_freshness:
                data_freshness.record_success(ticker, 'yfinance')
            self._cache[cache_key] = {'data': info, 'source': 'yfinance', 'ts': datetime.now()}
            return info, 'yfinance'

        # Record yfinance failure
        if data_freshness:
            data_freshness.record_failure(ticker, 'yfinance exhausted, trying fallback', 'yfinance')

        # Attempt 2: Yahoo Finance JSON endpoint (no API key needed)
        info = self._try_yahoo_json(ticker)
        if info:
            if data_freshness:
                data_freshness.record_success(ticker, 'yahoo_json')
            self._cache[cache_key] = {'data': info, 'source': 'yahoo_json', 'ts': datetime.now()}
            logger.info(f"Fallback to yahoo_json succeeded for {ticker}")
            return info, 'yahoo_json'

        if data_freshness:
            data_freshness.record_failure(ticker, 'All sources exhausted', 'yahoo_json')

        return {}, 'none'

    def get_stock_price(self, ticker: str) -> Optional[float]:
        """Quick price fetch with fallback."""
        info, source = self.get_stock_info(ticker, retries=1)
        if not info:
            return None
        # Try multiple price keys
        for key in ['currentPrice', 'regularMarketPrice', 'previousClose', 'price']:
            val = info.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    def _try_yfinance(self, ticker: str, retries: int = 2) -> Optional[Dict]:
        """Try to get data from yfinance with retries."""
        import time as _time
        for attempt in range(retries + 1):
            try:
                info = yf.Ticker(ticker).info
                if info and len(info) >= 3:
                    return info
            except Exception as e:
                logger.debug(f"yfinance attempt {attempt+1} for {ticker}: {e}")
            if attempt < retries:
                _time.sleep(0.5)
        return None

    def _try_yahoo_json(self, ticker: str) -> Optional[Dict]:
        """
        Fallback: fetch from Yahoo Finance v8 JSON endpoint directly.
        No API key needed — same data source yfinance uses internally.
        """
        import urllib.request
        import json
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; InvestmentMonitor/1.0)',
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            result = data.get('chart', {}).get('result', [])
            if not result:
                return None

            meta = result[0].get('meta', {})
            if not meta:
                return None

            # Map Yahoo JSON fields to yfinance-compatible keys
            info = {
                'currentPrice': meta.get('regularMarketPrice'),
                'regularMarketPrice': meta.get('regularMarketPrice'),
                'previousClose': meta.get('chartPreviousClose') or meta.get('previousClose'),
                'currency': meta.get('currency'),
                'exchangeName': meta.get('exchangeName'),
                'instrumentType': meta.get('instrumentType'),
                'regularMarketDayHigh': meta.get('regularMarketDayHigh'),
                'regularMarketDayLow': meta.get('regularMarketDayLow'),
                'regularMarketVolume': meta.get('regularMarketVolume'),
                '_source': 'yahoo_json_fallback',
            }
            # Only return if we got a valid price
            if info['currentPrice'] is not None:
                return info

        except Exception as e:
            logger.debug(f"Yahoo JSON fallback failed for {ticker}: {e}")

        return None


    def _save_to_disk(self) -> None:
        """Persist non-stale cache entries to disk on process exit."""
        now = datetime.now()
        serializable = {
            key: {'data': entry['data'], 'source': entry['source'], 'ts': entry['ts'].isoformat()}
            for key, entry in self._cache.items()
            if now - entry['ts'] < self._cache_duration
        }
        if not serializable:
            return
        try:
            self._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self._CACHE_FILE, 'w') as f:
                json.dump(serializable, f)
            logger.info("API fallback cache saved (%d entries)", len(serializable))
        except Exception as e:
            logger.warning("Failed to save API fallback cache: %s", e)

    def _load_from_disk(self) -> None:
        """Warm the cache from the last persisted snapshot."""
        if not self._CACHE_FILE.exists():
            return
        try:
            with open(self._CACHE_FILE) as f:
                data = json.load(f)
            now = datetime.now()
            loaded = 0
            for key, entry in data.items():
                ts = datetime.fromisoformat(entry['ts'])
                if now - ts < self._cache_duration:
                    self._cache[key] = {'data': entry['data'], 'source': entry['source'], 'ts': ts}
                    loaded += 1
            if loaded:
                logger.info("API fallback cache warmed (%d fresh entries)", loaded)
        except Exception as e:
            logger.warning("Failed to load API fallback cache: %s", e)


# Singleton
api_fallback = APIFallbackChain()
