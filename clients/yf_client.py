"""
Central yfinance client — batch OHLCV download with in-process TTL cache.

Replace individual `yf.Ticker(t).history(...)` calls with this client so that:
  1. Multiple tickers are fetched in one HTTP round-trip via yf.download().
  2. The same ticker+period combo is never downloaded twice within a scan cycle
     (15-minute TTL by default), even when called from different engine modules.

Non-OHLCV data (`.info`, `.dividends`, `.calendar`, `.fast_info`, etc.) cannot
be batched and should continue using yf.Ticker() directly.

Usage
-----
    from clients.yf_client import yf_client

    # Single ticker — drop-in replacement for yf.Ticker(t).history(period="6mo")
    df = yf_client.get_history_single("AAPL", period="6mo")

    # Batch — fetches all tickers in one request, returns dict
    data = yf_client.get_history(["AAPL", "MSFT", "GOOG"], period="6mo")
    aapl_df = data["AAPL"]   # DataFrame with Close/Open/High/Low/Volume
"""

import logging
import threading
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 900  # 15 minutes


class YFClient:
    """Thread-safe yfinance wrapper with batch download and in-process TTL cache."""

    def __init__(self, ttl: int = _DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    # ── Cache helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _key(*parts) -> str:
        return "|".join(str(p) for p in parts)

    def _get(self, key: str):
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.monotonic() < entry[1]:
                return entry[0]
        return None

    def _set(self, key: str, data) -> None:
        with self._lock:
            self._cache[key] = (data, time.monotonic() + self._ttl)

    def invalidate(self) -> None:
        """Clear all cached data — call at the start of a new scan cycle."""
        with self._lock:
            self._cache.clear()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_history(
        self,
        tickers: List[str],
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch-download OHLCV history for multiple tickers.

        Returns a dict mapping TICKER → DataFrame. Tickers that failed to
        download map to an empty DataFrame so callers can safely check
        `df.empty` without extra error handling.
        """
        if not tickers:
            return {}

        unique = sorted(set(t.upper() for t in tickers))
        cache_key = self._key("hist", unique, period, interval, start, end)
        cached = self._get(cache_key)
        if cached is not None:
            return cached

        result = self._download(unique, period, interval, start, end)
        self._set(cache_key, result)
        return result

    def get_history_single(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convenience wrapper — returns the DataFrame for a single ticker."""
        data = self.get_history(
            [ticker], period=period, interval=interval, start=start, end=end
        )
        return data.get(ticker.upper(), pd.DataFrame())

    # ── Download internals ─────────────────────────────────────────────────────

    def _download(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """Try yf.download() for all tickers at once; fall back per-ticker."""
        kwargs: Dict = {
            "tickers": tickers,
            "interval": interval,
            "auto_adjust": True,
            "progress": False,
            "threads": True,
            "group_by": "ticker",
        }
        if start:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        try:
            raw = yf.download(**kwargs)

            if raw is None or raw.empty:
                raise ValueError("empty response")

            result: Dict[str, pd.DataFrame] = {}

            if isinstance(raw.columns, pd.MultiIndex):
                # Multi-ticker response: level-0 = ticker, level-1 = field
                available = set(raw.columns.get_level_values(0))
                for t in tickers:
                    if t in available:
                        df = raw[t].dropna(how="all")
                        result[t] = df if not df.empty else pd.DataFrame()
                    else:
                        result[t] = pd.DataFrame()
            else:
                # Single-ticker flat DataFrame (yfinance behaviour for 1 ticker)
                result[tickers[0]] = raw if not raw.empty else pd.DataFrame()

            logger.debug(
                "yf_client: batch downloaded %d ticker(s) period=%s", len(tickers), period
            )
            return result

        except Exception as exc:
            logger.warning(
                "yf_client: batch download failed (%s) — falling back to per-ticker", exc
            )
            return self._download_individual(tickers, period, interval, start, end)

    def _download_individual(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                df = (
                    t.history(start=start, end=end, interval=interval, auto_adjust=True)
                    if start
                    else t.history(period=period, interval=interval, auto_adjust=True)
                )
                result[ticker] = df if not df.empty else pd.DataFrame()
            except Exception as exc:
                logger.warning("yf_client: failed to fetch %s: %s", ticker, exc)
                result[ticker] = pd.DataFrame()
        return result


# Module-level singleton used by all engine modules
yf_client = YFClient()
