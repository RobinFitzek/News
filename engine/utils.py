"""
Shared utilities for engine modules.

Centralises yfinance data fetching behind a per-ticker TTL cache so that
multiple engine modules hitting the same ticker in a short window reuse
the same download instead of each making an independent network call.
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL cache (thread-safe, dict-based, no extra dependency)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_history_cache: dict = {}   # key -> {"data": DataFrame, "ts": datetime}
_info_cache: dict = {}      # key -> {"data": dict,      "ts": datetime}

_DEFAULT_HISTORY_TTL = timedelta(minutes=15)
_DEFAULT_INFO_TTL = timedelta(minutes=30)


def _is_fresh(entry: dict, ttl: timedelta) -> bool:
    return (datetime.now() - entry["ts"]) < ttl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ticker_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    ttl: Optional[timedelta] = None,
) -> pd.DataFrame:
    """Fetch OHLCV history for *ticker* via yfinance with a per-key TTL cache.

    Parameters
    ----------
    ticker : str
        Stock symbol (case-insensitive, auto-uppercased).
    period : str
        yfinance period string, e.g. ``"3mo"``, ``"1y"``.
    interval : str
        Bar width, e.g. ``"1d"``, ``"1wk"``.
    ttl : timedelta | None
        How long to consider a cached entry fresh.  Defaults to 15 min.

    Returns
    -------
    pd.DataFrame
        OHLCV dataframe (may be empty if the ticker is invalid).
    """
    import yfinance as yf

    ticker = ticker.upper().strip()
    cache_key = f"{ticker}|{period}|{interval}"
    effective_ttl = ttl or _DEFAULT_HISTORY_TTL

    with _cache_lock:
        if cache_key in _history_cache and _is_fresh(_history_cache[cache_key], effective_ttl):
            return _history_cache[cache_key]["data"]

    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as e:
        logger.warning("yfinance history fetch failed for %s: %s", ticker, e)
        hist = pd.DataFrame()

    with _cache_lock:
        _history_cache[cache_key] = {"data": hist, "ts": datetime.now()}

    return hist


def get_ticker_info(
    ticker: str,
    ttl: Optional[timedelta] = None,
) -> dict:
    """Fetch ticker info dict via yfinance with caching.

    Returns an empty dict on failure instead of raising.
    """
    import yfinance as yf

    ticker = ticker.upper().strip()
    effective_ttl = ttl or _DEFAULT_INFO_TTL

    with _cache_lock:
        if ticker in _info_cache and _is_fresh(_info_cache[ticker], effective_ttl):
            return _info_cache[ticker]["data"]

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.warning("yfinance info fetch failed for %s: %s", ticker, e)
        info = {}

    with _cache_lock:
        _info_cache[ticker] = {"data": info, "ts": datetime.now()}

    return info


def get_current_price(ticker: str) -> Optional[float]:
    """Return the latest close price for *ticker*, or None."""
    hist = get_ticker_history(ticker, period="5d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Helpers reused across engine modules
# ---------------------------------------------------------------------------

def safe_float_list(series: pd.Series) -> list:
    """Convert a pandas Series to a list, replacing NaN with None."""
    return [round(float(v), 2) if pd.notna(v) else None for v in series]


def invalidate_cache(ticker: str = None):
    """Clear cache entries.  If *ticker* is given, only that ticker is evicted."""
    with _cache_lock:
        if ticker is None:
            _history_cache.clear()
            _info_cache.clear()
        else:
            ticker = ticker.upper().strip()
            keys_to_remove = [k for k in _history_cache if k.startswith(f"{ticker}|")]
            for k in keys_to_remove:
                del _history_cache[k]
            _info_cache.pop(ticker, None)
