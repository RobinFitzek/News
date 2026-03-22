"""
Pairs Trading / Statistical Arbitrage (item #40)

Detects cointegrated watchlist pairs using the Engle-Granger test (via statsmodels).
Monitors daily spread z-scores and flags mean-reversion entry opportunities when |z| > 2.0.

Cointegrated pairs (p-value < 0.05, ≥90d of shared history) are stored in
`pairs_signals` and re-tested weekly since cointegration breaks down over time.

Dependencies:
    statsmodels>=0.14  (pip install statsmodels)
    pandas, numpy, yfinance (already in requirements.txt)
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional statsmodels import
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.stattools import coint
    _statsmodels_available = True
except ImportError:
    _statsmodels_available = False
    logger.warning(
        "statsmodels not installed. Pairs trading engine disabled. "
        "Install with: pip install statsmodels"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_HISTORY_DAYS = 90
COINTEGRATION_PVALUE_THRESHOLD = 0.05
ZSCORE_ENTRY_THRESHOLD = 2.0
ZSCORE_EXIT_THRESHOLD = 0.5
LOOKBACK_DAYS = 90


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    """Create pairs_signals table if missing."""
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS pairs_signals (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_a       TEXT NOT NULL,
                ticker_b       TEXT NOT NULL,
                pvalue         REAL,
                hedge_ratio    REAL,
                zscore         REAL,
                spread_mean    REAL,
                spread_std     REAL,
                signal         TEXT,   -- 'long_spread' | 'short_spread' | 'hold' | 'exit'
                tested_at      TEXT NOT NULL,
                UNIQUE(ticker_a, ticker_b)
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pairs_signals_tickers ON pairs_signals (ticker_a, ticker_b)"
        )
    except Exception as e:
        logger.debug(f"pairs_trader.ensure_schema: {e}")


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
def _fetch_prices(ticker: str, days: int = LOOKBACK_DAYS + 30) -> Optional[pd.Series]:
    """Return closing prices as a Series (indexed by date)."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d")
        if hist.empty or len(hist) < MIN_HISTORY_DAYS:
            return None
        return hist["Close"].rename(ticker)
    except Exception as e:
        logger.debug(f"pairs_trader: price fetch failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Cointegration test
# ---------------------------------------------------------------------------
def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Run Engle-Granger cointegration test on two price series.

    Returns:
        {
            "pvalue": float,
            "cointegrated": bool,
            "hedge_ratio": float,   # OLS coefficient (regression A on B)
            "spread_mean": float,
            "spread_std": float,
            "current_zscore": float,
        }
    """
    if not _statsmodels_available:
        return {"error": "statsmodels not installed", "cointegrated": False}

    # Align on common dates
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if len(df) < MIN_HISTORY_DAYS:
        return {"error": "Insufficient shared history", "cointegrated": False}

    a_vals = df.iloc[:, 0].values
    b_vals = df.iloc[:, 1].values

    try:
        _, pvalue, _ = coint(a_vals, b_vals)
    except Exception as e:
        return {"error": str(e), "cointegrated": False}

    # OLS hedge ratio: regress A on B
    hedge_ratio = float(np.cov(a_vals, b_vals)[0, 1] / np.var(b_vals))

    # Spread = A - hedge_ratio * B
    spread = a_vals - hedge_ratio * b_vals
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread))
    current_zscore = float((spread[-1] - spread_mean) / spread_std) if spread_std > 0 else 0.0

    return {
        "pvalue": round(float(pvalue), 5),
        "cointegrated": float(pvalue) < COINTEGRATION_PVALUE_THRESHOLD,
        "hedge_ratio": round(hedge_ratio, 4),
        "spread_mean": round(spread_mean, 4),
        "spread_std": round(spread_std, 4),
        "current_zscore": round(current_zscore, 3),
    }


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------
def _determine_signal(zscore: float) -> str:
    """Map z-score to trade signal."""
    if zscore > ZSCORE_ENTRY_THRESHOLD:
        return "short_spread"   # A overvalued vs B → short A, long B
    elif zscore < -ZSCORE_ENTRY_THRESHOLD:
        return "long_spread"    # A undervalued vs B → long A, short B
    elif abs(zscore) < ZSCORE_EXIT_THRESHOLD:
        return "exit"           # Spread reverted — close position
    return "hold"


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------
def run_weekly_scan() -> list[dict]:
    """
    Run cointegration tests on all active watchlist pairs and store results.
    Intended to be called weekly from the scheduler.

    Returns list of cointegrated pairs found.
    """
    ensure_schema()

    if not _statsmodels_available:
        logger.warning("pairs_trader.run_weekly_scan: statsmodels not installed")
        return []

    try:
        from core.database import db
        tickers = [
            r["ticker"]
            for r in (db.query("SELECT ticker FROM watchlist WHERE active = 1") or [])
        ]
    except Exception as e:
        logger.error(f"pairs_trader: watchlist fetch failed: {e}")
        return []

    if len(tickers) < 2:
        return []

    # Fetch prices for all tickers (cached per call)
    price_cache: dict[str, Optional[pd.Series]] = {}
    for t in tickers:
        price_cache[t] = _fetch_prices(t)

    valid_tickers = [t for t, s in price_cache.items() if s is not None]
    logger.info(f"pairs_trader: testing {len(valid_tickers)*(len(valid_tickers)-1)//2} pairs from {len(valid_tickers)} tickers")

    cointegrated_pairs = []
    from core.database import db

    for i, a in enumerate(valid_tickers):
        for b in valid_tickers[i + 1:]:
            result = test_cointegration(price_cache[a], price_cache[b])
            if result.get("error"):
                continue

            signal = _determine_signal(result["current_zscore"])
            now = datetime.now().isoformat()

            try:
                db.execute(
                    """
                    INSERT INTO pairs_signals
                        (ticker_a, ticker_b, pvalue, hedge_ratio, zscore,
                         spread_mean, spread_std, signal, tested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker_a, ticker_b) DO UPDATE SET
                        pvalue=excluded.pvalue,
                        hedge_ratio=excluded.hedge_ratio,
                        zscore=excluded.zscore,
                        spread_mean=excluded.spread_mean,
                        spread_std=excluded.spread_std,
                        signal=excluded.signal,
                        tested_at=excluded.tested_at
                    """,
                    (
                        a, b,
                        result["pvalue"], result["hedge_ratio"], result["current_zscore"],
                        result["spread_mean"], result["spread_std"],
                        signal, now,
                    ),
                )
            except Exception as e:
                logger.debug(f"pairs_trader: DB upsert failed ({a}/{b}): {e}")

            if result["cointegrated"]:
                cointegrated_pairs.append({
                    "ticker_a": a,
                    "ticker_b": b,
                    **result,
                    "signal": signal,
                })

    logger.info(f"pairs_trader: found {len(cointegrated_pairs)} cointegrated pairs")
    return cointegrated_pairs


def get_active_pairs() -> list[dict]:
    """Return stored pairs with active signals (long_spread or short_spread)."""
    ensure_schema()
    try:
        from core.database import db
        rows = db.query(
            """
            SELECT * FROM pairs_signals
            WHERE signal IN ('long_spread', 'short_spread')
            ORDER BY ABS(zscore) DESC
            """
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"pairs_trader.get_active_pairs: {e}")
        return []


def get_all_pairs() -> list[dict]:
    """Return all stored pairs, ordered by cointegration strength (lowest p-value first)."""
    ensure_schema()
    try:
        from core.database import db
        rows = db.query(
            "SELECT * FROM pairs_signals ORDER BY pvalue ASC"
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"pairs_trader.get_all_pairs: {e}")
        return []
