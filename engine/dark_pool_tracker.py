"""
Dark Pool & Institutional Block Trade Detection (items #26 / #52)

Detects unusual institutional accumulation using:
1. Volume anomaly detection — daily volume > 3× 20-day average signals potential block trades
2. Price-volume divergence — large volume with small price move = likely block/dark pool
3. Combines with 13F smart money signals for confirmation

Note: Actual dark pool print data requires paid feeds (FINRA ATS data).
This engine uses publicly available volume data as a heuristic proxy.
Results cached in `dark_pool_signals` table.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)

VOLUME_SPIKE_MULTIPLIER = 3.0     # >3× avg volume = potential block
LOW_MOVE_THRESHOLD_PCT = 0.5      # price move <0.5% despite high volume = dark pool signal
HISTORY_DAYS = 30                 # days of history for avg volume calculation
CACHE_HOURS = 6                   # hours before re-checking a ticker
BLOCK_TRADE_THRESHOLD_USD = 5_000_000  # flag if estimated block value > $5M


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS dark_pool_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                signal_date     TEXT NOT NULL,
                volume          INTEGER,
                avg_volume_20d  INTEGER,
                volume_ratio    REAL,
                price_move_pct  REAL,
                estimated_value REAL DEFAULT 0,
                is_large_block  INTEGER DEFAULT 0,
                signal_type     TEXT NOT NULL,  -- 'volume_spike' | 'dark_pool_proxy' | 'smart_money'
                description     TEXT,
                detected_at     TEXT NOT NULL,
                UNIQUE(ticker, signal_date, signal_type)
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_dark_pool_ticker "
            "ON dark_pool_signals (ticker, signal_date DESC)"
        )
        # Migrate: add columns if missing (existing DBs)
        try:
            db.execute("ALTER TABLE dark_pool_signals ADD COLUMN estimated_value REAL DEFAULT 0")
        except Exception:
            pass
        try:
            db.execute("ALTER TABLE dark_pool_signals ADD COLUMN is_large_block INTEGER DEFAULT 0")
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"dark_pool_tracker.ensure_schema: {e}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyze_ticker(ticker: str) -> list[dict]:
    """
    Analyze a ticker for unusual volume / dark pool proxy signals.
    Returns list of signal dicts for the last 30 days.
    """
    try:
        hist = yf.Ticker(ticker).history(period=f"{HISTORY_DAYS + 5}d")
        if hist is None or hist.empty or len(hist) < 5:
            return []

        hist = hist.sort_index()
        volumes = hist["Volume"].values
        closes = hist["Close"].values
        opens = hist["Open"].values

        # 20-day average volume (excluding most recent)
        if len(volumes) < 5:
            return []
        avg_vol_20 = int(np.mean(volumes[:-1][-20:])) if len(volumes) > 1 else int(volumes[0])

        signals = []
        # Check last 10 trading days for anomalies
        check_days = min(10, len(hist))
        for i in range(-check_days, 0):
            try:
                date_idx = hist.index[i]
                date_str = str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx)[:10]
                vol = int(volumes[i])
                close = float(closes[i])
                open_p = float(opens[i])

                if avg_vol_20 == 0:
                    continue
                vol_ratio = vol / avg_vol_20
                price_move_pct = abs((close - open_p) / open_p * 100) if open_p > 0 else 0

                # Estimate trade value (volume × closing price)
                estimated_value = vol * close if close > 0 else 0
                is_large_block = estimated_value >= BLOCK_TRADE_THRESHOLD_USD

                if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:
                    if price_move_pct < LOW_MOVE_THRESHOLD_PCT:
                        signal_type = "dark_pool_proxy"
                        desc = (
                            f"Volume {vol_ratio:.1f}× avg with only {price_move_pct:.2f}% price move — "
                            f"possible institutional dark pool accumulation"
                        )
                    else:
                        signal_type = "volume_spike"
                        desc = (
                            f"Volume {vol_ratio:.1f}× avg ({vol:,} vs {avg_vol_20:,} avg) — "
                            f"unusual block trade activity"
                        )
                    if is_large_block:
                        desc += f" [${estimated_value / 1e6:.1f}M block]"
                    signals.append({
                        "ticker": ticker.upper(),
                        "signal_date": date_str,
                        "volume": vol,
                        "avg_volume_20d": avg_vol_20,
                        "volume_ratio": round(vol_ratio, 2),
                        "price_move_pct": round(price_move_pct, 2),
                        "estimated_value": round(estimated_value, 2),
                        "is_large_block": is_large_block,
                        "signal_type": signal_type,
                        "description": desc,
                    })
            except Exception:
                continue

        return signals
    except Exception as e:
        logger.debug(f"dark_pool_tracker._analyze_ticker({ticker}): {e}")
        return []


def _save_signals(signals: list[dict]) -> None:
    if not signals:
        return
    try:
        from core.database import db
        now = datetime.now().isoformat()
        for s in signals:
            db.execute(
                """
                INSERT INTO dark_pool_signals
                    (ticker, signal_date, volume, avg_volume_20d, volume_ratio,
                     price_move_pct, estimated_value, is_large_block,
                     signal_type, description, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, signal_date, signal_type) DO UPDATE SET
                    volume=excluded.volume,
                    volume_ratio=excluded.volume_ratio,
                    price_move_pct=excluded.price_move_pct,
                    estimated_value=excluded.estimated_value,
                    is_large_block=excluded.is_large_block,
                    description=excluded.description,
                    detected_at=excluded.detected_at
                """,
                (
                    s["ticker"],
                    s["signal_date"],
                    s.get("volume"),
                    s.get("avg_volume_20d"),
                    s.get("volume_ratio"),
                    s.get("price_move_pct"),
                    s.get("estimated_value", 0),
                    1 if s.get("is_large_block") else 0,
                    s["signal_type"],
                    s.get("description"),
                    now,
                ),
            )
    except Exception as e:
        logger.debug(f"dark_pool_tracker._save_signals: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scan_watchlist() -> int:
    """
    Scan all active watchlist tickers for dark pool / block trade signals.
    Returns number of new signals detected.
    """
    ensure_schema()
    try:
        from core.database import db
        tickers = [
            r["ticker"]
            for r in (db.query("SELECT ticker FROM watchlist WHERE active = 1") or [])
        ]
    except Exception:
        return 0

    total_signals = 0
    for ticker in tickers:
        signals = _analyze_ticker(ticker)
        if signals:
            _save_signals(signals)
            total_signals += len(signals)

    logger.info(f"dark_pool_tracker.scan_watchlist: {total_signals} signals across {len(tickers)} tickers")
    return total_signals


def get_ticker_signals(ticker: str, days: int = 30) -> list[dict]:
    """Return dark pool signals for a ticker in the last N days."""
    ensure_schema()
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = db.query(
            """
            SELECT * FROM dark_pool_signals
            WHERE ticker = ? AND signal_date >= ?
            ORDER BY signal_date DESC
            """,
            (ticker.upper(), cutoff),
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"dark_pool_tracker.get_ticker_signals: {e}")
        return []


def get_all_recent_signals(days: int = 7, min_volume_ratio: float = 0.0) -> list[dict]:
    """
    Return all dark pool signals from last N days across watchlist,
    optionally filtered by minimum volume ratio.
    """
    ensure_schema()
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = db.query(
            """
            SELECT * FROM dark_pool_signals
            WHERE signal_date >= ? AND volume_ratio >= ?
            ORDER BY signal_date DESC, volume_ratio DESC
            """,
            (cutoff, min_volume_ratio),
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"dark_pool_tracker.get_all_recent_signals: {e}")
        return []


def get_top_signals(days: int = 7, top_n: int = 20) -> list[dict]:
    """Return the N highest-ratio volume anomalies in the last N days."""
    all_signals = get_all_recent_signals(days=days)
    # Prioritize dark_pool_proxy, then sort by volume_ratio
    all_signals.sort(
        key=lambda s: (s["signal_type"] == "dark_pool_proxy", s.get("volume_ratio", 0)),
        reverse=True,
    )
    return all_signals[:top_n]


def get_large_block_trades(days: int = 2, threshold_usd: float = BLOCK_TRADE_THRESHOLD_USD) -> list[dict]:
    """
    Return signals where estimated trade value exceeds threshold (default $5M)
    within the last N days. Used for high-priority alerting.
    """
    all_signals = get_all_recent_signals(days=days)
    return [
        s for s in all_signals
        if s.get("estimated_value", 0) >= threshold_usd
    ]
