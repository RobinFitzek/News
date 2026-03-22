"""
Portfolio-Level Anomaly Detection (items #46 / #55)

Detects systemic risk patterns at the whole-portfolio level, not per-ticker:

1. Systemic direction flag  — >70% of holdings moving same direction intraday
2. Correlation spike       — rolling 20d avg pairwise correlation jumps >0.2 above 90d baseline
3. Beta creep              — effective portfolio beta exceeds 1.5× target beta

Results stored in `portfolio_anomalies` table. Alerts via webhook when anomalies fire.
Intended to be called from the 15-min price-alert scheduler tick.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_anomalies (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                type         TEXT NOT NULL,
                severity     REAL DEFAULT 1.0,
                description  TEXT,
                details      TEXT,          -- JSON extras
                triggered_at TEXT NOT NULL,
                resolved_at  TEXT
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_anomalies_type "
            "ON portfolio_anomalies (type, triggered_at DESC)"
        )
    except Exception as e:
        logger.debug(f"portfolio_anomaly.ensure_schema: {e}")


def _record_anomaly(anomaly_type: str, severity: float, description: str, details: dict = None) -> None:
    try:
        import json
        from core.database import db
        db.execute(
            """
            INSERT INTO portfolio_anomalies (type, severity, description, details, triggered_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                anomaly_type,
                severity,
                description,
                json.dumps(details or {}),
                datetime.now().isoformat(),
            ),
        )
    except Exception as e:
        logger.debug(f"portfolio_anomaly._record_anomaly: {e}")


def _send_alert(message: str) -> None:
    try:
        from engine.webhook_notifier import webhook_notifier
        webhook_notifier.reload()
        webhook_notifier.send_custom(f"⚠️ Portfolio Anomaly\n{message}")
    except Exception as e:
        logger.debug(f"portfolio_anomaly._send_alert: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_portfolio_tickers() -> list[str]:
    try:
        from core.database import db
        rows = db.query(
            "SELECT DISTINCT ticker FROM portfolio_trades WHERE exit_date IS NULL"
        ) or []
        return [r["ticker"] for r in rows if r.get("ticker")]
    except Exception:
        return []


def _fetch_intraday_returns(tickers: list[str]) -> dict[str, float]:
    """Return today's intraday % return for each ticker (open to last price)."""
    returns = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            open_p = getattr(info, "open", None)
            last_p = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            if open_p and last_p and float(open_p) > 0:
                returns[t] = (float(last_p) - float(open_p)) / float(open_p)
        except Exception:
            pass
    return returns


def _fetch_close_history(tickers: list[str], days: int = 95) -> Optional[pd.DataFrame]:
    """Return aligned daily close price DataFrame for tickers."""
    try:
        if not tickers:
            return None
        data = {}
        for t in tickers:
            hist = yf.Ticker(t).history(period=f"{days}d")
            if not hist.empty:
                data[t] = hist["Close"]
        if len(data) < 2:
            return None
        return pd.DataFrame(data).dropna()
    except Exception as e:
        logger.debug(f"portfolio_anomaly._fetch_close_history: {e}")
        return None


# ---------------------------------------------------------------------------
# Check 1 — Systemic direction flag
# ---------------------------------------------------------------------------
def check_systemic_direction(tickers: list[str], threshold: float = 0.70) -> Optional[dict]:
    """Return anomaly dict if >threshold% of holdings move same direction."""
    if not tickers:
        return None
    returns = _fetch_intraday_returns(tickers)
    if not returns:
        return None

    vals = list(returns.values())
    n_up = sum(1 for r in vals if r > 0.005)    # up >0.5%
    n_down = sum(1 for r in vals if r < -0.005)  # down >0.5%
    total = len(vals)

    for direction, count in [("up", n_up), ("down", n_down)]:
        ratio = count / total
        if ratio >= threshold:
            return {
                "type": "systemic_direction",
                "severity": round(ratio, 2),
                "description": f"{count}/{total} holdings moving {direction} today ({ratio:.0%})",
                "details": {
                    "direction": direction,
                    "ratio": ratio,
                    "returns": returns,
                },
            }
    return None


# ---------------------------------------------------------------------------
# Check 2 — Correlation spike
# ---------------------------------------------------------------------------
def check_correlation_spike(price_df: pd.DataFrame, spike_threshold: float = 0.2) -> Optional[dict]:
    """
    Compare 20d average pairwise correlation vs 90d baseline.
    Flag if 20d > 90d + spike_threshold.
    """
    if price_df is None or len(price_df) < 25:
        return None

    returns = price_df.pct_change().dropna()

    def _avg_pairwise_corr(ret_df: pd.DataFrame) -> float:
        corr_mat = ret_df.corr().values
        n = corr_mat.shape[0]
        if n < 2:
            return 0.0
        upper = corr_mat[np.triu_indices(n, k=1)]
        return float(np.nanmean(upper))

    baseline_90d = _avg_pairwise_corr(returns.tail(90)) if len(returns) >= 90 else None
    recent_20d = _avg_pairwise_corr(returns.tail(20))

    if baseline_90d is None:
        return None

    delta = recent_20d - baseline_90d
    if delta > spike_threshold:
        return {
            "type": "correlation_spike",
            "severity": round(delta, 3),
            "description": (
                f"Portfolio correlation spike: 20d avg={recent_20d:.3f} "
                f"vs 90d baseline={baseline_90d:.3f} (Δ+{delta:.3f})"
            ),
            "details": {
                "recent_20d": round(recent_20d, 4),
                "baseline_90d": round(baseline_90d, 4),
                "delta": round(delta, 4),
            },
        }
    return None


# ---------------------------------------------------------------------------
# Check 3 — Beta creep
# ---------------------------------------------------------------------------
def check_beta_creep(price_df: pd.DataFrame, target_beta: float = 1.0, multiplier: float = 1.5) -> Optional[dict]:
    """Flag if effective portfolio beta exceeds target_beta * multiplier."""
    if price_df is None or len(price_df) < 30:
        return None

    try:
        spy = yf.Ticker("SPY").history(period="95d")["Close"]
        spy_rets = spy.pct_change().dropna()

        port_rets = price_df.pct_change().dropna().mean(axis=1)
        aligned = pd.concat([port_rets, spy_rets], axis=1).dropna()
        aligned.columns = ["portfolio", "spy"]

        cov = float(np.cov(aligned["portfolio"], aligned["spy"])[0, 1])
        var_spy = float(aligned["spy"].var())
        beta = cov / var_spy if var_spy > 0 else 1.0
        threshold = target_beta * multiplier

        if beta > threshold:
            return {
                "type": "beta_creep",
                "severity": round(beta, 3),
                "description": (
                    f"Portfolio beta={beta:.2f} exceeds {multiplier}× target "
                    f"({target_beta}). Excessive market sensitivity."
                ),
                "details": {
                    "beta": round(beta, 3),
                    "target_beta": target_beta,
                    "threshold": round(threshold, 2),
                },
            }
    except Exception as e:
        logger.debug(f"portfolio_anomaly.check_beta_creep: {e}")

    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_anomaly_checks() -> list[dict]:
    """
    Run all portfolio anomaly checks.
    Stores results in DB and fires webhook alerts for new anomalies.
    Returns list of anomalies detected (may be empty).
    """
    ensure_schema()

    tickers = _get_portfolio_tickers()
    if not tickers:
        return []

    anomalies = []

    # Check 1: systemic direction
    anomaly = check_systemic_direction(tickers)
    if anomaly:
        anomalies.append(anomaly)
        _record_anomaly(**anomaly)
        _send_alert(anomaly["description"])

    # Fetch price history once for checks 2 & 3
    price_df = _fetch_close_history(tickers)

    # Check 2: correlation spike
    anomaly = check_correlation_spike(price_df)
    if anomaly:
        anomalies.append(anomaly)
        _record_anomaly(**anomaly)
        _send_alert(anomaly["description"])

    # Check 3: beta creep
    anomaly = check_beta_creep(price_df)
    if anomaly:
        anomalies.append(anomaly)
        _record_anomaly(**anomaly)
        _send_alert(anomaly["description"])

    if anomalies:
        logger.info(f"portfolio_anomaly: {len(anomalies)} anomaly(ies) detected")
    return anomalies


def get_active_anomalies(hours: int = 24) -> list[dict]:
    """Return anomalies triggered in the last N hours (not yet resolved)."""
    ensure_schema()
    try:
        from core.database import db
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = db.query(
            """
            SELECT * FROM portfolio_anomalies
            WHERE triggered_at >= ? AND resolved_at IS NULL
            ORDER BY triggered_at DESC
            """,
            (cutoff,),
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"portfolio_anomaly.get_active_anomalies: {e}")
        return []
