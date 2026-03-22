"""
Cross-Asset Composite Signals (item #47)

After each macro snapshot, evaluates named composite patterns across asset classes:
  - "Flight to Safety"  — gold+, VIX+, equities-
  - "Risk-On Surge"     — equities+, VIX-, credit spreads- (HYG/LQD ratio up)
  - "Inflation Spike"   — gold+, bonds-, DXY-

Patterns are config dicts: { name, conditions: [{series, direction, threshold}], min_match }
Results stored in `macro_composite_signals` table.
Alerts via webhook when new patterns fire.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite pattern definitions
# Each condition: { "series": <name>, "direction": "up"|"down", "threshold": float pct }
# "direction": "up" = today's 1d return > threshold, "down" = < -threshold
# "min_match" = how many conditions must fire for the composite to trigger
# ---------------------------------------------------------------------------
COMPOSITE_PATTERNS = [
    {
        "name": "Flight to Safety",
        "description": (
            "Gold rising, VIX spiking, equities selling off simultaneously — "
            "classic risk-off regime. Consider reducing equity exposure."
        ),
        "conditions": [
            {"series": "GLD",  "direction": "up",   "threshold": 0.5},   # gold ETF up >0.5%
            {"series": "^VIX", "direction": "up",   "threshold": 5.0},   # VIX up >5%
            {"series": "SPY",  "direction": "down",  "threshold": 0.5},  # equities down >0.5%
        ],
        "min_match": 3,
    },
    {
        "name": "Risk-On Surge",
        "description": (
            "Equities rallying, VIX falling, credit spreads tightening — "
            "broad risk-on appetite. Growth and momentum names favoured."
        ),
        "conditions": [
            {"series": "SPY",  "direction": "up",   "threshold": 0.5},   # equities up >0.5%
            {"series": "^VIX", "direction": "down",  "threshold": 5.0},  # VIX down >5%
            {"series": "HYG",  "direction": "up",   "threshold": 0.3},   # HY credit ETF up (spreads tighten)
        ],
        "min_match": 3,
    },
    {
        "name": "Inflation Spike",
        "description": (
            "Gold surging, bonds selling off, USD falling — "
            "stagflation / inflation surprise signal. Favour real assets and TIPS."
        ),
        "conditions": [
            {"series": "GLD",  "direction": "up",   "threshold": 0.7},   # gold up >0.7%
            {"series": "TLT",  "direction": "down",  "threshold": 0.5},  # long-bond ETF down >0.5%
            {"series": "UUP",  "direction": "down",  "threshold": 0.3},  # USD ETF down >0.3%
        ],
        "min_match": 2,  # 2-of-3 sufficient for this one
    },
    {
        "name": "Recession Warning",
        "description": (
            "Yield curve inverting, credit spreads widening, VIX elevated — "
            "early recession warning cluster. Rotate to defensives."
        ),
        "conditions": [
            {"series": "^VIX", "direction": "up",   "threshold": 3.0},   # VIX up >3%
            {"series": "HYG",  "direction": "down",  "threshold": 0.5},  # HY credit widening
            {"series": "SPY",  "direction": "down",  "threshold": 1.0},  # equities notably down
        ],
        "min_match": 3,
    },
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_composite_signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                description  TEXT,
                conditions_matched INTEGER NOT NULL,
                conditions_total   INTEGER NOT NULL,
                details      TEXT,          -- JSON: per-condition results
                triggered_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_composite_signals_pattern "
            "ON macro_composite_signals (pattern_name, triggered_at DESC)"
        )
    except Exception as e:
        logger.debug(f"composite_signals.ensure_schema: {e}")


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------
def _fetch_1d_returns(symbols: list[str]) -> dict[str, Optional[float]]:
    """
    Fetch previous-close to latest-price return (%) for each symbol.
    Returns dict mapping symbol -> return as a percentage (e.g., 1.5 for +1.5%).
    """
    returns = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).fast_info
            prev_close = getattr(info, "previous_close", None)
            last_price = (
                getattr(info, "last_price", None)
                or getattr(info, "regular_market_price", None)
            )
            if prev_close and last_price and float(prev_close) > 0:
                ret_pct = (float(last_price) - float(prev_close)) / float(prev_close) * 100
                returns[sym] = round(ret_pct, 3)
            else:
                returns[sym] = None
        except Exception:
            returns[sym] = None
    return returns


# ---------------------------------------------------------------------------
# Pattern evaluation
# ---------------------------------------------------------------------------
def _evaluate_pattern(pattern: dict, returns: dict[str, Optional[float]]) -> Optional[dict]:
    """
    Evaluate a single composite pattern against current returns.
    Returns a result dict if min_match conditions are met, else None.
    """
    conditions = pattern["conditions"]
    min_match = pattern["min_match"]

    matched = []
    not_matched = []

    for cond in conditions:
        sym = cond["series"]
        direction = cond["direction"]
        threshold = cond["threshold"]
        ret = returns.get(sym)

        if ret is None:
            not_matched.append({**cond, "actual": None, "fired": False, "reason": "no data"})
            continue

        if direction == "up":
            fired = ret > threshold
        else:  # "down"
            fired = ret < -threshold

        detail = {**cond, "actual": ret, "fired": fired}
        if fired:
            matched.append(detail)
        else:
            not_matched.append(detail)

    if len(matched) >= min_match:
        return {
            "pattern_name": pattern["name"],
            "description": pattern["description"],
            "conditions_matched": len(matched),
            "conditions_total": len(conditions),
            "details": {"matched": matched, "not_matched": not_matched},
        }
    return None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _was_recently_triggered(pattern_name: str, within_hours: int = 6) -> bool:
    """Suppress duplicate alerts: return True if pattern fired in last N hours."""
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(hours=within_hours)).isoformat()
        rows = db.query(
            "SELECT id FROM macro_composite_signals WHERE pattern_name = ? AND triggered_at >= ? LIMIT 1",
            (pattern_name, cutoff),
        )
        return bool(rows)
    except Exception:
        return False


def _record_signal(result: dict) -> None:
    try:
        from core.database import db
        db.execute(
            """
            INSERT INTO macro_composite_signals
                (pattern_name, description, conditions_matched, conditions_total, details, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result["pattern_name"],
                result["description"],
                result["conditions_matched"],
                result["conditions_total"],
                json.dumps(result["details"]),
                datetime.now().isoformat(),
            ),
        )
    except Exception as e:
        logger.debug(f"composite_signals._record_signal: {e}")


def _send_alert(result: dict) -> None:
    try:
        from engine.webhook_notifier import webhook_notifier
        webhook_notifier.reload()
        matched = result["conditions_matched"]
        total = result["conditions_total"]
        msg = (
            f"📊 Composite Signal: {result['pattern_name']} ({matched}/{total} conditions)\n"
            f"{result['description']}"
        )
        webhook_notifier.send_custom(msg)
    except Exception as e:
        logger.debug(f"composite_signals._send_alert: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_composite_signals() -> list[dict]:
    """
    Evaluate all composite patterns against current market returns.
    Records triggered patterns in DB and fires webhook alerts.
    Returns list of triggered pattern results.
    """
    ensure_schema()

    # Collect all unique symbols needed across all patterns
    all_symbols = list({c["series"] for p in COMPOSITE_PATTERNS for c in p["conditions"]})
    returns = _fetch_1d_returns(all_symbols)
    logger.debug(f"composite_signals: fetched returns for {len(returns)} symbols: {returns}")

    triggered = []
    for pattern in COMPOSITE_PATTERNS:
        result = _evaluate_pattern(pattern, returns)
        if result:
            if not _was_recently_triggered(result["pattern_name"], within_hours=6):
                _record_signal(result)
                _send_alert(result)
                logger.info(
                    f"composite_signals: '{result['pattern_name']}' triggered "
                    f"({result['conditions_matched']}/{result['conditions_total']})"
                )
            triggered.append(result)

    return triggered


def get_active_composite_signals(hours: int = 24) -> list[dict]:
    """Return composite signals triggered in the last N hours."""
    ensure_schema()
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = db.query(
            """
            SELECT * FROM macro_composite_signals
            WHERE triggered_at >= ?
            ORDER BY triggered_at DESC
            """,
            (cutoff,),
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"composite_signals.get_active_composite_signals: {e}")
        return []


def get_latest_per_pattern() -> list[dict]:
    """Return the most recent signal for each distinct pattern name."""
    ensure_schema()
    try:
        from core.database import db
        rows = db.query(
            """
            SELECT * FROM macro_composite_signals
            WHERE id IN (
                SELECT MAX(id) FROM macro_composite_signals GROUP BY pattern_name
            )
            ORDER BY triggered_at DESC
            """
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"composite_signals.get_latest_per_pattern: {e}")
        return []
