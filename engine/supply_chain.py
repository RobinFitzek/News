"""
Supply Chain Risk Mapper (items #44 / #58)

For each watchlist ticker, queries Perplexity AI once to map its top 5 suppliers
and key customers. Results cached in `supply_chain_map` (refreshed quarterly).

During geo scans, the geo engine can call `get_geo_elevated_tickers()` to find
holdings whose suppliers are in flagged regions — raising their geo_risk_score
by up to +2 pts even if the ticker itself isn't directly in the scan.

Cost: 1 Perplexity call per ticker per quarter (very low budget impact).
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DAYS = 90  # Refresh every 90 days


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS supply_chain_map (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT NOT NULL,
                related_ticker    TEXT,           -- supplier/customer ticker (if known)
                company_name      TEXT NOT NULL,
                relationship_type TEXT NOT NULL,  -- 'supplier' | 'customer' | 'partner'
                description       TEXT,
                cached_at         TEXT NOT NULL,
                UNIQUE(ticker, company_name, relationship_type)
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_supply_chain_ticker "
            "ON supply_chain_map (ticker, cached_at)"
        )
    except Exception as e:
        logger.debug(f"supply_chain.ensure_schema: {e}")


# ---------------------------------------------------------------------------
# Perplexity fetch
# ---------------------------------------------------------------------------
_SUPPLY_CHAIN_PROMPT = """
What are the top 5 suppliers and top 3 key customers of {company} ({ticker})?
For each, provide: company name, their ticker symbol if publicly traded, and relationship type (supplier/customer).

Respond ONLY as a JSON array with objects:
[
  {{"company_name": "...", "ticker": "..." or null, "relationship_type": "supplier"|"customer", "description": "..."}},
  ...
]
Do not include any explanation outside the JSON array.
"""


def _fetch_supply_chain(ticker: str, company_name: Optional[str] = None) -> list[dict]:
    """Query Perplexity for supply chain data. Returns parsed list."""
    try:
        from clients.perplexity_client import perplexity_client
        name = company_name or ticker
        prompt = _SUPPLY_CHAIN_PROMPT.format(company=name, ticker=ticker)
        response = perplexity_client.search(prompt)
        if not response or not response.strip():
            return []

        # Extract JSON array from response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []

        raw = json.loads(json_match.group())
        validated = []
        for item in raw:
            if isinstance(item, dict) and item.get("company_name") and item.get("relationship_type"):
                validated.append({
                    "company_name": str(item["company_name"])[:200],
                    "related_ticker": (str(item.get("ticker") or ""))[:20] or None,
                    "relationship_type": str(item["relationship_type"])[:20],
                    "description": str(item.get("description") or "")[:500],
                })
        return validated[:8]  # cap at 8 entries
    except Exception as e:
        logger.debug(f"supply_chain._fetch_supply_chain({ticker}): {e}")
        return []


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _is_cached(ticker: str) -> bool:
    """Return True if supply chain data is fresh (< CACHE_DAYS old)."""
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(days=CACHE_DAYS)).isoformat()
        rows = db.query(
            "SELECT id FROM supply_chain_map WHERE ticker = ? AND cached_at >= ? LIMIT 1",
            (ticker.upper(), cutoff),
        )
        return bool(rows)
    except Exception:
        return False


def _save_supply_chain(ticker: str, entries: list[dict]) -> None:
    try:
        from core.database import db
        now = datetime.now().isoformat()
        for e in entries:
            db.execute(
                """
                INSERT INTO supply_chain_map
                    (ticker, related_ticker, company_name, relationship_type, description, cached_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, company_name, relationship_type) DO UPDATE SET
                    related_ticker=excluded.related_ticker,
                    description=excluded.description,
                    cached_at=excluded.cached_at
                """,
                (
                    ticker.upper(),
                    e.get("related_ticker"),
                    e["company_name"],
                    e["relationship_type"],
                    e.get("description"),
                    now,
                ),
            )
    except Exception as ex:
        logger.debug(f"supply_chain._save_supply_chain({ticker}): {ex}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_supply_chain(ticker: str, force_refresh: bool = False) -> dict:
    """
    Return supply chain for a ticker, fetching from Perplexity if not cached.

    Returns:
        {
            "ticker": str,
            "suppliers": list[dict],
            "customers": list[dict],
            "cached_at": str or None,
            "source": "cache" | "perplexity",
        }
    """
    ensure_schema()
    ticker = ticker.upper()

    # Try from cache first
    if not force_refresh and _is_cached(ticker):
        return _get_cached_supply_chain(ticker)

    # Fetch company name from watchlist
    company_name = None
    try:
        from core.database import db
        row = db.query("SELECT company_name FROM watchlist WHERE ticker = ?", (ticker,))
        if row:
            company_name = row[0].get("company_name")
    except Exception:
        pass

    entries = _fetch_supply_chain(ticker, company_name)
    if entries:
        _save_supply_chain(ticker, entries)

    return _get_cached_supply_chain(ticker)


def _get_cached_supply_chain(ticker: str) -> dict:
    try:
        from core.database import db
        rows = db.query(
            "SELECT * FROM supply_chain_map WHERE ticker = ? ORDER BY relationship_type",
            (ticker,),
        ) or []
        suppliers = [dict(r) for r in rows if r["relationship_type"] == "supplier"]
        customers = [dict(r) for r in rows if r["relationship_type"] == "customer"]
        partners = [dict(r) for r in rows if r["relationship_type"] == "partner"]
        cached_at = rows[0]["cached_at"] if rows else None
        return {
            "ticker": ticker,
            "suppliers": suppliers,
            "customers": customers,
            "partners": partners,
            "cached_at": cached_at,
            "source": "cache" if rows else "empty",
        }
    except Exception as e:
        return {"ticker": ticker, "suppliers": [], "customers": [], "partners": [], "error": str(e)}


def get_geo_elevated_tickers(flagged_regions: list[str]) -> list[dict]:
    """
    After a geo scan, cross-check supplier companies against flagged regions.
    Returns tickers whose suppliers are in flagged regions, with a +2 geo_risk elevation.

    Called from the geopolitical scan post-processing.
    """
    ensure_schema()
    if not flagged_regions:
        return []

    try:
        from core.database import db
        elevated = []
        # Get all tickers with supplier data
        tickers = [
            r["ticker"]
            for r in (db.query("SELECT DISTINCT ticker FROM supply_chain_map WHERE relationship_type='supplier'") or [])
        ]
        region_keywords = [r.lower() for r in flagged_regions]

        for ticker in tickers:
            suppliers = db.query(
                "SELECT company_name, description FROM supply_chain_map WHERE ticker = ? AND relationship_type = 'supplier'",
                (ticker,),
            ) or []
            for s in suppliers:
                text = f"{s.get('company_name','')} {s.get('description','')}".lower()
                if any(kw in text for kw in region_keywords):
                    elevated.append({
                        "ticker": ticker,
                        "reason": f"Supplier '{s['company_name']}' in flagged region",
                        "geo_risk_elevation": 2,
                    })
                    break  # one flagged supplier is enough

        return elevated
    except Exception as e:
        logger.debug(f"supply_chain.get_geo_elevated_tickers: {e}")
        return []


def refresh_stale_tickers() -> int:
    """Quarterly job: refresh supply chain data for tickers not updated in >90 days."""
    ensure_schema()
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(days=CACHE_DAYS)).isoformat()
        # Tickers in watchlist with stale or missing supply chain data
        watchlist = db.query("SELECT ticker, company_name FROM watchlist WHERE active = 1") or []
        cached_recent = {
            r["ticker"]
            for r in (db.query(
                "SELECT DISTINCT ticker FROM supply_chain_map WHERE cached_at >= ?", (cutoff,)
            ) or [])
        }
        stale = [r for r in watchlist if r["ticker"] not in cached_recent]
        refreshed = 0
        for row in stale:
            entries = _fetch_supply_chain(row["ticker"], row.get("company_name"))
            if entries:
                _save_supply_chain(row["ticker"], entries)
                refreshed += 1
        logger.info(f"supply_chain.refresh_stale_tickers: refreshed {refreshed}/{len(stale)} tickers")
        return refreshed
    except Exception as e:
        logger.error(f"supply_chain.refresh_stale_tickers: {e}")
        return 0
