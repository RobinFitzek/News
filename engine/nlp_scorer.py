"""
NLP Sentiment Scorer — continuous free headline scoring (item #38 / #57)

Fetches RSS headlines from Reuters, AP, MarketWatch and scores them using
VADER (vaderSentiment) — a pure-Python sentiment analyser, zero GPU or API cost.

For each watchlist ticker, hourly snapshots are stored in `ticker_sentiment`.
The pipeline Stage 1 injects an anomaly when the compound score delta versus
the last analysis exceeds 0.3 (either direction).

Dashboard widget: "Sentiment Movers" — tickers with biggest NLP shift in 24h.

Dependencies:
    vaderSentiment>=3.3.2  (pip install vaderSentiment)
    The rss_client.py already handles RSS fetching via stdlib xml — we reuse it.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VADER import — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_available = True
except ImportError:
    _vader_available = False
    logger.warning(
        "vaderSentiment not installed. NLP scoring disabled. "
        "Install with: pip install vaderSentiment"
    )

# ---------------------------------------------------------------------------
# RSS feeds to monitor
# ---------------------------------------------------------------------------
HEADLINE_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.ap.org/ap/us-business",
    "https://feeds.marketwatch.com/marketwatch/topstories",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
]


def _is_available() -> bool:
    return _vader_available


def _fetch_headlines(max_per_feed: int = 30) -> list[dict]:
    """Fetch recent headlines from RSS feeds using the existing rss_client."""
    try:
        from clients.rss_client import rss_client
        return rss_client.fetch_headlines(HEADLINE_FEEDS, max_per_feed=max_per_feed)
    except Exception as e:
        logger.debug(f"nlp_scorer: RSS fetch failed: {e}")
        return []


def _mentions_ticker(text: str, ticker: str, company_name: Optional[str] = None) -> bool:
    """Check if a headline mentions a ticker symbol or company name."""
    text_upper = text.upper()
    # Exact ticker match (word boundary)
    if re.search(rf'\b{re.escape(ticker.upper())}\b', text_upper):
        return True
    # Company name match (case-insensitive, partial)
    if company_name:
        name_parts = [p for p in company_name.split() if len(p) > 3]
        if name_parts and all(part.upper() in text_upper for part in name_parts[:2]):
            return True
    return False


def score_ticker(ticker: str, headlines: list[dict], company_name: Optional[str] = None) -> dict:
    """
    Score all headlines mentioning `ticker` with VADER.

    Returns:
        {
            "ticker": str,
            "compound": float,    # -1.0 to +1.0 aggregate
            "positive": float,
            "neutral": float,
            "negative": float,
            "headline_count": int,
            "scored_at": str (ISO),
        }
    """
    if not _vader_available:
        return {
            "ticker": ticker, "compound": 0.0,
            "positive": 0.0, "neutral": 1.0, "negative": 0.0,
            "headline_count": 0, "scored_at": datetime.now().isoformat(),
            "error": "vaderSentiment not installed",
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = []

    for h in headlines:
        text = f"{h.get('title', '')} {h.get('summary', '')}".strip()
        if _mentions_ticker(text, ticker, company_name):
            vs = analyzer.polarity_scores(text)
            scores.append(vs)

    if not scores:
        return {
            "ticker": ticker, "compound": 0.0,
            "positive": 0.0, "neutral": 1.0, "negative": 0.0,
            "headline_count": 0, "scored_at": datetime.now().isoformat(),
        }

    avg_compound = sum(s["compound"] for s in scores) / len(scores)
    avg_pos = sum(s["pos"] for s in scores) / len(scores)
    avg_neu = sum(s["neu"] for s in scores) / len(scores)
    avg_neg = sum(s["neg"] for s in scores) / len(scores)

    return {
        "ticker": ticker,
        "compound": round(avg_compound, 4),
        "positive": round(avg_pos, 4),
        "neutral": round(avg_neu, 4),
        "negative": round(avg_neg, 4),
        "headline_count": len(scores),
        "scored_at": datetime.now().isoformat(),
    }


def run_hourly_scoring() -> list[dict]:
    """
    Score all active watchlist tickers and store snapshots in `ticker_sentiment`.
    Intended to be called from the scheduler every hour.

    Returns list of scored results.
    """
    if not _vader_available:
        logger.warning("nlp_scorer.run_hourly_scoring: vaderSentiment not installed, skipping")
        return []

    from core.database import db

    watchlist = db.query(
        "SELECT ticker, company_name FROM watchlist WHERE active = 1"
    ) or []

    if not watchlist:
        return []

    headlines = _fetch_headlines(max_per_feed=50)
    if not headlines:
        logger.debug("nlp_scorer: no headlines fetched")
        return []

    results = []
    for row in watchlist:
        ticker = row["ticker"]
        company_name = row.get("company_name") or row.get("name")
        result = score_ticker(ticker, headlines, company_name)
        if result.get("headline_count", 0) > 0:
            # Store snapshot
            try:
                db.execute(
                    """
                    INSERT INTO ticker_sentiment
                        (ticker, compound_score, positive, neutral, negative, headline_count, scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        result["compound"],
                        result["positive"],
                        result["neutral"],
                        result["negative"],
                        result["headline_count"],
                        result["scored_at"],
                    ),
                )
            except Exception as e:
                logger.debug(f"nlp_scorer: DB insert failed for {ticker}: {e}")
            results.append(result)

    logger.info(f"nlp_scorer: scored {len(results)} tickers from {len(headlines)} headlines")
    return results


def get_sentiment_delta(ticker: str, vs_hours: int = 24) -> Optional[float]:
    """
    Return the compound score delta for a ticker over the last `vs_hours` hours.
    Used by Stage 1 pipeline to detect sentiment anomalies.
    Returns None if insufficient data.
    """
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(hours=vs_hours)).isoformat()
        rows = db.query(
            """
            SELECT compound_score FROM ticker_sentiment
            WHERE ticker = ?
            ORDER BY scored_at DESC
            LIMIT 2
            """,
            (ticker,),
        ) or []
        if len(rows) < 2:
            return None
        latest = float(rows[0]["compound_score"])
        previous = float(rows[1]["compound_score"])
        return round(latest - previous, 4)
    except Exception:
        return None


def get_sentiment_movers(hours: int = 24, top_n: int = 10) -> list[dict]:
    """
    Return tickers with the biggest compound-score shift in the last `hours` hours.
    Used for the dashboard "Sentiment Movers" widget.
    """
    try:
        from core.database import db
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = db.query(
            """
            SELECT ticker,
                   MAX(compound_score) AS max_score,
                   MIN(compound_score) AS min_score,
                   (MAX(compound_score) - MIN(compound_score)) AS delta,
                   MAX(scored_at) AS latest_at,
                   SUM(headline_count) AS total_headlines
            FROM ticker_sentiment
            WHERE scored_at >= ?
            GROUP BY ticker
            HAVING total_headlines > 0
            ORDER BY ABS(delta) DESC
            LIMIT ?
            """,
            (cutoff, top_n),
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"nlp_scorer.get_sentiment_movers: {e}")
        return []


def ensure_schema() -> None:
    """Create ticker_sentiment table if it doesn't exist."""
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS ticker_sentiment (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker         TEXT NOT NULL,
                compound_score REAL,
                positive       REAL,
                neutral        REAL,
                negative       REAL,
                headline_count INTEGER DEFAULT 0,
                scored_at      TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_sentiment_ticker ON ticker_sentiment (ticker, scored_at DESC)"
        )
    except Exception as e:
        logger.debug(f"nlp_scorer.ensure_schema: {e}")
