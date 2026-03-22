"""
Portfolio Q&A — Natural Language Portfolio Queries (item #37)

Answers free-text questions about the current portfolio using Gemini.
Builds a rich structured context from live DB data (holdings, latest analyses,
geo scores, sector weights, recent geo scan) and sends it alongside the user's
question to Gemini for synthesis.

Rate-limited to 1 query per 30 seconds; cost deducted from Gemini budget.
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Rate limit: minimum seconds between queries
RATE_LIMIT_SECONDS = 30
_last_query_time: float = 0.0


def _get_portfolio_context() -> dict:
    """Build structured context from live DB data."""
    from core.database import db

    context: dict = {}

    # --- Current holdings ---
    try:
        holdings = db.query(
            """
            SELECT ticker, shares, entry_price, entry_date,
                   (shares * entry_price) AS cost_basis
            FROM portfolio_trades
            WHERE exit_date IS NULL
            ORDER BY cost_basis DESC
            LIMIT 30
            """
        ) or []
        context["holdings"] = [dict(h) for h in holdings]
    except Exception as e:
        context["holdings"] = []
        logger.debug(f"portfolio_qa: holdings fetch failed: {e}")

    # --- Latest analysis per held ticker ---
    try:
        held_tickers = [h["ticker"] for h in context.get("holdings", [])]
        analyses = []
        for ticker in held_tickers[:20]:  # cap to 20 to keep prompt size sane
            row = db.query(
                """
                SELECT ticker, signal, risk_score, geo_risk_score, bull_case, bear_case,
                       recommendation, timestamp
                FROM analysis_history
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (ticker,),
            )
            if row:
                analyses.append(dict(row[0]))
        context["latest_analyses"] = analyses
    except Exception as e:
        context["latest_analyses"] = []
        logger.debug(f"portfolio_qa: analyses fetch failed: {e}")

    # --- Sector distribution ---
    try:
        sector_rows = db.query(
            """
            SELECT w.sector, COUNT(*) as count, AVG(ah.risk_score) as avg_risk
            FROM watchlist w
            LEFT JOIN (
                SELECT ticker, risk_score, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
                FROM analysis_history
            ) ah ON ah.ticker = w.ticker AND ah.rn = 1
            WHERE w.active = 1 AND w.sector IS NOT NULL
            GROUP BY w.sector
            ORDER BY count DESC
            """
        ) or []
        context["sector_distribution"] = [dict(r) for r in sector_rows]
    except Exception as e:
        context["sector_distribution"] = []
        logger.debug(f"portfolio_qa: sector fetch failed: {e}")

    # --- Top geo exposures ---
    try:
        geo_rows = db.query(
            """
            SELECT ticker, geo_risk_score, geopolitical_context
            FROM analysis_history
            WHERE geo_risk_score IS NOT NULL AND geo_risk_score > 0
            ORDER BY timestamp DESC, geo_risk_score DESC
            LIMIT 10
            """
        ) or []
        # Deduplicate by ticker (keep highest score)
        seen = set()
        top_geo = []
        for r in geo_rows:
            if r["ticker"] not in seen:
                seen.add(r["ticker"])
                top_geo.append({"ticker": r["ticker"], "geo_risk_score": r["geo_risk_score"]})
        context["top_geo_exposures"] = top_geo
    except Exception as e:
        context["top_geo_exposures"] = []
        logger.debug(f"portfolio_qa: geo fetch failed: {e}")

    # --- Latest geopolitical scan summary ---
    try:
        geo_scan = db.query(
            """
            SELECT scan_date, severity_avg, raw_text
            FROM geopolitical_events
            ORDER BY scan_date DESC
            LIMIT 1
            """
        )
        if geo_scan:
            scan = geo_scan[0]
            # Truncate raw_text to keep prompt manageable
            summary_text = (scan.get("raw_text") or "")[:1500]
            context["latest_geo_scan"] = {
                "date": scan.get("scan_date"),
                "severity_avg": scan.get("severity_avg"),
                "summary": summary_text,
            }
    except Exception as e:
        logger.debug(f"portfolio_qa: geo scan fetch failed: {e}")

    return context


def _build_prompt(user_question: str, context: dict) -> str:
    """Compose the Gemini prompt from user question + portfolio context."""
    holdings_json = json.dumps(context.get("holdings", []), indent=2)
    analyses_json = json.dumps(context.get("latest_analyses", []), indent=2)
    sectors_json = json.dumps(context.get("sector_distribution", []), indent=2)
    geo_exp_json = json.dumps(context.get("top_geo_exposures", []), indent=2)

    geo_scan = context.get("latest_geo_scan", {})
    geo_summary = (
        f"Date: {geo_scan.get('date')}  |  Avg Severity: {geo_scan.get('severity_avg')}\n"
        f"{geo_scan.get('summary', '')}"
        if geo_scan else "No recent geopolitical scan."
    )

    prompt = f"""You are a portfolio analyst AI. Answer the following question about the user's investment portfolio using ONLY the data provided below. Be concise, specific, and actionable. If data is insufficient to answer fully, say so clearly.

USER QUESTION:
{user_question}

--- PORTFOLIO DATA ---

Current Holdings:
{holdings_json}

Latest Signal Analysis per Holding:
{analyses_json}

Sector Distribution (watchlist):
{sectors_json}

Top Geopolitical Risk Exposures:
{geo_exp_json}

Latest Geopolitical Scan Summary:
{geo_summary}

--- INSTRUCTIONS ---
- Answer the question directly and factually.
- Reference specific tickers and numbers from the data where relevant.
- If you identify risks or opportunities, be specific (e.g., "NVDA has geo_risk_score=8, suggesting elevated Taiwan/chip-war exposure").
- Keep the answer under 300 words.
- Do NOT invent data that isn't in the provided context.
"""
    return prompt


def ask(question: str) -> dict:
    """
    Answer a free-text question about the user's portfolio.

    Returns:
        {
            "answer": str,
            "sources": list of tickers/datasets used,
            "rate_limited": bool,
            "error": str or None,
        }
    """
    global _last_query_time

    # Rate limit
    now = time.time()
    elapsed = now - _last_query_time
    if elapsed < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - elapsed)
        return {
            "answer": None,
            "sources": [],
            "rate_limited": True,
            "error": f"Rate limit: please wait {wait}s before the next query.",
        }

    if not question or not question.strip():
        return {"answer": None, "sources": [], "rate_limited": False, "error": "Empty question."}

    try:
        context = _get_portfolio_context()
        prompt = _build_prompt(question.strip(), context)

        from clients.gemini_client import gemini_client
        answer = gemini_client.generate(prompt, task_type="analyze")

        _last_query_time = time.time()

        # Determine which sources were used
        sources = []
        holdings = context.get("holdings", [])
        if holdings:
            sources.append(f"{len(holdings)} portfolio holdings")
        if context.get("latest_analyses"):
            sources.append(f"{len(context['latest_analyses'])} latest analyses")
        if context.get("sector_distribution"):
            sources.append("sector distribution")
        if context.get("top_geo_exposures"):
            sources.append("geopolitical exposure scores")
        if context.get("latest_geo_scan"):
            sources.append("latest geo scan")

        return {
            "answer": answer,
            "sources": sources,
            "rate_limited": False,
            "error": None,
        }
    except Exception as e:
        logger.error(f"portfolio_qa.ask failed: {e}")
        return {
            "answer": None,
            "sources": [],
            "rate_limited": False,
            "error": str(e),
        }
