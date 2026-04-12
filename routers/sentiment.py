"""
Sentiment analysis: NLP VADER, analyst consensus, patterns, short interest, options flow.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/sentiment/{ticker}")
async def api_sentiment(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get sentiment summary including analyst consensus and contrarian signals."""
    from engine.sentiment_analyzer import sentiment_analyzer
    try:
        summary = sentiment_analyzer.get_sentiment_summary(ticker)
        return summary
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/nlp-sentiment/{ticker}")
async def api_nlp_sentiment(ticker: str, days: int = 7, username: str = Depends(require_api_key_or_session)):
    """
    Return NLP VADER sentiment trend for a ticker from stored snapshots (#38/#57).
    """
    ticker = ticker.upper()
    try:
        from core.database import db
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = db.query(
            """
            SELECT compound_score, positive, neutral, negative, headline_count, scored_at
            FROM ticker_sentiment
            WHERE ticker = ? AND scored_at >= ?
            ORDER BY scored_at ASC
            """,
            (ticker, cutoff),
        ) or []
        return {
            "ticker": ticker,
            "days": days,
            "snapshots": [dict(r) for r in rows],
            "latest": dict(rows[-1]) if rows else None,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/nlp-sentiment/movers")
async def api_nlp_sentiment_movers(hours: int = 24, username: str = Depends(require_api_key_or_session)):
    """Return tickers with biggest VADER sentiment shift in the last N hours (#38)."""
    from engine.nlp_scorer import get_sentiment_movers
    return {"movers": get_sentiment_movers(hours=hours)}


@router.get("/api/patterns/{ticker}")
async def api_patterns(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Detect chart patterns for a ticker."""
    from engine.pattern_recognition import pattern_recognizer
    try:
        patterns = pattern_recognizer.detect_patterns(ticker)
        return {"ticker": ticker.upper(), "patterns": patterns}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/short-interest/{ticker}")
async def api_short_interest(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get short interest data and squeeze setup analysis."""
    from engine.short_interest import short_interest_tracker
    try:
        data = short_interest_tracker.get_short_data(ticker)
        squeeze = short_interest_tracker.check_squeeze_setup(ticker)
        history = short_interest_tracker.get_history(ticker, days=60)
        return {
            "current": data,
            "squeeze": squeeze,
            "history": history,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/options-flow/{ticker}")
async def api_options_flow(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get options flow summary and unusual activity."""
    from engine.options_flow import options_flow
    try:
        summary = options_flow.get_options_summary(ticker)
        unusual = options_flow.detect_unusual_activity(ticker)
        return {
            "summary": summary,
            "unusual_activity": unusual,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/pairs")
async def api_pairs_all(username: str = Depends(require_api_key_or_session)):
    """Return all tested pairs ordered by cointegration strength (#40)."""
    from engine.pairs_trader import get_all_pairs, ensure_schema
    ensure_schema()
    return {"pairs": get_all_pairs()}


@router.get("/api/pairs/active")
async def api_pairs_active(username: str = Depends(require_api_key_or_session)):
    """Return pairs with active long_spread / short_spread signals (#40)."""
    from engine.pairs_trader import get_active_pairs
    return {"pairs": get_active_pairs()}


@router.post("/api/pairs/scan")
async def api_pairs_scan(request: Request, username: str = Depends(require_api_key_or_session)):
    """Trigger a manual pairs cointegration scan (#40)."""
    from engine.pairs_trader import run_weekly_scan
    try:
        pairs = run_weekly_scan()
        return {"cointegrated_pairs": len(pairs), "pairs": pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
