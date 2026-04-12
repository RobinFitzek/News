"""
Insider trading, politicians, smart-money, institutional.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/insider-activity", response_class=HTMLResponse)
async def insider_activity_page(request: Request, username: str = Depends(require_auth)):
    """Insider trading activity page"""
    from engine.insider_tracker import insider_tracker

    # Get recent insider signals from database
    top_signals = db.get_top_insider_signals(limit=20)

    return templates.TemplateResponse("insider_activity.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "top_signals": top_signals
    })


@router.post("/insider-activity/scan")
@limiter.limit("3/hour")
async def scan_insider_activity(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Scan watchlist for insider activity"""
    csrf.verify_token(request, csrf_token)

    from engine.insider_tracker import insider_tracker

    try:
        # Scan watchlist
        results = insider_tracker.scan_watchlist_insiders(days_back=90)

        # Save to database
        for result in results:
            if result.get('recent_transactions'):
                db.save_insider_transactions_bulk(result['recent_transactions'])

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/insider-activity/{ticker}", response_class=HTMLResponse)
async def insider_detail_page(
    request: Request,
    ticker: str,
    username: str = Depends(require_auth)
):
    """Detailed insider activity for a specific ticker"""
    from engine.insider_tracker import insider_tracker

    ticker = ticker.upper()

    # Get comprehensive analysis
    analysis = insider_tracker.get_insider_analysis(ticker, days_back=180)

    # Save transactions to database
    if analysis.get('transactions'):
        db.save_insider_transactions_bulk(analysis['transactions'])

    return templates.TemplateResponse("insider_detail.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "ticker": ticker,
        "analysis": analysis
    })


@router.get("/api/institutional/{ticker}")
async def api_institutional(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get institutional holder data, ownership changes, and smart money activity."""
    from engine.institutional_tracker import institutional_tracker
    try:
        holders = institutional_tracker.get_institutional_holders(ticker)
        changes = institutional_tracker.get_ownership_changes(ticker)
        smart_money = institutional_tracker.get_smart_money_activity(ticker)
        return {
            "holders": holders,
            "changes": changes,
            "smart_money": smart_money,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/smart-money/{ticker}")
async def api_smart_money(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Lightweight smart money badge check for a ticker (queries local DB only)."""
    from engine.institutional_tracker import institutional_tracker
    try:
        return institutional_tracker.get_smart_money_activity(ticker)
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/smart-money/refresh")
async def api_refresh_13f(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually trigger 13F data refresh for all top filers."""
    from engine.institutional_tracker import institutional_tracker
    try:
        results = institutional_tracker.refresh_top_filer_holdings()
        total = sum(results.values())
        return {"filers_refreshed": len(results), "total_holdings": total, "details": results}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/insider-activity")
async def api_insider_activity(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Insider activity — JSON for React SPA"""
    return {"signals": db.get_top_insider_signals(limit=50) or []}


@router.post("/api/insider-activity/scan")
async def api_scan_insider_activity(
    request: Request,
    username: str = Depends(require_auth)
):
    """Scan watchlist for insider activity — JSON for React SPA"""
    _verify_spa_csrf(request)
    from engine.insider_tracker import insider_tracker
    try:
        results = insider_tracker.scan_watchlist_insiders(days_back=90)
        for result in results:
            if result.get('recent_transactions'):
                db.save_insider_transactions_bulk(result['recent_transactions'])
        return {"success": True, "count": len(results)}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/politicians/recent")
async def api_politicians_recent(
    request: Request,
    ticker: str = None,
    days: int = 30,
    username: str = Depends(require_auth)
):
    """Recent Senate trade disclosures, optionally filtered by ticker."""
    from engine.politician_tracker import politician_tracker
    trades = politician_tracker.get_recent_trades(ticker=ticker, days=days)
    return {"trades": trades, "count": len(trades)}


@router.get("/api/politicians/top-tickers")
async def api_politicians_top_tickers(
    request: Request,
    days: int = 90,
    top_n: int = 20,
    username: str = Depends(require_auth)
):
    """Most actively traded tickers by politicians in the last N days."""
    from engine.politician_tracker import politician_tracker
    return {"tickers": politician_tracker.get_top_traded_tickers(days=days, top_n=top_n)}


@router.get("/api/politicians/features/{ticker}")
async def api_politicians_features(
    ticker: str,
    request: Request,
    username: str = Depends(require_auth)
):
    """9 political trade features for a ticker (today)."""
    from engine.politician_tracker import politician_tracker
    from datetime import date
    features = politician_tracker.get_features_for_date(ticker.upper(), date.today())
    return {"ticker": ticker.upper(), "date": date.today().isoformat(), **features}
