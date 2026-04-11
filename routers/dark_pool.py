"""
Dark-pool activity pages and JSON feeds.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/dark-pool", response_class=HTMLResponse)
async def dark_pool_page(request: Request, days: int = 7, username: str = Depends(require_auth)):
    """Dark pool & institutional block trade activity page (#52)."""
    from engine.dark_pool_tracker import get_top_signals, ensure_schema
    ensure_schema()
    signals = get_top_signals(days=days, top_n=50)
    return templates.TemplateResponse("dark_pool.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "signals": signals,
        "days": days,
    })


@router.get("/dark-pool/{ticker}", response_class=HTMLResponse)
async def dark_pool_ticker_page(request: Request, ticker: str, username: str = Depends(require_auth)):
    """Dark pool signals for a specific ticker (#52)."""
    from engine.dark_pool_tracker import get_ticker_signals, ensure_schema
    ensure_schema()
    ticker = ticker.upper()
    signals = get_ticker_signals(ticker, days=30)
    return templates.TemplateResponse("dark_pool.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "signals": signals,
        "ticker": ticker,
        "days": 30,
    })


@router.post("/dark-pool/scan")
@limiter.limit("2/hour")
async def dark_pool_scan(request: Request, csrf_token: str = Form(...), username: str = Depends(require_auth)):
    """Trigger a dark pool / volume anomaly scan (#52)."""
    csrf.verify_token(request, csrf_token)
    from engine.dark_pool_tracker import scan_watchlist
    count = scan_watchlist()
    return {"success": True, "signals_found": count}


@router.get("/api/dark-pool")
async def api_dark_pool_signals(days: int = 7, min_ratio: float = 0.0,
                                 username: str = Depends(require_api_key_or_session)):
    """Return recent dark pool / volume anomaly signals (#52)."""
    from engine.dark_pool_tracker import get_all_recent_signals
    return {"signals": get_all_recent_signals(days=days, min_volume_ratio=min_ratio)}


@router.get("/api/dark-pool/{ticker}")
async def api_dark_pool_ticker(ticker: str, days: int = 30,
                                username: str = Depends(require_api_key_or_session)):
    """Return dark pool signals for a single ticker (#52)."""
    from engine.dark_pool_tracker import get_ticker_signals
    return {"ticker": ticker.upper(), "signals": get_ticker_signals(ticker, days=days)}
