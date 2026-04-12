"""
Paper-trading stats, equity curve, settings, reset.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/paper-trading", response_class=HTMLResponse)
async def paper_trading_page(request: Request, username: str = Depends(require_auth)):
    """Paper trading simulation page"""
    from engine.paper_trading import paper_trader
    
    summary = paper_trader.get_portfolio_summary()
    trades = paper_trader.get_trade_log(limit=50)
    settings = paper_trader.get_settings()
    
    return templates.TemplateResponse("paper_trading.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "summary": summary,
        "trades": trades,
        "settings": settings,
    })


@router.post("/paper-trading/reset")
async def reset_paper_trading(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Reset paper trading portfolio"""
    csrf.verify_token(request, csrf_token)
    
    from engine.paper_trading import paper_trader
    paper_trader.reset_portfolio()
    
    audit_log.log("paper_trading_reset", username=username, ip=request.client.host)
    
    return RedirectResponse(url="/paper-trading?reset=1", status_code=303)


@router.post("/paper-trading/settings")
async def save_paper_trading_settings(
    request: Request,
    csrf_token: str = Form(...),
    starting_capital: float = Form(10000),
    position_size_pct: float = Form(5),
    min_confidence: int = Form(70),
    max_positions: int = Form(10),
    auto_execute: bool = Form(False),
    username: str = Depends(require_auth)
):
    """Save paper trading settings"""
    csrf.verify_token(request, csrf_token)
    
    from engine.paper_trading import paper_trader
    paper_trader.update_settings(
        starting_capital=starting_capital,
        position_size_pct=position_size_pct,
        min_confidence=min_confidence,
        max_positions=max_positions,
        auto_execute=1 if auto_execute else 0
    )
    
    return RedirectResponse(url="/paper-trading?saved=1", status_code=303)


@router.get("/api/paper-trading/summary")
async def api_paper_trading_summary(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get paper trading portfolio summary"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_portfolio_summary()


@router.get("/api/paper-trading/equity-curve")
async def api_paper_trading_equity_curve(
    request: Request, 
    days: int = 30,
    username: str = Depends(require_api_key_or_session)
):
    """Get equity curve data for charting"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_equity_curve(days_back=min(days, 365))


@router.get("/api/paper-trading/risk-metrics")
async def api_paper_risk_metrics(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get risk-adjusted metrics for paper trading portfolio"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_risk_metrics()


@router.get("/api/paper-trading/spy-correlation")
async def api_paper_spy_correlation(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get portfolio beta and alpha vs SPY"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_spy_correlation()


@router.get("/api/paper-trading")
async def api_paper_trading(
    request: Request,
    username: str = Depends(require_auth)
):
    """Paper trading summary — JSON for React SPA"""
    from engine.paper_trading import paper_trader
    summary = paper_trader.get_portfolio_summary()
    trades = paper_trader.get_trade_log(limit=50)
    settings = paper_trader.get_settings()
    return {"summary": summary, "trades": trades or [], "settings": settings}
