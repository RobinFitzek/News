"""
Automated paper-trade engine endpoints.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/paper-trading/auto")
async def api_auto_paper_trading(username: str = Depends(require_api_key_or_session)):
    """Provides automated paper trading tracking (legacy endpoint)."""
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {
            "summary": auto_paper_trader.get_performance_summary(),
            "open_positions": auto_paper_trader.get_open_positions(),
            "should_trust": auto_paper_trader.should_trust_signals()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auto-trade/status")
async def api_auto_trade_status(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auto-trade/trust-gate")
async def api_auto_trade_trust_gate(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_trust_gate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auto-trade/positions")
async def api_auto_trade_positions(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {"positions": auto_paper_trader.get_open_positions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/auto-trade/log")
async def api_auto_trade_log(
    page: int = 1,
    username: str = Depends(require_api_key_or_session)
):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        trades = auto_paper_trader.get_trade_log(limit=20, page=page)
        total = auto_paper_trader.get_trade_log_count()
        return {"trades": trades, "total": total, "page": page, "pages": max(1, (total + 19) // 20)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auto-trade/close/{trade_id}")
async def api_auto_trade_close(
    trade_id: int,
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        result = auto_paper_trader.manual_close(trade_id)
        audit_log.log("auto_trade_manual_close", username=username,
                      ip=request.client.host, details=f"trade_id={trade_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auto-trade/toggle")
async def api_auto_trade_toggle(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    from core.database import db as _db
    current = _db.get_setting("auto_trade_enabled") or False
    _db.set_setting("auto_trade_enabled", not current)
    audit_log.log("auto_trade_toggle", username=username, ip=request.client.host,
                  details=f"enabled={not current}")
    return {"enabled": not current}


@router.get("/api/auto-trade/pending-confirm")
async def api_auto_trade_pending(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {"pending": auto_paper_trader.get_pending_confirmations()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auto-trade/confirm/{token}")
async def api_auto_trade_confirm(
    token: str,
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    from engine.auto_paper_trader import auto_paper_trader
    result = auto_paper_trader.confirm_trade(token)
    if result.get("success"):
        audit_log.log("auto_trade_confirmed", username=username,
                      ip=request.client.host, details=f"token={token[:8]}…")
    return result


@router.post("/api/auto-trade/skip/{token}")
async def api_auto_trade_skip(
    token: str,
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    from engine.auto_paper_trader import auto_paper_trader
    result = auto_paper_trader.skip_trade(token)
    audit_log.log("auto_trade_skipped", username=username,
                  ip=request.client.host, details=f"token={token[:8]}…")
    return result


@router.get("/api/auto-trade/risk-gate-status")
async def api_auto_trade_risk_gate(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_risk_gate_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/auto-trade/propose")
async def api_auto_trade_propose(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually propose an auto-trade for a specific analysis (dashboard Auto-Execute button)."""
    from engine.auto_paper_trader import auto_paper_trader
    from core.database import db as _db
    import yfinance as yf

    data = await request.json()
    analysis_id = data.get("analysis_id")
    if not analysis_id:
        raise HTTPException(status_code=400, detail="analysis_id required")

    sig = _db.query_one(
        "SELECT id, ticker, signal, score, timestamp FROM analysis_history WHERE id = ?",
        (analysis_id,)
    )
    if not sig:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        cfg = auto_paper_trader._get_config()
        hist = yf.Ticker(sig['ticker']).history(period="5d")
        if hist.empty:
            raise HTTPException(status_code=502, detail="Could not fetch current price")
        entry_price = float(hist['Close'].iloc[-1])
        portfolio_value = auto_paper_trader._estimate_portfolio_value()
        position_usd = portfolio_value * cfg["position_size_pct"]
        shares = position_usd / entry_price if entry_price > 0 else 0
        direction = 'LONG' if sig['signal'] in ('STRONG_BUY', 'BUY') else 'SHORT'

        gate = auto_paper_trader._run_risk_gate(sig['ticker'], position_usd)
        if not gate["allowed"]:
            return {"queued": False, "reason": gate["reason"]}

        auto_paper_trader._create_pending(sig, sig['ticker'], direction, entry_price, shares, position_usd, cfg)
        return {"queued": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/action/auto-trade/confirm/{token}", response_class=HTMLResponse)
async def action_auto_trade_confirm(token: str, request: Request):
    """One-click email confirm link — no auth required (token is the credential)."""
    from engine.auto_paper_trader import auto_paper_trader
    result = auto_paper_trader.confirm_trade(token)
    if result.get("success"):
        ticker = result.get("ticker", "")
        direction = result.get("direction", "")
        html = (
            "<html><body style='font-family:Arial;text-align:center;padding:60px;'>"
            f"<h2 style='color:#10b981'>✅ Trade Approved</h2>"
            f"<p>{direction} {ticker} has been entered.</p>"
            "<p><a href='/paper-trading'>View in Paper Trading →</a></p>"
            "</body></html>"
        )
    else:
        err = result.get("error", "Unknown error")
        html = (
            "<html><body style='font-family:Arial;text-align:center;padding:60px;'>"
            f"<h2 style='color:#ef4444'>❌ Could Not Approve</h2>"
            f"<p>{err}</p>"
            "<p><a href='/paper-trading'>Go to Paper Trading →</a></p>"
            "</body></html>"
        )
    return HTMLResponse(content=html)


@router.get("/action/auto-trade/skip/{token}", response_class=HTMLResponse)
async def action_auto_trade_skip(token: str, request: Request):
    """One-click email skip link — no auth required (token is the credential)."""
    from engine.auto_paper_trader import auto_paper_trader
    auto_paper_trader.skip_trade(token)
    html = (
        "<html><body style='font-family:Arial;text-align:center;padding:60px;'>"
        "<h2 style='color:#6b7280'>⏭ Trade Skipped</h2>"
        "<p>The trade proposal has been declined.</p>"
        "<p><a href='/paper-trading'>Go to Paper Trading →</a></p>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@router.get("/api/auto-trade/equity-curve")
async def api_auto_trade_equity_curve(days: int = 30, username: str = Depends(require_api_key_or_session)):
    """Cumulative auto-trade PnL by date for charting."""
    from core.database import db as _db
    cutoff = (datetime.now() - timedelta(days=min(days, 365))).strftime('%Y-%m-%d')
    rows = _db.query("""
        SELECT date(exit_date) as trade_date, pnl_pct
        FROM auto_paper_trades
        WHERE status = 'closed' AND exit_date >= ? AND pnl_pct IS NOT NULL
        ORDER BY exit_date ASC
    """, (cutoff,))

    if not rows:
        return []

    # Build cumulative curve
    from collections import OrderedDict
    daily: dict = OrderedDict()
    cumulative = 0.0
    for r in rows:
        d = r['trade_date']
        cumulative += float(r['pnl_pct']) * 100
        daily[d] = round(cumulative, 2)

    return [{"date": d, "cumulative_pnl_pct": v} for d, v in daily.items()]
