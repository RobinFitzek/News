"""
Idea discovery engine — scan, list, promote, dismiss.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/discover", response_class=HTMLResponse)
async def discover_page(request: Request, username: str = Depends(require_auth)):
    """Stock discovery page"""
    from clients.perplexity_client import pplx_client

    usage = pplx_client.get_usage()

    return templates.TemplateResponse("discover.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "perplexity_configured": pplx_client.is_configured(),
        "api_usage": usage
    })


@router.post("/discover")
@limiter.limit("5/hour")
async def run_discovery(
    request: Request,
    sector: str = Form(None),
    focus: str = Form("balanced"),
    limit: int = Form(5),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Run Perplexity-powered stock discovery"""
    csrf.verify_token(request, csrf_token)

    from engine.discovery_engine import discovery_engine

    # Run discovery
    result = discovery_engine.discover_with_perplexity(
        sector=sector if sector else None,
        focus=focus,
        limit=min(limit, 10)  # Cap at 10
    )

    # Return JSON response
    return {
        "success": result['success'],
        "stocks": result.get('stocks', []),
        "error": result.get('error'),
        "raw_analysis": result.get('raw_analysis', ''),
        "timestamp": result.get('timestamp'),
        "api_usage": result.get('filtered_count', 0)
    }


@router.get("/discoveries", response_class=HTMLResponse)
async def discoveries_page(request: Request, username: str = Depends(require_auth)):
    """Auto-discovery results page"""
    status_filter = request.query_params.get('status', 'all')

    discoveries = db.get_recent_discoveries(
        days=30,
        status=status_filter if status_filter != 'all' else None
    )
    stats = db.get_discovery_stats()
    discovery_log = db.get_discovery_log(limit=10)

    return templates.TemplateResponse("discoveries.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "discoveries": discoveries,
        "stats": stats,
        "discovery_log": discovery_log,
        "status_filter": status_filter,
    })


@router.post("/discoveries/{discovery_id}/promote")
async def promote_discovery(
    request: Request,
    discovery_id: int,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Manually promote a discovery to watchlist"""
    csrf.verify_token(request, csrf_token)

    # Get the discovery
    discovery = db.query_one(
        "SELECT * FROM discovered_stocks WHERE id = ?", (discovery_id,)
    )
    if not discovery:
        raise HTTPException(status_code=404, detail="Discovery not found")

    ticker = discovery['ticker']
    db.add_to_watchlist(ticker, '')
    db.promote_discovery(ticker)

    audit_log.log("discovery_promoted", username=username,
                  ip=request.client.host, details={"ticker": ticker})

    return RedirectResponse(url="/discoveries?promoted=1", status_code=303)


@router.post("/discoveries/{discovery_id}/dismiss")
async def dismiss_discovery(
    request: Request,
    discovery_id: int,
    csrf_token: str = Form(...),
    reason: str = Form(""),
    username: str = Depends(require_auth)
):
    """Dismiss a discovery"""
    csrf.verify_token(request, csrf_token)
    db.dismiss_discovery(discovery_id, reason)
    return RedirectResponse(url="/discoveries", status_code=303)


@router.get("/api/discovery/stats")
async def api_discovery_stats(request: Request, username: str = Depends(require_api_key_or_session)):
    """JSON stats for discovery dashboard widget"""
    return db.get_discovery_stats()


@router.post("/discovery/run-now")
@limiter.limit("2/hour")
async def run_discovery_now(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Manual trigger for discovery run"""
    csrf.verify_token(request, csrf_token)

    import threading
    from engine.auto_discovery import auto_discovery

    audit_log.log("manual_discovery_triggered", username=username,
                  ip=request.client.host)

    def _run():
        auto_discovery.run_daily_discovery()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return RedirectResponse(
        url="/discoveries?message=Discovery+started+in+background",
        status_code=303
    )


@router.post("/discoveries/bulk-promote")
async def bulk_promote_discoveries(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Promote multiple discoveries to watchlist in one action."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    ids_raw = form.getlist("discovery_ids")
    ids = []
    for v in ids_raw:
        try:
            ids.append(int(v))
        except (ValueError, TypeError):
            pass
    if ids:
        db.bulk_promote_discoveries(ids)
        # Also add to watchlist
        for disc_id in ids:
            disc = db.query_one("SELECT ticker FROM discovered_stocks WHERE id = ?", (disc_id,))
            if disc:
                db.add_to_watchlist(disc['ticker'])
    return RedirectResponse(url=f"/discoveries?promoted={len(ids)}", status_code=303)


@router.post("/discoveries/bulk-dismiss")
async def bulk_dismiss_discoveries(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Dismiss multiple discoveries at once."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    ids_raw = form.getlist("discovery_ids")
    ids = []
    for v in ids_raw:
        try:
            ids.append(int(v))
        except (ValueError, TypeError):
            pass
    if ids:
        db.bulk_dismiss_discoveries(ids, reason="bulk_dismiss")
    return RedirectResponse(url=f"/discoveries?dismissed={len(ids)}", status_code=303)


@router.get("/api/discoveries")
async def api_discoveries(
    request: Request,
    status: str = "all",
    username: str = Depends(require_api_key_or_session)
):
    """Discoveries list + stats — JSON for React SPA"""
    items = db.get_recent_discoveries(
        days=30,
        status=status if status != "all" else None
    )
    stats = db.get_discovery_stats()
    log = db.get_discovery_log(limit=10)
    return {"discoveries": items or [], "stats": stats, "log": log or [], "status_filter": status}


@router.post("/api/discoveries/{discovery_id}/promote")
async def api_promote_discovery(
    request: Request,
    discovery_id: int,
    username: str = Depends(require_auth)
):
    """Promote a discovery — JSON for React SPA"""
    _verify_spa_csrf(request)
    discovery = db.query_one("SELECT * FROM discovered_stocks WHERE id = ?", (discovery_id,))
    if not discovery:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Discovery not found")
    ticker = discovery['ticker']
    db.add_to_watchlist(ticker, '')
    db.promote_discovery(ticker)
    audit_log.log("discovery_promoted", username=username, ip=request.client.host, details={"ticker": ticker})
    return {"status": "promoted", "ticker": ticker}


@router.post("/api/discoveries/{discovery_id}/dismiss")
async def api_dismiss_discovery(
    request: Request,
    discovery_id: int,
    username: str = Depends(require_auth)
):
    """Dismiss a discovery — JSON for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    db.dismiss_discovery(discovery_id, data.get("reason", ""))
    return {"status": "dismissed"}
