"""
Scheduler start/stop/run-now endpoints.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.post("/scheduler/start")
async def start_scheduler(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Start the scheduler"""
    csrf.verify_token(request, csrf_token)
    scheduler.start()
    return RedirectResponse(url="/", status_code=303)


@router.post("/scheduler/stop")
async def stop_scheduler(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Stop the scheduler"""
    csrf.verify_token(request, csrf_token)
    scheduler.stop()
    return RedirectResponse(url="/", status_code=303)


@router.post("/scheduler/run-now")
async def run_scan_now(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Trigger immediate scan"""
    csrf.verify_token(request, csrf_token)

    try:
        # Log the scan attempt
        audit_log.log("manual_scan_triggered", username=username, ip=request.client.host, details={"source": "web_dashboard"})

        # Check if any AI provider is configured
        if not provider_registry.get_all_providers_with_status():
            return RedirectResponse(url="/?message=error&detail=No+AI+provider+configured.+Add+one+in+Settings.", status_code=303)

        # Check watchlist
        watchlist = db.get_watchlist(active_only=True)
        if not watchlist or len(watchlist) == 0:
            return RedirectResponse(url="/?message=error&detail=Watchlist+is+empty", status_code=303)

        # Run the scan
        if scheduler.trigger_manual_scan():
            # Success message
            return RedirectResponse(url="/?message=success&detail=Scan+started+in+background.+Check+dashboard+for+status.", status_code=303)
        else:
            return RedirectResponse(url="/?message=warning&detail=Scan+already+running", status_code=303)

    except Exception as e:
        # Log the error
        audit_log.log("manual_scan_failed", username=username, ip=request.client.host, details={"error": str(e)})
        print(f"❌ Manual scan error: {e}")

        # Return with error message
        error_msg = str(e)[:100]  # Limit error message length
        return RedirectResponse(url=f"/?message=error&detail={error_msg}", status_code=303)


@router.post("/api/scheduler/start")
async def api_start_scheduler(
    request: Request,
    username: str = Depends(require_auth)
):
    """Start scheduler — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    scheduler.start()
    return {"status": "started"}


@router.post("/api/scheduler/stop")
async def api_stop_scheduler(
    request: Request,
    username: str = Depends(require_auth)
):
    """Stop scheduler — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    scheduler.stop()
    return {"status": "stopped"}


@router.post("/api/scheduler/run-now")
async def api_run_now(
    request: Request,
    username: str = Depends(require_auth)
):
    """Trigger immediate scan — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    try:
        audit_log.log("manual_scan_triggered", username=username,
                      ip=request.client.host, details={"source": "react_spa"})
        if scheduler.trigger_manual_scan():
            return {"status": "scanning", "message": "Scan started in background"}
        else:
            return {"status": "already_scanning", "message": "Scan already running"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
