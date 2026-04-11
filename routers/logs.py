"""
System logs, dev logs, login-failure unlocks.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, username: str = Depends(require_auth)):
    """System logs"""
    from engine.alert_manager import alert_manager

    dev_mode = db.get_setting('development_mode') or False
    alert_filter = (request.query_params.get('alert_filter') or 'active').lower()
    if alert_filter not in ('active', 'all'):
        alert_filter = 'active'
    
    # Load system logs if in dev mode
    system_logs = ""
    if dev_mode:
        try:
            log_file = PROJECT_ROOT / "logs" / "application.log"
            if log_file.exists():
                # Read last 500 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    system_logs = ''.join(lines[-500:])
        except Exception as e:
            system_logs = f"Error loading logs: {e}"

    dedup_alerts = alert_manager.get_active_alerts(include_acknowledged=(alert_filter == 'all'))
    alert_summary = alert_manager.get_alert_summary()
    login_fail_summary = db.get_login_failures_summary(hours=24)
    recent_login_failures = db.get_recent_login_failures(limit=30, hours=24)
    
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "scheduler_logs": db.get_scheduler_logs(limit=50),
        "alerts": db.get_alerts(limit=50),
        "dedup_alerts": dedup_alerts,
        "alert_summary": alert_summary,
        "alert_filter": alert_filter,
        "login_fail_summary": login_fail_summary,
        "recent_login_failures": recent_login_failures,
        "dev_mode": dev_mode,
        "system_logs": system_logs
    })


@router.post("/logs/login-failures/unlock")
async def unlock_login_failures(
    request: Request,
    unlock_username: str = Form(""),
    unlock_ip: str = Form(""),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Clear lockout/failure records for a username and/or IP."""
    csrf.verify_token(request, csrf_token)

    target_user = (unlock_username or '').strip().lower()
    target_ip = (unlock_ip or '').strip()

    if not target_user and not target_ip:
        return RedirectResponse(url="/logs?unlock=error", status_code=303)

    deleted_user = 0
    deleted_ip = 0
    if target_user:
        deleted_user = db.clear_login_failures(target_user)
    if target_ip:
        deleted_ip = db.clear_login_failures_for_ip(target_ip)

    audit_log.log(
        "login_failures_unlocked",
        username=username,
        ip=request.client.host,
        details={
            "target_user": target_user,
            "target_ip": target_ip,
            "deleted_user": deleted_user,
            "deleted_ip": deleted_ip,
        }
    )

    return RedirectResponse(url="/logs?unlock=1", status_code=303)


@router.post("/logs/login-failures/unlock-all")
async def unlock_all_login_failures(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Clear all login failure records from the last 24 hours."""
    csrf.verify_token(request, csrf_token)

    deleted = db.clear_recent_login_failures(hours=24)
    audit_log.log(
        "login_failures_unlocked_all",
        username=username,
        ip=request.client.host,
        details={"window_hours": 24, "deleted": deleted}
    )
    return RedirectResponse(url="/logs?unlock=all", status_code=303)


@router.post("/toggle-dev-mode")
@limiter.limit("10/minute")
async def toggle_dev_mode(request: Request, username: str = Depends(require_auth)):
    """Toggle development mode"""
    try:
        data = await request.json()
        
        # Validate CSRF token from header
        token = request.headers.get('X-CSRF-Token', '')
        if not csrf.validate_token(token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
        
        enabled = data.get('enabled', False)
        
        # Update setting
        db.set_setting('development_mode', enabled)
        
        # Log the action
        audit_log.log(
            event_type="toggle_dev_mode",
            username=username,
            ip=request.client.host,
            details={"enabled": enabled}
        )
        
        return {"success": True, "dev_mode": enabled}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dev-logs")
@limiter.limit("30/minute")
async def get_dev_logs(request: Request, username: str = Depends(require_auth)):
    """Get fresh system logs (dev mode only)"""
    dev_mode = db.get_setting('development_mode') or False
    
    if not dev_mode:
        raise HTTPException(status_code=403, detail="Dev mode not enabled")
    
    try:
        log_file = PROJECT_ROOT / "logs" / "application.log"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Return last 500 lines
                logs = ''.join(lines[-500:])
        else:
            logs = "No log file found"
        
        return {"logs": logs}
    except Exception as e:
        return {"logs": f"Error loading logs: {e}"}


@router.get("/api/logs")
async def api_logs(
    request: Request,
    alert_filter: str = "active",
    username: str = Depends(require_auth)
):
    """System logs — JSON for React SPA"""
    from engine.alert_manager import alert_manager
    if alert_filter not in ("active", "all"):
        alert_filter = "active"
    dedup_alerts = alert_manager.get_active_alerts(include_acknowledged=(alert_filter == "all"))
    alert_summary = alert_manager.get_alert_summary()
    login_fail_summary = db.get_login_failures_summary(hours=24)
    recent_login_failures = db.get_recent_login_failures(limit=30, hours=24)
    return {
        "scheduler_logs": db.get_scheduler_logs(limit=50) or [],
        "alerts": db.get_alerts(limit=50) or [],
        "dedup_alerts": list(dedup_alerts) if dedup_alerts else [],
        "alert_summary": alert_summary,
        "alert_filter": alert_filter,
        "login_fail_summary": login_fail_summary,
        "recent_login_failures": recent_login_failures or [],
    }
