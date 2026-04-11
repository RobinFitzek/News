"""
SPA JSON auth endpoints and CSRF token helpers.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/csrf-token")
async def get_csrf_token_for_spa(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Provide CSRF token for React SPA clients"""
    return {"token": csrf.get_token(request)}


@router.get("/api/auth/csrf")
async def get_auth_csrf_token(request: Request):
    """Public CSRF token endpoint for pre-auth forms (login, totp). No session required."""
    return {"token": csrf.get_token(request)}


def _verify_spa_csrf(request: Request):
    """Verify CSRF token sent as X-CSRF-Token header (for React SPA)"""
    from fastapi import HTTPException
    token = request.headers.get("X-CSRF-Token", "")
    if not token or not csrf.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


@router.post("/api/auth/login")
@limiter.limit("5/minute")
async def api_auth_login(request: Request):
    """JSON login endpoint for React SPA."""
    from fastapi import HTTPException
    from core.config import ENABLE_HTTPS
    from slowapi.util import get_remote_address

    _verify_spa_csrf(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing credentials")

    client_ip = get_remote_address(request)

    try:
        db.cleanup_old_login_failures(days=30)
    except Exception:
        pass

    lockout = db.get_login_lockout_info(username, client_ip)
    if lockout.get("locked"):
        remaining_minutes = max(1, int((lockout.get("remaining_seconds", 0) + 59) / 60))
        audit_log.log("login_locked", username=username, ip=client_ip,
                      details={"remaining_minutes": remaining_minutes})
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "locked", "minutes": remaining_minutes})

    if db.verify_user(username, password):
        db.clear_login_failures(username)

        totp_info = auth_manager.get_user_totp_info(username)
        if totp_info.get("enabled"):
            import secrets as _sec
            pending_token = _sec.token_urlsafe(24)
            db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (f"_pending_totp_{pending_token}", f'"{username}"')
            )
            from fastapi.responses import JSONResponse
            return JSONResponse({"success": False, "requires_totp": True, "token": pending_token})

        session_id = auth_manager.create_session(
            username,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent", "")
        )
        db.update_last_login(username)
        audit_log.log("login_success", username=username, ip=client_ip)

        force_password_change = db.user_must_change_password(username)
        from fastapi.responses import JSONResponse
        resp = JSONResponse({
            "success": True,
            "redirect": "/change-password" if force_password_change else "/"
        })
        resp.set_cookie(
            key="session_id", value=session_id,
            httponly=True, secure=ENABLE_HTTPS, samesite="lax", max_age=86400
        )
        return resp

    db.record_login_failure(username, client_ip)
    audit_log.log("login_failed", username=username, ip=client_ip)

    post_fail_lockout = db.get_login_lockout_info(username, client_ip)
    if post_fail_lockout.get("locked"):
        remaining_minutes = max(1, int((post_fail_lockout.get("remaining_seconds", 0) + 59) / 60))
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "locked", "minutes": remaining_minutes})

    from fastapi.responses import JSONResponse
    return JSONResponse({"success": False, "error": "invalid"})


@router.post("/api/auth/totp")
@limiter.limit("10/minute")
async def api_auth_totp(request: Request):
    """JSON TOTP verification endpoint for React SPA."""
    from fastapi import HTTPException
    from core.config import ENABLE_HTTPS
    from slowapi.util import get_remote_address

    _verify_spa_csrf(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    token = (body.get("token") or "").strip()
    code = (body.get("code") or "").strip()

    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    import json as _json
    row = db.query_one("SELECT value FROM settings WHERE key = ?", (f"_pending_totp_{token}",))
    if not row:
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "expired"})

    try:
        username = _json.loads(row["value"])
    except Exception:
        username = row["value"]

    user_row = db.query_one("SELECT totp_secret FROM users WHERE username = ?", (username,))
    if user_row and user_row.get("totp_secret"):
        if not auth_manager.verify_totp(user_row["totp_secret"], code):
            if not auth_manager.use_backup_code(username, code):
                audit_log.log("login_totp_failed", username=username, ip=get_remote_address(request))
                from fastapi.responses import JSONResponse
                return JSONResponse({"success": False, "error": "invalid"})

    db.execute("DELETE FROM settings WHERE key = ?", (f"_pending_totp_{token}",))

    client_ip = get_remote_address(request)
    session_id = auth_manager.create_session(
        username,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent", "")
    )
    db.update_last_login(username)
    audit_log.log("login_success_2fa", username=username, ip=client_ip)

    force_pw = db.user_must_change_password(username)
    from fastapi.responses import JSONResponse
    resp = JSONResponse({
        "success": True,
        "redirect": "/change-password" if force_pw else "/"
    })
    resp.set_cookie(
        key="session_id", value=session_id,
        httponly=True, secure=ENABLE_HTTPS, samesite="lax", max_age=86400
    )
    return resp


@router.get("/api/auth/2fa/status")
async def api_auth_2fa_status(request: Request, username: str = Depends(require_auth)):
    """2FA status for React settings page."""
    totp_info = auth_manager.get_user_totp_info(username)
    return {
        "enabled": bool(totp_info.get("enabled")),
        "backup_codes_remaining": totp_info.get("backup_codes_remaining", 0),
    }


@router.post("/api/auth/2fa/setup-init")
async def api_auth_2fa_setup_init(request: Request, username: str = Depends(require_auth)):
    """Initialize 2FA setup — returns QR code and backup codes as JSON."""
    _verify_spa_csrf(request)
    import json as _json

    secret = auth_manager.generate_totp_secret()
    uri = auth_manager.get_totp_uri(username, secret)
    try:
        qr_b64 = auth_manager.generate_qr_code_base64(uri)
    except Exception:
        qr_b64 = None
    backup_codes = auth_manager.generate_backup_codes()

    db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (f"_pending_totp_setup_{username}", _json.dumps({"secret": secret, "backup_codes": backup_codes}))
    )

    return {
        "qr_code": qr_b64,
        "manual_key": secret,
        "backup_codes": backup_codes,
    }


@router.post("/api/auth/2fa/enable")
async def api_auth_2fa_enable(request: Request, username: str = Depends(require_auth)):
    """Enable 2FA — JSON version of /settings/2fa/enable."""
    from fastapi import HTTPException
    import json as _json

    _verify_spa_csrf(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    code = (body.get("code") or "").strip()

    row = db.query_one("SELECT value FROM settings WHERE key = ?", (f"_pending_totp_setup_{username}",))
    if not row:
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "expired"})

    data = _json.loads(row["value"])
    secret = data["secret"]
    backup_codes = data["backup_codes"]

    if not auth_manager.verify_totp(secret, code):
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "invalid_code"})

    auth_manager.save_totp_for_user(username, secret, backup_codes)
    db.execute("DELETE FROM settings WHERE key = ?", (f"_pending_totp_setup_{username}",))
    audit_log.log("2fa_enabled", username=username, ip=request.client.host)

    from fastapi.responses import JSONResponse
    return JSONResponse({"success": True})


@router.post("/api/auth/2fa/disable")
async def api_auth_2fa_disable(request: Request, username: str = Depends(require_auth)):
    """Disable 2FA — JSON version of /settings/2fa/disable."""
    from fastapi import HTTPException

    _verify_spa_csrf(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    password = body.get("password") or ""

    if not db.verify_user(username, password):
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": False, "error": "wrong_password"})

    auth_manager.disable_totp_for_user(username)
    audit_log.log("2fa_disabled", username=username, ip=request.client.host)

    from fastapi.responses import JSONResponse
    return JSONResponse({"success": True})
