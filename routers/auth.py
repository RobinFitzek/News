"""
Session auth, login, logout, 2FA, password change.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/login")
async def login_page(request: Request):
    """Login page — served by React SPA"""
    username = auth_manager.get_current_user(request)
    if username:
        if db.user_must_change_password(username):
            return RedirectResponse(url="/change-password", status_code=303)
        return RedirectResponse(url="/", status_code=303)
    from fastapi.responses import FileResponse
    index = PROJECT_ROOT / "static" / "react" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=503, detail="React build not found. Run: cd frontend && npm run build")


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    from core.config import ENABLE_HTTPS
    from slowapi.util import get_remote_address

    client_ip = get_remote_address(request)

    # Best-effort housekeeping for stale failure records
    try:
        db.cleanup_old_login_failures(days=30)
    except Exception:
        pass

    # Backoff/lockout gate before password verification
    lockout = db.get_login_lockout_info(username, client_ip)
    if lockout.get('locked'):
        remaining_minutes = max(1, int((lockout.get('remaining_seconds', 0) + 59) / 60))
        audit_log.log("login_locked", username=username, ip=client_ip,
                      details={"remaining_minutes": remaining_minutes})
        return RedirectResponse(url=f"/login?error=locked&minutes={remaining_minutes}", status_code=303)

    if db.verify_user(username, password):
        db.clear_login_failures(username)

        # Check if 2FA is required (#33)
        totp_info = auth_manager.get_user_totp_info(username)
        if totp_info.get('enabled'):
            # Store pending auth state in a short-lived cookie and redirect to TOTP step
            import secrets as _sec
            pending_token = _sec.token_urlsafe(24)
            db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (f'_pending_totp_{pending_token}', f'"{username}"')
            )
            resp = RedirectResponse(url=f"/login/totp?token={pending_token}", status_code=303)
            return resp

        # Create session
        session_id = auth_manager.create_session(
            username,
            ip_address=client_ip,
            user_agent=request.headers.get('user-agent', '')
        )
        db.update_last_login(username)

        # Audit log successful login
        audit_log.log("login_success", username=username, ip=client_ip)

        force_password_change = db.user_must_change_password(username)

        # Redirect to dashboard with session cookie
        response = RedirectResponse(
            url="/change-password" if force_password_change else "/",
            status_code=303
        )
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=ENABLE_HTTPS,
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return response

    # Audit log failed login
    db.record_login_failure(username, client_ip)
    audit_log.log("login_failed", username=username, ip=client_ip)

    post_fail_lockout = db.get_login_lockout_info(username, client_ip)
    if post_fail_lockout.get('locked'):
        remaining_minutes = max(1, int((post_fail_lockout.get('remaining_seconds', 0) + 59) / 60))
        return RedirectResponse(url=f"/login?error=locked&minutes={remaining_minutes}", status_code=303)

    # Invalid credentials - redirect back to login with error
    return RedirectResponse(url="/login?error=invalid", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    session_id = request.cookies.get("session_id")
    if session_id:
        auth_manager.destroy_session(session_id)

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id")
    return response


@router.get("/login/totp")
async def login_totp_page(request: Request):
    """TOTP page — served by React SPA"""
    from fastapi.responses import FileResponse
    index = PROJECT_ROOT / "static" / "react" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/login")


@router.post("/login/totp")
@limiter.limit("10/minute")
async def login_totp_verify(
    request: Request,
    token: str = Form(...),
    code: str = Form(...),
    csrf_token: str = Form(...),
):
    """Verify TOTP code or backup code and complete login."""
    from core.config import ENABLE_HTTPS
    from slowapi.util import get_remote_address
    csrf.verify_token(request, csrf_token)

    # Retrieve pending username from DB
    row = db.query_one("SELECT value FROM settings WHERE key = ?", (f'_pending_totp_{token}',))
    if not row:
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    import json as _json
    try:
        username = _json.loads(row['value'])
    except Exception:
        username = row['value']

    # Get user TOTP secret
    user_row = db.query_one("SELECT totp_secret FROM users WHERE username = ?", (username,))
    if not user_row or not user_row.get('totp_secret'):
        # TOTP setup broken — let them through
        pass
    else:
        code = code.strip()
        # Try TOTP first, then backup codes
        if not auth_manager.verify_totp(user_row['totp_secret'], code):
            if not auth_manager.use_backup_code(username, code):
                audit_log.log("login_totp_failed", username=username,
                              ip=get_remote_address(request))
                return RedirectResponse(url=f"/login/totp?token={token}&error=invalid_code", status_code=303)

    # Clean up pending token
    db.execute("DELETE FROM settings WHERE key = ?", (f'_pending_totp_{token}',))

    # Create full session
    client_ip = get_remote_address(request)
    session_id = auth_manager.create_session(
        username,
        ip_address=client_ip,
        user_agent=request.headers.get('user-agent', '')
    )
    db.update_last_login(username)
    audit_log.log("login_success_2fa", username=username, ip=client_ip)

    force_pw = db.user_must_change_password(username)
    response = RedirectResponse(url="/change-password" if force_pw else "/", status_code=303)
    response.set_cookie(
        key="session_id", value=session_id,
        httponly=True, secure=ENABLE_HTTPS, samesite="lax", max_age=86400
    )
    return response


@router.get("/settings/2fa/setup")
async def settings_2fa_setup(request: Request, username: str = Depends(require_auth)):
    """2FA setup page — Jinja2 template with QR code."""
    import json as _json
    totp_info = auth_manager.get_user_totp_info(username)

    qr_b64 = None
    backup_codes = None
    secret_pending = None

    error = request.query_params.get("error")
    success = request.query_params.get("success")

    if not totp_info.get("enabled"):
        # Generate a fresh secret + QR for the setup form (store in settings as temp)
        existing_pending = db.query_one(
            "SELECT value FROM settings WHERE key = ?",
            (f"_pending_totp_setup_{username}",)
        )
        if existing_pending:
            data = _json.loads(existing_pending["value"])
            secret_pending = data["secret"]
            backup_codes = data["backup_codes"]
        else:
            secret_pending = auth_manager.generate_totp_secret()
            backup_codes = auth_manager.generate_backup_codes()
            db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (f"_pending_totp_setup_{username}",
                 _json.dumps({"secret": secret_pending, "backup_codes": backup_codes}))
            )
        try:
            uri = auth_manager.get_totp_uri(username, secret_pending)
            qr_b64 = auth_manager.generate_qr_code_base64(uri)
        except RuntimeError:
            qr_b64 = None  # pyotp/qrcode not installed

    return templates.TemplateResponse("2fa_setup.html", {
        "request": request,
        "username": username,
        "totp_info": totp_info,
        "qr_b64": qr_b64,
        "secret_pending": secret_pending,
        "backup_codes": backup_codes,
        "error": error,
        "success": success,
        "csrf_token": request.state.csrf_token,
    })


@router.post("/settings/2fa/enable")
async def settings_2fa_enable(
    request: Request,
    code: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Confirm TOTP code and enable 2FA."""
    csrf.verify_token(request, csrf_token)
    import json as _json
    row = db.query_one("SELECT value FROM settings WHERE key = ?", (f'_pending_totp_setup_{username}',))
    if not row:
        return RedirectResponse(url="/settings/2fa/setup?error=expired", status_code=303)
    data = _json.loads(row['value'])
    secret = data['secret']
    backup_codes = data['backup_codes']
    if not auth_manager.verify_totp(secret, code.strip()):
        return RedirectResponse(url="/settings/2fa/setup?error=invalid_code", status_code=303)
    auth_manager.save_totp_for_user(username, secret, backup_codes)
    db.execute("DELETE FROM settings WHERE key = ?", (f'_pending_totp_setup_{username}',))
    audit_log.log("2fa_enabled", username=username, ip=request.client.host)
    return RedirectResponse(url="/settings/2fa/setup?success=1", status_code=303)


@router.post("/settings/2fa/disable")
async def settings_2fa_disable(
    request: Request,
    password: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Disable 2FA after password confirmation."""
    csrf.verify_token(request, csrf_token)
    if not db.verify_user(username, password):
        return RedirectResponse(url="/settings/2fa/setup?error=wrong_password", status_code=303)
    auth_manager.disable_totp_for_user(username)
    audit_log.log("2fa_disabled", username=username, ip=request.client.host)
    return RedirectResponse(url="/settings/2fa/setup?success=disabled", status_code=303)


@router.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, username: str = Depends(require_auth_basic)):
    """First-login (or manual) password change page."""
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "error": request.query_params.get("error"),
        "must_change_password": db.user_must_change_password(username),
    })


@router.post("/change-password")
async def change_password_submit(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth_basic)
):
    """Handle password change and clear first-login requirement."""
    csrf.verify_token(request, csrf_token)

    if not db.verify_user(username, current_password):
        return RedirectResponse(url="/change-password?error=current", status_code=303)

    if new_password != confirm_password:
        return RedirectResponse(url="/change-password?error=match", status_code=303)

    if len(new_password) < 10:
        return RedirectResponse(url="/change-password?error=length", status_code=303)

    if new_password == current_password:
        return RedirectResponse(url="/change-password?error=reuse", status_code=303)

    db.update_password(username, new_password)
    audit_log.log("password_change", username=username, ip=request.client.host)
    return RedirectResponse(url="/?password_changed=1", status_code=303)
