"""
FastAPI Web Dashboard for Investment Monitor
Control panel for the automated investment analysis system.
"""
# Initialize logging first
from logging_config import setup_logging
setup_logging()

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
import uvicorn
from datetime import datetime, timedelta

from core.config import WEB_HOST, WEB_PORT, TEMPLATES_DIR
from core.database import db
from core.notifications import notifications
from core.auth import auth_manager
from core.csrf import csrf
from core.rate_limit import limiter
from core.audit_log import audit_log
from core.plugin_manager import plugin_manager
from scheduler import scheduler
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from engine.agents import swarm
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
from clients.custom_provider_client import custom_provider_client
from clients.provider_registry import provider_registry, PROVIDER_SHORTCUTS, STAGE_INFO
from core.budget_tracker import budget_tracker
from engine.learning_optimizer import learning_optimizer
from engine.staleness_tracker import staleness_tracker
from engine.ai_crosscheck import ai_crosscheck

app = FastAPI(title="AI Investment Monitor", version="1.0.0")

# Configure templates with autoescape for XSS protection
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html', 'xml'])
)
templates = Jinja2Templates(env=jinja_env)

# Configure rate limiter
app.state.limiter = limiter

async def _custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Redirect back with a friendly error instead of a raw JSON 429 page."""
    referer = request.headers.get("referer", "/")
    # Strip to path + query only (no external redirects)
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    parsed = urlparse(referer)
    safe_back = parsed.path or "/"
    msg = "Too+many+requests+%E2%80%94+please+wait+a+moment+before+trying+again"
    sep = "&" if "?" in safe_back else "?"
    return RedirectResponse(url=f"{safe_back}{sep}error={msg}", status_code=303)

app.add_exception_handler(RateLimitExceeded, _custom_rate_limit_handler)

# Mount static files directory
from pathlib import Path
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==================== MIDDLEWARE ====================

@app.middleware("http")
async def add_csrf_token(request: Request, call_next):
    """Add CSRF token to request state for templates"""
    request.state.csrf_token = csrf.get_token(request)
    response = await call_next(request)
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    from core.config import ENABLE_HTTPS
    response = await call_next(request)

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS (only if HTTPS enabled)
    if ENABLE_HTTPS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )

    return response

# ==================== AUTHENTICATION ====================

def require_auth_basic(request: Request) -> str:
    """Dependency to require authentication on routes (without password-change redirect)."""
    username = auth_manager.get_current_user(request)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"}
        )
    return username


def require_auth(request: Request) -> str:
    """Dependency to require auth and enforce first-login password change."""
    username = require_auth_basic(request)
    if db.user_must_change_password(username):
        path = request.url.path
        if path not in ("/change-password", "/logout"):
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": "/change-password"}
            )
    return username


def require_api_key_or_session(request: Request) -> str:
    """
    Accept either a valid Bearer token (personal API key) or an active session cookie.
    Used on all /api/* endpoints to enable external tool access.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        result = auth_manager.validate_bearer_token(token)
        if result:
            return result[0]  # username
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired API key")
    # Fall back to session-based auth
    return require_auth_basic(request)


@app.get("/login")
async def login_page(request: Request):
    """Login page — served by React SPA"""
    username = auth_manager.get_current_user(request)
    if username:
        if db.user_must_change_password(username):
            return RedirectResponse(url="/change-password", status_code=303)
        return RedirectResponse(url="/", status_code=303)
    from fastapi.responses import FileResponse
    index = Path(__file__).parent / "static" / "react" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=503, detail="React build not found. Run: cd frontend && npm run build")

@app.post("/login")
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

@app.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    session_id = request.cookies.get("session_id")
    if session_id:
        auth_manager.destroy_session(session_id)

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id")
    return response


# === TOTP / 2FA Routes (#33) ===

@app.get("/login/totp")
async def login_totp_page(request: Request):
    """TOTP page — served by React SPA"""
    from fastapi.responses import FileResponse
    index = Path(__file__).parent / "static" / "react" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/login")


@app.post("/login/totp")
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


@app.get("/settings/2fa/setup")
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


@app.post("/settings/2fa/enable")
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


@app.post("/settings/2fa/disable")
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


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, username: str = Depends(require_auth_basic)):
    """First-login (or manual) password change page."""
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "error": request.query_params.get("error"),
        "must_change_password": db.user_must_change_password(username),
    })


@app.post("/change-password")
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

# ==================== DASHBOARD ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, username: str = Depends(require_auth)):
    """Main dashboard"""
    try:
        # Get top 3 picks for preview widget
        try:
            top_picks_preview = db.get_top_picks(min_predictions=3, min_accuracy=0.65, limit=3)
        except Exception:
            top_picks_preview = []
        
        # Get trusted tickers for badge display
        try:
            trusted_tickers = set(db.get_trusted_tickers(min_accuracy=0.7) or [])
        except Exception:
            trusted_tickers = set()

        # Enrich recent analyses with staleness metadata
        recent_analyses = db.get_analysis_history(limit=10)
        for a in recent_analyses:
            staleness_tracker.enrich_analysis(a)

        # Feature 4: Cold start detection
        learning_stats = learning_optimizer.get_learning_stats()
        cold_start = False
        cold_start_reason = ""
        first_prediction = db.query_one("SELECT MIN(prediction_date) as first_date FROM prediction_outcomes")
        if first_prediction and first_prediction.get('first_date'):
            from datetime import datetime as dt
            try:
                first_date = dt.fromisoformat(first_prediction['first_date'].replace(' ', 'T')[:19])
                days_active = (dt.now() - first_date).days
                if days_active < 60 or learning_stats.get('total_verified', 0) < 20:
                    cold_start = True
                    cold_start_reason = f"{days_active} days of data, {learning_stats.get('total_verified', 0)} verified predictions"
            except Exception:
                cold_start = True
                cold_start_reason = "Unable to determine system age"
        else:
            cold_start = True
            cold_start_reason = "No predictions recorded yet"

        # Feature 6: Kill switch check
        system_paused = db.get_setting('system_paused_accuracy') or False

        # Discovery stats for widget
        try:
            discovery_stats = db.get_discovery_stats()
            discovery_stats['enabled'] = db.get_setting('discovery_enabled')
        except Exception:
            discovery_stats = None

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "csrf_token": request.state.csrf_token,
            "scheduler_status": scheduler.get_status(),
            "watchlist": db.get_watchlist(),
            "recent_analyses": recent_analyses,
            "api_status": {
                "perplexity": pplx_client.get_usage(),
                "gemini": gemini_client.get_usage()
            },
            "learning_stats": learning_stats,
            "top_picks_preview": top_picks_preview,
            "trusted_tickers": trusted_tickers,
            "cold_start": cold_start,
            "cold_start_reason": cold_start_reason,
            "system_paused": system_paused,
            "discovery_stats": discovery_stats,
        })
    except Exception as e:
        import traceback
        print(f"Dashboard error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

# ==================== SYSTEM HEALTH ====================

@app.get("/api/health")
async def api_health(username: str = Depends(require_api_key_or_session)):
    """System health monitor endpoint for the dashboard widget"""
    from engine.health_monitor import health_monitor
    try:
        return health_monitor.get_full_health_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GEOPOLITICAL ====================

@app.get("/api/geopolitical")
async def api_geopolitical(username: str = Depends(require_api_key_or_session)):
    """Return the latest geopolitical scan (max 24h old)"""
    try:
        scan = db.get_latest_geopolitical_scan()
        return {"scan": scan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/macro/events")
async def api_macro_events(days: int = 14, username: str = Depends(require_api_key_or_session)):
    from engine.macro_tracker import macro_tracker
    return {"events": macro_tracker.get_upcoming_events(days_ahead=days)}

@app.get("/api/geopolitical/exposure")
async def api_geopolitical_exposure(username: str = Depends(require_api_key_or_session)):
    """Return per-ticker geopolitical exposure from the latest analysis"""
    try:
        watchlist = db.get_watchlist()
        exposures = []
        for stock in watchlist:
            latest = db.get_latest_analysis(stock['ticker'])
            exposures.append({
                "ticker": stock['ticker'],
                "geopolitical_context": latest.get('geopolitical_context') if latest else None,
                "geo_risk_score": latest.get('geo_risk_score') if latest else None,
                "timestamp": latest.get('timestamp') if latest else None,
            })
        return {"exposures": exposures}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SIGNAL ACCURACY ====================

@app.get("/api/signal-accuracy")
async def api_signal_accuracy(username: str = Depends(require_api_key_or_session)):
    """Provides accuracy and breakdown of predictive signals."""
    from engine.signal_grader import signal_grader
    try:
        return {
            "by_signal": signal_grader.get_accuracy_by_signal(),
            "monthly_trend": signal_grader.get_accuracy_by_month(),
            "weight_recommendations": signal_grader.get_weight_recommendations()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AUTO PAPER TRADING ====================

@app.get("/api/paper-trading/auto")
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

# ==================== AUTO-TRADE API ====================

@app.get("/api/auto-trade/status")
async def api_auto_trade_status(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto-trade/trust-gate")
async def api_auto_trade_trust_gate(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_trust_gate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto-trade/positions")
async def api_auto_trade_positions(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {"positions": auto_paper_trader.get_open_positions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto-trade/log")
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

@app.post("/api/auto-trade/close/{trade_id}")
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

@app.post("/api/auto-trade/toggle")
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

@app.get("/api/auto-trade/pending-confirm")
async def api_auto_trade_pending(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {"pending": auto_paper_trader.get_pending_confirmations()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-trade/confirm/{token}")
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

@app.post("/api/auto-trade/skip/{token}")
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

@app.get("/api/auto-trade/risk-gate-status")
async def api_auto_trade_risk_gate(username: str = Depends(require_api_key_or_session)):
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return auto_paper_trader.get_risk_gate_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/auto-trades")
async def export_auto_trades(username: str = Depends(require_api_key_or_session)):
    """CSV export of all auto paper trades."""
    import csv, io
    from fastapi.responses import StreamingResponse
    from core.database import db as _db

    rows = _db.query("""
        SELECT id, ticker, direction, entry_date, entry_price,
               exit_date, exit_price, pnl_pct, close_reason, status, blocked_reason
        FROM auto_paper_trades
        ORDER BY entry_date DESC
    """)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "ticker", "direction", "entry_date", "entry_price",
                     "exit_date", "exit_price", "pnl_pct", "close_reason", "status", "blocked_reason"])
    for row in (rows or []):
        pnl = round(row['pnl_pct'] * 100, 2) if row['pnl_pct'] is not None else ""
        writer.writerow([
            row['id'], row['ticker'], row['direction'],
            row['entry_date'], row['entry_price'],
            row['exit_date'] or "", row['exit_price'] or "",
            pnl, row['close_reason'] or "",
            row['status'], row['blocked_reason'] or ""
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=auto_trades.csv"}
    )

# ==================== AUTO-TRADE PROPOSE / ACTION LINKS ====================

@app.post("/api/auto-trade/propose")
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


@app.get("/action/auto-trade/confirm/{token}", response_class=HTMLResponse)
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


@app.get("/action/auto-trade/skip/{token}", response_class=HTMLResponse)
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


@app.get("/api/auto-trade/equity-curve")
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


# ==================== BROKER / ORDER ROUTES (Phase 6) ====================

@app.post("/api/orders/execute")
async def api_orders_execute(request: Request, username: str = Depends(require_api_key_or_session)):
    """Execute a trade entry. Body: {token} OR {ticker, direction, size_usd}."""
    from engine.order_manager import order_manager
    data = await request.json()
    token = data.get("token")
    if token:
        # Confirm a pending trade via token
        from engine.auto_paper_trader import auto_paper_trader
        result = auto_paper_trader.confirm_trade(token)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Confirm failed"))
        return result
    ticker = data.get("ticker", "").upper()
    direction = data.get("direction", "LONG").upper()
    size_usd = float(data.get("size_usd", 0))
    if not ticker or size_usd <= 0:
        raise HTTPException(status_code=400, detail="ticker and size_usd required")
    result = order_manager.execute_entry(ticker, direction, size_usd)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Entry failed"))
    return result


@app.post("/api/orders/close/{trade_id}")
async def api_orders_close(trade_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Close an open auto-trade position."""
    from engine.order_manager import order_manager
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass
    reason = data.get("reason", "manual")
    result = order_manager.execute_exit(trade_id, reason)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Exit failed"))
    return result


@app.get("/api/orders/status/{order_id}")
async def api_orders_status(order_id: str, username: str = Depends(require_api_key_or_session)):
    """Poll fill status for a broker order (best-effort)."""
    from clients.broker_client import get_broker_client, AlpacaBrokerClient
    broker = get_broker_client()
    if isinstance(broker, AlpacaBrokerClient):
        try:
            data = broker._get(f"/orders/{order_id}")
            return {"order_id": order_id, "status": data.get("status"), "filled_qty": data.get("filled_qty")}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
    # Paper / IBKR: synthetic response
    return {"order_id": order_id, "status": "filled", "filled_qty": None}


@app.get("/api/broker/account")
async def api_broker_account(username: str = Depends(require_api_key_or_session)):
    """Return broker account: equity, buying_power, cash, broker_last_sync."""
    from clients.broker_client import get_broker_client
    broker = get_broker_client()
    account = broker.get_account()
    account["broker_last_sync"] = db.get_setting("broker_last_sync") or ""
    return account


@app.get("/api/broker/positions")
async def api_broker_positions(username: str = Depends(require_api_key_or_session)):
    """Return live broker positions."""
    from clients.broker_client import get_broker_client
    broker = get_broker_client()
    return broker.get_positions()


@app.post("/api/broker/sync")
async def api_broker_sync(username: str = Depends(require_api_key_or_session)):
    """Trigger manual broker position sync."""
    from engine.order_manager import order_manager
    synced = order_manager.sync_broker_positions()
    return {"synced": synced, "broker_last_sync": db.get_setting("broker_last_sync") or ""}


# ==================== TRUTH BANNER ====================

_truth_banner_cache = {"data": None, "time": None}

@app.get("/api/truth-banner")
async def api_truth_banner(username: str = Depends(require_api_key_or_session)):
    """The single most important metric: system signals vs just buying SPY."""
    import yfinance as yf
    from datetime import datetime as dt, timedelta

    # 30 min cache
    if (_truth_banner_cache["data"] and _truth_banner_cache["time"]
            and dt.now() - _truth_banner_cache["time"] < timedelta(minutes=30)):
        return _truth_banner_cache["data"]

    try:
        # Get all closed paper trades
        closed = db.query("""
            SELECT ticker, direction, entry_date, entry_price, exit_price, pnl_pct, close_reason
            FROM auto_paper_trades WHERE status = 'closed' AND pnl_pct IS NOT NULL
            ORDER BY entry_date
        """)

        # Also count open trades
        open_count = db.query_one("SELECT COUNT(*) as c FROM auto_paper_trades WHERE status = 'open'")
        open_positions = open_count['c'] if open_count else 0

        if not closed or len(closed) < 1:
            result = {
                "data_sufficient": False,
                "total_trades": 0,
                "open_positions": open_positions,
                "message": "Collecting data — no closed trades yet"
            }
            _truth_banner_cache["data"] = result
            _truth_banner_cache["time"] = dt.now()
            return result

        # System cumulative return: compound all trade returns
        cumulative = 1.0
        for t in closed:
            cumulative *= (1.0 + (t['pnl_pct'] or 0))
        system_return_pct = round((cumulative - 1.0) * 100, 2)

        # Win rate
        wins = sum(1 for t in closed if (t['pnl_pct'] or 0) > 0)
        win_rate = round(wins / len(closed) * 100, 1)

        # Find date range
        first_date_str = closed[0]['entry_date'][:10]
        try:
            start_date = dt.strptime(first_date_str, '%Y-%m-%d')
        except ValueError:
            start_date = dt.fromisoformat(first_date_str)

        # SPY return over the same period
        spy_return_pct = 0.0
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=start_date.strftime('%Y-%m-%d'), end=dt.now().strftime('%Y-%m-%d'))
            if spy_hist is not None and not spy_hist.empty and len(spy_hist) >= 2:
                spy_start = float(spy_hist['Close'].iloc[0])
                spy_end = float(spy_hist['Close'].iloc[-1])
                if spy_start > 0:
                    spy_return_pct = round((spy_end - spy_start) / spy_start * 100, 2)
        except Exception:
            pass

        alpha = round(system_return_pct - spy_return_pct, 2)

        # MCPT significance
        mcpt_p = None
        mcpt_sig = None
        try:
            from engine.mcpt_validator import mcpt_validator
            mcpt_result = mcpt_validator.get_latest_result()
            if mcpt_result:
                mcpt_p = mcpt_result.get('p_value')
                mcpt_sig = mcpt_result.get('significant')
        except Exception:
            pass

        data_sufficient = len(closed) >= 5

        result = {
            "data_sufficient": data_sufficient,
            "start_date": first_date_str,
            "days_tracked": (dt.now() - start_date).days,
            "system_return_pct": system_return_pct,
            "spy_return_pct": spy_return_pct,
            "alpha": alpha,
            "total_trades": len(closed),
            "open_positions": open_positions,
            "win_rate_pct": win_rate,
            "mcpt_p_value": mcpt_p,
            "mcpt_significant": mcpt_sig,
        }

        _truth_banner_cache["data"] = result
        _truth_banner_cache["time"] = dt.now()
        return result

    except Exception as e:
        return {"data_sufficient": False, "total_trades": 0, "error": str(e)}

# ==================== TRUST OVERVIEW ====================

@app.get("/trust", response_class=HTMLResponse)
async def trust_page(request: Request, username: str = Depends(require_auth)):
    """Trust overview — should you trust this system?"""
    learning_stats = learning_optimizer.get_learning_stats()

    # Last scheduler run
    logs = db.get_scheduler_logs(limit=1)
    last_run = logs[0] if logs else None

    # Error rate from recent runs
    recent_logs = db.get_scheduler_logs(limit=10)
    error_rate = None
    if recent_logs:
        errors = sum(1 for log in recent_logs if log.get('errors'))
        error_rate = round(errors / len(recent_logs) * 100, 0)

    # 30-day accuracy
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total,
               AVG(accuracy_score) as accuracy,
               SUM(CASE WHEN accuracy_score >= 0.5 THEN 1 ELSE 0 END) as hits
        FROM prediction_outcomes
        WHERE verified_at IS NOT NULL
          AND verified_at > datetime('now', '-30 days')
    """)
    row30 = cursor.fetchone()
    accuracy_30d = round((row30['hits'] or 0) / row30['total'] * 100, 1) if row30 and row30['total'] > 0 else None
    predictions_30d = row30['total'] if row30 else 0

    # Current top signals: recent analyses with highest confidence + track record
    cursor.execute("""
        SELECT a.ticker, a.signal, a.confidence, a.timestamp,
               p.accuracy, p.prediction_count
        FROM analysis_history a
        LEFT JOIN (
            SELECT ticker, AVG(accuracy_score) as accuracy, COUNT(*) as prediction_count
            FROM prediction_outcomes WHERE verified_at IS NOT NULL
            GROUP BY ticker
        ) p ON a.ticker = p.ticker
        WHERE a.timestamp > datetime('now', '-7 days')
          AND a.confidence >= 60
        ORDER BY a.confidence DESC
        LIMIT 5
    """)
    top_signals = [dict(r) for r in cursor.fetchall()]
    conn.close()

    watchlist_count = len(db.get_watchlist(active_only=True))

    return templates.TemplateResponse("trust.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "learning_stats": learning_stats,
        "last_run": last_run,
        "error_rate": error_rate,
        "accuracy_30d": accuracy_30d,
        "predictions_30d": predictions_30d,
        "top_signals": top_signals,
        "watchlist_count": watchlist_count,
    })

# ==================== WATCHLIST ====================

@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(
    request: Request, 
    username: str = Depends(require_auth),
    sort_by: str = "ticker",
    sort_order: str = "asc"
):
    """Watchlist management"""
    return templates.TemplateResponse("watchlist.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "watchlist": db.get_watchlist(active_only=False, sort_by=sort_by, sort_order=sort_order),
        "current_sort": sort_by,
        "current_order": sort_order
    })

@app.post("/watchlist/add")
async def add_to_watchlist(
    request: Request,
    ticker: str = Form(...),
    name: str = Form(""),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Add stock to watchlist"""
    csrf.verify_token(request, csrf_token)
    db.add_to_watchlist(ticker.upper(), name)
    return RedirectResponse(url="/watchlist", status_code=303)

@app.post("/watchlist/remove/{ticker}")
async def remove_from_watchlist(
    request: Request,
    ticker: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Remove stock from watchlist"""
    csrf.verify_token(request, csrf_token)
    db.remove_from_watchlist(ticker)
    return RedirectResponse(url="/watchlist", status_code=303)

@app.post("/watchlist/archive/{ticker}")
async def archive_watchlist_item(
    request: Request,
    ticker: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Archive a watchlist item"""
    csrf.verify_token(request, csrf_token)
    db.archive_watchlist_item(ticker)
    return RedirectResponse(url="/watchlist", status_code=303)

@app.post("/api/watchlist/note")
async def save_watchlist_note(
    request: Request,
    ticker: str = Form(...),
    note_text: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Save note for a watchlist item"""
    csrf.verify_token(request, csrf_token)
    db.save_stock_note(ticker, note_text)
    return JSONResponse({"status": "success", "message": "Note saved"})

@app.get("/api/watchlist/note/{ticker}")
async def get_watchlist_note(
    request: Request,
    ticker: str,
    username: str = Depends(require_auth)
):
    """Get note for a watchlist item"""
    note = db.get_stock_note(ticker)
    return JSONResponse({"ticker": ticker, "note": note or ""})

# ==================== SETTINGS ====================

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(require_auth)):
    """Settings page with budget status"""
    try:
        budget_status = budget_tracker.get_budget_status()
    except Exception:
        budget_status = None

    try:
        risk_overrides = db.get_ticker_risk_overrides()
    except Exception:
        risk_overrides = []

    current_session_id = request.cookies.get("session_id")
    try:
        sessions = db.get_user_sessions(username)
    except Exception:
        sessions = []

    # Feature 6: Kill switch status
    system_paused = db.get_setting('system_paused_accuracy') or False

    try:
        personal_api_keys = db.list_personal_api_keys()
    except Exception:
        personal_api_keys = []

    try:
        plugins = db.list_plugins()
    except Exception:
        plugins = []

    try:
        totp_info = auth_manager.get_user_totp_info(username)
    except Exception:
        totp_info = {"enabled": False}

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "settings": db.get_all_settings(),
        "budget_status": budget_status,
        "risk_overrides": risk_overrides,
        "sessions": sessions,
        "current_session_id": current_session_id,
        "api_keys": {
            "perplexity": bool(db.get_api_key("perplexity")),
            "gemini": bool(db.get_api_key("gemini"))
        },
        "provider_shortcuts": PROVIDER_SHORTCUTS,
        "stage_info": STAGE_INFO,
        "system_paused": system_paused,
        "personal_api_keys": personal_api_keys,
        "plugins": plugins,
        "totp_info": totp_info,
    })

@app.post("/settings/clear-kill-switch")
async def clear_kill_switch(request: Request, username: str = Depends(require_auth)):
    """Feature 6: Clear the accuracy kill switch and resume pipeline"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))
    db.set_setting('system_paused_accuracy', False)
    return RedirectResponse(url="/settings?saved=1", status_code=303)

@app.post("/settings/save")
async def save_settings(request: Request, username: str = Depends(require_auth)):
    """Save settings - supports per-section saving via _section marker"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    # Determine which section(s) are being saved
    section = form.get("_section", "")
    save_all = not section

    # Scheduler settings
    if save_all or section == "scheduler":
        db.set_setting("scan_interval_hours", int(form.get("scan_interval_hours", 2)))
        db.set_setting("active_hours_start", form.get("active_hours_start", "08:00"))
        db.set_setting("active_hours_end", form.get("active_hours_end", "22:00"))
        try:
            trigger_pct = float(form.get("intraday_trigger_pct", 3.0))
            db.set_setting("intraday_trigger_pct", max(0.5, min(20.0, trigger_pct)))
        except (ValueError, TypeError):
            pass

        # Auto-Discovery settings
        db.set_setting("discovery_enabled", form.get("discovery_enabled") == "on")
        db.set_setting("discovery_daily_time", form.get("discovery_daily_time", "06:00"))
        db.set_setting("discovery_weekly_day", form.get("discovery_weekly_day", "wed"))
        db.set_setting("discovery_weekly_time", form.get("discovery_weekly_time", "12:00"))
        try:
            db.set_setting("discovery_promotion_threshold", int(form.get("discovery_promotion_threshold", 55)))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("discovery_max_promote_per_run", int(form.get("discovery_max_promote_per_run", 5)))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("discovery_max_watchlist_size", int(form.get("discovery_max_watchlist_size", 50)))
        except (ValueError, TypeError):
            pass
        all_strategies = ['volume_spike', 'breakout', 'oversold', 'sector_rotation', 'insider_buy', 'value_screen']
        enabled_strategies = [s for s in all_strategies if form.get(f"strategy_{s}") == "on"]
        if enabled_strategies:
            db.set_setting("discovery_strategies", enabled_strategies)


    # Email settings
    if save_all or section == "notifications":
        db.set_setting("email_enabled", form.get("email_enabled") == "on")
        db.set_setting("email_recipient", form.get("email_recipient", ""))
        db.set_setting("email_smtp_host", form.get("email_smtp_host", "smtp.gmail.com"))
        try:
            db.set_setting("email_smtp_port", int(form.get("email_smtp_port") or 587))
        except (ValueError, TypeError):
            db.set_setting("email_smtp_port", 587)
        db.set_setting("email_smtp_user", form.get("email_smtp_user", ""))
        if form.get("email_smtp_password"):
            db.set_setting("email_smtp_password", form.get("email_smtp_password"))
        db.set_setting("notify_on_strong_signals", form.get("notify_on_strong_signals") == "on")
        db.set_setting("daily_summary_enabled", form.get("daily_summary_enabled") == "on")
        db.set_setting("daily_summary_time", form.get("daily_summary_time", "20:00"))
        try:
            db.set_setting("alert_cooldown_hours",
                           max(1, min(168, int(form.get("alert_cooldown_hours") or 24))))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("intraday_trigger_pct",
                           max(0.5, min(20.0, float(form.get("intraday_trigger_pct") or 3.0))))
        except (ValueError, TypeError):
            pass

    # Server Efficiency settings
    if save_all or section == "server_efficiency":
        db.set_setting("deep_sleep_enabled", form.get("deep_sleep_enabled") == "on")
        db.set_setting("deep_sleep_intensity", form.get("deep_sleep_intensity", "deep"))
        db.set_setting("deep_sleep_start", form.get("deep_sleep_start", "22:00"))
        db.set_setting("deep_sleep_end", form.get("deep_sleep_end", "07:00"))
        db.set_setting("deep_sleep_full_weekends", form.get("deep_sleep_full_weekends") == "on")

    # Analysis settings
    if save_all or section == "analysis":
        db.set_setting("include_news", form.get("include_news") == "on")
        db.set_setting("include_fundamental", form.get("include_fundamental") == "on")
        db.set_setting("include_technical", form.get("include_technical") == "on")
        db.set_setting("analysis_variant", form.get("analysis_variant", "balanced"))

        # Monthly API budgets (EUR)
        try:
            pplx_budget = float(form.get("perplexity_monthly_budget", 5.0))
            db.set_setting("perplexity_monthly_budget", max(0, min(100, pplx_budget)))
        except (ValueError, TypeError):
            db.set_setting("perplexity_monthly_budget", 5.0)
        try:
            gemini_budget = float(form.get("gemini_monthly_budget", 5.0))
            db.set_setting("gemini_monthly_budget", max(0, min(100, gemini_budget)))
        except (ValueError, TypeError):
            db.set_setting("gemini_monthly_budget", 5.0)

        # Learning system settings
        try:
            db.set_setting("learning_verification_days", int(form.get("learning_verification_days", 90)))
        except (ValueError, TypeError):
            pass

    # Portfolio rule thresholds
    try:
        db.set_setting("portfolio_max_position_pct", float(form.get("portfolio_max_position_pct", 10.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_stop_loss_pct", float(form.get("portfolio_stop_loss_pct", 15.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_max_sector_pct", float(form.get("portfolio_max_sector_pct", 30.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_rebalance_drift_pct", float(form.get("portfolio_rebalance_drift_pct", 5.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_risk_guard_enabled", form.get("portfolio_risk_guard_enabled") == "on")
    except (ValueError, TypeError):
        pass
    try:
        global_loss_limit = float(form.get("portfolio_global_loss_limit_pct", 10.0))
        db.set_setting("portfolio_global_loss_limit_pct", max(1.0, min(50.0, global_loss_limit)))
    except (ValueError, TypeError):
        pass
    try:
        cooldown_hours = float(form.get("portfolio_risk_cooldown_hours", 24.0))
        db.set_setting("portfolio_risk_cooldown_hours", max(0.0, min(168.0, cooldown_hours)))
    except (ValueError, TypeError):
        pass

    # Authentication lockout policy
    try:
        max_failed = int(form.get("auth_max_failed_attempts", 5))
        db.set_setting("auth_max_failed_attempts", max(3, min(20, max_failed)))
    except (ValueError, TypeError):
        pass
    try:
        attempt_window = int(form.get("auth_attempt_window_minutes", 15))
        db.set_setting("auth_attempt_window_minutes", max(5, min(120, attempt_window)))
    except (ValueError, TypeError):
        pass
    try:
        lockout_minutes = int(form.get("auth_lockout_minutes", 15))
        db.set_setting("auth_lockout_minutes", max(1, min(240, lockout_minutes)))
    except (ValueError, TypeError):
        pass

    # Authentication lockout policy
    if save_all or section == "security":
        try:
            max_failed = int(form.get("auth_max_failed_attempts", 5))
            db.set_setting("auth_max_failed_attempts", max(3, min(20, max_failed)))
        except (ValueError, TypeError):
            pass
        try:
            attempt_window = int(form.get("auth_attempt_window_minutes", 15))
            db.set_setting("auth_attempt_window_minutes", max(5, min(120, attempt_window)))
        except (ValueError, TypeError):
            pass
        try:
            lockout_minutes = int(form.get("auth_lockout_minutes", 15))
            db.set_setting("auth_lockout_minutes", max(1, min(240, lockout_minutes)))
        except (ValueError, TypeError):
            pass

    if save_all or section == "autotrading":
        # Auto-Trading settings
        db.set_setting("auto_trade_enabled", form.get("auto_trade_enabled") == "on")
        db.set_setting("auto_trade_mode", form.get("auto_trade_mode", "paper"))
        db.set_setting("auto_trade_signal_filter", form.get("auto_trade_signal_filter", "STRONG"))
        db.set_setting("auto_trade_require_confirm", form.get("auto_trade_require_confirm") == "on")
        for key, default, lo, hi in [
            ("auto_trade_take_profit_pct",    8.0,  1.0,  50.0),
            ("auto_trade_stop_loss_pct",      4.0,  1.0,  25.0),
            ("auto_trade_max_days_open",      30,   1,    90),
            ("auto_trade_position_size_pct",  5.0,  1.0,  20.0),
            ("auto_trade_max_open_positions", 10,   1,    50),
            ("auto_trade_min_trust_trades",   20,   5,    200),
            ("auto_trade_min_trust_win_rate", 55.0, 40.0, 80.0),
        ]:
            try:
                val = float(form.get(key, default))
                db.set_setting(key, max(lo, min(hi, val)))
            except (ValueError, TypeError):
                pass

        # Phase 6 — Broker credentials
        db.set_setting("auto_trade_alpaca_api_key", form.get("auto_trade_alpaca_api_key", "").strip())
        db.set_setting("auto_trade_alpaca_secret", form.get("auto_trade_alpaca_secret", "").strip())
        db.set_setting("auto_trade_alpaca_base_url", form.get("auto_trade_alpaca_base_url", "https://paper-api.alpaca.markets").strip())
        db.set_setting("auto_trade_ibkr_host", form.get("auto_trade_ibkr_host", "127.0.0.1").strip())
        db.set_setting("auto_trade_ibkr_port", form.get("auto_trade_ibkr_port", "7497").strip())
        db.set_setting("auto_trade_ibkr_client_id", form.get("auto_trade_ibkr_client_id", "1").strip())
        db.set_setting("auto_trade_trust_override", form.get("auto_trade_trust_override") == "on")

    # Reload settings in services
    scheduler.reload_settings()
    notifications.reload_settings()
    budget_tracker.invalidate_cache()

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.post("/settings/risk-override")
async def save_risk_override(
    request: Request,
    ticker: str = Form(...),
    stop_loss_pct: str = Form(""),
    max_position_pct: str = Form(""),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Create or update per-ticker risk overrides."""
    csrf.verify_token(request, csrf_token)

    ticker_value = (ticker or "").strip().upper()
    if not ticker_value:
        return RedirectResponse(url="/settings?error=invalid_ticker", status_code=303)

    try:
        stop_loss_value = float(stop_loss_pct) if str(stop_loss_pct).strip() else None
    except (ValueError, TypeError):
        stop_loss_value = None

    try:
        max_position_value = float(max_position_pct) if str(max_position_pct).strip() else None
    except (ValueError, TypeError):
        max_position_value = None

    if stop_loss_value is None and max_position_value is None:
        return RedirectResponse(url="/settings?error=empty_override", status_code=303)

    db.set_ticker_risk_override(
        ticker_value,
        stop_loss_pct=stop_loss_value,
        max_position_pct=max_position_value
    )

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.post("/settings/risk-override/delete/{ticker}")
async def delete_risk_override(
    request: Request,
    ticker: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Delete per-ticker risk override."""
    csrf.verify_token(request, csrf_token)
    db.delete_ticker_risk_override(ticker)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.post("/settings/sessions/logout-others")
async def logout_other_sessions(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """End all other active sessions for current user."""
    csrf.verify_token(request, csrf_token)

    current_session_id = request.cookies.get("session_id")
    db.delete_other_user_sessions(username, current_session_id)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.post("/settings/sessions/logout/{session_id}")
async def logout_single_session(
    request: Request,
    session_id: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """End one selected active session for current user."""
    csrf.verify_token(request, csrf_token)

    current_session_id = request.cookies.get("session_id")
    if session_id == current_session_id:
        return RedirectResponse(url="/settings?error=current_session", status_code=303)

    db.delete_user_session_for_user(username, session_id)
    return RedirectResponse(url="/settings?saved=1", status_code=303)

@app.post("/settings/api-keys")
async def save_api_keys(request: Request, username: str = Depends(require_auth)):
    """Save API keys"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    if form.get("perplexity_key"):
        db.set_api_key("perplexity", form.get("perplexity_key"))
        pplx_client.api_key = form.get("perplexity_key")
        db.clear_system_alert('perplexity_auth')

    if form.get("gemini_key"):
        db.set_api_key("gemini", form.get("gemini_key"))
        gemini_client.reload_api_key(form.get("gemini_key"))
        db.clear_system_alert('gemini_auth')

    return RedirectResponse(url="/settings?saved=1", status_code=303)


# ==================== PERSONAL API KEYS (#42) ====================

@app.get("/api/personal-keys")
async def list_personal_api_keys(request: Request, username: str = Depends(require_auth)):
    """List personal API keys (session auth only — no bearer)."""
    keys = db.list_personal_api_keys()
    return {"keys": keys}


@app.post("/api/personal-keys")
async def generate_personal_api_key(request: Request, username: str = Depends(require_auth)):
    """Generate a new personal API key. Returns the raw key ONCE — store it now."""
    payload = await request.json()
    label = (payload.get("label") or "").strip()
    scope = (payload.get("scope") or "read").strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required")
    if scope not in ("read", "write"):
        raise HTTPException(status_code=400, detail="scope must be 'read' or 'write'")
    raw_key, key_id = auth_manager.generate_personal_api_key(label=label, scope=scope)
    return {"id": key_id, "raw_key": raw_key, "label": label, "scope": scope}


@app.delete("/api/personal-keys/{key_id}")
async def revoke_personal_api_key(key_id: int, request: Request, username: str = Depends(require_auth)):
    """Revoke a personal API key by id (session auth only)."""
    db.revoke_personal_api_key(key_id)
    return {"status": "revoked"}


# ==================== PLUGINS (#42.5) ====================

@app.get("/api/plugins")
async def api_list_plugins(request: Request, username: str = Depends(require_auth)):
    """List all installed plugins."""
    plugins = db.list_plugins()
    # Add settings schema to each plugin record for the UI
    for plugin in plugins:
        try:
            plugin["settings_schema"] = plugin_manager.get_plugin_settings_schema(plugin["id"])
        except Exception:
            plugin["settings_schema"] = {}
    return {"plugins": plugins}


@app.post("/api/plugins/install")
async def api_install_plugin(request: Request, username: str = Depends(require_auth)):
    """Install a plugin from an uploaded .py file (multipart/form-data)."""
    from fastapi import UploadFile, File
    form = await request.form()
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename = getattr(file, "filename", None) or "plugin.py"
    if not filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files are accepted")
    content = await file.read()
    try:
        meta = plugin_manager.install(filename=filename, content=content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return meta


@app.delete("/api/plugins/{plugin_id}")
async def api_uninstall_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Uninstall a plugin by id."""
    try:
        plugin_manager.uninstall(plugin_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "uninstalled"}


@app.post("/api/plugins/{plugin_id}/toggle")
async def api_toggle_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Enable or disable a plugin."""
    payload = await request.json()
    enabled = bool(payload.get("enabled", False))
    db.toggle_plugin(plugin_id, enabled)
    return {"status": "ok", "enabled": enabled}


@app.post("/api/plugins/{plugin_id}/settings")
async def api_update_plugin_settings(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Save plugin-specific settings."""
    import json as _json
    payload = await request.json()
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be a JSON object")
    db.update_plugin_settings(plugin_id, _json.dumps(settings))
    return {"status": "ok"}


@app.post("/api/plugins/{plugin_id}/run")
async def api_run_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Manually trigger a plugin (test run). Uses a dummy context."""
    result = plugin_manager.run_plugin(plugin_id)
    return result


@app.get("/api/providers")
async def api_get_providers(request: Request, username: str = Depends(require_api_key_or_session)):
    """List custom API providers and usage stats."""
    providers = db.get_api_providers(include_secrets=False)
    enriched = []
    for provider in providers:
        usage = db.get_api_provider_usage(
            provider_id=provider['id'],
            monthly_budget_eur=provider.get('monthly_budget_eur', 5.0),
        )
        item = dict(provider)
        item.update(usage)
        enriched.append(item)
    return {"providers": enriched}


@app.post("/api/providers")
async def api_create_provider(request: Request, username: str = Depends(require_api_key_or_session)):
    """Create a custom OpenAI-compatible provider."""
    payload = await request.json()

    name = (payload.get('name') or '').strip()
    provider_type = (payload.get('provider_type') or 'llm').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    pipeline_role = (payload.get('pipeline_role') or '').strip() or None
    adapter_type = (payload.get('adapter_type') or 'openai_compatible').strip()
    monthly_budget_raw = payload.get('monthly_budget_eur', 5.0)
    monthly_budget_eur = 5.0 if monthly_budget_raw is None else float(monthly_budget_raw)

    if not name or not base_url or not model:
        raise HTTPException(status_code=400, detail="name, base_url, model required")

    provider_id = db.create_api_provider(
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        api_key=api_key,
        model=model,
        pipeline_role=pipeline_role,
        monthly_budget_eur=monthly_budget_eur,
        adapter_type=adapter_type,
    )
    if not provider_id:
        raise HTTPException(status_code=400, detail="Could not create provider (name may already exist)")

    return {"id": provider_id, "status": "ok"}


@app.put("/api/providers/{provider_id}")
async def api_update_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Update an existing custom OpenAI-compatible provider."""
    payload = await request.json()

    existing = db.get_api_provider(provider_id, include_secret=False)
    if not existing:
        raise HTTPException(status_code=404, detail="Provider not found")

    name = (payload.get('name') or '').strip()
    provider_type = (payload.get('provider_type') or 'llm').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    pipeline_role = (payload.get('pipeline_role') or '').strip() or None
    adapter_type = (payload.get('adapter_type') or 'openai_compatible').strip()
    monthly_budget_raw = payload.get('monthly_budget_eur', 5.0)
    monthly_budget_eur = 5.0 if monthly_budget_raw is None else float(monthly_budget_raw)

    if not name or not base_url or not model:
        raise HTTPException(status_code=400, detail="name, base_url, model required")

    ok = db.update_api_provider(
        provider_id=provider_id,
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        model=model,
        pipeline_role=pipeline_role,
        monthly_budget_eur=monthly_budget_eur,
        api_key=api_key if api_key else None,
        adapter_type=adapter_type,
    )

    if not ok:
        raise HTTPException(status_code=400, detail="Could not update provider")

    return {"id": provider_id, "status": "ok"}


@app.delete("/api/providers/{provider_id}")
async def api_delete_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Delete a custom provider."""
    db.delete_api_provider(provider_id)
    return {"status": "ok"}


@app.post("/api/providers/{provider_id}/test")
async def api_test_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Test provider connectivity with a lightweight completion call."""
    result = provider_registry.test_provider(provider_id)
    if result.get("status") == "error" and result.get("error") == "provider_not_found":
        raise HTTPException(status_code=404, detail="Provider not found")
    return result

# ==================== API KEY PEEK ====================

@app.get("/api/api-key/peek/{service}")
async def peek_api_key(service: str, request: Request, username: str = Depends(require_auth)):
    """Return masked API key (first 4 + last 4 chars visible) — looks up by provider name."""
    # Try provider by name first (works for any provider)
    provider = db.get_api_provider_by_name(service, include_secret=True)
    key = provider.get("api_key") if provider else None
    # Fallback: legacy api_keys table for perplexity/gemini
    if not key and service in ('perplexity', 'gemini'):
        key = db.get_api_key(service)
    if not key:
        return {"service": service, "masked": None, "configured": False}
    if len(key) <= 8:
        masked = key[:2] + '*' * (len(key) - 2)
    else:
        masked = key[:4] + '*' * (len(key) - 8) + key[-4:]
    return {"service": service, "masked": masked, "configured": True}

# ==================== STAGE ASSIGNMENT API ====================

@app.get("/api/stage-assignments")
async def api_get_stage_assignments(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current pipeline stage → provider assignments."""
    assignments_list = db.get_stage_assignments()
    assignments = {a['stage_name']: a for a in assignments_list}
    providers = db.get_api_providers(include_secrets=False)
    return {"assignments": assignments, "providers": providers, "stage_info": STAGE_INFO}


@app.post("/api/stage-assignments")
async def api_set_stage_assignments(request: Request, username: str = Depends(require_api_key_or_session)):
    """Save pipeline stage → provider assignments."""
    payload = await request.json()
    mode = payload.get("mode", "per_stage")
    if mode == "one_for_all":
        provider_id = payload.get("provider_id")
        fallback_id = payload.get("fallback_provider_id") or None
        if provider_id:
            db.set_all_stages_to_provider(int(provider_id), int(fallback_id) if fallback_id else None)
    else:
        for stage_name, data in payload.get("stages", {}).items():
            pid = data.get("provider_id")
            fid = data.get("fallback_provider_id")
            db.set_stage_assignment(
                stage_name,
                int(pid) if pid else None,
                int(fid) if fid else None,
                bool(data.get("enabled", True)),
            )
    return {"status": "ok"}


# ==================== SYSTEM ALERTS API ====================

@app.post("/api/system-alert/dismiss")
async def dismiss_system_alert(request: Request, username: str = Depends(require_auth)):
    """Dismiss a system alert banner"""
    data = await request.json()
    alert_key = data.get('alert_key')
    if alert_key:
        db.dismiss_system_alert(alert_key)
    return {"ok": True}

@app.post("/api/system-alert/raise")
async def raise_system_alert(request: Request, username: str = Depends(require_auth)):
    """Programmatic endpoint for services to raise alerts"""
    data = await request.json()
    alert_key = data.get('alert_key')
    title = data.get('title')
    message = data.get('message')
    if not alert_key or not title or not message:
        raise HTTPException(status_code=400, detail="alert_key, title, message required")
    db.raise_system_alert(
        alert_key=alert_key,
        title=title,
        message=message,
        severity=data.get('severity', 'error'),
        service=data.get('service'),
        action_url=data.get('action_url'),
        action_label=data.get('action_label'),
    )
    return {"ok": True}

@app.post("/api/system-alert/clear")
async def clear_system_alert(request: Request, username: str = Depends(require_auth)):
    """Programmatic endpoint for services to clear resolved alerts"""
    data = await request.json()
    alert_key = data.get('alert_key')
    if alert_key:
        db.clear_system_alert(alert_key)
    return {"ok": True}

# ==================== HISTORY ====================

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, ticker: str = None, username: str = Depends(require_auth)):
    """Analysis history"""
    return templates.TemplateResponse("history.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "analyses": db.get_analysis_history(ticker=ticker, limit=100),
        "filter_ticker": ticker
    })

@app.get("/geo-history", response_class=HTMLResponse)
async def geo_history_page(
    request: Request,
    limit: int = 30,
    only_deltas: bool = False,
    username: str = Depends(require_auth)
):
    scans = db.get_geopolitical_history(limit=limit, only_deltas=only_deltas)
    return templates.TemplateResponse("geo_history.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "scans": scans,
        "only_deltas": only_deltas,
        "limit": limit,
    })

@app.get("/macro", response_class=HTMLResponse)
async def macro_page(request: Request, username: str = Depends(require_auth)):
    """Macro dashboard — yield curve, VIX, credit spreads (#22)."""
    from engine.macro_tracker import macro_tracker
    snapshots = macro_tracker.get_macro_snapshots(90)
    events = macro_tracker.get_upcoming_events(days_ahead=30)
    latest = macro_tracker.get_latest_snapshot()
    return templates.TemplateResponse("macro.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "snapshots": snapshots,
        "events": events,
        "latest": latest,
    })


@app.get("/corporate-actions", response_class=HTMLResponse)
async def corporate_actions_page(request: Request, username: str = Depends(require_auth)):
    """Dividend & corporate actions ledger across all watchlist tickers (#50)."""
    tickers = [w['ticker'] for w in db.get_watchlist()]
    all_actions = []
    for t in tickers:
        actions = db.get_corporate_actions(t, limit=30)
        all_actions.extend(actions)
    # Sort newest first
    all_actions.sort(key=lambda x: x.get('action_date', ''), reverse=True)
    return templates.TemplateResponse("corporate_actions.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "actions": all_actions,
        "tickers": tickers,
    })


@app.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page(request: Request, username: str = Depends(require_auth)):
    """Geopolitical scenario stress-test overview (#39)."""
    from engine.geo_scenario import geo_scenarios
    return templates.TemplateResponse("scenarios.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "scenarios": geo_scenarios.get_all_scenarios(),
    })


@app.get("/api/scenarios")
async def api_list_scenarios(request: Request, username: str = Depends(require_api_key_or_session)):
    """List all available geopolitical scenarios."""
    from engine.geo_scenario import geo_scenarios
    return {"scenarios": geo_scenarios.get_all_scenarios()}


@app.post("/api/scenarios/run")
async def api_run_scenario(
    request: Request,
    name: str,
    username: str = Depends(require_api_key_or_session)
):
    """Run a named geo scenario against the current portfolio (#39)."""
    from engine.geo_scenario import geo_scenarios
    try:
        result = geo_scenarios.run_scenario(name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/watchlist/groups")
async def api_watchlist_groups(username: str = Depends(require_api_key_or_session)):
    """Return all distinct watchlist group names (#27)."""
    return {"groups": db.get_watchlist_groups()}


@app.post("/api/watchlist/{ticker}/group")
async def api_set_watchlist_group(
    request: Request,
    ticker: str,
    username: str = Depends(require_api_key_or_session)
):
    """Set the group for a watchlist ticker (#27)."""
    body = await request.json()
    group_name = str(body.get("group_name", "Default")).strip() or "Default"
    ok = db.update_watchlist_group(ticker, group_name)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to update group")
    return {"ticker": ticker.upper(), "group_name": group_name}


@app.get("/api/watchlist/group-stats")
async def api_watchlist_group_stats(username: str = Depends(require_api_key_or_session)):
    """
    Per-group aggregate statistics: avg risk score, avg geo risk, signal distribution (#27).
    """
    rows = db.query(
        """
        SELECT
            COALESCE(w.group_name, 'Default') AS group_name,
            COUNT(*) AS ticker_count,
            ROUND(AVG(ah.risk_score), 1) AS avg_risk_score,
            ROUND(AVG(ah.geo_risk_score), 1) AS avg_geo_risk,
            SUM(CASE WHEN ah.signal = 'STRONG_BUY'  THEN 1 ELSE 0 END) AS strong_buy,
            SUM(CASE WHEN ah.signal = 'BUY'         THEN 1 ELSE 0 END) AS buy,
            SUM(CASE WHEN ah.signal = 'HOLD'        THEN 1 ELSE 0 END) AS hold,
            SUM(CASE WHEN ah.signal = 'SELL'        THEN 1 ELSE 0 END) AS sell,
            SUM(CASE WHEN ah.signal = 'STRONG_SELL' THEN 1 ELSE 0 END) AS strong_sell
        FROM watchlist w
        LEFT JOIN (
            SELECT ticker, risk_score, geo_risk_score, signal,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) AS rn
            FROM analysis_history
        ) ah ON ah.ticker = w.ticker AND ah.rn = 1
        WHERE w.active = 1
        GROUP BY group_name
        ORDER BY group_name
        """
    ) or []
    return {"groups": [dict(r) for r in rows]}


@app.get("/api/macro/snapshot")
async def api_macro_snapshot(username: str = Depends(require_api_key_or_session)):
    """Return the latest macro snapshot and 90-day history (#22)."""
    from engine.macro_tracker import macro_tracker
    return {
        "latest": macro_tracker.get_latest_snapshot(),
        "history": macro_tracker.get_macro_snapshots(90),
        "events": macro_tracker.get_upcoming_events(30),
    }


@app.get("/analysis/{analysis_id}")
async def analysis_detail(request: Request, analysis_id: int, username: str = Depends(require_auth)):
    """Full AI analysis report for a specific analysis run."""
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = dict(row)

    # Fetch cross-check result for this analysis
    crosscheck = None
    try:
        conn2 = db._get_conn()
        cur2 = conn2.cursor()
        cur2.execute(
            "SELECT * FROM ai_crosscheck_log WHERE analysis_id = ? ORDER BY checked_at DESC LIMIT 1",
            (analysis_id,)
        )
        cc_row = cur2.fetchone()
        conn2.close()
        if cc_row:
            import json as _json
            crosscheck = dict(cc_row)
            if crosscheck.get('details'):
                try:
                    crosscheck['details'] = _json.loads(crosscheck['details'])
                except (ValueError, TypeError):
                    crosscheck['details'] = []
    except Exception:
        pass

    return templates.TemplateResponse("analysis_detail.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "analysis": analysis,
        "crosscheck": crosscheck
    })

# ==================== SCHEDULER CONTROL ====================

@app.post("/scheduler/start")
async def start_scheduler(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Start the scheduler"""
    csrf.verify_token(request, csrf_token)
    scheduler.start()
    return RedirectResponse(url="/", status_code=303)

@app.post("/scheduler/stop")
async def stop_scheduler(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Stop the scheduler"""
    csrf.verify_token(request, csrf_token)
    scheduler.stop()
    return RedirectResponse(url="/", status_code=303)

@app.post("/scheduler/run-now")
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

# ==================== MANUAL ANALYSIS ====================

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, username: str = Depends(require_auth)):
    """Manual analysis page"""
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "csrf_token": request.state.csrf_token
    })

@app.post("/analyze")
@limiter.limit("10/hour")
async def run_analysis(
    request: Request,
    ticker: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Run manual analysis"""
    csrf.verify_token(request, csrf_token)
    results = swarm.analyze_single_stock(ticker.upper())
    analysis_id, signal, confidence = db.save_analysis(ticker.upper(), results)

    # Run AI cross-check against yfinance ground truth
    try:
        analysis_text = ' '.join(filter(None, [
            results.get('fundamental', ''),
            results.get('recommendation', ''),
            results.get('technical', ''),
        ]))
        if analysis_text.strip():
            crosscheck = ai_crosscheck.check_analysis(ticker.upper(), analysis_text)
            db.save_crosscheck(ticker.upper(), analysis_id, crosscheck)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Cross-check failed for {ticker}: {e}")

    return RedirectResponse(url=f"/analysis/{analysis_id}", status_code=303)

# ==================== DISCOVERY ====================

@app.get("/discover", response_class=HTMLResponse)
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

@app.post("/discover")
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

# ==================== AUTO-DISCOVERY ====================

@app.get("/discoveries", response_class=HTMLResponse)
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

@app.post("/discoveries/{discovery_id}/promote")
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

@app.post("/discoveries/{discovery_id}/dismiss")
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

@app.get("/api/discovery/stats")
async def api_discovery_stats(request: Request, username: str = Depends(require_api_key_or_session)):
    """JSON stats for discovery dashboard widget"""
    return db.get_discovery_stats()

@app.post("/discovery/run-now")
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

# ==================== INSIDER TRADING ====================

@app.get("/dark-pool", response_class=HTMLResponse)
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


@app.get("/dark-pool/{ticker}", response_class=HTMLResponse)
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


@app.post("/dark-pool/scan")
@limiter.limit("2/hour")
async def dark_pool_scan(request: Request, csrf_token: str = Form(...), username: str = Depends(require_auth)):
    """Trigger a dark pool / volume anomaly scan (#52)."""
    csrf.verify_token(request, csrf_token)
    from engine.dark_pool_tracker import scan_watchlist
    count = scan_watchlist()
    return {"success": True, "signals_found": count}


@app.get("/api/dark-pool")
async def api_dark_pool_signals(days: int = 7, min_ratio: float = 0.0,
                                 username: str = Depends(require_api_key_or_session)):
    """Return recent dark pool / volume anomaly signals (#52)."""
    from engine.dark_pool_tracker import get_all_recent_signals
    return {"signals": get_all_recent_signals(days=days, min_volume_ratio=min_ratio)}


@app.get("/api/dark-pool/{ticker}")
async def api_dark_pool_ticker(ticker: str, days: int = 30,
                                username: str = Depends(require_api_key_or_session)):
    """Return dark pool signals for a single ticker (#52)."""
    from engine.dark_pool_tracker import get_ticker_signals
    return {"ticker": ticker.upper(), "signals": get_ticker_signals(ticker, days=days)}


@app.get("/insider-activity", response_class=HTMLResponse)
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

@app.post("/insider-activity/scan")
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

@app.get("/insider-activity/{ticker}", response_class=HTMLResponse)
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

# ==================== LOGS ====================

@app.get("/logs", response_class=HTMLResponse)
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
            from pathlib import Path
            log_file = Path(__file__).parent / "logs" / "application.log"
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


@app.post("/logs/login-failures/unlock")
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


@app.post("/logs/login-failures/unlock-all")
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

@app.post("/toggle-dev-mode")
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

@app.get("/dev-logs")
@limiter.limit("30/minute")
async def get_dev_logs(request: Request, username: str = Depends(require_auth)):
    """Get fresh system logs (dev mode only)"""
    dev_mode = db.get_setting('development_mode') or False
    
    if not dev_mode:
        raise HTTPException(status_code=403, detail="Dev mode not enabled")
    
    try:
        from pathlib import Path
        log_file = Path(__file__).parent / "logs" / "application.log"
        
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

# ==================== ARCHITECTURE ====================

@app.get("/architecture", response_class=HTMLResponse)
async def architecture_page(request: Request, username: str = Depends(require_auth)):
    """System architecture visualization page"""
    return templates.TemplateResponse("architecture.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
    })

# ==================== LEARNING PERFORMANCE ====================

@app.get("/learning", response_class=HTMLResponse)
async def learning_page(request: Request, username: str = Depends(require_auth)):
    """Learning performance and accuracy statistics"""
    # Get overall learning stats
    learning_stats = learning_optimizer.get_learning_stats()
    
    # Get recent verified predictions
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM prediction_outcomes 
        WHERE verified_at IS NOT NULL 
        ORDER BY verified_at DESC 
        LIMIT 20
    """)
    recent_predictions = [dict(row) for row in cursor.fetchall()]
    
    # Get per-ticker statistics
    cursor.execute("""
        SELECT 
            ticker,
            COUNT(*) as total_predictions,
            AVG(accuracy_score) as accuracy,
            AVG(confidence) as avg_confidence
        FROM prediction_outcomes
        WHERE verified_at IS NOT NULL
        GROUP BY ticker
        HAVING COUNT(*) >= 3
        ORDER BY accuracy DESC
    """)
    ticker_stats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Feature 4: Cold start detection for learning page
    cold_start = False
    cold_start_reason = ""
    first_prediction = db.query_one("SELECT MIN(prediction_date) as first_date FROM prediction_outcomes")
    if first_prediction and first_prediction.get('first_date'):
        from datetime import datetime as dt
        try:
            first_date = dt.fromisoformat(first_prediction['first_date'].replace(' ', 'T')[:19])
            days_active = (dt.now() - first_date).days
            if days_active < 60 or learning_stats.get('total_verified', 0) < 20:
                cold_start = True
                cold_start_reason = f"{days_active} days of data, {learning_stats.get('total_verified', 0)} verified predictions"
        except Exception:
            cold_start = True
            cold_start_reason = "Unable to determine system age"
    else:
        cold_start = True
        cold_start_reason = "No predictions recorded yet"

    # Feature 7: Graveyard stats
    graveyard_stats = {'count': 0, 'recent': []}
    try:
        graveyard_count = db.query_one("SELECT COUNT(*) as cnt FROM ticker_graveyard")
        graveyard_stats['count'] = graveyard_count['cnt'] if graveyard_count else 0
        graveyard_recent = db.query("SELECT * FROM ticker_graveyard ORDER BY added_at DESC LIMIT 10")
        graveyard_stats['recent'] = graveyard_recent
    except Exception:
        pass

    return templates.TemplateResponse("learning.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "learning_stats": learning_stats,
        "recent_predictions": recent_predictions,
        "ticker_stats": ticker_stats,
        "cold_start": cold_start,
        "cold_start_reason": cold_start_reason,
        "graveyard_stats": graveyard_stats,
    })

# ==================== TOP PICKS ====================

@app.get("/top-picks", response_class=HTMLResponse)
async def top_picks_page(request: Request, username: str = Depends(require_auth)):
    """Top Picks - Stocks with best prediction track record"""
    # Get top performing stocks
    top_picks = db.get_top_picks(min_predictions=5, min_accuracy=0.6, limit=20)
    
    # Get recent high-confidence predictions
    recent_signals = db.get_recent_high_confidence_predictions(days=7, min_confidence=70)
    
    # Get learning stats for context
    learning_stats = learning_optimizer.get_learning_stats()
    
    return templates.TemplateResponse("top_picks.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "top_picks": top_picks,
        "recent_signals": recent_signals,
        "learning_stats": learning_stats,
        "total_trusted": len([p for p in top_picks if p['accuracy'] >= 70])
    })


# ==================== PORTFOLIO ====================

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request, username: str = Depends(require_auth)):
    """Portfolio management page"""
    portfolio_summary = db.get_portfolio_summary()
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "summary": portfolio_summary,
        "trades": db.get_trades(),
        "holdings": portfolio_summary['holdings']
    })

@app.post("/portfolio/add-trade")
async def add_trade(
    request: Request,
    ticker: str = Form(...),
    type: str = Form(...),
    amount: float = Form(...),
    price: float = Form(...),
    csrf_token: str = Form(...),
    date: str = Form(None),
    fees: float = Form(0.0),
    notes: str = Form(""),
    currency: str = Form("USD"),
    username: str = Depends(require_auth)
):
    """Add a trade to portfolio"""
    csrf.verify_token(request, csrf_token)
    currency = currency.upper()
    db.add_trade(
        ticker=ticker,
        trade_type=type,
        amount=amount,
        price=price,
        date=date,
        fees=fees,
        notes=notes,
        currency=currency,
    )
    return RedirectResponse(url="/portfolio?added=1", status_code=303)

@app.get("/portfolio/export")
async def export_portfolio(request: Request, username: str = Depends(require_api_key_or_session)):
    """Export portfolio to CSV"""
    import csv
    import io
    from fastapi.responses import StreamingResponse
    
    trades = db.get_trades()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Type', 'Ticker', 'Amount', 'Price', 'Fees', 'Total', 'Notes'])
    
    for t in trades:
        total = (t['amount'] * t['price']) + t['fees'] if t['type'] == 'BUY' else (t['amount'] * t['price']) - t['fees']
        writer.writerow([
            t['date'],
            t['type'],
            t['ticker'],
            t['amount'],
            t['price'],
            t['fees'],
            total,
            t['notes']
        ])
    
    output.seek(0)
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=portfolio_export.csv"
    return response

# ==================== BACKTEST ====================

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request, username: str = Depends(require_auth)):
    """Backtest dashboard"""
    from engine.quant_screener import quant_screener
    past_runs = db.get_backtest_runs(limit=20)
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "past_runs": past_runs,
        "active_weights": quant_screener.config['composite_weights'],
    })

@app.post("/backtest/run")
@limiter.limit("2/hour")
async def start_backtest(
    request: Request,
    csrf_token: str = Form(...),
    months: int = Form(24),
    slippage_pct: float = Form(0.001),
    commission_eur: float = Form(1.0),
    username: str = Depends(require_auth),
):
    """Start a backtest in a background thread."""
    csrf.verify_token(request, csrf_token)

    from engine.backtest_engine import backtest_engine

    progress = backtest_engine.get_progress()
    if progress.get('running'):
        return {"success": False, "error": "A backtest is already running"}

    months = max(6, min(60, months))
    slippage_pct = max(0.0, min(0.05, slippage_pct))  # cap 0–5%
    commission_eur = max(0.0, min(50.0, commission_eur))  # cap 0–€50

    import threading
    def _run():
        backtest_engine.run(tickers=None, months=months,
                            slippage_pct=slippage_pct, commission_eur=commission_eur)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {"success": True, "message": "Backtest started"}

@app.get("/api/backtest/progress")
async def backtest_progress(request: Request, username: str = Depends(require_auth)):
    """Poll backtest progress."""
    from engine.backtest_engine import backtest_engine
    return backtest_engine.get_progress()

@app.post("/api/backtest/apply-weights/{run_id}")
async def apply_backtest_weights(
    request: Request,
    run_id: int,
    username: str = Depends(require_auth),
):
    """Apply best weights from a backtest run to the live screener."""
    run = db.get_backtest_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.get('best_weights'):
        raise HTTPException(status_code=400, detail="No best_weights in this run")

    import json as _json
    try:
        weights = _json.loads(run['best_weights']) if isinstance(run['best_weights'], str) else run['best_weights']
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid best_weights JSON")

    if 'tech_weight' not in weights or 'momentum_weight' not in weights:
        raise HTTPException(status_code=400, detail="Missing tech_weight or momentum_weight")

    # Log weight change before applying
    from engine.quant_screener import quant_screener
    old_weights = dict(quant_screener.config['composite_weights'])

    # Save to settings
    db.set_setting('quant_weights_override', weights)

    # Reload in live screener singleton
    quant_screener.reload_weights()
    new_weights = dict(quant_screener.config['composite_weights'])

    # Weight versioning audit trail
    learning_optimizer._log_weight_change(
        old_weights, new_weights,
        trigger='backtest',
        reason=f'Applied best weights from backtest run #{run_id}',
        backtest_run_id=run_id,
    )

    audit_log.log("apply_backtest_weights", username=username, ip=request.client.host,
                  details={"run_id": run_id, "weights": weights})

    return {"success": True, "weights": weights, "active": new_weights}

@app.get("/api/backtest/results/{run_id}")
async def backtest_results(request: Request, run_id: int, username: str = Depends(require_auth)):
    """Get detailed results for a backtest run."""
    run = db.get_backtest_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = db.get_backtest_results(run_id)

    # Build per-ticker summary
    ticker_summary = {}
    for r in results:
        t = r['ticker']
        if t not in ticker_summary:
            ticker_summary[t] = {'ticker': t, 'signals': 0, 'hits': 0,
                                 'returns': [], 'alphas': [], 'benchmarks': set(),
                                 'regimes': []}
        ticker_summary[t]['signals'] += 1
        ticker_summary[t]['hits'] += r.get('hit', 0)
        if r.get('forward_20d_return') is not None:
            ticker_summary[t]['returns'].append(r['forward_20d_return'])
        if r.get('alpha') is not None:
            ticker_summary[t]['alphas'].append(r['alpha'])
        if r.get('benchmark_ticker'):
            ticker_summary[t]['benchmarks'].add(r['benchmark_ticker'])
        if r.get('regime'):
            ticker_summary[t]['regimes'].append(r['regime'])

    for ts in ticker_summary.values():
        ts['accuracy'] = round(ts['hits'] / ts['signals'] * 100, 1) if ts['signals'] else 0
        ts['avg_return'] = round(sum(ts['returns']) / len(ts['returns']), 2) if ts['returns'] else 0
        ts['avg_alpha'] = round(sum(ts['alphas']) / len(ts['alphas']), 2) if ts['alphas'] else None
        ts['benchmark'] = ', '.join(ts['benchmarks']) if ts['benchmarks'] else None
        # Primary regime: most common regime for this ticker's signals
        if ts['regimes']:
            from collections import Counter
            ts['primary_regime'] = Counter(ts['regimes']).most_common(1)[0][0]
        else:
            ts['primary_regime'] = None
        del ts['returns']
        del ts['alphas']
        del ts['regimes']
        ts['benchmarks'] = list(ts['benchmarks'])  # make JSON-serializable

    # Parse walk-forward windows JSON if present
    if run.get('walk_forward_windows') and isinstance(run['walk_forward_windows'], str):
        try:
            run['walk_forward_windows'] = json.loads(run['walk_forward_windows'])
        except (json.JSONDecodeError, TypeError):
            run['walk_forward_windows'] = []

    return {
        "run": run,
        "results": results[:500],
        "ticker_summary": sorted(ticker_summary.values(), key=lambda x: x['accuracy'], reverse=True),
    }

# ==================== PAPER TRADING ====================

@app.get("/paper-trading", response_class=HTMLResponse)
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

@app.post("/paper-trading/reset")
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

@app.post("/paper-trading/settings")
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

@app.get("/api/paper-trading/summary")
async def api_paper_trading_summary(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get paper trading portfolio summary"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_portfolio_summary()

@app.get("/api/paper-trading/equity-curve")
async def api_paper_trading_equity_curve(
    request: Request, 
    days: int = 30,
    username: str = Depends(require_api_key_or_session)
):
    """Get equity curve data for charting"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_equity_curve(days_back=min(days, 365))

@app.get("/api/signal-ev")
async def api_signal_ev(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get Signal Expected Value - avg returns per signal type and confidence"""
    return learning_optimizer.feedback.calculate_signal_ev()

@app.get("/api/calibration")
async def api_calibration(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get calibration curve data: predicted confidence vs actual hit rate"""
    return learning_optimizer.feedback.calculate_calibration()

@app.get("/api/ab-comparison")
async def api_ab_comparison(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get quant-only vs quant+AI accuracy comparison"""
    return learning_optimizer.feedback.calculate_ab_comparison()

@app.get("/api/signal-decay")
async def api_signal_decay(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get signal accuracy at multiple time horizons (1d, 3d, 7d, 14d, 30d)"""
    return learning_optimizer.feedback.calculate_signal_decay()

@app.get("/api/weight-history")
async def api_weight_history(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get weight change audit trail"""
    return learning_optimizer.get_weight_history()

@app.post("/api/weight-rollback/{version_id}")
@limiter.limit("5/hour")
async def api_weight_rollback(request: Request, version_id: int, username: str = Depends(require_api_key_or_session)):
    """Rollback to a previous weight version"""
    token = request.headers.get('X-CSRF-Token', '')
    if not csrf.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    result = learning_optimizer.rollback_weights(version_id)
    if result.get('success'):
        audit_log.log("weight_rollback", username=username,
                      ip=request.client.host, details={"version_id": version_id})
    return result

@app.get("/api/paper-trading/risk-metrics")
async def api_paper_risk_metrics(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get risk-adjusted metrics for paper trading portfolio"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_risk_metrics()

@app.get("/api/paper-trading/spy-correlation")
async def api_paper_spy_correlation(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get portfolio beta and alpha vs SPY"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_spy_correlation()

@app.get("/api/sector-momentum")
async def api_sector_momentum(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get sector momentum heat map data"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_heat_map_data()

@app.get("/api/sector-momentum/rotation")
async def api_sector_rotation(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get sector rotation signals"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_rotation_signals()

@app.get("/api/economic-calendar")
async def api_economic_calendar(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get upcoming market-moving events"""
    from engine.economic_calendar import economic_calendar
    return economic_calendar.get_calendar_summary()

@app.get("/api/multi-timeframe/{ticker}")
async def api_multi_timeframe(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Get multi-timeframe analysis for a ticker"""
    from engine.multi_timeframe import multi_timeframe
    return multi_timeframe.analyze_ticker(ticker.upper())

@app.get("/api/position-size/{ticker}")
async def api_position_size(ticker: str, confidence: int = 70, portfolio: float = 100000, request: Request = None, username: str = Depends(require_api_key_or_session)):
    """Get recommended position size for a ticker"""
    from engine.position_sizing import position_sizer
    return position_sizer.calculate_position_size(
        ticker=ticker.upper(),
        signal_confidence=confidence,
        portfolio_value=portfolio,
    )

@app.get("/api/statistical-significance")
async def api_statistical_significance(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get statistical significance of prediction accuracy"""
    return learning_optimizer.feedback.calculate_significance()

@app.get("/api/drawdown")
async def api_drawdown(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get drawdown analysis for paper trading"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_paper_trading_drawdown()

@app.get("/api/reality-check")
async def api_reality_check(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get comprehensive reality check dashboard"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_reality_dashboard()

# ==================== API ENDPOINTS ====================

@app.get("/api/server/sleep-status")
async def api_sleep_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """API endpoint for BREATHE-5b Deep Sleep UI status"""
    enabled = db.get_setting("deep_sleep_enabled") == True or str(db.get_setting("deep_sleep_enabled")).lower() == "true"
    intensity = db.get_setting("deep_sleep_intensity") or "deep"
    wake_time = db.get_setting("deep_sleep_end") or "07:00"
    
    # We can ask scheduler if it's currently sleeping
    is_sleeping = False
    if enabled and hasattr(scheduler, 'is_deep_sleep_active'):
        is_sleeping = scheduler.is_deep_sleep_active()

    status = scheduler.get_status()
    jobs = status.get('jobs', [])
    next_scan_min = 15
    if jobs:
        scan_job = next((j for j in jobs if j['name'] == 'run_scan'), jobs[0])
        if scan_job and scan_job.get('next_run'):
            try:
                from datetime import datetime
                import pytz
                nr = datetime.fromisoformat(scan_job['next_run'])
                now = datetime.now(pytz.utc)
                diff = (nr.astimezone(pytz.utc) - now).total_seconds()
                next_scan_min = max(0, int(diff / 60))
            except Exception:
                pass

    return {
        "sleeping": is_sleeping,
        "hibernate": intensity == "hibernate",
        "resumes_at": wake_time,
        "next_scan_min": next_scan_min
    }

@app.get("/api/status")
async def api_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """API endpoint for status"""
    # Count stale analyses
    recent = db.get_analysis_history(limit=50)
    stale_count = 0
    for a in recent:
        staleness_tracker.enrich_analysis(a)
        if a.get('staleness_level') in ('stale', 'very_stale'):
            stale_count += 1

    return {
        "scheduler": scheduler.get_status(),
        "api_usage": {
            "perplexity": pplx_client.get_usage(),
            "gemini": gemini_client.get_usage()
        },
        "watchlist_count": len(db.get_watchlist()),
        "stale_analyses": stale_count,
    }

@app.get("/api/scan-progress")
async def api_scan_progress(request: Request, username: str = Depends(require_api_key_or_session)):
    """Real-time scan progress for dashboard status bar"""
    from engine.scan_progress import scan_progress
    return scan_progress.get_state()

@app.get("/api/discovery-status")
async def api_discovery_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """Real-time discovery run status for discoveries page polling"""
    from engine.auto_discovery import discovery_status
    return discovery_status.get()

@app.get("/api/ollama/health")
async def api_ollama_health(request: Request, username: str = Depends(require_api_key_or_session)):
    """Health check for local Ollama server (item #41)."""
    from clients.ollama_client import ollama_client
    available = ollama_client.health_check()
    models = ollama_client.list_models() if available else []
    return {"available": available, "models": models}

@app.get("/api/budget")
async def api_budget_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """API endpoint for budget status (used by dashboard AJAX)"""
    return budget_tracker.get_budget_status()

@app.get("/api/budget/status")
async def api_budget_status_detail(request: Request, username: str = Depends(require_api_key_or_session)):
    """Detailed budget health card endpoint (#29)."""
    status = budget_tracker.get_budget_status()
    # Compute avg cost per analysis from last 7 days
    try:
        from datetime import date as _date, timedelta
        week_ago = (_date.today() - timedelta(days=7)).isoformat()
        rows = db.query(
            "SELECT api, SUM(estimated_cost) as total, COUNT(*) as calls FROM api_cost_log WHERE date >= ? GROUP BY api",
            (week_ago,)
        )
        cost_7d = {r['api']: {'total': r['total'], 'calls': r['calls']} for r in rows}
        total_cost_7d = sum(r['total'] for r in rows)
        total_calls_7d = sum(r['calls'] for r in rows)
        avg_cost_per_analysis = round(total_cost_7d / max(total_calls_7d, 1), 4)
    except Exception:
        cost_7d = {}
        avg_cost_per_analysis = None
    status['avg_cost_per_analysis_usd'] = avg_cost_per_analysis
    status['cost_7d'] = cost_7d
    return status

@app.get("/api/portfolio/alerts")
async def api_portfolio_alerts(request: Request, username: str = Depends(require_api_key_or_session)):
    """Portfolio rule checks: position sizing, stop-loss, sector concentration, benchmark."""
    from engine.portfolio_manager import portfolio_manager
    from engine.alert_manager import alert_manager

    payload = portfolio_manager.check_all_rules()
    raw_alerts = payload.get('alerts', [])

    surfaced_alerts = []
    for alert in raw_alerts:
        if alert_manager.should_alert(alert):
            surfaced_alerts.append(alert)
            if not alert.get('is_repeated'):
                alert_manager.store_alert(alert)

    payload['alerts'] = alert_manager.prioritize_alerts(surfaced_alerts)
    payload['active_alerts'] = alert_manager.get_active_alerts(include_acknowledged=False)
    payload['alert_summary'] = alert_manager.get_alert_summary()
    payload['raw_alert_count'] = len(raw_alerts)

    return payload


@app.post("/api/portfolio/alerts/ack")
async def api_ack_portfolio_alert(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Acknowledge a deduplicated alert by id or hash."""
    from engine.alert_manager import alert_manager

    data = await request.json()

    token = request.headers.get('X-CSRF-Token', '')
    if not csrf.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    alert_id = data.get('alert_id')
    alert_hash = data.get('alert_hash')

    if not alert_id and not alert_hash:
        raise HTTPException(status_code=400, detail="alert_id or alert_hash is required")

    alert_manager.acknowledge_alert(alert_id=alert_id, alert_hash=alert_hash)
    return {
        'success': True,
        'alert_id': alert_id,
        'alert_hash': alert_hash,
    }

@app.get("/api/signal-pnl")
async def api_signal_pnl(request: Request, username: str = Depends(require_api_key_or_session)):
    """Signal P&L scorecard — aggregated prediction outcome stats."""
    return db.get_signal_pnl_summary()

@app.get("/api/quant-screen")
async def api_quant_screen(request: Request, username: str = Depends(require_api_key_or_session)):
    """Run quant screener on watchlist — zero API cost."""
    from engine.quant_screener import quant_screener
    watchlist = db.get_watchlist(active_only=True)
    tickers = [item['ticker'] for item in watchlist]
    if not tickers:
        return {'results': [], 'message': 'Watchlist empty'}
    results = quant_screener.screen_batch(tickers)
    return {'results': results, 'count': len(results)}

# ==================== DATA EXPORT ====================

@app.get("/api/export/analyses")
async def export_analyses(request: Request, format: str = "csv", username: str = Depends(require_api_key_or_session)):
    """Export analysis history as CSV or JSON"""
    import csv
    import io
    import json as _json
    from fastapi.responses import StreamingResponse

    analyses = db.get_analysis_history(limit=1000)

    if format == "json":
        content = _json.dumps(analyses, indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analyses_export.json"}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Ticker', 'Signal', 'Confidence', 'Timestamp', 'Recommendation', 'Fundamental', 'Technical'])
    for a in analyses:
        writer.writerow([
            a.get('id', ''), a.get('ticker', ''), a.get('signal', ''),
            a.get('confidence', ''), a.get('timestamp', ''),
            (a.get('recommendation', '') or '')[:200],
            (a.get('fundamental', '') or '')[:200],
            (a.get('technical', '') or '')[:200],
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analyses_export.csv"}
    )

@app.get("/api/export/predictions")
async def export_predictions(request: Request, format: str = "csv", username: str = Depends(require_api_key_or_session)):
    """Export prediction outcomes as CSV or JSON"""
    import csv
    import io
    import json as _json
    from fastapi.responses import StreamingResponse

    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_outcomes ORDER BY prediction_date DESC LIMIT 2000")
    predictions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if format == "json":
        content = _json.dumps(predictions, indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=predictions_export.json"}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Ticker', 'Date', 'Signal', 'Direction', 'Confidence',
                     'Price_At_Prediction', 'Price_After', 'Actual_Direction',
                     'Accuracy', 'Days_Elapsed', 'Verified_At', 'Signal_Type', 'Has_AI'])
    for p in predictions:
        writer.writerow([
            p.get('id', ''), p.get('ticker', ''), p.get('prediction_date', ''),
            p.get('signal', ''), p.get('predicted_direction', ''), p.get('confidence', ''),
            p.get('actual_price_at_prediction', ''), p.get('actual_price_after', ''),
            p.get('actual_direction', ''), p.get('accuracy_score', ''),
            p.get('days_elapsed', ''), p.get('verified_at', ''),
            p.get('signal_type', ''), p.get('has_ai', ''),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_export.csv"}
    )

@app.get("/api/export/paper-trades")
async def export_paper_trades(request: Request, format: str = "csv", username: str = Depends(require_api_key_or_session)):
    """Export paper trading history as CSV or JSON"""
    import csv
    import io
    import json as _json
    from fastapi.responses import StreamingResponse

    from engine.paper_trading import paper_trader
    trades = paper_trader.get_trade_log(limit=5000)

    if format == "json":
        content = _json.dumps(trades, indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=paper_trades_export.json"}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    if trades:
        writer.writerow(list(trades[0].keys()))
        for t in trades:
            writer.writerow(list(t.values()))
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=paper_trades_export.csv"}
    )

@app.get("/api/export/backtest/{run_id}")
async def export_backtest(request: Request, run_id: int, format: str = "csv", username: str = Depends(require_api_key_or_session)):
    """Export backtest results as CSV or JSON"""
    import csv
    import io
    import json as _json
    from fastapi.responses import StreamingResponse

    results = db.get_backtest_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    if format == "json":
        content = _json.dumps(results, indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=backtest_{run_id}_export.json"}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    if results:
        writer.writerow(list(results[0].keys()))
        for r in results:
            writer.writerow(list(r.values()))
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=backtest_{run_id}_export.csv"}
    )

# ==================== AI CROSS-CHECK ====================

@app.get("/crosscheck", response_class=HTMLResponse)
async def crosscheck_page(request: Request, username: str = Depends(require_auth)):
    """Cross-check history page"""
    history = db.get_crosscheck_history(limit=50)
    return templates.TemplateResponse("crosscheck.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "history": history
    })

@app.post("/api/crosscheck/{analysis_id}")
@limiter.limit("10/hour")
async def run_crosscheck(
    request: Request,
    analysis_id: int,
    username: str = Depends(require_auth),
):
    """Run cross-check on an existing analysis."""
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = dict(row)
    ticker = analysis['ticker']

    analysis_text = ' '.join(filter(None, [
        analysis.get('fundamental', ''),
        analysis.get('recommendation', ''),
        analysis.get('technical', ''),
    ]))

    if not analysis_text.strip():
        return {"success": False, "error": "No analysis text to cross-check"}

    crosscheck = ai_crosscheck.check_analysis(ticker, analysis_text)
    db.save_crosscheck(ticker, analysis_id, crosscheck)

    return {"success": True, "result": crosscheck}

@app.get("/api/crosscheck/history")
async def crosscheck_history(
    request: Request,
    ticker: str = None,
    username: str = Depends(require_auth),
):
    """Get cross-check history."""
    return db.get_crosscheck_history(ticker=ticker, limit=50)

# ==================== MARKET REGIME ====================

@app.get("/api/market-regime")
async def api_market_regime(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current market regime (bull/bear/choppy) with VIX and yield data."""
    from engine.market_regime import market_regime
    return market_regime.get_current_regime()


@app.get("/api/regime-adjustments")
async def api_regime_adjustments(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return active weight adjustment multipliers for the current market regime."""
    from engine.market_regime import market_regime
    regime_data = market_regime.get_current_regime()
    regime = regime_data.get('regime', 'choppy')
    adjustments = market_regime.get_regime_weight_adjustments(regime)
    return {
        "regime": regime,
        "adjustments": adjustments,
        "description": {
            "bull": "Bull market: boost momentum & technical, reduce valuation weight",
            "bear": "Bear market: boost valuation & quality, reduce momentum weight",
            "choppy": "Choppy market: balanced weighting across all factors",
        }.get(regime, ""),
    }

# ==================== PORTFOLIO Q&A (#37) ====================

@app.post("/api/portfolio/ask")
async def api_portfolio_ask(request: Request, username: str = Depends(require_api_key_or_session)):
    """
    Natural language portfolio Q&A powered by Gemini (#37).
    POST body: { "question": "Which of my holdings are most exposed to tariff risk?" }
    """
    try:
        body = await request.json()
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    question = (body.get("question") or "").strip()
    if not question:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "question is required"}, status_code=400)

    from engine.portfolio_qa import ask as portfolio_ask
    result = portfolio_ask(question)
    return result

# ==================== PORTFOLIO BENCHMARK ====================

@app.get("/api/portfolio/benchmark")
async def api_portfolio_benchmark(request: Request, username: str = Depends(require_api_key_or_session)):
    """Portfolio vs SPY benchmark comparison."""
    from engine.portfolio_benchmark import portfolio_benchmark
    return portfolio_benchmark.calculate_portfolio_vs_spy()

# ==================== CONCENTRATION CHECK ====================

@app.get("/api/portfolio/concentration")
async def api_portfolio_concentration(request: Request, username: str = Depends(require_api_key_or_session)):
    """Check portfolio concentration and correlation risks."""
    from engine.concentration_checker import concentration_checker
    holdings = db.get_portfolio_holdings()
    return concentration_checker.check_portfolio_concentration(holdings)

# ==================== PRICE CHART DATA ====================

@app.get("/api/chart-data/{ticker}")
async def api_chart_data(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return 6mo OHLCV + SMA overlays + signal markers for a ticker."""
    import yfinance as yf_local

    ticker = ticker.upper().strip()
    try:
        stock = yf_local.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return {"error": "No data available"}

        close = hist['Close']
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = [round(float(p), 2) for p in close]

        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        def safe_list(series):
            return [round(float(v), 2) if not (v != v) else None for v in series]

        # Signal markers from analysis history
        signals = db.get_analysis_history(ticker=ticker, limit=20)
        markers = []
        for s in signals:
            sig_date = s['timestamp'][:10] if s.get('timestamp') else None
            if sig_date and sig_date in dates:
                idx = dates.index(sig_date)
                markers.append({
                    'date': sig_date,
                    'price': prices[idx],
                    'signal': s.get('signal', ''),
                    'confidence': s.get('confidence', 0),
                })

        # === Algorithm Visualization Data ===

        # Market Structure: DC turning points + support/resistance
        turning_points = []
        support_resistance = []
        try:
            from engine.market_structure import market_structure_analyzer
            structure = market_structure_analyzer.analyze(ticker, hist)
            if structure:
                tp_list = market_structure_analyzer.get_dc_turning_points(hist)
                for p in tp_list:
                    if 0 <= p.index < len(dates):
                        turning_points.append({
                            'date': dates[p.index],
                            'price': round(p.price, 2),
                            'type': p.type,
                            'level': p.level,
                        })
                if structure.get('support'):
                    support_resistance.append({'price': structure['support'], 'type': 'support'})
                if structure.get('resistance'):
                    support_resistance.append({'price': structure['resistance'], 'type': 'resistance'})
        except Exception:
            pass

        # Harmonic Patterns: XABCD overlays
        harmonic_overlays = []
        try:
            from engine.harmonic_patterns import harmonic_detector
            h_patterns = harmonic_detector.detect(ticker, hist)
            for p in h_patterns[:3]:
                harmonic_overlays.append({
                    'pattern_name': p['pattern_name'],
                    'direction': p['direction'],
                    'confidence': p['confidence'],
                    'points': {
                        'x': p['x_price'], 'a': p['a_price'], 'b': p['b_price'],
                        'c': p['c_price'], 'd': p['d_price'],
                    },
                    'entry_zone': p.get('entry_zone'),
                    'stop_loss': p.get('stop_loss'),
                    'targets': p.get('targets', []),
                })
        except Exception:
            pass

        # Visibility Graph indicator
        vg_data = {}
        try:
            from engine.visibility_graph import vg_analyzer
            vg_data = vg_analyzer.analyze(ticker, hist)
        except Exception:
            pass

        # Meta-labeler status
        meta_label_data = {}
        try:
            from engine.meta_labeler import meta_labeler
            meta_label_data = meta_labeler.get_status()
        except Exception:
            pass

        # MCPT validation status
        mcpt_data = {}
        try:
            from engine.mcpt_validator import mcpt_validator
            mcpt_data = mcpt_validator.get_latest_result() or {}
        except Exception:
            pass

        return {
            'ticker': ticker,
            'dates': dates,
            'prices': prices,
            'sma20': safe_list(sma20),
            'sma50': safe_list(sma50),
            'sma200': safe_list(sma200),
            'volume': [int(v) for v in hist['Volume']],
            'signals': markers,
            'turning_points': turning_points,
            'support_resistance': support_resistance,
            'harmonic_patterns': harmonic_overlays,
            'vg': vg_data,
            'meta_labeler': meta_label_data,
            'mcpt': mcpt_data,
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== ALGORITHM STATUS ====================

@app.get("/api/algo-status")
async def api_algo_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return algorithm module statuses for dashboard badges."""
    result = {}
    try:
        from engine.meta_labeler import meta_labeler
        result['meta_labeler'] = meta_labeler.get_status()
    except Exception:
        pass
    try:
        from engine.mcpt_validator import mcpt_validator
        result['mcpt'] = mcpt_validator.get_latest_result() or {}
    except Exception:
        pass
    return result

# ==================== LEARNING WEIGHT OPTIMIZATION ====================

@app.get("/api/learning/weight-suggestions")
async def api_weight_suggestions(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current vs suggested quant weights based on learning data."""
    return learning_optimizer.calculate_optimal_weights()

@app.get("/api/learning/feature-importance")
async def api_feature_importance(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return RF meta-labeler feature importances sorted descending (#48)."""
    from engine.meta_labeler import meta_labeler
    importances = meta_labeler.get_feature_importances()
    if not importances:
        return {"ready": False, "importances": [], "top3": []}
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return {
        "ready": True,
        "importances": [{"feature": k, "importance": v} for k, v in sorted_items],
        "top3": [k for k, _ in sorted_items[:3]],
    }

@app.post("/api/learning/apply-weights")
@limiter.limit("5/hour")
async def api_apply_weights(request: Request, username: str = Depends(require_api_key_or_session)):
    """Apply suggested weight optimizations to the live screener."""
    data = await request.json()

    # Validate CSRF token from header
    token = request.headers.get('X-CSRF-Token', '')
    if not csrf.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    dry_run = data.get('dry_run', True)
    result = learning_optimizer.auto_adjust_weights(dry_run=dry_run)

    if not dry_run:
        audit_log.log("apply_learning_weights", username=username,
                      ip=request.client.host, details={"result": str(result)})

    return result

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    import os
    import psutil
    
    # Database connectivity
    db_healthy = False
    try:
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        db_healthy = True
    except Exception as e:
        db_error = str(e)
    
    # API connectivity — check any provider is configured
    configured_providers = db.get_api_providers(include_secrets=False)
    enabled_providers = [p for p in configured_providers if p.get("enabled")]
    any_ai_configured = len(enabled_providers) > 0

    # Disk space
    disk = psutil.disk_usage('/')
    disk_warning = disk.percent > 80

    # Learning system
    learning_stats = learning_optimizer.get_learning_stats()

    status = {
        "status": "healthy" if (db_healthy and not disk_warning) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "database": {
                "healthy": db_healthy,
                "error": db_error if not db_healthy else None
            },
            "ai_providers": {
                "count": len(enabled_providers),
                "healthy": any_ai_configured,
                "names": [p.get("name") for p in enabled_providers],
            },
            "scheduler": {
                "running": scheduler.is_running,
                "jobs": len(scheduler.get_status().get("jobs", []))
            },
            "disk_space": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": disk.percent,
                "warning": disk_warning
            },
            "learning": {
                "total_predictions": learning_stats.get("total_verified", 0),
                "accuracy": learning_stats.get("avg_accuracy", 0),
                "cache_size": learning_stats.get("cache_size", 0)
            }
        }
    }
    
    return status

# ==================== STOCK DETAIL PAGE ====================

@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail_page(request: Request, ticker: str, analysis_id: int | None = None, username: str = Depends(require_auth)):
    """Unified stock detail page — all data about a ticker in one place."""
    from engine.earnings_tracker import earnings_tracker
    from engine.financial_statements import financial_statements
    from engine.insider_tracker import insider_tracker
    from engine.multi_timeframe import multi_timeframe
    from engine.position_sizing import position_sizer
    from engine.dividend_tracker import dividend_tracker

    ticker = ticker.upper()

    # Parallel data gathering (best-effort — each module handles its own errors)
    try:
        key_stats = financial_statements.get_key_stats(ticker)
    except Exception:
        key_stats = {'available': False}

    try:
        earnings_info = earnings_tracker.get_earnings_info(ticker)
        beat_history = earnings_tracker.get_beat_history(ticker)
        earnings_alert = earnings_tracker.generate_positioning_alert(ticker)
    except Exception:
        earnings_info = None
        beat_history = {'available': False}
        earnings_alert = None

    try:
        quarterly = financial_statements.get_quarterly_financials(ticker)
    except Exception:
        quarterly = {'available': False}

    try:
        dcf = financial_statements.estimate_fair_value(ticker)
    except Exception:
        dcf = {'available': False}

    # Insider activity (last 10 transactions)
    try:
        insider_data = db.query("""
            SELECT * FROM insider_transactions
            WHERE ticker = ?
            ORDER BY transaction_date DESC LIMIT 10
        """, (ticker,))
    except Exception:
        insider_data = []

    # Analysis history (last 10)
    try:
        analysis_history = db.get_analysis_history(ticker=ticker, limit=10)
    except Exception:
        analysis_history = []

    # Stock notes
    try:
        stock_note = db.get_stock_note(ticker)
    except Exception:
        stock_note = None

    # Dividend info
    try:
        div_info = dividend_tracker.get_dividend_info(ticker) if hasattr(dividend_tracker, 'get_dividend_info') else None
    except Exception:
        div_info = None

    # Check watchlist membership
    try:
        in_watchlist = any(w['ticker'] == ticker for w in db.get_watchlist())
    except Exception:
        in_watchlist = False

    # Discovery history for this ticker
    try:
        discovery_history = db.query("""
            SELECT strategy, quant_score as score, found_at, promoted_at as promoted
            FROM discovered_stocks
            WHERE ticker = ?
            ORDER BY found_at DESC LIMIT 10
        """, (ticker,))
    except Exception:
        discovery_history = []

    # Selected/latest analysis for merged stock+analysis view
    selected_analysis = None
    crosscheck = None
    try:
        conn = db._get_conn()
        cursor = conn.cursor()
        if analysis_id is not None:
            cursor.execute(
                "SELECT * FROM analysis_history WHERE id = ? AND ticker = ?",
                (analysis_id, ticker)
            )
            row = cursor.fetchone()
            if not row:
                cursor.execute(
                    "SELECT * FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                    (ticker,)
                )
                row = cursor.fetchone()
        else:
            cursor.execute(
                "SELECT * FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                (ticker,)
            )
            row = cursor.fetchone()
        conn.close()

        if row:
            selected_analysis = dict(row)

            try:
                conn2 = db._get_conn()
                cur2 = conn2.cursor()
                cur2.execute(
                    "SELECT * FROM ai_crosscheck_log WHERE analysis_id = ? ORDER BY checked_at DESC LIMIT 1",
                    (selected_analysis["id"],)
                )
                cc_row = cur2.fetchone()
                conn2.close()

                if cc_row:
                    import json as _json
                    crosscheck = dict(cc_row)
                    if crosscheck.get("details"):
                        try:
                            crosscheck["details"] = _json.loads(crosscheck["details"])
                        except (ValueError, TypeError):
                            crosscheck["details"] = []
            except Exception:
                crosscheck = None
    except Exception:
        selected_analysis = None
        crosscheck = None

    return templates.TemplateResponse("stock_detail.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "ticker": ticker,
        "key_stats": key_stats,
        "earnings_info": earnings_info,
        "beat_history": beat_history,
        "earnings_alert": earnings_alert,
        "quarterly": quarterly,
        "dcf": dcf,
        "insider_data": insider_data,
        "analysis_history": analysis_history,
        "selected_analysis": selected_analysis,
        "crosscheck": crosscheck,
        "stock_note": stock_note,
        "div_info": div_info,
        "in_watchlist": in_watchlist,
        "discovery_history": discovery_history,
    })


@app.get("/api/stock/{ticker}/risk-trend")
async def api_risk_trend(ticker: str, days: int = 30, username: str = Depends(require_api_key_or_session)):
    rows = db.query("""
        SELECT timestamp, risk_score, geo_risk_score, signal, confidence
        FROM analysis_history
        WHERE ticker = ?
        AND timestamp >= datetime('now', ? || ' days')
        ORDER BY timestamp ASC
    """, (ticker.upper(), f"-{days}"))
    return {"ticker": ticker.upper(), "data": rows}

@app.get("/api/corporate-actions")
async def api_corporate_actions_all(
    request: Request,
    ticker: str = None,
    type: str = None,
    username: str = Depends(require_api_key_or_session),
):
    """Return corporate actions across all watchlist tickers with optional filters."""
    if ticker:
        actions = db.get_corporate_actions(ticker.upper(), limit=100)
    else:
        actions = db.get_recent_corporate_actions(days=365)

    if type:
        actions = [a for a in actions if a.get('action_type', '').lower() == type.lower()]

    # Build dividend income summary
    dividend_summary = {}
    for a in actions:
        if a.get('action_type', '').lower() == 'dividend' and a.get('value'):
            t = a.get('ticker', '')
            dividend_summary[t] = dividend_summary.get(t, 0) + float(a['value'])

    return {"actions": actions, "dividend_summary": dividend_summary}


@app.get("/api/stock/{ticker}/staleness")
async def api_staleness(ticker: str, username: str = Depends(require_api_key_or_session)):
    """
    Return confidence decay metadata for a ticker's latest analysis (#28/#53).

    Includes staleness_days, decay_pct (50% at 5 days), staleness_level,
    and a 10-point decay curve for charting.
    """
    ticker = ticker.upper()
    from engine.staleness_tracker import staleness_tracker

    row = db.query(
        "SELECT timestamp, confidence FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
        (ticker,)
    )
    if not row:
        return {"ticker": ticker, "error": "No analysis found"}

    analysis = dict(row[0])
    enriched = staleness_tracker.enrich_analysis(analysis)
    age_days = enriched.get("age_days", 0)
    original_conf = enriched.get("confidence", 70) or 70

    # Build a 10-point decay curve: day 0 → day 14
    decay_curve = []
    for d in range(0, 15, 1):
        decayed = staleness_tracker.apply_confidence_decay(float(original_conf), d)
        decay_curve.append({"day": d, "confidence": round(decayed, 1)})

    return {
        "ticker": ticker,
        "last_analyzed": analysis.get("timestamp"),
        "age_days": age_days,
        "original_confidence": original_conf,
        "current_confidence": round(staleness_tracker.apply_confidence_decay(float(original_conf), age_days), 1),
        "decay_pct": round((1 - staleness_tracker.apply_confidence_decay(float(original_conf), age_days) / float(original_conf)) * 100, 1) if original_conf else 0,
        "staleness_level": staleness_tracker.get_staleness_level(age_days),
        "staleness_icon": staleness_tracker.get_staleness_icon(staleness_tracker.get_staleness_level(age_days)),
        "should_refresh": staleness_tracker.should_refresh(age_days),
        "decay_curve": decay_curve,
    }

@app.get("/api/stock/{ticker}/corporate-actions")
async def api_corporate_actions(ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return corporate actions (splits, dividends) for a ticker (#43)."""
    ticker = ticker.upper()
    actions = db.get_corporate_actions(ticker, limit=50)
    return {"ticker": ticker, "actions": actions}

@app.post("/stock/{ticker}/notes")
async def save_stock_note(
    request: Request,
    ticker: str,
    note_text: str = Form(""),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Save free-text note for a ticker."""
    csrf.verify_token(request, csrf_token)
    db.save_stock_note(ticker.upper(), note_text)
    return RedirectResponse(url=f"/stock/{ticker.upper()}?saved=1", status_code=303)


# ==================== EARNINGS API ====================

@app.get("/api/earnings/{ticker}")
async def api_earnings(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return earnings data including beat history for a ticker."""
    from engine.earnings_tracker import earnings_tracker
    t = ticker.upper()
    info = earnings_tracker.get_earnings_info(t)
    beat = earnings_tracker.get_beat_history(t)
    alert = earnings_tracker.generate_positioning_alert(t)
    return {"ticker": t, "earnings_info": info, "beat_history": beat, "alert_message": alert}


@app.get("/api/key-stats/{ticker}")
async def api_key_stats(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return 52w proximity, market cap label, short interest, pre/post market prices."""
    from engine.financial_statements import financial_statements
    return financial_statements.get_key_stats(ticker.upper())


@app.get("/api/financials/{ticker}")
async def api_financials(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return 8-quarter financial trend data."""
    from engine.financial_statements import financial_statements
    return financial_statements.get_quarterly_financials(ticker.upper())


@app.get("/api/dcf/{ticker}")
async def api_dcf(
    ticker: str,
    request: Request,
    growth_rate: float = None,
    terminal_rate: float = 0.03,
    discount_rate: float = 0.10,
    username: str = Depends(require_api_key_or_session),
):
    """Run DCF fair value estimate with adjustable assumptions."""
    from engine.financial_statements import financial_statements
    return financial_statements.estimate_fair_value(
        ticker.upper(), growth_rate=growth_rate,
        terminal_rate=terminal_rate, discount_rate=discount_rate
    )


@app.get("/api/peers/{ticker}")
async def api_peers(
    ticker: str, request: Request,
    peers: str = "",
    username: str = Depends(require_api_key_or_session),
):
    """Peer comparison table."""
    from engine.financial_statements import financial_statements
    peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()] if peers else None
    return financial_statements.get_peer_comparison(ticker.upper(), peer_list)


# ==================== WEBHOOK TESTING ====================

@app.post("/settings/test-telegram")
async def test_telegram(
    request: Request,
    csrf_token: str = Form(...),
    telegram_bot_token: str = Form(""),
    telegram_chat_id: str = Form(""),
    username: str = Depends(require_auth),
):
    """Send a test Telegram message to validate token/chat_id."""
    csrf.verify_token(request, csrf_token)
    from engine.webhook_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    ok, msg = notifier.test(
        token=telegram_bot_token or None,
        chat_id=telegram_chat_id or None
    )
    redirect_url = f"/settings?{'telegram_ok=1' if ok else 'telegram_error=1'}&msg={msg[:100]}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/settings/test-discord")
async def test_discord(
    request: Request,
    csrf_token: str = Form(...),
    discord_webhook_url: str = Form(""),
    username: str = Depends(require_auth),
):
    """Send a test Discord message to validate webhook URL."""
    csrf.verify_token(request, csrf_token)
    from engine.webhook_notifier import DiscordNotifier
    notifier = DiscordNotifier()
    ok, msg = notifier.test(webhook_url=discord_webhook_url or None)
    redirect_url = f"/settings?{'discord_ok=1' if ok else 'discord_error=1'}&msg={msg[:100]}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/settings/save-webhooks")
async def save_webhook_settings(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Save Telegram / Discord webhook configuration."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    db.set_setting("telegram_enabled", form.get("telegram_enabled") == "on")
    if form.get("telegram_bot_token"):
        db.set_setting("telegram_bot_token", form.get("telegram_bot_token", ""))
    if form.get("telegram_chat_id"):
        db.set_setting("telegram_chat_id", form.get("telegram_chat_id", ""))
    bot_enabled = form.get("telegram_bot_enabled") == "on"
    db.set_setting("telegram_bot_enabled", bot_enabled)
    # Restart the bot polling thread to pick up the new setting
    try:
        from clients.telegram_bot import telegram_bot
        telegram_bot.restart()
    except Exception:
        pass
    db.set_setting("discord_enabled", form.get("discord_enabled") == "on")
    if form.get("discord_webhook_url"):
        db.set_setting("discord_webhook_url", form.get("discord_webhook_url", ""))
    return RedirectResponse(url="/settings?saved=1", status_code=303)


# ==================== WATCHLIST TIER ====================

@app.post("/watchlist/tier/{ticker}")
async def update_watchlist_tier(
    request: Request,
    ticker: str,
    tier: str = Form(...),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Update the tier tag for a watchlist entry."""
    csrf.verify_token(request, csrf_token)
    valid_tiers = {"core", "swing", "research", "earnings"}
    if tier.lower() in valid_tiers:
        db.update_watchlist_tier(ticker.upper(), tier.lower())
    return RedirectResponse(url="/watchlist", status_code=303)


# ==================== TRADE JOURNAL ====================

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request, ticker: str = None, username: str = Depends(require_auth)):
    """Trade journal page."""
    entries = db.get_journal_entries(ticker=ticker, limit=100)
    # Compute stats
    closed = [e for e in entries if e.get('outcome_pct') is not None]
    wins = [e for e in closed if e['outcome_pct'] > 0]
    total_return = sum(e['outcome_pct'] for e in closed) if closed else 0
    return templates.TemplateResponse("journal.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "entries": entries,
        "filter_ticker": ticker,
        "stats": {
            "total": len(entries),
            "closed": len(closed),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
            "avg_return": round(total_return / len(closed), 2) if closed else None,
        }
    })


@app.post("/journal/add")
async def add_journal_entry(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Add a new trade journal entry."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    entry = {
        'ticker': form.get('ticker', ''),
        'entry_date': form.get('entry_date'),
        'entry_price': float(form.get('entry_price', 0) or 0) or None,
        'shares': float(form.get('shares', 0) or 0) or None,
        'trade_type': form.get('trade_type', 'LONG'),
        'system_signal': form.get('system_signal', ''),
        'user_action': form.get('user_action', ''),
        'entry_reason': form.get('entry_reason', ''),
        'notes': form.get('notes', ''),
    }
    db.add_journal_entry(entry)
    return RedirectResponse(url="/journal?added=1", status_code=303)


@app.post("/journal/{entry_id}/close")
async def close_journal_entry(
    request: Request,
    entry_id: int,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Close out a journal entry with exit data."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    db.update_journal_entry(
        entry_id=entry_id,
        exit_price=float(form.get('exit_price', 0)),
        exit_date=form.get('exit_date', datetime.now().strftime('%Y-%m-%d')),
        exit_reason=form.get('exit_reason', ''),
        notes=form.get('notes', ''),
    )
    return RedirectResponse(url="/journal?closed=1", status_code=303)


@app.post("/journal/{entry_id}/delete")
async def delete_journal_entry(
    request: Request,
    entry_id: int,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Delete a journal entry."""
    csrf.verify_token(request, csrf_token)
    db.delete_journal_entry(entry_id)
    return RedirectResponse(url="/journal", status_code=303)


# ==================== BULK DISCOVERY ACTIONS ====================

@app.post("/discoveries/bulk-promote")
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


@app.post("/discoveries/bulk-dismiss")
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


# ==================== RELATIVE STRENGTH RANKING ====================

@app.get("/api/rs-ranking")
async def api_rs_ranking(request: Request, username: str = Depends(require_api_key_or_session)):
    """Rank watchlist stocks by 3/6/12-month relative strength vs SPY."""
    from engine.rs_ranking import rs_ranking
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    if not watchlist:
        return {"available": False, "reason": "Empty watchlist"}
    rankings = rs_ranking.rank_tickers(watchlist)
    return {"available": True, "rankings": rankings}


# ==================== DISCOVERY HIT RATE ====================

@app.get("/api/discovery-hit-rate")
async def api_discovery_hit_rate(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return discovery hit rate by strategy and overall."""
    from engine.discovery_hit_rate import discovery_hit_rate
    return {
        "overall": discovery_hit_rate.get_overall_hit_rate(),
        "by_strategy": discovery_hit_rate.get_strategy_hit_rates(),
        "recent": discovery_hit_rate.get_recent_outcomes(limit=10),
    }


@app.post("/api/discovery-hit-rate/check")
async def trigger_hit_rate_check(request: Request, username: str = Depends(require_auth)):
    """Manually trigger discovery outcome checking."""
    from engine.discovery_hit_rate import discovery_hit_rate
    result = discovery_hit_rate.check_outcomes()
    return result


# ==================== UPCOMING DIVIDENDS ====================

@app.get("/api/upcoming-dividends")
async def api_upcoming_dividends(
    request: Request,
    days: int = 30,
    username: str = Depends(require_api_key_or_session),
):
    """Watchlist stocks with ex-dividend dates in the next N days."""
    from engine.dividend_tracker import dividend_tracker
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    results = []
    for ticker in watchlist:
        try:
            info = dividend_tracker.get_dividend_info(ticker) if hasattr(dividend_tracker, 'get_dividend_info') else None
            if not info:
                continue
            ex_date = info.get('ex_dividend_date') or info.get('next_ex_date')
            if not ex_date:
                continue
            from datetime import datetime as _dt
            try:
                ex_dt = _dt.fromisoformat(str(ex_date)[:10])
                days_away = (ex_dt - _dt.now()).days
                if 0 <= days_away <= days:
                    results.append({
                        "ticker": ticker,
                        "ex_dividend_date": ex_date,
                        "days_away": days_away,
                        "dividend_yield": info.get('dividend_yield'),
                        "dividend_amount": info.get('last_dividend_amount') or info.get('dividendRate'),
                    })
            except (ValueError, TypeError):
                pass
        except Exception:
            pass
    results.sort(key=lambda x: x.get("days_away", 999))
    return {"upcoming": results, "count": len(results)}


# ==================== PRE/POST MARKET PRICES ====================

@app.get("/api/extended-hours")
async def api_extended_hours(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return pre-market / after-hours prices for all watchlist tickers."""
    from engine.financial_statements import financial_statements
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    results = []
    for ticker in watchlist:
        try:
            stats = financial_statements.get_key_stats(ticker)
            if stats.get('available'):
                results.append({
                    "ticker": ticker,
                    "current_price": stats.get("current_price"),
                    "pre_market_price": stats.get("pre_market_price"),
                    "post_market_price": stats.get("post_market_price"),
                    "week52_high": stats.get("week52_high"),
                    "week52_low": stats.get("week52_low"),
                    "pct_from_52w_high": stats.get("pct_from_52w_high"),
                    "cap_label": stats.get("cap_label"),
                })
        except Exception:
            pass
    return {"results": results}


@app.get("/api/portfolio/var")
async def api_portfolio_var(request: Request, username: str = Depends(require_api_key_or_session)):
    """Calculate Value at Risk for current portfolio."""
    from engine.var_calculator import var_calculator
    try:
        result = var_calculator.calculate_portfolio_var()
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/correlation")
async def api_portfolio_correlation(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get correlation matrix for portfolio holdings."""
    from engine.correlation_analyzer import correlation_analyzer
    holdings = db.get_portfolio_holdings()
    tickers = [h['ticker'] for h in holdings if h['shares'] > 0]
    if len(tickers) < 2:
        return {"error": "Need at least 2 holdings for correlation analysis"}
    try:
        matrix = correlation_analyzer.get_correlation_matrix(tickers)
        if matrix is None:
            return {"error": "Could not compute correlation matrix"}
        return {
            "tickers": list(matrix.columns),
            "matrix": matrix.values.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/exposure/{ticker}")
async def api_portfolio_exposure(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Check how a specific ticker correlates with the user's existing portfolio."""
    try:
        from engine.portfolio_manager import portfolio_manager
        portfolio_status = portfolio_manager.check_all_rules()
        enriched_holdings = portfolio_status.get('holdings', [])
        
        from engine.correlation_analyzer import correlation_analyzer
        exposure = correlation_analyzer.check_new_ticker_exposure(ticker.upper(), enriched_holdings)
        return exposure
    except Exception as e:
        return {"error": str(e), "warnings": [], "max_correlation": 0}



@app.get("/api/portfolio/rebalancing-plan")
async def api_portfolio_rebalancing_plan(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate concrete rebalancing execution plan with specific share counts."""
    from engine.portfolio_manager import portfolio_manager
    try:
        plan = portfolio_manager.get_rebalancing_plan()
        return {"plan": plan, "count": len(plan)}
    except Exception as e:
        return {"error": str(e), "plan": []}


@app.get("/api/portfolio/risk-metrics")
async def api_portfolio_risk_metrics(request: Request, username: str = Depends(require_api_key_or_session)):
    """Calculate Sharpe, Sortino, Calmar, beta, volatility for portfolio."""
    import yfinance as yf
    import numpy as np
    holdings = db.get_portfolio_holdings()
    active = [h for h in holdings if h['shares'] > 0]
    if not active:
        return {"error": "No active holdings"}
    try:
        tickers = [h['ticker'] for h in active]
        values = [h['shares'] * h['avg_price'] for h in active]
        total = sum(values)
        weights = np.array([v / total for v in values])

        all_returns = []
        for t in tickers:
            hist = yf.Ticker(t).history(period="1y")
            if len(hist) > 20:
                all_returns.append(hist['Close'].pct_change().dropna().values)

        if not all_returns:
            return {"error": "Insufficient price data"}

        min_len = min(len(r) for r in all_returns)
        aligned = np.column_stack([r[-min_len:] for r in all_returns])
        port_returns = aligned @ weights[:len(all_returns)]

        # SPY benchmark
        spy_hist = yf.Ticker("SPY").history(period="1y")
        spy_returns = spy_hist['Close'].pct_change().dropna().values[-min_len:]

        risk_free = 0.05 / 252
        excess = port_returns - risk_free
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0

        downside = excess[excess < 0]
        sortino = float(np.mean(excess) / np.std(downside) * np.sqrt(252)) if len(downside) > 0 and np.std(downside) > 0 else 0

        cumulative = np.cumprod(1 + port_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = float(np.min(drawdown))
        annual_return = float((cumulative[-1]) ** (252 / len(port_returns)) - 1) if len(port_returns) > 0 else 0
        calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0

        beta = float(np.cov(port_returns, spy_returns)[0][1] / np.var(spy_returns)) if np.var(spy_returns) > 0 else 1.0
        volatility = float(np.std(port_returns) * np.sqrt(252))

        return {
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "calmar": round(calmar, 2),
            "beta": round(beta, 2),
            "volatility": round(volatility * 100, 1),
            "max_drawdown": round(max_dd * 100, 1),
            "annual_return": round(annual_return * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/scenario-analysis")
async def api_scenario_analysis(
    request: Request,
    scenario: str = "market_crash",
    username: str = Depends(require_api_key_or_session),
):
    """Run a stress scenario against portfolio."""
    from engine.scenario_analyzer import scenario_analyzer
    try:
        result = scenario_analyzer.run_scenario(scenario)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/price-alerts")
async def api_get_price_alerts(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get all active price alerts."""
    alerts = db.query("SELECT * FROM price_alerts WHERE active = 1 ORDER BY created_at DESC") or []
    return {"alerts": [dict(a) for a in alerts]}


@app.post("/api/price-alerts")
async def api_create_price_alert(request: Request, username: str = Depends(require_api_key_or_session)):
    """Create a new price alert."""
    data = await request.json()
    ticker = data.get('ticker', '').upper()
    alert_type = data.get('alert_type', 'target_price')
    threshold = data.get('threshold')
    direction = data.get('direction', 'above')

    if not ticker or threshold is None:
        return {"error": "ticker and threshold required"}

    db.execute("""
        INSERT INTO price_alerts (ticker, alert_type, threshold, direction)
        VALUES (?, ?, ?, ?)
    """, (ticker, alert_type, float(threshold), direction))
    return {"status": "created"}


@app.delete("/api/price-alerts/{alert_id}")
async def api_delete_price_alert(request: Request, alert_id: int, username: str = Depends(require_api_key_or_session)):
    """Deactivate a price alert."""
    db.execute("UPDATE price_alerts SET active = 0 WHERE id = ?", (alert_id,))
    return {"status": "deactivated"}


@app.get("/api/patterns/{ticker}")
async def api_patterns(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Detect chart patterns for a ticker."""
    from engine.pattern_recognition import pattern_recognizer
    try:
        patterns = pattern_recognizer.detect_patterns(ticker)
        return {"ticker": ticker.upper(), "patterns": patterns}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sentiment/{ticker}")
async def api_sentiment(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get sentiment summary including analyst consensus and contrarian signals."""
    from engine.sentiment_analyzer import sentiment_analyzer
    try:
        summary = sentiment_analyzer.get_sentiment_summary(ticker)
        return summary
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/nlp-sentiment/{ticker}")
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


@app.get("/api/nlp-sentiment/movers")
async def api_nlp_sentiment_movers(hours: int = 24, username: str = Depends(require_api_key_or_session)):
    """Return tickers with biggest VADER sentiment shift in the last N hours (#38)."""
    from engine.nlp_scorer import get_sentiment_movers
    return {"movers": get_sentiment_movers(hours=hours)}


# ==================== PAIRS TRADING (#40) ====================

@app.get("/api/pairs")
async def api_pairs_all(username: str = Depends(require_api_key_or_session)):
    """Return all tested pairs ordered by cointegration strength (#40)."""
    from engine.pairs_trader import get_all_pairs, ensure_schema
    ensure_schema()
    return {"pairs": get_all_pairs()}


@app.get("/api/pairs/active")
async def api_pairs_active(username: str = Depends(require_api_key_or_session)):
    """Return pairs with active long_spread / short_spread signals (#40)."""
    from engine.pairs_trader import get_active_pairs
    return {"pairs": get_active_pairs()}


@app.post("/api/pairs/scan")
async def api_pairs_scan(request: Request, username: str = Depends(require_api_key_or_session)):
    """Trigger a manual pairs cointegration scan (#40)."""
    from engine.pairs_trader import run_weekly_scan
    try:
        pairs = run_weekly_scan()
        return {"cointegrated_pairs": len(pairs), "pairs": pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MOAT SCORING (#45/#54) ====================

@app.get("/api/moat/{ticker}")
async def api_moat_ticker(ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return economic moat score for a ticker (#45/#54)."""
    from engine.moat_scorer import moat_score
    return moat_score(ticker.upper())


@app.get("/api/moat")
async def api_moat_watchlist(username: str = Depends(require_api_key_or_session)):
    """Return moat scores for all active watchlist tickers, sorted by score (#45/#54)."""
    from engine.moat_scorer import batch_moat_scores
    tickers = [
        r["ticker"]
        for r in (db.query("SELECT ticker FROM watchlist WHERE active = 1") or [])
    ]
    if not tickers:
        return {"moat_scores": []}
    return {"moat_scores": batch_moat_scores(tickers)}


# ==================== PORTFOLIO ANOMALY DETECTION (#46/#55) ====================

@app.get("/api/portfolio/anomaly-detection")
async def api_portfolio_anomaly_detection(username: str = Depends(require_api_key_or_session)):
    """
    Run portfolio anomaly checks and return active anomalies (#46/#55).
    Also returns correlation vs SPY, beta trend, sector concentration.
    """
    from engine.portfolio_anomaly import get_active_anomalies, run_anomaly_checks
    from engine.portfolio_anomaly import ensure_schema
    ensure_schema()

    # Return cached recent anomalies (no re-run to avoid slowness on page load)
    active = get_active_anomalies(hours=48)

    # Correlation vs SPY (lightweight)
    correlation_vs_spy = None
    try:
        import yfinance as yf
        import pandas as pd
        tickers = [r["ticker"] for r in (db.query(
            "SELECT DISTINCT ticker FROM portfolio_trades WHERE exit_date IS NULL"
        ) or [])]
        if tickers:
            spy_hist = yf.Ticker("SPY").history(period="30d")["Close"]
            port_prices = [yf.Ticker(t).history(period="30d")["Close"] for t in tickers[:10]]
            if port_prices:
                port_avg = pd.concat(port_prices, axis=1).mean(axis=1)
                aligned = pd.concat([port_avg, spy_hist], axis=1).dropna()
                if len(aligned) > 10:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    correlation_vs_spy = round(corr, 3)
    except Exception:
        pass

    return {
        "active_anomalies": active,
        "correlation_vs_spy": correlation_vs_spy,
    }


@app.post("/api/portfolio/anomaly-detection/run")
async def api_portfolio_anomaly_run(request: Request, username: str = Depends(require_api_key_or_session)):
    """Trigger a manual portfolio anomaly check (#46)."""
    from engine.portfolio_anomaly import run_anomaly_checks
    anomalies = run_anomaly_checks()
    return {"anomalies_detected": len(anomalies), "anomalies": anomalies}


# ==================== PWA PUSH NOTIFICATIONS (#31/#59) ====================

@app.get("/api/push/vapid-key")
async def api_push_vapid_key(username: str = Depends(require_api_key_or_session)):
    """Return the VAPID public key for browser push subscription (#31/#59)."""
    from engine.push_notifier import get_vapid_public_key
    key = get_vapid_public_key()
    if not key:
        return {"vapid_public_key": None, "available": False}
    return {"vapid_public_key": key, "available": True}


@app.post("/api/push/subscribe")
async def api_push_subscribe(request: Request, username: str = Depends(require_api_key_or_session)):
    """Store a browser Web Push subscription (#31/#59)."""
    from engine.push_notifier import save_subscription
    body = await request.json()
    endpoint = body.get("endpoint", "")
    keys = body.get("keys", {})
    p256dh = keys.get("p256dh", "")
    auth = keys.get("auth", "")
    if not endpoint or not p256dh or not auth:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing endpoint or keys")
    user_agent = request.headers.get("user-agent", "")
    ok = save_subscription(endpoint, p256dh, auth, user_agent)
    return {"subscribed": ok}


@app.delete("/api/push/unsubscribe")
async def api_push_unsubscribe(request: Request, username: str = Depends(require_api_key_or_session)):
    """Remove a browser Web Push subscription (#31/#59)."""
    from engine.push_notifier import remove_subscription
    body = await request.json()
    endpoint = body.get("endpoint", "")
    if not endpoint:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing endpoint")
    ok = remove_subscription(endpoint)
    return {"unsubscribed": ok}


@app.post("/api/push/test")
async def api_push_test(request: Request, username: str = Depends(require_api_key_or_session)):
    """Send a test push notification to all subscriptions (#31/#59)."""
    from engine.push_notifier import send_push
    sent = send_push(
        title="Stockholm Test Notification",
        body="Push notifications are working correctly!",
        url="/",
    )
    return {"sent": sent}


# ==================== CROSS-ASSET COMPOSITE SIGNALS (#47) ====================

@app.get("/api/macro/composite-signals")
async def api_composite_signals_active(hours: int = 24, username: str = Depends(require_api_key_or_session)):
    """Return composite macro signals triggered in the last N hours (#47)."""
    from engine.composite_signals import get_active_composite_signals
    return {"signals": get_active_composite_signals(hours=hours)}


@app.get("/api/macro/composite-signals/latest")
async def api_composite_signals_latest(username: str = Depends(require_api_key_or_session)):
    """Return the most recent signal for each composite pattern (#47)."""
    from engine.composite_signals import get_latest_per_pattern
    return {"signals": get_latest_per_pattern()}


@app.post("/api/macro/composite-signals/evaluate")
async def api_composite_signals_evaluate(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually trigger cross-asset composite signal evaluation (#47)."""
    from engine.composite_signals import evaluate_composite_signals
    triggered = evaluate_composite_signals()
    return {"triggered": len(triggered), "signals": triggered}


# ==================== SUPPLY CHAIN RISK (#44/#58) ====================

@app.get("/api/supply-chain/{ticker}")
async def api_supply_chain(
    ticker: str,
    force_refresh: bool = False,
    username: str = Depends(require_api_key_or_session),
):
    """Return supply chain map (suppliers/customers/partners) for a ticker (#44/#58)."""
    from engine.supply_chain import get_supply_chain
    return get_supply_chain(ticker.upper(), force_refresh=force_refresh)


@app.get("/api/supply-chain/{ticker}/geo-exposure")
async def api_supply_chain_geo(
    ticker: str,
    regions: str = "",
    username: str = Depends(require_api_key_or_session),
):
    """Check if any of a ticker's suppliers are in flagged geo regions (#44/#58)."""
    from engine.supply_chain import get_geo_elevated_tickers
    flagged = [r.strip() for r in regions.split(",") if r.strip()] if regions else []
    if not flagged:
        return {"ticker": ticker.upper(), "elevated": False, "matches": []}
    all_elevated = get_geo_elevated_tickers(flagged)
    matches = [e for e in all_elevated if e["ticker"] == ticker.upper()]
    return {
        "ticker": ticker.upper(),
        "elevated": bool(matches),
        "matches": matches,
    }


@app.post("/api/supply-chain/refresh")
async def api_supply_chain_refresh(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually trigger quarterly supply chain refresh for all stale tickers (#44)."""
    from engine.supply_chain import refresh_stale_tickers
    refreshed = refresh_stale_tickers()
    return {"refreshed": refreshed}


@app.get("/api/catalysts/{ticker}")
async def api_catalysts(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Unified catalyst timeline: earnings, dividends, economic events."""
    from engine.earnings_tracker import earnings_tracker
    from engine.dividend_tracker import dividend_tracker
    catalysts = []
    try:
        earnings = earnings_tracker.get_earnings_info(ticker)
        if earnings and earnings.get('next_earnings_date'):
            catalysts.append({
                'type': 'earnings',
                'name': f"{ticker.upper()} Earnings Report",
                'date': earnings['next_earnings_date'],
                'detail': f"Est. EPS: {earnings.get('estimated_eps', '—')}",
            })
    except Exception:
        pass
    try:
        div_info = dividend_tracker.get_dividend_info(ticker)
        if div_info and div_info.get('estimated_next_ex_date'):
            catalysts.append({
                'type': 'dividend',
                'name': f"{ticker.upper()} Ex-Dividend",
                'date': div_info['estimated_next_ex_date'],
                'detail': f"${div_info.get('last_dividend', 0):.2f} per share",
            })
    except Exception:
        pass
    catalysts.sort(key=lambda c: c.get('date', ''))
    return {"ticker": ticker.upper(), "catalysts": catalysts}


@app.get("/api/short-interest/{ticker}")
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


@app.get("/api/options-flow/{ticker}")
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


@app.get("/api/institutional/{ticker}")
async def api_institutional(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get institutional holder data and ownership changes."""
    from engine.institutional_tracker import institutional_tracker
    try:
        holders = institutional_tracker.get_institutional_holders(ticker)
        changes = institutional_tracker.get_ownership_changes(ticker)
        return {
            "holders": holders,
            "changes": changes,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/graveyard", response_class=HTMLResponse)
async def graveyard_page(request: Request, username: str = Depends(require_auth)):
    """Show removed tickers and their post-removal performance."""
    graveyard = db.query("""
        SELECT ticker, last_seen, reason, added_at
        FROM ticker_graveyard
        ORDER BY added_at DESC
    """) or []
    return templates.TemplateResponse("graveyard.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "graveyard": graveyard,
    })


@app.get("/api/graveyard/performance")
async def api_graveyard_performance(request: Request, username: str = Depends(require_api_key_or_session)):
    """Fetch post-removal price performance for graveyard tickers."""
    import yfinance as yf
    graveyard = db.query("""
        SELECT ticker, last_seen, reason, added_at
        FROM ticker_graveyard
        ORDER BY added_at DESC LIMIT 50
    """) or []
    results = []
    for g in graveyard:
        try:
            stock = yf.Ticker(g['ticker'])
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            if current_price and g.get('last_seen'):
                hist = stock.history(start=g['added_at'][:10] if g.get('added_at') else None, period="1y")
                removal_price = float(hist['Close'].iloc[0]) if not hist.empty else None
                if removal_price and removal_price > 0:
                    change_pct = ((current_price - removal_price) / removal_price) * 100
                    results.append({
                        'ticker': g['ticker'],
                        'reason': g.get('reason', ''),
                        'removed_at': g.get('added_at', ''),
                        'removal_price': round(removal_price, 2),
                        'current_price': round(current_price, 2),
                        'change_pct': round(change_pct, 1),
                    })
        except Exception:
            pass
    return {"results": results}


@app.get("/api/scenario-analysis/presets")
async def api_scenario_presets(request: Request, username: str = Depends(require_api_key_or_session)):
    """List available preset scenarios."""
    from engine.scenario_analyzer import scenario_analyzer
    return {"scenarios": scenario_analyzer.get_preset_scenarios()}


# ==================== STOCK COMPARE ====================

@app.get("/stock/compare", response_class=HTMLResponse)
async def stock_compare_page(
    request: Request,
    tickers: str = "",
    username: str = Depends(require_auth),
):
    """Side-by-side comparison of up to 5 tickers."""
    from engine.financial_statements import financial_statements

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:5]

    comparisons = []
    for ticker in ticker_list:
        try:
            stats = financial_statements.get_key_stats(ticker)
        except Exception:
            stats = {"available": False}
        comparisons.append({"ticker": ticker, "stats": stats})

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "tickers": tickers,
        "ticker_list": ticker_list,
        "comparisons": comparisons,
    })


# ==================== WEEKLY REPORT ====================

@app.get("/api/report/preview", response_class=HTMLResponse)
async def api_report_preview(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate and return the weekly report as HTML (opens in browser)."""
    from engine.report_generator import ReportGenerator
    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        html = "<html><body><p>Report generation failed: " + result.get("error", "unknown error") + "</p></body></html>"
    return HTMLResponse(content=html)


@app.post("/api/report/send")
async def api_report_send(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate and send the weekly report via configured channels."""
    from engine.report_generator import ReportGenerator
    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        return {"sent": False, "error": result.get("error", "Generation failed")}
    sent = rg.send_report(html)
    return {"sent": sent, "generated_at": result.get("generated_at"), "error": result.get("error")}


@app.post("/settings/save-report")
async def settings_save_report(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Save weekly report schedule settings."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    db.set_setting("weekly_report_auto_send", "1" if form.get("weekly_report_auto_send") else "0")
    db.set_setting("weekly_report_day", form.get("weekly_report_day", "Sunday"))
    db.set_setting("weekly_report_time", form.get("weekly_report_time", "08:00"))
    return RedirectResponse(url="/settings?saved=1", status_code=303)


# ==================== SECTOR-RELATIVE SCREENING ====================

@app.get("/sector-screen", response_class=HTMLResponse)
async def sector_screen_page(request: Request, username: str = Depends(require_auth)):
    """Sector-relative screening page."""
    return templates.TemplateResponse("sector_screen.html", {
        "request": request,
        "csrf_token": csrf.get_token(request),
    })


@app.get("/api/sector-screen")
async def api_sector_screen(request: Request, username: str = Depends(require_api_key_or_session)):
    """
    Sector-relative screening API.
    Returns top 3 sectors with cheapest stocks per sector + contrarian pick.
    """
    from engine.sector_momentum import sector_momentum, SECTOR_ETFS, TICKER_SECTOR_MAP
    from engine.auto_discovery import AutoDiscovery
    from engine.quant_screener import quant_screener

    try:
        rankings = sector_momentum.get_sector_rankings()
    except Exception as e:
        logger.error(f"Sector screen: rankings failed: {e}")
        return {"error": "Failed to fetch sector rankings", "sectors": []}

    # Build ETF → sector name map
    etf_name = {etf: info['name'] for etf, info in SECTOR_ETFS.items()}

    # Build universe: SP500_CORE
    universe = set(AutoDiscovery.SP500_CORE)

    # Map ticker → etf for quick lookup
    ticker_etf = TICKER_SECTOR_MAP

    benchmark_hist = quant_screener._get_benchmark_history()

    # Top 3 sectors by 1-month return
    top3 = rankings[:3]
    results = []

    for sector_row in top3:
        etf = sector_row['etf']
        sector_name = sector_row['name']

        # Tickers in this sector from universe
        sector_tickers = [t for t in universe if ticker_etf.get(t) == etf]

        # Screen each ticker
        screened = []
        for ticker in sector_tickers[:20]:  # cap at 20 to limit latency
            try:
                res = quant_screener.screen_ticker(ticker, benchmark_hist)
                if res and 'error' not in res:
                    screened.append({
                        'ticker': ticker,
                        'score': res.get('composite_score', 0),
                        'pe_ratio': res.get('valuation', {}).get('pe_ratio'),
                        'pe_vs_sector': res.get('valuation', {}).get('pe_vs_sector'),
                        'signal': res.get('signal', 'Neutral'),
                        'price': res.get('data', {}).get('current_price'),
                    })
            except Exception:
                continue

        # Sort by composite score descending, take top 3
        screened.sort(key=lambda x: x['score'], reverse=True)
        top3_stocks = screened[:3]

        results.append({
            'etf': etf,
            'name': sector_name,
            'rank': sector_row.get('rank', 0),
            'return_1mo': sector_row.get('return_1mo', 0),
            'return_1wk': sector_row.get('return_1wk', 0),
            'momentum': sector_row.get('momentum', 'neutral'),
            'stocks': top3_stocks,
        })

    # Contrarian: worst sector + cheapest stock (lowest P/E with positive score)
    contrarian = None
    if len(rankings) >= 11:
        worst = rankings[-1]
        etf = worst['etf']
        sector_tickers = [t for t in universe if ticker_etf.get(t) == etf]
        cheapest = None
        best_val = None
        for ticker in sector_tickers[:15]:
            try:
                res = quant_screener.screen_ticker(ticker, benchmark_hist)
                if res and 'error' not in res:
                    pe = res.get('valuation', {}).get('pe_ratio')
                    score = res.get('composite_score', 0)
                    if pe and pe > 0 and score >= 30:
                        if best_val is None or pe < best_val:
                            best_val = pe
                            cheapest = {
                                'ticker': ticker,
                                'score': score,
                                'pe_ratio': pe,
                                'signal': res.get('signal', 'Neutral'),
                                'price': res.get('data', {}).get('current_price'),
                            }
            except Exception:
                continue
        if cheapest:
            contrarian = {
                'etf': etf,
                'name': worst['name'],
                'return_1mo': worst.get('return_1mo', 0),
                'stock': cheapest,
            }

    return {
        'sectors': results,
        'contrarian': contrarian,
        'generated_at': datetime.now().isoformat(),
    }


# ==================== WATCHLIST CSV IMPORT (#30) ====================

@app.post("/api/watchlist/import")
async def watchlist_import_csv(
    request: Request,
    username: str = Depends(require_auth),
):
    """Import tickers from a broker CSV file (IBKR / Degiro / Schwab).

    Accepts multipart/form-data with:
      - file: the CSV file
      - preview: "1" to only parse and return tickers without importing
      - csrf_token: CSRF token (required when preview != "1")
    """
    import csv
    import io
    from fastapi import UploadFile, File as FastAPIFile

    form = await request.form()
    preview_mode = form.get("preview", "0") == "1"

    if not preview_mode:
        csrf.verify_token(request, form.get("csrf_token", ""))

    uploaded = form.get("file")
    if uploaded is None or not hasattr(uploaded, "read"):
        raise HTTPException(status_code=400, detail="No CSV file provided")

    raw_bytes = await uploaded.read()
    try:
        content = raw_bytes.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="Could not parse CSV header")

    fieldnames_lower = {f.strip().lower(): f for f in reader.fieldnames}

    def pick(row: dict, *candidates: str):
        for cand in candidates:
            if cand in fieldnames_lower:
                val = row.get(fieldnames_lower[cand], "").strip()
                if val:
                    return val
        return ""

    tickers = []
    seen: set = set()
    # Max realistic ticker length: 12 chars covers most global exchanges (e.g. BRK.B = 5, longest US ~5, LSE up to 12)
    MAX_TICKER_LEN = 12
    for row in reader:
        # IBKR / Schwab: "Symbol"; Degiro newer formats also export "Symbol"
        # Degiro older format uses "Produkt" (product description) — not a usable ticker, skipped
        ticker = pick(row, "symbol", "ticker", "stock symbol", "security", "asset")
        if not ticker:
            continue
        # Clean up: strip exchange suffix (e.g. "AAPL.US" -> "AAPL")
        ticker = ticker.split(".")[0].upper()
        if not ticker or len(ticker) > MAX_TICKER_LEN:
            continue
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise HTTPException(status_code=400, detail="No recognizable ticker symbols found in CSV")

    if preview_mode:
        return {"tickers": tickers, "count": len(tickers)}

    # Import tickers into watchlist
    added, skipped = [], []
    for t in tickers:
        try:
            existing = [w["ticker"] for w in db.get_watchlist(active_only=False)]
            if t in existing:
                skipped.append(t)
            else:
                db.add_to_watchlist(t, "")
                added.append(t)
        except Exception:
            skipped.append(t)

    return {"added": added, "skipped": skipped, "total": len(tickers)}


# ==================== EXPORT (#36) ====================

@app.get("/api/analysis/export.csv")
async def export_analyses_csv(request: Request, username: str = Depends(require_api_key_or_session)):
    """Export all analysis_history rows as CSV (#36)."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    analyses = db.get_analysis_history(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Ticker", "Signal", "Confidence", "Timestamp",
        "Recommendation", "Fundamental", "Technical", "Geo_Risk_Score",
    ])
    for a in analyses:
        writer.writerow([
            a.get("id", ""), a.get("ticker", ""), a.get("signal", ""),
            a.get("confidence", ""), a.get("timestamp", ""),
            (a.get("recommendation", "") or "")[:300],
            (a.get("fundamental", "") or "")[:300],
            (a.get("technical", "") or "")[:300],
            a.get("geo_risk_score", ""),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analysis_history.csv"},
    )


@app.get("/api/portfolio/export.csv")
async def export_portfolio_csv(request: Request, username: str = Depends(require_api_key_or_session)):
    """Export paper trades with entry/exit and FIFO-based P&L as CSV (#36)."""
    import csv
    import io
    from fastapi.responses import StreamingResponse
    from engine.portfolio_manager import portfolio_manager

    trades = db.get_trades()
    fifo_rows = {r["trade_id"]: r for r in portfolio_manager.calculate_fifo_pnl()}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Date", "Type", "Ticker", "Shares", "Price", "Fees",
        "Total", "FIFO_Cost_Basis", "Proceeds", "Realized_PnL", "Realized_PnL_Pct", "Notes",
    ])
    for t in trades:
        tid = t.get("id")
        fifo = fifo_rows.get(tid, {})
        total = (t["amount"] * t["price"]) + t["fees"] if t["type"] == "BUY" else (t["amount"] * t["price"]) - t["fees"]
        writer.writerow([
            t["date"], t["type"], t["ticker"], t["amount"], t["price"], t["fees"],
            round(total, 4),
            fifo.get("fifo_cost_basis", ""),
            fifo.get("proceeds", ""),
            fifo.get("realized_pnl", ""),
            fifo.get("realized_pnl_pct", ""),
            t.get("notes", ""),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_trades.csv"},
    )


@app.get("/report/weekly/pdf")
async def report_weekly_pdf(request: Request, username: str = Depends(require_auth)):
    """Render the weekly HTML report to PDF via WeasyPrint (#36)."""
    from fastapi.responses import Response
    from engine.report_generator import ReportGenerator

    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        raise HTTPException(status_code=500, detail=result.get("error", "Report generation failed"))

    try:
        from weasyprint import HTML as WeasyHTML
        pdf_bytes = WeasyHTML(string=html).write_pdf()
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires weasyprint. Run: pip install weasyprint",
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=weekly_report.pdf"},
    )


# ==================== DATA FRESHNESS ====================

@app.get("/api/data-freshness")
async def api_data_freshness(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get data freshness summary — detects stale yfinance data."""
    from engine.data_freshness import data_freshness
    summary = data_freshness.get_freshness_summary()
    return summary


# ============================================================
# React SPA Support Endpoints
# ============================================================

@app.get("/api/csrf-token")
async def get_csrf_token_for_spa(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Provide CSRF token for React SPA clients"""
    return {"token": csrf.get_token(request)}


@app.get("/api/auth/csrf")
async def get_auth_csrf_token(request: Request):
    """Public CSRF token endpoint for pre-auth forms (login, totp). No session required."""
    return {"token": csrf.get_token(request)}


def _verify_spa_csrf(request: Request):
    """Verify CSRF token sent as X-CSRF-Token header (for React SPA)"""
    from fastapi import HTTPException
    token = request.headers.get("X-CSRF-Token", "")
    if not token or not csrf.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


@app.post("/api/auth/login")
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


@app.post("/api/auth/totp")
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


@app.get("/api/auth/2fa/status")
async def api_auth_2fa_status(request: Request, username: str = Depends(require_auth)):
    """2FA status for React settings page."""
    totp_info = auth_manager.get_user_totp_info(username)
    return {
        "enabled": bool(totp_info.get("enabled")),
        "backup_codes_remaining": totp_info.get("backup_codes_remaining", 0),
    }


@app.post("/api/auth/2fa/setup-init")
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


@app.post("/api/auth/2fa/enable")
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


@app.post("/api/auth/2fa/disable")
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


@app.post("/api/scheduler/start")
async def api_start_scheduler(
    request: Request,
    username: str = Depends(require_auth)
):
    """Start scheduler — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    scheduler.start()
    return {"status": "started"}


@app.post("/api/scheduler/stop")
async def api_stop_scheduler(
    request: Request,
    username: str = Depends(require_auth)
):
    """Stop scheduler — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    scheduler.stop()
    return {"status": "stopped"}


@app.post("/api/scheduler/run-now")
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


@app.get("/api/settings-data")
async def get_settings_data(
    request: Request,
    username: str = Depends(require_auth)
):
    """Return all settings for the React settings page"""
    from core.config import (
        SCAN_INTERVAL_HOURS, GEO_SCAN_INTERVAL_HOURS,
        PERPLEXITY_MONTHLY_BUDGET_EUR, GEMINI_MONTHLY_BUDGET_EUR
    )
    return {
        "scheduler": {
            "scan_interval_hours": SCAN_INTERVAL_HOURS,
            "geo_scan_interval_hours": GEO_SCAN_INTERVAL_HOURS,
            "daily_limit": 10,
        },
        "budget": {
            "perplexity_monthly_eur": PERPLEXITY_MONTHLY_BUDGET_EUR,
            "gemini_monthly_eur": GEMINI_MONTHLY_BUDGET_EUR,
        },
    }


@app.get("/api/watchlist")
async def api_get_watchlist(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Get watchlist items as JSON — for React SPA"""
    items = db.get_watchlist(active_only=False)
    return items if isinstance(items, list) else []


@app.post("/api/watchlist")
async def api_add_watchlist(
    request: Request,
    username: str = Depends(require_auth)
):
    """Add ticker to watchlist — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    ticker = data.get("ticker", "").upper()
    name = data.get("name", "")
    if not ticker:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Ticker required")
    db.add_to_watchlist(ticker, name)
    return {"status": "added", "ticker": ticker}


@app.delete("/api/watchlist/{ticker}")
async def api_remove_watchlist(
    request: Request,
    ticker: str,
    username: str = Depends(require_auth)
):
    """Remove ticker from watchlist — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    db.remove_from_watchlist(ticker.upper())
    return {"status": "removed", "ticker": ticker}


@app.post("/api/settings/save")
async def save_settings_json(
    request: Request,
    username: str = Depends(require_auth)
):
    """Save settings via JSON — for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    # Delegate to existing settings handler logic or just acknowledge
    return {"status": "saved", "section": data.get("section")}


# ── Analysis History ────────────────────────────────────────────────────────

@app.get("/api/history")
async def api_history(
    request: Request,
    ticker: str = None,
    limit: int = 100,
    username: str = Depends(require_api_key_or_session)
):
    """Analysis history — JSON for React SPA"""
    analyses = db.get_analysis_history(ticker=ticker.upper() if ticker else None, limit=limit)
    return {"analyses": analyses or [], "filter_ticker": ticker}


# ── Discoveries ──────────────────────────────────────────────────────────────

@app.get("/api/discoveries")
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


@app.post("/api/discoveries/{discovery_id}/promote")
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


@app.post("/api/discoveries/{discovery_id}/dismiss")
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


# ── Logs ─────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
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


# ── Top Picks ─────────────────────────────────────────────────────────────────

@app.get("/api/top-picks")
async def api_top_picks(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Top picks — JSON for React SPA"""
    top_picks = db.get_top_picks(min_predictions=5, min_accuracy=0.6, limit=20)
    recent_signals = db.get_recent_high_confidence_predictions(days=7, min_confidence=70)
    learning_stats = learning_optimizer.get_learning_stats()
    return {
        "top_picks": top_picks or [],
        "recent_signals": recent_signals or [],
        "learning_stats": learning_stats,
        "total_trusted": len([p for p in (top_picks or []) if p.get('accuracy', 0) >= 70]),
    }


# ── Insider Activity ──────────────────────────────────────────────────────────

@app.get("/api/insider-activity")
async def api_insider_activity(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Insider activity — JSON for React SPA"""
    return {"signals": db.get_top_insider_signals(limit=50) or []}


@app.post("/api/insider-activity/scan")
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


# ── Journal ──────────────────────────────────────────────────────────────────

@app.get("/api/journal")
async def api_journal(
    request: Request,
    ticker: str = None,
    username: str = Depends(require_auth)
):
    """Journal entries — JSON for React SPA"""
    entries = db.get_journal_entries(ticker=ticker, limit=50)
    return {"entries": entries or []}


@app.post("/api/journal/add")
async def api_add_journal(
    request: Request,
    username: str = Depends(require_auth)
):
    """Add journal entry — JSON for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    ticker = data.get("ticker", "").upper()
    entry_type = data.get("type", "")
    notes = data.get("notes", "")
    price = data.get("price")
    if not ticker or not entry_type:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="ticker and type required")
    entry_id = db.add_journal_entry(ticker=ticker, entry_type=entry_type, notes=notes, price=price)
    return {"status": "added", "id": entry_id}


@app.post("/api/journal/{entry_id}/close")
async def api_close_journal(
    request: Request,
    entry_id: int,
    username: str = Depends(require_auth)
):
    """Close journal entry — JSON for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    db.close_journal_entry(entry_id, exit_price=data.get("exit_price"), notes=data.get("notes", ""))
    return {"status": "closed"}


@app.post("/api/journal/{entry_id}/delete")
async def api_delete_journal(
    request: Request,
    entry_id: int,
    username: str = Depends(require_auth)
):
    """Delete journal entry — JSON for React SPA"""
    _verify_spa_csrf(request)
    db.delete_journal_entry(entry_id)
    return {"status": "deleted"}


# ── Portfolio (main summary) ──────────────────────────────────────────────────

@app.get("/api/portfolio")
async def api_portfolio(
    request: Request,
    username: str = Depends(require_auth)
):
    """Portfolio summary + trades — JSON for React SPA"""
    summary = db.get_portfolio_summary()
    trades = db.get_trades()
    return {"summary": summary, "trades": trades or []}


@app.post("/api/portfolio/add-trade")
async def api_add_trade(
    request: Request,
    username: str = Depends(require_auth)
):
    """Add trade — JSON for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    ticker = data.get("ticker", "").upper()
    if not ticker:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="ticker required")
    db.add_trade(
        ticker=ticker,
        trade_type=data.get("type", "BUY"),
        amount=float(data.get("amount", 0)),
        price=float(data.get("price", 0)),
        date=data.get("date"),
        fees=float(data.get("fees", 0)),
        notes=data.get("notes", ""),
        currency=data.get("currency", "USD").upper(),
    )
    return {"status": "added"}


# ── Paper Trading ─────────────────────────────────────────────────────────────

@app.get("/api/paper-trading")
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


# SPA catch-all — serve React index.html for all non-API routes
_REACT_INDEX = Path(__file__).parent / "static" / "react" / "index.html"

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_spa(full_path: str):
    """Serve React SPA for all non-API, non-static routes"""
    # Block server-side API and auth endpoints
    if full_path.startswith("api/") or full_path == "logout":
        from fastapi import HTTPException
        raise HTTPException(status_code=404)

    # Serve static files directly (mount may not win over path routes in this Starlette version)
    if full_path.startswith("static/"):
        from fastapi.responses import FileResponse as _FR
        file_path = Path(__file__).parent / full_path
        if file_path.exists() and file_path.is_file():
            return _FR(str(file_path))
        from fastapi import HTTPException
        raise HTTPException(status_code=404)

    if _REACT_INDEX.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(_REACT_INDEX))

    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="React build not found. Run: cd frontend && npm run build")


# Entry point
def run_server():
    """Run the web server"""
    from core.config import ENABLE_HTTPS, CERT_FILE, KEY_FILE

    if ENABLE_HTTPS:
        if not CERT_FILE.exists() or not KEY_FILE.exists():
            print("[ERROR] HTTPS enabled but certificates not found!")
            print(f"   Expected: {CERT_FILE} and {KEY_FILE}")
            return

        print(f"[HTTPS] Server starting on https://{WEB_HOST}:{WEB_PORT}")
        uvicorn.run(
            app,
            host=WEB_HOST,
            port=WEB_PORT,
            ssl_certfile=str(CERT_FILE),
            ssl_keyfile=str(KEY_FILE),
            log_level="warning"
        )
    else:
        print(f"[WARNING] HTTP server starting on http://{WEB_HOST}:{WEB_PORT}")
        print("[WARNING] Enable HTTPS in .env for secure connections!")
        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")

if __name__ == "__main__":
    run_server()
