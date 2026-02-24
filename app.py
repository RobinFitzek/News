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
from datetime import datetime

from core.config import WEB_HOST, WEB_PORT, TEMPLATES_DIR
from core.database import db
from core.notifications import notifications
from core.auth import auth_manager
from core.csrf import csrf
from core.rate_limit import limiter
from core.audit_log import audit_log
from scheduler import scheduler
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from engine.agents import swarm
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
from clients.custom_provider_client import custom_provider_client
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    # If already logged in, redirect to dashboard or forced password change
    username = auth_manager.get_current_user(request)
    if username:
        if db.user_must_change_password(username):
            return RedirectResponse(url="/change-password", status_code=303)
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": request.query_params.get("error"),
        "show_default_credentials": db.user_must_change_password("admin"),
    })

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

        # Active system alerts (service errors)
        try:
            system_alerts = db.get_active_system_alerts()
        except Exception:
            system_alerts = []

        try:
            custom_provider_cards = db.get_enabled_api_provider_cards()
        except Exception:
            custom_provider_cards = []

        api_cards = []
        pplx_usage = pplx_client.get_usage()
        gemini_usage = gemini_client.get_usage()

        api_cards.append({
            "key": "perplexity",
            "name": "Perplexity",
            "is_configured": pplx_usage.get("is_configured", False),
            "used_today": pplx_usage.get("used_today", 0),
            "daily_limit": max(1, pplx_usage.get("daily_limit", 1)),
            "hint": "Requests Today",
        })
        api_cards.append({
            "key": "gemini-flash",
            "name": "Gemini Flash",
            "is_configured": gemini_usage.get("is_configured", False),
            "used_today": gemini_usage.get("flash", {}).get("used_today", 0),
            "daily_limit": max(1, gemini_usage.get("flash", {}).get("daily_limit", 1)),
            "hint": "Requests Today",
        })
        api_cards.append({
            "key": "gemini-pro",
            "name": "Gemini Pro",
            "is_configured": gemini_usage.get("is_configured", False),
            "used_today": gemini_usage.get("pro", {}).get("used_today", 0),
            "daily_limit": max(1, gemini_usage.get("pro", {}).get("daily_limit", 1)),
            "hint": "Requests Today",
        })

        for provider in custom_provider_cards:
            api_cards.append({
                "key": f"provider-{provider['id']}",
                "name": provider.get("name", "Custom Provider"),
                "is_configured": provider.get("is_configured", False),
                "used_today": provider.get("used_today", 0),
                "daily_limit": max(1, provider.get("daily_limit", 1)),
                "hint": provider.get("pipeline_role", "custom").replace('_', ' ').title() if provider.get("pipeline_role") else "Custom",
            })

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
            "system_alerts": system_alerts,
            "api_cards": api_cards,
        })
    except Exception as e:
        import traceback
        print(f"Dashboard error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

# ==================== SYSTEM HEALTH ====================

@app.get("/api/health")
async def api_health(username: str = Depends(require_auth)):
    """System health monitor endpoint for the dashboard widget"""
    from engine.health_monitor import health_monitor
    try:
        return health_monitor.get_full_health_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SIGNAL ACCURACY ====================

@app.get("/api/signal-accuracy")
async def api_signal_accuracy(username: str = Depends(require_auth)):
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
async def api_auto_paper_trading(username: str = Depends(require_auth)):
    """Provides automated paper trading tracking."""
    from engine.auto_paper_trader import auto_paper_trader
    try:
        return {
            "summary": auto_paper_trader.get_performance_summary(),
            "open_positions": auto_paper_trader.get_open_positions(),
            "should_trust": auto_paper_trader.should_trust_signals()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TRUTH BANNER ====================

_truth_banner_cache = {"data": None, "time": None}

@app.get("/api/truth-banner")
async def api_truth_banner(username: str = Depends(require_auth)):
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
async def watchlist_page(request: Request, username: str = Depends(require_auth)):
    """Watchlist management"""
    return templates.TemplateResponse("watchlist.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "watchlist": db.get_watchlist(active_only=False)
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
        "system_paused": system_paused,
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
        db.set_setting("email_smtp_port", int(form.get("email_smtp_port", 587)))
        db.set_setting("email_smtp_user", form.get("email_smtp_user", ""))
        if form.get("email_smtp_password"):
            db.set_setting("email_smtp_password", form.get("email_smtp_password"))
        db.set_setting("notify_on_strong_signals", form.get("notify_on_strong_signals") == "on")
        db.set_setting("daily_summary_enabled", form.get("daily_summary_enabled") == "on")
        db.set_setting("daily_summary_time", form.get("daily_summary_time", "20:00"))

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
    if save_all or section == "portfolio":
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


@app.get("/api/providers")
async def api_get_providers(request: Request, username: str = Depends(require_auth)):
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
async def api_create_provider(request: Request, username: str = Depends(require_auth)):
    """Create a custom OpenAI-compatible provider."""
    payload = await request.json()

    name = (payload.get('name') or '').strip()
    provider_type = (payload.get('provider_type') or 'llm').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    pipeline_role = (payload.get('pipeline_role') or '').strip() or None
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
    )
    if not provider_id:
        raise HTTPException(status_code=400, detail="Could not create provider (name may already exist)")

    return {"id": provider_id, "status": "ok"}


@app.put("/api/providers/{provider_id}")
async def api_update_provider(provider_id: int, request: Request, username: str = Depends(require_auth)):
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
    )

    if not ok:
        raise HTTPException(status_code=400, detail="Could not update provider")

    return {"id": provider_id, "status": "ok"}


@app.delete("/api/providers/{provider_id}")
async def api_delete_provider(provider_id: int, request: Request, username: str = Depends(require_auth)):
    """Delete a custom provider."""
    db.delete_api_provider(provider_id)
    return {"status": "ok"}


@app.post("/api/providers/{provider_id}/test")
async def api_test_provider(provider_id: int, request: Request, username: str = Depends(require_auth)):
    """Test provider connectivity with a lightweight completion call."""
    provider = db.get_api_provider(provider_id, include_secret=True)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    result = custom_provider_client.test_connection(provider)
    return result

# ==================== API KEY PEEK ====================

@app.get("/api/api-key/peek/{service}")
async def peek_api_key(service: str, request: Request, username: str = Depends(require_auth)):
    """Return masked API key (first 4 + last 4 chars visible)"""
    if service not in ('perplexity', 'gemini'):
        raise HTTPException(status_code=400, detail="Invalid service")
    key = db.get_api_key(service)
    if not key:
        return {"service": service, "masked": None, "configured": False}
    if len(key) <= 8:
        masked = key[:2] + '*' * (len(key) - 2)
    else:
        masked = key[:4] + '*' * (len(key) - 8) + key[-4:]
    return {"service": service, "masked": masked, "configured": True}

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

@app.get("/analysis/{analysis_id}")
async def analysis_detail(request: Request, analysis_id: int, username: str = Depends(require_auth)):
    """Redirect legacy analysis detail URL to unified stock detail page."""
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM analysis_history WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    ticker = row[0]
    return RedirectResponse(url=f"/stock/{ticker}?analysis_id={analysis_id}", status_code=303)

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

        # Check if APIs are configured
        pplx_key = db.get_api_key("perplexity")
        gemini_key = db.get_api_key("gemini")

        if not gemini_key:
            return RedirectResponse(url="/?message=error&detail=Gemini+API+not+configured", status_code=303)

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
async def api_discovery_stats(request: Request, username: str = Depends(require_auth)):
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
    enable_dev_mode_param = (request.query_params.get('enable_dev_mode') or '').strip().lower()
    if enable_dev_mode_param in ('1', 'true', 'yes', 'on') and not dev_mode:
        db.set_setting('development_mode', True)
        dev_mode = True
        audit_log.log(
            event_type="enable_dev_mode_via_logs_link",
            username=username,
            ip=request.client.host,
            details={"source": "settings_api_error_banner"}
        )

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
    username: str = Depends(require_auth)
):
    """Add a trade to portfolio"""
    csrf.verify_token(request, csrf_token)
    db.add_trade(
        ticker=ticker,
        trade_type=type,
        amount=amount,
        price=price,
        date=date,
        fees=fees,
        notes=notes
    )
    return RedirectResponse(url="/portfolio?added=1", status_code=303)

@app.get("/portfolio/export")
async def export_portfolio(request: Request, username: str = Depends(require_auth)):
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
    username: str = Depends(require_auth),
):
    """Start a backtest in a background thread."""
    csrf.verify_token(request, csrf_token)

    from engine.backtest_engine import backtest_engine

    progress = backtest_engine.get_progress()
    if progress.get('running'):
        return {"success": False, "error": "A backtest is already running"}

    months = max(6, min(60, months))

    import threading
    def _run():
        backtest_engine.run(tickers=None, months=months)

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
async def api_paper_trading_summary(request: Request, username: str = Depends(require_auth)):
    """Get paper trading portfolio summary"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_portfolio_summary()

@app.get("/api/paper-trading/equity-curve")
async def api_paper_trading_equity_curve(
    request: Request, 
    days: int = 30,
    username: str = Depends(require_auth)
):
    """Get equity curve data for charting"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_equity_curve(days_back=min(days, 365))

@app.get("/api/signal-ev")
async def api_signal_ev(request: Request, username: str = Depends(require_auth)):
    """Get Signal Expected Value - avg returns per signal type and confidence"""
    return learning_optimizer.feedback.calculate_signal_ev()

@app.get("/api/calibration")
async def api_calibration(request: Request, username: str = Depends(require_auth)):
    """Get calibration curve data: predicted confidence vs actual hit rate"""
    return learning_optimizer.feedback.calculate_calibration()

@app.get("/api/ab-comparison")
async def api_ab_comparison(request: Request, username: str = Depends(require_auth)):
    """Get quant-only vs quant+AI accuracy comparison"""
    return learning_optimizer.feedback.calculate_ab_comparison()

@app.get("/api/signal-decay")
async def api_signal_decay(request: Request, username: str = Depends(require_auth)):
    """Get signal accuracy at multiple time horizons (1d, 3d, 7d, 14d, 30d)"""
    return learning_optimizer.feedback.calculate_signal_decay()

@app.get("/api/weight-history")
async def api_weight_history(request: Request, username: str = Depends(require_auth)):
    """Get weight change audit trail"""
    return learning_optimizer.get_weight_history()

@app.post("/api/weight-rollback/{version_id}")
@limiter.limit("5/hour")
async def api_weight_rollback(request: Request, version_id: int, username: str = Depends(require_auth)):
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
async def api_paper_risk_metrics(request: Request, username: str = Depends(require_auth)):
    """Get risk-adjusted metrics for paper trading portfolio"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_risk_metrics()

@app.get("/api/paper-trading/spy-correlation")
async def api_paper_spy_correlation(request: Request, username: str = Depends(require_auth)):
    """Get portfolio beta and alpha vs SPY"""
    from engine.paper_trading import paper_trader
    return paper_trader.get_spy_correlation()

@app.get("/api/sector-momentum")
async def api_sector_momentum(request: Request, username: str = Depends(require_auth)):
    """Get sector momentum heat map data"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_heat_map_data()

@app.get("/api/sector-momentum/rotation")
async def api_sector_rotation(request: Request, username: str = Depends(require_auth)):
    """Get sector rotation signals"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_rotation_signals()

@app.get("/api/economic-calendar")
async def api_economic_calendar(request: Request, username: str = Depends(require_auth)):
    """Get upcoming market-moving events"""
    from engine.economic_calendar import economic_calendar
    return economic_calendar.get_calendar_summary()

@app.get("/api/multi-timeframe/{ticker}")
async def api_multi_timeframe(ticker: str, request: Request, username: str = Depends(require_auth)):
    """Get multi-timeframe analysis for a ticker"""
    from engine.multi_timeframe import multi_timeframe
    return multi_timeframe.analyze_ticker(ticker.upper())

@app.get("/api/position-size/{ticker}")
async def api_position_size(ticker: str, confidence: int = 70, portfolio: float = 100000, request: Request = None, username: str = Depends(require_auth)):
    """Get recommended position size for a ticker"""
    from engine.position_sizing import position_sizer
    return position_sizer.calculate_position_size(
        ticker=ticker.upper(),
        signal_confidence=confidence,
        portfolio_value=portfolio,
    )

@app.get("/api/statistical-significance")
async def api_statistical_significance(request: Request, username: str = Depends(require_auth)):
    """Get statistical significance of prediction accuracy"""
    return learning_optimizer.feedback.calculate_significance()

@app.get("/api/drawdown")
async def api_drawdown(request: Request, username: str = Depends(require_auth)):
    """Get drawdown analysis for paper trading"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_paper_trading_drawdown()

@app.get("/api/reality-check")
async def api_reality_check(request: Request, username: str = Depends(require_auth)):
    """Get comprehensive reality check dashboard"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_reality_dashboard()

# ==================== API ENDPOINTS ====================

@app.get("/api/status")
async def api_status(request: Request, username: str = Depends(require_auth)):
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
        "providers": db.get_enabled_api_provider_cards(),
        "watchlist_count": len(db.get_watchlist()),
        "stale_analyses": stale_count,
    }

@app.get("/api/scan-progress")
async def api_scan_progress(request: Request, username: str = Depends(require_auth)):
    """Real-time scan progress for dashboard status bar"""
    from engine.scan_progress import scan_progress
    return scan_progress.get_state()

@app.get("/api/budget")
async def api_budget_status(request: Request, username: str = Depends(require_auth)):
    """API endpoint for budget status (used by dashboard AJAX)"""
    return budget_tracker.get_budget_status()

@app.get("/api/portfolio/alerts")
async def api_portfolio_alerts(request: Request, username: str = Depends(require_auth)):
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
    username: str = Depends(require_auth)
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
async def api_signal_pnl(request: Request, username: str = Depends(require_auth)):
    """Signal P&L scorecard — aggregated prediction outcome stats."""
    return db.get_signal_pnl_summary()

@app.get("/api/quant-screen")
async def api_quant_screen(request: Request, username: str = Depends(require_auth)):
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
async def export_analyses(request: Request, format: str = "csv", username: str = Depends(require_auth)):
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
async def export_predictions(request: Request, format: str = "csv", username: str = Depends(require_auth)):
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
async def export_paper_trades(request: Request, format: str = "csv", username: str = Depends(require_auth)):
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
async def export_backtest(request: Request, run_id: int, format: str = "csv", username: str = Depends(require_auth)):
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
async def api_market_regime(request: Request, username: str = Depends(require_auth)):
    """Get current market regime (bull/bear/choppy) with VIX and yield data."""
    from engine.market_regime import market_regime
    return market_regime.get_current_regime()


@app.get("/api/regime-adjustments")
async def api_regime_adjustments(request: Request, username: str = Depends(require_auth)):
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

# ==================== PORTFOLIO BENCHMARK ====================

@app.get("/api/portfolio/benchmark")
async def api_portfolio_benchmark(request: Request, username: str = Depends(require_auth)):
    """Portfolio vs SPY benchmark comparison."""
    from engine.portfolio_benchmark import portfolio_benchmark
    return portfolio_benchmark.calculate_portfolio_vs_spy()

# ==================== CONCENTRATION CHECK ====================

@app.get("/api/portfolio/concentration")
async def api_portfolio_concentration(request: Request, username: str = Depends(require_auth)):
    """Check portfolio concentration and correlation risks."""
    from engine.concentration_checker import concentration_checker
    holdings = db.get_portfolio_holdings()
    return concentration_checker.check_portfolio_concentration(holdings)

# ==================== PRICE CHART DATA ====================

@app.get("/api/chart-data/{ticker}")
async def api_chart_data(request: Request, ticker: str, username: str = Depends(require_auth)):
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
async def api_algo_status(request: Request, username: str = Depends(require_auth)):
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
async def api_weight_suggestions(request: Request, username: str = Depends(require_auth)):
    """Get current vs suggested quant weights based on learning data."""
    return learning_optimizer.calculate_optimal_weights()

@app.post("/api/learning/apply-weights")
@limiter.limit("5/hour")
async def api_apply_weights(request: Request, username: str = Depends(require_auth)):
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
    
    # API connectivity
    perplexity_configured = bool(db.get_api_key("perplexity"))
    gemini_configured = bool(db.get_api_key("gemini"))
    
    # Disk space
    disk = psutil.disk_usage('/')
    disk_warning = disk.percent > 80
    
    # Learning system
    learning_stats = learning_optimizer.get_learning_stats()
    
    status = {
        "status": "healthy" if (db_healthy and perplexity_configured and gemini_configured and not disk_warning) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "database": {
                "healthy": db_healthy,
                "error": db_error if not db_healthy else None
            },
            "api_keys": {
                "perplexity": perplexity_configured,
                "gemini": gemini_configured
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

    selected_analysis = None
    crosscheck = None

    # Selected/latest analysis for merged stock+analysis view
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
            SELECT *
            FROM discovered_stocks
            WHERE ticker = ?
            ORDER BY found_at DESC LIMIT 10
        """, (ticker,))
    except Exception:
        discovery_history = []

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
async def api_earnings(ticker: str, request: Request, username: str = Depends(require_auth)):
    """Return earnings data including beat history for a ticker."""
    from engine.earnings_tracker import earnings_tracker
    t = ticker.upper()
    info = earnings_tracker.get_earnings_info(t)
    beat = earnings_tracker.get_beat_history(t)
    alert = earnings_tracker.generate_positioning_alert(t)
    return {"ticker": t, "earnings_info": info, "beat_history": beat, "alert_message": alert}


@app.get("/api/key-stats/{ticker}")
async def api_key_stats(ticker: str, request: Request, username: str = Depends(require_auth)):
    """Return 52w proximity, market cap label, short interest, pre/post market prices."""
    from engine.financial_statements import financial_statements
    return financial_statements.get_key_stats(ticker.upper())


@app.get("/api/financials/{ticker}")
async def api_financials(ticker: str, request: Request, username: str = Depends(require_auth)):
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
    username: str = Depends(require_auth),
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
    username: str = Depends(require_auth),
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
async def api_rs_ranking(request: Request, username: str = Depends(require_auth)):
    """Rank watchlist stocks by 3/6/12-month relative strength vs SPY."""
    from engine.rs_ranking import rs_ranking
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    if not watchlist:
        return {"available": False, "reason": "Empty watchlist"}
    rankings = rs_ranking.rank_tickers(watchlist)
    return {"available": True, "rankings": rankings}


# ==================== DISCOVERY HIT RATE ====================

@app.get("/api/discovery-hit-rate")
async def api_discovery_hit_rate(request: Request, username: str = Depends(require_auth)):
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
    username: str = Depends(require_auth),
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
async def api_extended_hours(request: Request, username: str = Depends(require_auth)):
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
async def api_portfolio_var(request: Request, username: str = Depends(require_auth)):
    """Calculate Value at Risk for current portfolio."""
    from engine.var_calculator import var_calculator
    try:
        result = var_calculator.calculate_portfolio_var()
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/correlation")
async def api_portfolio_correlation(request: Request, username: str = Depends(require_auth)):
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
async def api_portfolio_exposure(ticker: str, request: Request, username: str = Depends(require_auth)):
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
async def api_portfolio_rebalancing_plan(request: Request, username: str = Depends(require_auth)):
    """Generate concrete rebalancing execution plan with specific share counts."""
    from engine.portfolio_manager import portfolio_manager
    try:
        plan = portfolio_manager.get_rebalancing_plan()
        return {"plan": plan, "count": len(plan)}
    except Exception as e:
        return {"error": str(e), "plan": []}


@app.get("/api/portfolio/risk-metrics")
async def api_portfolio_risk_metrics(request: Request, username: str = Depends(require_auth)):
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
    username: str = Depends(require_auth),
):
    """Run a stress scenario against portfolio."""
    from engine.scenario_analyzer import scenario_analyzer
    try:
        result = scenario_analyzer.run_scenario(scenario)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/price-alerts")
async def api_get_price_alerts(request: Request, username: str = Depends(require_auth)):
    """Get all active price alerts."""
    alerts = db.query("SELECT * FROM price_alerts WHERE active = 1 ORDER BY created_at DESC") or []
    return {"alerts": [dict(a) for a in alerts]}


@app.post("/api/price-alerts")
async def api_create_price_alert(request: Request, username: str = Depends(require_auth)):
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
async def api_delete_price_alert(request: Request, alert_id: int, username: str = Depends(require_auth)):
    """Deactivate a price alert."""
    db.execute("UPDATE price_alerts SET active = 0 WHERE id = ?", (alert_id,))
    return {"status": "deactivated"}


@app.get("/api/patterns/{ticker}")
async def api_patterns(request: Request, ticker: str, username: str = Depends(require_auth)):
    """Detect chart patterns for a ticker."""
    from engine.pattern_recognition import pattern_recognizer
    try:
        patterns = pattern_recognizer.detect_patterns(ticker)
        return {"ticker": ticker.upper(), "patterns": patterns}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sentiment/{ticker}")
async def api_sentiment(request: Request, ticker: str, username: str = Depends(require_auth)):
    """Get sentiment summary including analyst consensus and contrarian signals."""
    from engine.sentiment_analyzer import sentiment_analyzer
    try:
        summary = sentiment_analyzer.get_sentiment_summary(ticker)
        return summary
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/catalysts/{ticker}")
async def api_catalysts(request: Request, ticker: str, username: str = Depends(require_auth)):
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
async def api_short_interest(request: Request, ticker: str, username: str = Depends(require_auth)):
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
async def api_options_flow(request: Request, ticker: str, username: str = Depends(require_auth)):
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
async def api_institutional(request: Request, ticker: str, username: str = Depends(require_auth)):
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
async def api_graveyard_performance(request: Request, username: str = Depends(require_auth)):
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
async def api_scenario_presets(request: Request, username: str = Depends(require_auth)):
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
async def api_report_preview(request: Request, username: str = Depends(require_auth)):
    """Generate and return the weekly report as HTML (opens in browser)."""
    from engine.report_generator import ReportGenerator
    rg = ReportGenerator()
    result = rg.generate_weekly_report()
    html = result.get("html_content", "")
    if not html:
        html = "<html><body><p>Report generation failed: " + result.get("error", "unknown error") + "</p></body></html>"
    return HTMLResponse(content=html)


@app.post("/api/report/send")
async def api_report_send(request: Request, username: str = Depends(require_auth)):
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
async def api_sector_screen(request: Request, username: str = Depends(require_auth)):
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


# ==================== DATA FRESHNESS ====================

@app.get("/api/data-freshness")
async def api_data_freshness(request: Request, username: str = Depends(require_auth)):
    """Get data freshness summary — detects stale yfinance data."""
    from engine.data_freshness import data_freshness
    summary = data_freshness.get_freshness_summary()
    return summary


# Startup: check API keys and raise alerts for missing/broken ones
def _check_api_keys_on_startup():
    """Raise system alerts for missing API keys at startup."""
    if not db.get_api_key('perplexity'):
        db.raise_system_alert(
            'perplexity_auth',
            'Perplexity API Key Missing',
            'No Perplexity API key configured. News analysis will be unavailable.',
            severity='warning', service='perplexity',
            action_url='/settings', action_label='Add API Key')
    if not db.get_api_key('gemini'):
        db.raise_system_alert(
            'gemini_auth',
            'Gemini API Key Missing',
            'No Gemini API key configured. AI analysis will not work.',
            severity='error', service='gemini',
            action_url='/settings', action_label='Add API Key')

_check_api_keys_on_startup()


# Entry point
def run_server():
    """Run the web server"""
    from core.config import ENABLE_HTTPS, CERT_FILE, KEY_FILE

    if ENABLE_HTTPS:
        if not CERT_FILE.exists() or not KEY_FILE.exists():
            print("❌ HTTPS enabled but certificates not found!")
            print(f"   Expected: {CERT_FILE} and {KEY_FILE}")
            return

        print(f"🔒 HTTPS server starting on https://{WEB_HOST}:{WEB_PORT}")
        uvicorn.run(
            app,
            host=WEB_HOST,
            port=WEB_PORT,
            ssl_certfile=str(CERT_FILE),
            ssl_keyfile=str(KEY_FILE),
            log_level="warning"
        )
    else:
        print(f"⚠️  HTTP server starting on http://{WEB_HOST}:{WEB_PORT}")
        print("⚠️  Enable HTTPS in .env for secure connections!")
        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")

if __name__ == "__main__":
    run_server()
