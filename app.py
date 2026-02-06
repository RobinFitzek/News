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
from core.budget_tracker import budget_tracker
from engine.learning_optimizer import learning_optimizer

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
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self';"
    )

    return response

# ==================== AUTHENTICATION ====================

def require_auth(request: Request) -> str:
    """Dependency to require authentication on routes"""
    username = auth_manager.get_current_user(request)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"}
        )
    return username

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    # If already logged in, redirect to dashboard
    if auth_manager.get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": request.query_params.get("error")
    })

@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    from core.config import ENABLE_HTTPS
    from slowapi.util import get_remote_address

    if db.verify_user(username, password):
        # Create session
        session_id = auth_manager.create_session(username)
        db.update_last_login(username)

        # Audit log successful login
        audit_log.log("login_success", username=username, ip=get_remote_address(request))

        # Redirect to dashboard with session cookie
        response = RedirectResponse(url="/", status_code=303)
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
    audit_log.log("login_failed", username=username, ip=get_remote_address(request))

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

# ==================== DASHBOARD ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, username: str = Depends(require_auth)):
    """Main dashboard"""
    try:
        # Get top 3 picks for preview widget
        top_picks_preview = db.get_top_picks(min_predictions=3, min_accuracy=0.65, limit=3)
        
        # Get trusted tickers for badge display
        trusted_tickers = set(db.get_trusted_tickers(min_accuracy=0.7))
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "csrf_token": request.state.csrf_token,
            "scheduler_status": scheduler.get_status(),
            "watchlist": db.get_watchlist(),
            "recent_analyses": db.get_analysis_history(limit=10),
            "api_status": {
                "perplexity": pplx_client.get_usage(),
                "gemini": gemini_client.get_usage()
            },
            "learning_stats": learning_optimizer.get_learning_stats(),
            "top_picks_preview": top_picks_preview,
            "trusted_tickers": trusted_tickers
        })
    except Exception as e:
        import traceback
        print(f"Dashboard error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

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

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "settings": db.get_all_settings(),
        "budget_status": budget_status,
        "api_keys": {
            "perplexity": bool(db.get_api_key("perplexity")),
            "gemini": bool(db.get_api_key("gemini"))
        }
    })

@app.post("/settings/save")
async def save_settings(request: Request, username: str = Depends(require_auth)):
    """Save settings"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    # Scheduler settings
    db.set_setting("scan_interval_hours", int(form.get("scan_interval_hours", 2)))
    db.set_setting("active_hours_start", form.get("active_hours_start", "08:00"))
    db.set_setting("active_hours_end", form.get("active_hours_end", "22:00"))
    
    # Email settings
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
    db.set_setting("include_news", form.get("include_news") == "on")
    db.set_setting("include_fundamental", form.get("include_fundamental") == "on")
    db.set_setting("include_technical", form.get("include_technical") == "on")
    db.set_setting("analysis_variant", form.get("analysis_variant", "balanced"))

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

    # Learning system settings
    try:
        db.set_setting("learning_verification_days", int(form.get("learning_verification_days", 90)))
    except (ValueError, TypeError):
        pass

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
    
    # Reload settings in services
    scheduler.reload_settings()
    notifications.reload_settings()
    
    return RedirectResponse(url="/settings?saved=1", status_code=303)

@app.post("/settings/api-keys")
async def save_api_keys(request: Request, username: str = Depends(require_auth)):
    """Save API keys"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    if form.get("perplexity_key"):
        db.set_api_key("perplexity", form.get("perplexity_key"))
        pplx_client.api_key = form.get("perplexity_key")
    
    if form.get("gemini_key"):
        db.set_api_key("gemini", form.get("gemini_key"))
        gemini_client.reload_api_key(form.get("gemini_key"))
    
    return RedirectResponse(url="/settings?saved=1", status_code=303)

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

@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
async def analysis_detail(request: Request, analysis_id: int, username: str = Depends(require_auth)):
    """Single analysis detail"""
    # Get analysis by ID
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return templates.TemplateResponse("analysis_detail.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "analysis": dict(row)
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
@limiter.limit("3/hour")
async def run_scan_now(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Trigger immediate scan"""
    csrf.verify_token(request, csrf_token)

    try:
        # Log the scan attempt
        audit_log.log_event("manual_scan_triggered", username, {"source": "web_dashboard"})

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
        scheduler.trigger_manual_scan()

        # Success message
        return RedirectResponse(url="/?message=success&detail=Scan+started+successfully", status_code=303)

    except Exception as e:
        # Log the error
        audit_log.log_event("manual_scan_failed", username, {"error": str(e)})
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
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "scheduler_logs": db.get_scheduler_logs(limit=50),
        "alerts": db.get_alerts(limit=50)
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
    
    return templates.TemplateResponse("learning.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "learning_stats": learning_stats,
        "recent_predictions": recent_predictions,
        "ticker_stats": ticker_stats
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

# ==================== API ENDPOINTS ====================

@app.get("/api/status")
async def api_status(request: Request, username: str = Depends(require_auth)):
    """API endpoint for status"""
    return {
        "scheduler": scheduler.get_status(),
        "api_usage": {
            "perplexity": pplx_client.get_usage(),
            "gemini": gemini_client.get_usage()
        },
        "watchlist_count": len(db.get_watchlist())
    }

@app.get("/api/budget")
async def api_budget_status(request: Request, username: str = Depends(require_auth)):
    """API endpoint for budget status (used by dashboard AJAX)"""
    return budget_tracker.get_budget_status()

@app.get("/api/portfolio/alerts")
async def api_portfolio_alerts(request: Request, username: str = Depends(require_auth)):
    """Portfolio rule checks: position sizing, stop-loss, sector concentration, benchmark."""
    from engine.portfolio_manager import portfolio_manager
    return portfolio_manager.check_all_rules()

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
