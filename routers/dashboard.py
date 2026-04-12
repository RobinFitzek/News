"""
Dashboard, health, status, trust, budget, truth banner.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
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


@router.get("/api/health")
async def api_health(username: str = Depends(require_api_key_or_session)):
    """System health monitor endpoint for the dashboard widget"""
    from engine.health_monitor import health_monitor
    try:
        return health_monitor.get_full_health_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_truth_banner_cache = {"data": None, "time": None}


@router.get("/api/truth-banner")
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


@router.get("/trust", response_class=HTMLResponse)
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


@router.get("/api/server/sleep-status")
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


@router.get("/api/status")
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


@router.get("/api/scan-progress")
async def api_scan_progress(request: Request, username: str = Depends(require_api_key_or_session)):
    """Real-time scan progress for dashboard status bar"""
    from engine.scan_progress import scan_progress
    return scan_progress.get_state()


@router.get("/api/discovery-status")
async def api_discovery_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """Real-time discovery run status for discoveries page polling"""
    from engine.auto_discovery import discovery_status
    return discovery_status.get()


@router.get("/api/ollama/health")
async def api_ollama_health(request: Request, username: str = Depends(require_api_key_or_session)):
    """Health check for local Ollama server (item #41)."""
    from clients.ollama_client import ollama_client
    available = ollama_client.health_check()
    models = ollama_client.list_models() if available else []
    return {"available": available, "models": models}


@router.get("/api/budget")
async def api_budget_status(request: Request, username: str = Depends(require_api_key_or_session)):
    """API endpoint for budget status (used by dashboard AJAX)"""
    return budget_tracker.get_budget_status()


@router.get("/api/budget/status")
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


@router.get("/health")
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


@router.get("/api/data-freshness")
async def api_data_freshness(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get data freshness summary — detects stale yfinance data."""
    from engine.data_freshness import data_freshness
    summary = data_freshness.get_freshness_summary()
    return summary
