"""
Analyze, crosscheck, learning, top picks, history pages.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, ticker: str = None, username: str = Depends(require_auth)):
    """Analysis history"""
    return templates.TemplateResponse("history.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "analyses": db.get_analysis_history(ticker=ticker, limit=100),
        "filter_ticker": ticker
    })


@router.get("/analysis/{analysis_id}")
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


@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, username: str = Depends(require_auth)):
    """Manual analysis page"""
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "csrf_token": request.state.csrf_token
    })


@router.post("/analyze")
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


@router.get("/architecture", response_class=HTMLResponse)
async def architecture_page(request: Request, username: str = Depends(require_auth)):
    """System architecture visualization page"""
    return templates.TemplateResponse("architecture.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
    })


@router.get("/learning", response_class=HTMLResponse)
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


@router.get("/top-picks", response_class=HTMLResponse)
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


@router.get("/crosscheck", response_class=HTMLResponse)
async def crosscheck_page(request: Request, username: str = Depends(require_auth)):
    """Cross-check history page"""
    history = db.get_crosscheck_history(limit=50)
    return templates.TemplateResponse("crosscheck.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "history": history
    })


@router.post("/api/crosscheck/{analysis_id}")
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


@router.get("/api/crosscheck/history")
async def crosscheck_history(
    request: Request,
    ticker: str = None,
    username: str = Depends(require_auth),
):
    """Get cross-check history."""
    return db.get_crosscheck_history(ticker=ticker, limit=50)


@router.get("/api/algo-status")
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


@router.get("/api/learning/weight-suggestions")
async def api_weight_suggestions(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current vs suggested quant weights based on learning data."""
    return learning_optimizer.calculate_optimal_weights()


@router.get("/api/learning/feature-importance")
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


@router.post("/api/learning/apply-weights")
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


@router.get("/api/history")
async def api_history(
    request: Request,
    ticker: str = None,
    limit: int = 100,
    username: str = Depends(require_api_key_or_session)
):
    """Analysis history — JSON for React SPA"""
    analyses = db.get_analysis_history(ticker=ticker.upper() if ticker else None, limit=limit)
    return {"analyses": analyses or [], "filter_ticker": ticker}


@router.get("/api/top-picks")
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
