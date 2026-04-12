"""
Signal quality, calibration, A/B, drawdown, weights.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/signal-accuracy")
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


@router.get("/api/signal-ev")
async def api_signal_ev(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get Signal Expected Value - avg returns per signal type and confidence"""
    return learning_optimizer.feedback.calculate_signal_ev()


@router.get("/api/calibration")
async def api_calibration(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get calibration curve data: predicted confidence vs actual hit rate"""
    return learning_optimizer.feedback.calculate_calibration()


@router.get("/api/ab-comparison")
async def api_ab_comparison(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get quant-only vs quant+AI accuracy comparison"""
    return learning_optimizer.feedback.calculate_ab_comparison()


@router.get("/api/signal-decay")
async def api_signal_decay(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get signal accuracy at multiple time horizons (1d, 3d, 7d, 14d, 30d)"""
    return learning_optimizer.feedback.calculate_signal_decay()


@router.get("/api/weight-history")
async def api_weight_history(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get weight change audit trail"""
    return learning_optimizer.get_weight_history()


@router.post("/api/weight-rollback/{version_id}")
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


@router.get("/api/position-size/{ticker}")
async def api_position_size(ticker: str, confidence: int = 70, portfolio: float = 100000, request: Request = None, username: str = Depends(require_api_key_or_session)):
    """Get recommended position size for a ticker"""
    from engine.position_sizing import position_sizer
    return position_sizer.calculate_position_size(
        ticker=ticker.upper(),
        signal_confidence=confidence,
        portfolio_value=portfolio,
    )


@router.get("/api/statistical-significance")
async def api_statistical_significance(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get statistical significance of prediction accuracy"""
    return learning_optimizer.feedback.calculate_significance()


@router.get("/api/drawdown")
async def api_drawdown(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get drawdown analysis for paper trading"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_paper_trading_drawdown()


@router.get("/api/reality-check")
async def api_reality_check(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get comprehensive reality check dashboard"""
    from engine.drawdown_tracker import drawdown_tracker
    return drawdown_tracker.get_reality_dashboard()


@router.get("/api/signal-pnl")
async def api_signal_pnl(request: Request, username: str = Depends(require_api_key_or_session)):
    """Signal P&L scorecard — aggregated prediction outcome stats."""
    return db.get_signal_pnl_summary()


@router.get("/api/quant-screen")
async def api_quant_screen(request: Request, username: str = Depends(require_api_key_or_session)):
    """Run quant screener on watchlist — zero API cost."""
    from engine.quant_screener import quant_screener
    watchlist = db.get_watchlist(active_only=True)
    tickers = [item['ticker'] for item in watchlist]
    if not tickers:
        return {'results': [], 'message': 'Watchlist empty'}
    results = quant_screener.screen_batch(tickers)
    return {'results': results, 'count': len(results)}


@router.get("/api/discovery-hit-rate")
async def api_discovery_hit_rate(request: Request, username: str = Depends(require_api_key_or_session)):
    """Return discovery hit rate by strategy and overall."""
    from engine.discovery_hit_rate import discovery_hit_rate
    return {
        "overall": discovery_hit_rate.get_overall_hit_rate(),
        "by_strategy": discovery_hit_rate.get_strategy_hit_rates(),
        "recent": discovery_hit_rate.get_recent_outcomes(limit=10),
    }


@router.post("/api/discovery-hit-rate/check")
async def trigger_hit_rate_check(request: Request, username: str = Depends(require_auth)):
    """Manually trigger discovery outcome checking."""
    from engine.discovery_hit_rate import discovery_hit_rate
    result = discovery_hit_rate.check_outcomes()
    return result
