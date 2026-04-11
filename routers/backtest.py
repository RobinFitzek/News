"""
Backtest runs, progress, results, weight application.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/backtest", response_class=HTMLResponse)
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


@router.post("/backtest/run")
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


@router.get("/api/backtest/progress")
async def backtest_progress(request: Request, username: str = Depends(require_auth)):
    """Poll backtest progress."""
    from engine.backtest_engine import backtest_engine
    return backtest_engine.get_progress()


@router.post("/api/backtest/apply-weights/{run_id}")
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


@router.get("/api/backtest/results/{run_id}")
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


@router.post("/api/backtest/random-baseline")
async def api_backtest_random_baseline(
    request: Request,
    username: str = Depends(require_auth)
):
    """
    Run 500-simulation random portfolio baseline against latest backtest results.
    POST body (optional): {"n_simulations": 500, "portfolio_size": 20}
    """
    from engine.backtest_engine import BacktestEngine
    try:
        body = await request.json()
    except Exception:
        body = {}
    n_sim = int(body.get("n_simulations", 500))
    port_size = int(body.get("portfolio_size", 20))

    # Fetch latest backtest results from DB
    rows = db.query("SELECT * FROM backtest_results ORDER BY test_date DESC LIMIT 5000")
    if not rows:
        return {"error": "No backtest results found. Run a backtest first."}

    engine = BacktestEngine()
    # Convert DB rows to expected format
    results = [dict(r) for r in rows]
    return engine.run_random_baseline(results, n_simulations=n_sim, portfolio_size=port_size)
