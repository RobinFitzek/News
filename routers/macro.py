"""
Macro / geopolitical / fear-greed / market-regime / dividends.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/geopolitical")
async def api_geopolitical(username: str = Depends(require_api_key_or_session)):
    """Return the latest geopolitical scan (max 24h old)"""
    try:
        scan = db.get_latest_geopolitical_scan()
        return {"scan": scan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/macro/events")
async def api_macro_events(days: int = 14, username: str = Depends(require_api_key_or_session)):
    from engine.macro_tracker import macro_tracker
    return {"events": macro_tracker.get_upcoming_events(days_ahead=days)}


@router.get("/api/geopolitical/exposure")
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


@router.get("/geo-history", response_class=HTMLResponse)
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


@router.get("/macro", response_class=HTMLResponse)
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


@router.get("/corporate-actions", response_class=HTMLResponse)
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


@router.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page(request: Request, username: str = Depends(require_auth)):
    """Geopolitical scenario stress-test overview (#39)."""
    from engine.geo_scenario import geo_scenarios
    return templates.TemplateResponse("scenarios.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "scenarios": geo_scenarios.get_all_scenarios(),
    })


@router.get("/api/scenarios")
async def api_list_scenarios(request: Request, username: str = Depends(require_api_key_or_session)):
    """List all available geopolitical scenarios."""
    from engine.geo_scenario import geo_scenarios
    return {"scenarios": geo_scenarios.get_all_scenarios()}


@router.post("/api/scenarios/run")
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


@router.get("/api/macro/snapshot")
async def api_macro_snapshot(username: str = Depends(require_api_key_or_session)):
    """Return the latest macro snapshot and 90-day history (#22)."""
    from engine.macro_tracker import macro_tracker
    return {
        "latest": macro_tracker.get_latest_snapshot(),
        "history": macro_tracker.get_macro_snapshots(90),
        "events": macro_tracker.get_upcoming_events(30),
    }


@router.get("/api/economic-calendar")
async def api_economic_calendar(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get upcoming market-moving events"""
    from engine.economic_calendar import economic_calendar
    return economic_calendar.get_calendar_summary()


@router.get("/api/market-regime")
async def api_market_regime(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current market regime (bull/bear/choppy) with VIX and yield data."""
    from engine.market_regime import market_regime
    return market_regime.get_current_regime()


@router.get("/api/regime-adjustments")
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


@router.get("/api/corporate-actions")
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


@router.get("/api/upcoming-dividends")
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


@router.get("/api/extended-hours")
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


@router.get("/api/scenario-analysis")
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


@router.get("/api/macro/composite-signals")
async def api_composite_signals_active(hours: int = 24, username: str = Depends(require_api_key_or_session)):
    """Return composite macro signals triggered in the last N hours (#47)."""
    from engine.composite_signals import get_active_composite_signals
    return {"signals": get_active_composite_signals(hours=hours)}


@router.get("/api/macro/composite-signals/latest")
async def api_composite_signals_latest(username: str = Depends(require_api_key_or_session)):
    """Return the most recent signal for each composite pattern (#47)."""
    from engine.composite_signals import get_latest_per_pattern
    return {"signals": get_latest_per_pattern()}


@router.post("/api/macro/composite-signals/evaluate")
async def api_composite_signals_evaluate(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually trigger cross-asset composite signal evaluation (#47)."""
    from engine.composite_signals import evaluate_composite_signals
    triggered = evaluate_composite_signals()
    return {"triggered": len(triggered), "signals": triggered}


@router.get("/api/scenario-analysis/presets")
async def api_scenario_presets(request: Request, username: str = Depends(require_api_key_or_session)):
    """List available preset scenarios."""
    from engine.scenario_analyzer import scenario_analyzer
    return {"scenarios": scenario_analyzer.get_preset_scenarios()}


@router.get("/api/fear-greed/current")
async def api_fear_greed_current(request: Request, username: str = Depends(require_auth)):
    """Current CNN Fear & Greed index value."""
    from engine.fear_greed_tracker import fear_greed_tracker
    value = fear_greed_tracker.get_current_fear_greed()
    label = fear_greed_tracker.get_fg_label(value) if value is not None else None
    vix = fear_greed_tracker.get_latest_vix_features()
    return {"fg_value": value, "fg_label": label, **vix, "fetched_at": datetime.now().isoformat()}


@router.get("/api/fear-greed/history")
async def api_fear_greed_history(request: Request, username: str = Depends(require_auth)):
    """Historical Fear & Greed data (from 2011)."""
    from engine.fear_greed_tracker import fear_greed_tracker
    df = fear_greed_tracker.fetch_fear_greed_history()
    if df.empty:
        return {"data": [], "error": "Could not fetch Fear & Greed history"}
    df["date"] = df["date"].astype(str)
    return {"data": df.to_dict(orient="records"), "count": len(df)}


@router.get("/api/fear-greed/sensitivity/{ticker}")
async def api_fg_sensitivity(
    ticker: str,
    request: Request,
    lookback: int = 60,
    username: str = Depends(require_auth)
):
    """60-day rolling F&G sensitivity factor for a ticker (positive=risk-on, negative=defensive)."""
    from engine.fear_greed_tracker import fear_greed_tracker
    factor = fear_greed_tracker.get_fg_sensitivity_factor(ticker.upper(), lookback_days=lookback)
    return {
        "ticker": ticker.upper(),
        "fg_sensitivity": factor,
        "lookback_days": lookback,
        "interpretation": (
            "Risk-on (correlated with greed)" if factor and factor > 0.3
            else "Defensive (anti-correlated with fear)" if factor and factor < -0.3
            else "Neutral" if factor is not None else "Insufficient data"
        )
    }


@router.get("/api/fear-greed/features/{ticker}")
async def api_fg_features(
    ticker: str,
    request: Request,
    username: str = Depends(require_auth)
):
    """All F&G + VIX features for a ticker (input snapshot for LSTM model)."""
    from engine.fear_greed_tracker import fear_greed_tracker
    features = fear_greed_tracker.get_features_for_ticker(ticker.upper())
    return {"ticker": ticker.upper(), **features}
