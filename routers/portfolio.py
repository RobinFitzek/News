"""
Portfolio, journal, graveyard — positions, notes, metrics.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/portfolio", response_class=HTMLResponse)
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


@router.post("/portfolio/add-trade")
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


@router.get("/portfolio/export")
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


@router.get("/api/portfolio/alerts")
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


@router.post("/api/portfolio/alerts/ack")
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


@router.post("/api/portfolio/ask")
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

    # Audit log for training (#56)
    try:
        audit_log.log(
            "portfolio_question_asked",
            username=username,
            ip=request.client.host if request.client else "unknown",
            details={"question": question, "rate_limited": result.get("rate_limited", False)},
        )
    except Exception:
        pass

    return result


@router.get("/api/portfolio/benchmark")
async def api_portfolio_benchmark(request: Request, username: str = Depends(require_api_key_or_session)):
    """Portfolio vs SPY benchmark comparison."""
    from engine.portfolio_benchmark import portfolio_benchmark
    return portfolio_benchmark.calculate_portfolio_vs_spy()


@router.get("/api/portfolio/concentration")
async def api_portfolio_concentration(request: Request, username: str = Depends(require_api_key_or_session)):
    """Check portfolio concentration and correlation risks."""
    from engine.concentration_checker import concentration_checker
    holdings = db.get_portfolio_holdings()
    return concentration_checker.check_portfolio_concentration(holdings)


@router.get("/journal", response_class=HTMLResponse)
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


@router.post("/journal/add")
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


@router.post("/journal/{entry_id}/close")
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


@router.post("/journal/{entry_id}/delete")
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


@router.get("/api/portfolio/var")
async def api_portfolio_var(request: Request, username: str = Depends(require_api_key_or_session)):
    """Calculate Value at Risk for current portfolio."""
    from engine.var_calculator import var_calculator
    try:
        result = var_calculator.calculate_portfolio_var()
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/portfolio/correlation")
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


@router.get("/api/portfolio/exposure/{ticker}")
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


@router.get("/api/portfolio/rebalancing-plan")
async def api_portfolio_rebalancing_plan(request: Request, username: str = Depends(require_api_key_or_session)):
    """Generate concrete rebalancing execution plan with specific share counts."""
    from engine.portfolio_manager import portfolio_manager
    try:
        plan = portfolio_manager.get_rebalancing_plan()
        return {"plan": plan, "count": len(plan)}
    except Exception as e:
        return {"error": str(e), "plan": []}


@router.get("/api/portfolio/risk-metrics")
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


@router.get("/api/portfolio/anomaly-detection")
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


@router.post("/api/portfolio/anomaly-detection/run")
async def api_portfolio_anomaly_run(request: Request, username: str = Depends(require_api_key_or_session)):
    """Trigger a manual portfolio anomaly check (#46)."""
    from engine.portfolio_anomaly import run_anomaly_checks
    anomalies = run_anomaly_checks()
    return {"anomalies_detected": len(anomalies), "anomalies": anomalies}


@router.get("/graveyard", response_class=HTMLResponse)
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


@router.get("/api/graveyard/performance")
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


@router.get("/api/portfolio/export.csv")
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


@router.get("/api/journal")
async def api_journal(
    request: Request,
    ticker: str = None,
    username: str = Depends(require_auth)
):
    """Journal entries — JSON for React SPA"""
    entries = db.get_journal_entries(ticker=ticker, limit=50)
    return {"entries": entries or []}


@router.post("/api/journal/add")
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


@router.post("/api/journal/{entry_id}/close")
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


@router.post("/api/journal/{entry_id}/delete")
async def api_delete_journal(
    request: Request,
    entry_id: int,
    username: str = Depends(require_auth)
):
    """Delete journal entry — JSON for React SPA"""
    _verify_spa_csrf(request)
    db.delete_journal_entry(entry_id)
    return {"status": "deleted"}


@router.get("/api/portfolio")
async def api_portfolio(
    request: Request,
    username: str = Depends(require_auth)
):
    """Portfolio summary + trades — JSON for React SPA"""
    summary = db.get_portfolio_summary()
    trades = db.get_trades()
    return {"summary": summary, "trades": trades or []}


@router.post("/api/portfolio/add-trade")
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
