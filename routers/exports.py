"""
CSV/JSON exports — analyses, predictions, trades, backtests.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/api/export/auto-trades")
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


@router.get("/api/export/analyses")
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


@router.get("/api/export/predictions")
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


@router.get("/api/export/paper-trades")
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


@router.get("/api/export/backtest/{run_id}")
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


@router.get("/api/analysis/export.csv")
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
