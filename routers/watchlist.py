"""
Watchlist add/remove/archive/notes/groups.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/watchlist", response_class=HTMLResponse)
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


@router.post("/watchlist/add")
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


@router.post("/watchlist/remove/{ticker}")
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


@router.post("/watchlist/archive/{ticker}")
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


@router.post("/api/watchlist/note")
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


@router.get("/api/watchlist/note/{ticker}")
async def get_watchlist_note(
    request: Request,
    ticker: str,
    username: str = Depends(require_auth)
):
    """Get note for a watchlist item"""
    note = db.get_stock_note(ticker)
    return JSONResponse({"ticker": ticker, "note": note or ""})


@router.get("/api/watchlist/groups")
async def api_watchlist_groups(username: str = Depends(require_api_key_or_session)):
    """Return all distinct watchlist group names (#27)."""
    return {"groups": db.get_watchlist_groups()}


@router.post("/api/watchlist/{ticker}/group")
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


@router.get("/api/watchlist/group-stats")
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


@router.post("/watchlist/tier/{ticker}")
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


@router.post("/api/watchlist/import")
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


@router.get("/api/watchlist")
async def api_get_watchlist(
    request: Request,
    username: str = Depends(require_api_key_or_session)
):
    """Get watchlist items as JSON — for React SPA"""
    items = db.get_watchlist(active_only=False)
    return items if isinstance(items, list) else []


@router.post("/api/watchlist")
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


@router.delete("/api/watchlist/{ticker}")
async def api_remove_watchlist(
    request: Request,
    ticker: str,
    username: str = Depends(require_auth)
):
    """Remove ticker from watchlist — JSON endpoint for React SPA"""
    _verify_spa_csrf(request)
    db.remove_from_watchlist(ticker.upper())
    return {"status": "removed", "ticker": ticker}
