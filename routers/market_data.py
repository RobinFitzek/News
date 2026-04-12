"""
Sector momentum, chart data, stock detail pages, screeners, comparison.

Fundamental, sentiment, model, and alert endpoints live in their own routers:
  - routers/fundamentals.py
  - routers/sentiment.py
  - routers/models.py
  - routers/alerts.py
"""
from core.web_deps import *  # noqa: F401,F403
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# === Sector Momentum ===

@router.get("/api/sector-momentum")
async def api_sector_momentum(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get sector momentum heat map data"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_heat_map_data()


@router.get("/api/sector-momentum/rotation")
async def api_sector_rotation(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get sector rotation signals"""
    from engine.sector_momentum import sector_momentum
    return sector_momentum.get_rotation_signals()


@router.get("/api/multi-timeframe/{ticker}")
async def api_multi_timeframe(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Get multi-timeframe analysis for a ticker"""
    from engine.multi_timeframe import multi_timeframe
    return multi_timeframe.analyze_ticker(ticker.upper())


# === Chart Data ===

@router.get("/api/chart-data/{ticker}")
async def api_chart_data(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return 6mo OHLCV + SMA overlays + signal markers for a ticker."""
    from engine.utils import get_ticker_history, safe_float_list

    ticker = ticker.upper().strip()
    try:
        hist = get_ticker_history(ticker, period="6mo")
        if hist.empty:
            return {"error": "No data available"}

        close = hist['Close']
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = [round(float(p), 2) for p in close]

        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        safe_list = safe_float_list

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


# === Stock Detail Page ===

@router.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail_page(request: Request, ticker: str, analysis_id: int | None = None, username: str = Depends(require_auth)):
    """Unified stock detail page -- all data about a ticker in one place."""
    from engine.earnings_tracker import earnings_tracker
    from engine.financial_statements import financial_statements
    from engine.insider_tracker import insider_tracker
    from engine.multi_timeframe import multi_timeframe
    from engine.position_sizing import position_sizer
    from engine.dividend_tracker import dividend_tracker

    ticker = ticker.upper()

    # Parallel data gathering (best-effort -- each module handles its own errors)
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
            SELECT strategy, quant_score as score, found_at, promoted_at as promoted
            FROM discovered_stocks
            WHERE ticker = ?
            ORDER BY found_at DESC LIMIT 10
        """, (ticker,))
    except Exception:
        discovery_history = []

    # Selected/latest analysis for merged stock+analysis view
    selected_analysis = None
    crosscheck = None
    try:
        if analysis_id is not None:
            selected_analysis = db.query_one(
                "SELECT * FROM analysis_history WHERE id = ? AND ticker = ?",
                (analysis_id, ticker)
            )
        if not selected_analysis:
            selected_analysis = db.query_one(
                "SELECT * FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                (ticker,)
            )

        if selected_analysis:
            try:
                cc_row = db.query_one(
                    "SELECT * FROM ai_crosscheck_log WHERE analysis_id = ? ORDER BY checked_at DESC LIMIT 1",
                    (selected_analysis["id"],)
                )
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


# === Ticker-Level API Endpoints ===

@router.get("/api/stock/{ticker}/risk-trend")
async def api_risk_trend(ticker: str, days: int = 30, username: str = Depends(require_api_key_or_session)):
    rows = db.query("""
        SELECT timestamp, risk_score, geo_risk_score, signal, confidence
        FROM analysis_history
        WHERE ticker = ?
        AND timestamp >= datetime('now', ? || ' days')
        ORDER BY timestamp ASC
    """, (ticker.upper(), f"-{days}"))
    return {"ticker": ticker.upper(), "data": rows}


@router.get("/api/stock/{ticker}/staleness")
async def api_staleness(ticker: str, username: str = Depends(require_api_key_or_session)):
    """
    Return confidence decay metadata for a ticker's latest analysis (#28/#53).
    """
    ticker = ticker.upper()
    from engine.staleness_tracker import staleness_tracker

    row = db.query(
        "SELECT timestamp, confidence FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
        (ticker,)
    )
    if not row:
        return {"ticker": ticker, "error": "No analysis found"}

    analysis = dict(row[0])
    enriched = staleness_tracker.enrich_analysis(analysis)
    age_days = enriched.get("age_days", 0)
    original_conf = enriched.get("confidence", 70) or 70

    # Build a 10-point decay curve: day 0 -> day 14
    decay_curve = []
    for d in range(0, 15, 1):
        decayed = staleness_tracker.apply_confidence_decay(float(original_conf), d)
        decay_curve.append({"day": d, "confidence": round(decayed, 1)})

    return {
        "ticker": ticker,
        "last_analyzed": analysis.get("timestamp"),
        "age_days": age_days,
        "original_confidence": original_conf,
        "current_confidence": round(staleness_tracker.apply_confidence_decay(float(original_conf), age_days), 1),
        "decay_pct": round((1 - staleness_tracker.apply_confidence_decay(float(original_conf), age_days) / float(original_conf)) * 100, 1) if original_conf else 0,
        "staleness_level": staleness_tracker.get_staleness_level(age_days),
        "staleness_icon": staleness_tracker.get_staleness_icon(staleness_tracker.get_staleness_level(age_days)),
        "should_refresh": staleness_tracker.should_refresh(age_days),
        "decay_curve": decay_curve,
    }


@router.get("/api/stock/{ticker}/corporate-actions")
async def api_corporate_actions(ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return corporate actions (splits, dividends) for a ticker (#43)."""
    ticker = ticker.upper()
    actions = db.get_corporate_actions(ticker, limit=50)
    return {"ticker": ticker, "actions": actions}


@router.post("/stock/{ticker}/notes")
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


@router.get("/api/rs-ranking")
async def api_rs_ranking(request: Request, username: str = Depends(require_api_key_or_session)):
    """Rank watchlist stocks by 3/6/12-month relative strength vs SPY."""
    from engine.rs_ranking import rs_ranking
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    if not watchlist:
        return {"available": False, "reason": "Empty watchlist"}
    rankings = rs_ranking.rank_tickers(watchlist)
    return {"available": True, "rankings": rankings}


# === Comparison & Screening Pages ===

@router.get("/stock/compare", response_class=HTMLResponse)
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


@router.get("/sector-screen", response_class=HTMLResponse)
async def sector_screen_page(request: Request, username: str = Depends(require_auth)):
    """Sector-relative screening page."""
    return templates.TemplateResponse("sector_screen.html", {
        "request": request,
        "csrf_token": csrf.get_token(request),
    })


@router.get("/api/sector-screen")
async def api_sector_screen(request: Request, username: str = Depends(require_api_key_or_session)):
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

    # Build ETF -> sector name map
    etf_name = {etf: info['name'] for etf, info in SECTOR_ETFS.items()}

    # Build universe: SP500_CORE
    universe = set(AutoDiscovery.SP500_CORE)

    # Map ticker -> etf for quick lookup
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
