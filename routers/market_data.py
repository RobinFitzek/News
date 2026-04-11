"""
Stock detail pages, fundamentals, chart data, screeners.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


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


@router.get("/api/chart-data/{ticker}")
async def api_chart_data(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return 6mo OHLCV + SMA overlays + signal markers for a ticker."""
    import yfinance as yf_local

    ticker = ticker.upper().strip()
    try:
        stock = yf_local.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return {"error": "No data available"}

        close = hist['Close']
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = [round(float(p), 2) for p in close]

        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        def safe_list(series):
            return [round(float(v), 2) if not (v != v) else None for v in series]

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


@router.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail_page(request: Request, ticker: str, analysis_id: int | None = None, username: str = Depends(require_auth)):
    """Unified stock detail page — all data about a ticker in one place."""
    from engine.earnings_tracker import earnings_tracker
    from engine.financial_statements import financial_statements
    from engine.insider_tracker import insider_tracker
    from engine.multi_timeframe import multi_timeframe
    from engine.position_sizing import position_sizer
    from engine.dividend_tracker import dividend_tracker

    ticker = ticker.upper()

    # Parallel data gathering (best-effort — each module handles its own errors)
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
        conn = db._get_conn()
        cursor = conn.cursor()
        if analysis_id is not None:
            cursor.execute(
                "SELECT * FROM analysis_history WHERE id = ? AND ticker = ?",
                (analysis_id, ticker)
            )
            row = cursor.fetchone()
            if not row:
                cursor.execute(
                    "SELECT * FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                    (ticker,)
                )
                row = cursor.fetchone()
        else:
            cursor.execute(
                "SELECT * FROM analysis_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                (ticker,)
            )
            row = cursor.fetchone()
        conn.close()

        if row:
            selected_analysis = dict(row)

            try:
                conn2 = db._get_conn()
                cur2 = conn2.cursor()
                cur2.execute(
                    "SELECT * FROM ai_crosscheck_log WHERE analysis_id = ? ORDER BY checked_at DESC LIMIT 1",
                    (selected_analysis["id"],)
                )
                cc_row = cur2.fetchone()
                conn2.close()

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

    Includes staleness_days, decay_pct (50% at 5 days), staleness_level,
    and a 10-point decay curve for charting.
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

    # Build a 10-point decay curve: day 0 → day 14
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


@router.get("/api/earnings/{ticker}")
async def api_earnings(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return earnings data including beat history for a ticker."""
    from engine.earnings_tracker import earnings_tracker
    t = ticker.upper()
    info = earnings_tracker.get_earnings_info(t)
    beat = earnings_tracker.get_beat_history(t)
    alert = earnings_tracker.generate_positioning_alert(t)
    return {"ticker": t, "earnings_info": info, "beat_history": beat, "alert_message": alert}


@router.get("/api/key-stats/{ticker}")
async def api_key_stats(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return 52w proximity, market cap label, short interest, pre/post market prices."""
    from engine.financial_statements import financial_statements
    return financial_statements.get_key_stats(ticker.upper())


@router.get("/api/financials/{ticker}")
async def api_financials(ticker: str, request: Request, username: str = Depends(require_api_key_or_session)):
    """Return 8-quarter financial trend data."""
    from engine.financial_statements import financial_statements
    return financial_statements.get_quarterly_financials(ticker.upper())


@router.get("/api/dcf/{ticker}")
async def api_dcf(
    ticker: str,
    request: Request,
    growth_rate: float = None,
    terminal_rate: float = 0.03,
    discount_rate: float = 0.10,
    username: str = Depends(require_api_key_or_session),
):
    """Run DCF fair value estimate with adjustable assumptions."""
    from engine.financial_statements import financial_statements
    return financial_statements.estimate_fair_value(
        ticker.upper(), growth_rate=growth_rate,
        terminal_rate=terminal_rate, discount_rate=discount_rate
    )


@router.get("/api/peers/{ticker}")
async def api_peers(
    ticker: str, request: Request,
    peers: str = "",
    username: str = Depends(require_api_key_or_session),
):
    """Peer comparison table."""
    from engine.financial_statements import financial_statements
    peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()] if peers else None
    return financial_statements.get_peer_comparison(ticker.upper(), peer_list)


@router.get("/api/rs-ranking")
async def api_rs_ranking(request: Request, username: str = Depends(require_api_key_or_session)):
    """Rank watchlist stocks by 3/6/12-month relative strength vs SPY."""
    from engine.rs_ranking import rs_ranking
    watchlist = [w['ticker'] for w in db.get_watchlist()]
    if not watchlist:
        return {"available": False, "reason": "Empty watchlist"}
    rankings = rs_ranking.rank_tickers(watchlist)
    return {"available": True, "rankings": rankings}


@router.get("/api/price-alerts")
async def api_get_price_alerts(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get all active price alerts."""
    alerts = db.query("SELECT * FROM price_alerts WHERE active = 1 ORDER BY created_at DESC") or []
    return {"alerts": [dict(a) for a in alerts]}


@router.post("/api/price-alerts")
async def api_create_price_alert(request: Request, username: str = Depends(require_api_key_or_session)):
    """Create a new price alert."""
    data = await request.json()
    ticker = data.get('ticker', '').upper()
    alert_type = data.get('alert_type', 'target_price')
    threshold = data.get('threshold')
    direction = data.get('direction', 'above')

    if not ticker or threshold is None:
        return {"error": "ticker and threshold required"}

    db.execute("""
        INSERT INTO price_alerts (ticker, alert_type, threshold, direction)
        VALUES (?, ?, ?, ?)
    """, (ticker, alert_type, float(threshold), direction))
    return {"status": "created"}


@router.delete("/api/price-alerts/{alert_id}")
async def api_delete_price_alert(request: Request, alert_id: int, username: str = Depends(require_api_key_or_session)):
    """Deactivate a price alert."""
    db.execute("UPDATE price_alerts SET active = 0 WHERE id = ?", (alert_id,))
    return {"status": "deactivated"}


@router.get("/api/patterns/{ticker}")
async def api_patterns(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Detect chart patterns for a ticker."""
    from engine.pattern_recognition import pattern_recognizer
    try:
        patterns = pattern_recognizer.detect_patterns(ticker)
        return {"ticker": ticker.upper(), "patterns": patterns}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/sentiment/{ticker}")
async def api_sentiment(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get sentiment summary including analyst consensus and contrarian signals."""
    from engine.sentiment_analyzer import sentiment_analyzer
    try:
        summary = sentiment_analyzer.get_sentiment_summary(ticker)
        return summary
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/nlp-sentiment/{ticker}")
async def api_nlp_sentiment(ticker: str, days: int = 7, username: str = Depends(require_api_key_or_session)):
    """
    Return NLP VADER sentiment trend for a ticker from stored snapshots (#38/#57).
    """
    ticker = ticker.upper()
    try:
        from core.database import db
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = db.query(
            """
            SELECT compound_score, positive, neutral, negative, headline_count, scored_at
            FROM ticker_sentiment
            WHERE ticker = ? AND scored_at >= ?
            ORDER BY scored_at ASC
            """,
            (ticker, cutoff),
        ) or []
        return {
            "ticker": ticker,
            "days": days,
            "snapshots": [dict(r) for r in rows],
            "latest": dict(rows[-1]) if rows else None,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/nlp-sentiment/movers")
async def api_nlp_sentiment_movers(hours: int = 24, username: str = Depends(require_api_key_or_session)):
    """Return tickers with biggest VADER sentiment shift in the last N hours (#38)."""
    from engine.nlp_scorer import get_sentiment_movers
    return {"movers": get_sentiment_movers(hours=hours)}


@router.get("/api/pairs")
async def api_pairs_all(username: str = Depends(require_api_key_or_session)):
    """Return all tested pairs ordered by cointegration strength (#40)."""
    from engine.pairs_trader import get_all_pairs, ensure_schema
    ensure_schema()
    return {"pairs": get_all_pairs()}


@router.get("/api/pairs/active")
async def api_pairs_active(username: str = Depends(require_api_key_or_session)):
    """Return pairs with active long_spread / short_spread signals (#40)."""
    from engine.pairs_trader import get_active_pairs
    return {"pairs": get_active_pairs()}


@router.post("/api/pairs/scan")
async def api_pairs_scan(request: Request, username: str = Depends(require_api_key_or_session)):
    """Trigger a manual pairs cointegration scan (#40)."""
    from engine.pairs_trader import run_weekly_scan
    try:
        pairs = run_weekly_scan()
        return {"cointegrated_pairs": len(pairs), "pairs": pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/moat/{ticker}")
async def api_moat_ticker(ticker: str, username: str = Depends(require_api_key_or_session)):
    """Return economic moat score for a ticker (#45/#54)."""
    from engine.moat_scorer import moat_score
    return moat_score(ticker.upper())


@router.get("/api/moat")
async def api_moat_watchlist(username: str = Depends(require_api_key_or_session)):
    """Return moat scores for all active watchlist tickers, sorted by score (#45/#54)."""
    from engine.moat_scorer import batch_moat_scores
    tickers = [
        r["ticker"]
        for r in (db.query("SELECT ticker FROM watchlist WHERE active = 1") or [])
    ]
    if not tickers:
        return {"moat_scores": []}
    return {"moat_scores": batch_moat_scores(tickers)}


@router.get("/api/supply-chain/{ticker}")
async def api_supply_chain(
    ticker: str,
    force_refresh: bool = False,
    username: str = Depends(require_api_key_or_session),
):
    """Return supply chain map (suppliers/customers/partners) for a ticker (#44/#58)."""
    from engine.supply_chain import get_supply_chain
    return get_supply_chain(ticker.upper(), force_refresh=force_refresh)


@router.get("/api/supply-chain/{ticker}/geo-exposure")
async def api_supply_chain_geo(
    ticker: str,
    regions: str = "",
    username: str = Depends(require_api_key_or_session),
):
    """Check if any of a ticker's suppliers are in flagged geo regions (#44/#58)."""
    from engine.supply_chain import get_geo_elevated_tickers
    flagged = [r.strip() for r in regions.split(",") if r.strip()] if regions else []
    if not flagged:
        return {"ticker": ticker.upper(), "elevated": False, "matches": []}
    all_elevated = get_geo_elevated_tickers(flagged)
    matches = [e for e in all_elevated if e["ticker"] == ticker.upper()]
    return {
        "ticker": ticker.upper(),
        "elevated": bool(matches),
        "matches": matches,
    }


@router.post("/api/supply-chain/refresh")
async def api_supply_chain_refresh(request: Request, username: str = Depends(require_api_key_or_session)):
    """Manually trigger quarterly supply chain refresh for all stale tickers (#44)."""
    from engine.supply_chain import refresh_stale_tickers
    refreshed = refresh_stale_tickers()
    return {"refreshed": refreshed}


@router.get("/api/catalysts/{ticker}")
async def api_catalysts(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Unified catalyst timeline: earnings, dividends, economic events."""
    from engine.earnings_tracker import earnings_tracker
    from engine.dividend_tracker import dividend_tracker
    catalysts = []
    try:
        earnings = earnings_tracker.get_earnings_info(ticker)
        if earnings and earnings.get('next_earnings_date'):
            catalysts.append({
                'type': 'earnings',
                'name': f"{ticker.upper()} Earnings Report",
                'date': earnings['next_earnings_date'],
                'detail': f"Est. EPS: {earnings.get('estimated_eps', '—')}",
            })
    except Exception:
        pass
    try:
        div_info = dividend_tracker.get_dividend_info(ticker)
        if div_info and div_info.get('estimated_next_ex_date'):
            catalysts.append({
                'type': 'dividend',
                'name': f"{ticker.upper()} Ex-Dividend",
                'date': div_info['estimated_next_ex_date'],
                'detail': f"${div_info.get('last_dividend', 0):.2f} per share",
            })
    except Exception:
        pass
    catalysts.sort(key=lambda c: c.get('date', ''))
    return {"ticker": ticker.upper(), "catalysts": catalysts}


@router.get("/api/short-interest/{ticker}")
async def api_short_interest(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get short interest data and squeeze setup analysis."""
    from engine.short_interest import short_interest_tracker
    try:
        data = short_interest_tracker.get_short_data(ticker)
        squeeze = short_interest_tracker.check_squeeze_setup(ticker)
        history = short_interest_tracker.get_history(ticker, days=60)
        return {
            "current": data,
            "squeeze": squeeze,
            "history": history,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/options-flow/{ticker}")
async def api_options_flow(request: Request, ticker: str, username: str = Depends(require_api_key_or_session)):
    """Get options flow summary and unusual activity."""
    from engine.options_flow import options_flow
    try:
        summary = options_flow.get_options_summary(ticker)
        unusual = options_flow.detect_unusual_activity(ticker)
        return {
            "summary": summary,
            "unusual_activity": unusual,
        }
    except Exception as e:
        return {"error": str(e)}


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

    # Build ETF → sector name map
    etf_name = {etf: info['name'] for etf, info in SECTOR_ETFS.items()}

    # Build universe: SP500_CORE
    universe = set(AutoDiscovery.SP500_CORE)

    # Map ticker → etf for quick lookup
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


@router.get("/api/graham/screen")
async def api_graham_screen(
    request: Request,
    discount: float = 0.2,
    max_positions: int = 50,
    username: str = Depends(require_auth)
):
    """Graham intrinsic value screen across watchlist. discount=margin of safety (0.0–0.9)."""
    from engine.graham_screener import graham_screener
    tickers = [row["ticker"] for row in db.get_watchlist()]
    if not tickers:
        return {"error": "Watchlist is empty"}
    discount = max(0.0, min(discount, 0.9))
    return graham_screener.screen_watchlist(tickers, discount_factor=discount,
                                             max_positions=max_positions)


@router.get("/api/graham/ticker/{ticker}")
async def api_graham_ticker(
    ticker: str,
    request: Request,
    discount: float = 0.2,
    username: str = Depends(require_auth)
):
    """Graham intrinsic value for a single ticker."""
    from engine.graham_screener import graham_screener
    return graham_screener.screen_ticker(ticker.upper(), discount_factor=discount)


@router.get("/api/graham/backtest")
async def api_graham_backtest(
    request: Request,
    discount: float = 0.2,
    max_positions: int = 50,
    holding_days: int = 252,
    username: str = Depends(require_auth)
):
    """Graham back test across watchlist tickers."""
    from engine.graham_screener import graham_screener
    tickers = [row["ticker"] for row in db.get_watchlist()]
    if not tickers:
        return {"error": "Watchlist is empty"}
    return graham_screener.backtest(tickers, discount_factor=discount,
                                    max_positions=max_positions, holding_days=holding_days)


@router.get("/api/graham/aaa-yield")
async def api_graham_aaa_yield(request: Request, username: str = Depends(require_auth)):
    """Return current AAA bond yield used in Graham formula."""
    from engine.graham_screener import graham_screener
    yield_val = graham_screener.fetch_aaa_yield()
    return {"aaa_yield_pct": yield_val, "source": "FRED DAAA"}


@router.get("/api/lstm/predict/{ticker}")
async def api_lstm_predict(
    ticker: str,
    request: Request,
    username: str = Depends(require_auth)
):
    """LSTM buy signal prediction for a single ticker (requires trained model)."""
    from engine.lstm_predictor import lstm_predictor
    return lstm_predictor.predict(ticker.upper())


@router.get("/api/lstm/signals")
async def api_lstm_signals(request: Request, username: str = Depends(require_auth)):
    """LSTM buy signals across entire watchlist."""
    from engine.lstm_predictor import lstm_predictor
    tickers = [row["ticker"] for row in db.get_watchlist()]
    if not tickers:
        return {"signals": [], "error": "Watchlist empty"}
    signals = lstm_predictor.get_buy_signals(tickers)
    return {"signals": signals, "count": len(signals), "threshold": 0.50}


@router.post("/api/lstm/train")
async def api_lstm_train(
    request: Request,
    username: str = Depends(require_auth)
):
    """
    Trigger LSTM training on the watchlist (runs synchronously — may take several minutes).
    POST body (optional JSON): {"epochs": 20, "years_back": 3}
    """
    from engine.lstm_predictor import lstm_predictor
    try:
        body = await request.json()
    except Exception:
        body = {}
    epochs = int(body.get("epochs", 20))
    years_back = int(body.get("years_back", 3))
    tickers = [row["ticker"] for row in db.get_watchlist()]
    if not tickers:
        return {"error": "Watchlist empty"}
    result = lstm_predictor.train(tickers, epochs=epochs, years_back=years_back)
    return result


@router.get("/api/lstm/performance")
async def api_lstm_performance(request: Request, username: str = Depends(require_auth)):
    """LSTM trade log performance metrics: CAGR, max drawdown, win rate."""
    from engine.lstm_predictor import lstm_predictor
    return lstm_predictor.get_performance_metrics()


@router.get("/api/lstm/trade-history")
async def api_lstm_trade_history(
    request: Request,
    limit: int = 100,
    username: str = Depends(require_auth)
):
    """LSTM trade history log (expected vs actual returns, hold periods)."""
    from engine.lstm_predictor import lstm_predictor
    trades = lstm_predictor.get_trade_history(limit=limit)
    return {"trades": trades, "count": len(trades)}
