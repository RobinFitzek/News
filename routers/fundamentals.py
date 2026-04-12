"""
Stock fundamentals: earnings, financials, DCF, peers, moat, supply chain, catalysts.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


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
