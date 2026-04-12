"""
Quantitative model endpoints: Graham screener, LSTM predictor.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


# === Graham Screener ===

@router.get("/api/graham/screen")
async def api_graham_screen(
    request: Request,
    discount: float = 0.2,
    max_positions: int = 50,
    username: str = Depends(require_auth)
):
    """Graham intrinsic value screen across watchlist. discount=margin of safety (0.0-0.9)."""
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


# === LSTM Predictor ===

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
    Trigger LSTM training on the watchlist (runs synchronously -- may take several minutes).
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
