"""
Live broker integration: orders, account, positions, sync.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.post("/api/orders/execute")
async def api_orders_execute(request: Request, username: str = Depends(require_api_key_or_session)):
    """Execute a trade entry. Body: {token} OR {ticker, direction, size_usd}."""
    from engine.order_manager import order_manager
    data = await request.json()
    token = data.get("token")
    if token:
        # Confirm a pending trade via token
        from engine.auto_paper_trader import auto_paper_trader
        result = auto_paper_trader.confirm_trade(token)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Confirm failed"))
        return result
    ticker = data.get("ticker", "").upper()
    direction = data.get("direction", "LONG").upper()
    size_usd = float(data.get("size_usd", 0))
    if not ticker or size_usd <= 0:
        raise HTTPException(status_code=400, detail="ticker and size_usd required")
    result = order_manager.execute_entry(ticker, direction, size_usd)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Entry failed"))
    return result


@router.post("/api/orders/close/{trade_id}")
async def api_orders_close(trade_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Close an open auto-trade position."""
    from engine.order_manager import order_manager
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass
    reason = data.get("reason", "manual")
    result = order_manager.execute_exit(trade_id, reason)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Exit failed"))
    return result


@router.get("/api/orders/status/{order_id}")
async def api_orders_status(order_id: str, username: str = Depends(require_api_key_or_session)):
    """Poll fill status for a broker order (best-effort)."""
    from clients.broker_client import get_broker_client, AlpacaBrokerClient
    broker = get_broker_client()
    if isinstance(broker, AlpacaBrokerClient):
        try:
            data = broker._get(f"/orders/{order_id}")
            return {"order_id": order_id, "status": data.get("status"), "filled_qty": data.get("filled_qty")}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
    # Paper / IBKR: synthetic response
    return {"order_id": order_id, "status": "filled", "filled_qty": None}


@router.get("/api/broker/account")
async def api_broker_account(username: str = Depends(require_api_key_or_session)):
    """Return broker account: equity, buying_power, cash, broker_last_sync."""
    from clients.broker_client import get_broker_client
    broker = get_broker_client()
    account = broker.get_account()
    account["broker_last_sync"] = db.get_setting("broker_last_sync") or ""
    return account


@router.get("/api/broker/positions")
async def api_broker_positions(username: str = Depends(require_api_key_or_session)):
    """Return live broker positions."""
    from clients.broker_client import get_broker_client
    broker = get_broker_client()
    return broker.get_positions()


@router.post("/api/broker/sync")
async def api_broker_sync(username: str = Depends(require_api_key_or_session)):
    """Trigger manual broker position sync."""
    from engine.order_manager import order_manager
    synced = order_manager.sync_broker_positions()
    return {"synced": synced, "broker_last_sync": db.get_setting("broker_last_sync") or ""}
