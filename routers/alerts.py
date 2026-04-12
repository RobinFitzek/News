"""
Price alerts: create, list, deactivate.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


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
