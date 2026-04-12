"""
Push subscriptions, Telegram/Discord webhook tests.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.post("/settings/test-telegram")
async def test_telegram(
    request: Request,
    csrf_token: str = Form(...),
    telegram_bot_token: str = Form(""),
    telegram_chat_id: str = Form(""),
    username: str = Depends(require_auth),
):
    """Send a test Telegram message to validate token/chat_id."""
    csrf.verify_token(request, csrf_token)
    from engine.webhook_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    ok, msg = notifier.test(
        token=telegram_bot_token or None,
        chat_id=telegram_chat_id or None
    )
    redirect_url = f"/settings?{'telegram_ok=1' if ok else 'telegram_error=1'}&msg={msg[:100]}"
    return RedirectResponse(url=redirect_url, status_code=303)


@router.post("/settings/test-discord")
async def test_discord(
    request: Request,
    csrf_token: str = Form(...),
    discord_webhook_url: str = Form(""),
    username: str = Depends(require_auth),
):
    """Send a test Discord message to validate webhook URL."""
    csrf.verify_token(request, csrf_token)
    from engine.webhook_notifier import DiscordNotifier
    notifier = DiscordNotifier()
    ok, msg = notifier.test(webhook_url=discord_webhook_url or None)
    redirect_url = f"/settings?{'discord_ok=1' if ok else 'discord_error=1'}&msg={msg[:100]}"
    return RedirectResponse(url=redirect_url, status_code=303)


@router.post("/settings/save-webhooks")
async def save_webhook_settings(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth),
):
    """Save Telegram / Discord webhook configuration."""
    csrf.verify_token(request, csrf_token)
    form = await request.form()
    db.set_setting("telegram_enabled", form.get("telegram_enabled") == "on")
    if form.get("telegram_bot_token"):
        db.set_setting("telegram_bot_token", form.get("telegram_bot_token", ""))
    if form.get("telegram_chat_id"):
        db.set_setting("telegram_chat_id", form.get("telegram_chat_id", ""))
    bot_enabled = form.get("telegram_bot_enabled") == "on"
    db.set_setting("telegram_bot_enabled", bot_enabled)
    # Restart the bot polling thread to pick up the new setting
    try:
        from clients.telegram_bot import telegram_bot
        telegram_bot.restart()
    except Exception:
        pass
    db.set_setting("discord_enabled", form.get("discord_enabled") == "on")
    if form.get("discord_webhook_url"):
        db.set_setting("discord_webhook_url", form.get("discord_webhook_url", ""))
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.get("/api/push/vapid-key")
async def api_push_vapid_key(username: str = Depends(require_api_key_or_session)):
    """Return the VAPID public key for browser push subscription (#31/#59)."""
    from engine.push_notifier import get_vapid_public_key
    key = get_vapid_public_key()
    if not key:
        return {"vapid_public_key": None, "available": False}
    return {"vapid_public_key": key, "available": True}


@router.post("/api/push/subscribe")
async def api_push_subscribe(request: Request, username: str = Depends(require_api_key_or_session)):
    """Store a browser Web Push subscription (#31/#59)."""
    from engine.push_notifier import save_subscription
    body = await request.json()
    endpoint = body.get("endpoint", "")
    keys = body.get("keys", {})
    p256dh = keys.get("p256dh", "")
    auth = keys.get("auth", "")
    if not endpoint or not p256dh or not auth:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing endpoint or keys")
    user_agent = request.headers.get("user-agent", "")
    ok = save_subscription(endpoint, p256dh, auth, user_agent)
    return {"subscribed": ok}


@router.delete("/api/push/unsubscribe")
async def api_push_unsubscribe(request: Request, username: str = Depends(require_api_key_or_session)):
    """Remove a browser Web Push subscription (#31/#59)."""
    from engine.push_notifier import remove_subscription
    body = await request.json()
    endpoint = body.get("endpoint", "")
    if not endpoint:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing endpoint")
    ok = remove_subscription(endpoint)
    return {"unsubscribed": ok}


@router.post("/api/push/test")
async def api_push_test(request: Request, username: str = Depends(require_api_key_or_session)):
    """Send a test push notification to all subscriptions (#31/#59)."""
    from engine.push_notifier import send_push
    sent = send_push(
        title="Stockholm Test Notification",
        body="Push notifications are working correctly!",
        url="/",
    )
    return {"sent": sent}
