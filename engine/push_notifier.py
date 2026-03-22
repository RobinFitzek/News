"""
PWA Web Push Notification Backend (items #31 / #59)

Manages VAPID keys, stores browser push subscriptions, and sends Web Push
notifications for STRONG_BUY/SELL alerts, geo severity >= 8, and intraday breakouts.

Requires: pywebpush  (pip install pywebpush)
Gracefully degrades if pywebpush is not installed.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: pywebpush
# ---------------------------------------------------------------------------
try:
    from pywebpush import webpush, WebPushException
    _webpush_available = True
except ImportError:
    _webpush_available = False
    logger.debug("push_notifier: pywebpush not installed — push notifications disabled")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    try:
        from core.database import db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint     TEXT NOT NULL UNIQUE,
                p256dh       TEXT NOT NULL,
                auth         TEXT NOT NULL,
                user_agent   TEXT,
                created_at   TEXT NOT NULL,
                last_used_at TEXT
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_push_subscriptions_endpoint "
            "ON push_subscriptions (endpoint)"
        )
    except Exception as e:
        logger.debug(f"push_notifier.ensure_schema: {e}")


# ---------------------------------------------------------------------------
# VAPID key management
# ---------------------------------------------------------------------------
def _get_or_create_vapid_keys() -> dict:
    """
    Return VAPID public/private keys from DB settings, generating them on first use.
    Keys are stored as base64url-encoded strings.
    """
    from core.database import db

    pub = db.get_setting("vapid_public_key")
    priv = db.get_setting("vapid_private_key")

    if pub and priv:
        return {"public_key": pub, "private_key": priv}

    # Generate new VAPID key pair
    try:
        from py_vapid import Vapid
        vapid = Vapid()
        vapid.generate_keys()
        pub_key = vapid.public_key.public_bytes(
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding", "PublicFormat"]).Encoding.X962,
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["PublicFormat"]).PublicFormat.UncompressedPoint,
        )
        import base64
        pub_b64 = base64.urlsafe_b64encode(pub_key).rstrip(b"=").decode()
        priv_b64 = vapid.private_key  # already base64 in py_vapid

        db.save_setting("vapid_public_key", pub_b64)
        db.save_setting("vapid_private_key", str(priv_b64))
        logger.info("VAPID keys generated and stored")
        return {"public_key": pub_b64, "private_key": str(priv_b64)}
    except Exception as e:
        logger.warning(f"push_notifier: VAPID key generation failed: {e}")
        return {}


def get_vapid_public_key() -> Optional[str]:
    """Return the VAPID public key for browser subscription."""
    try:
        keys = _get_or_create_vapid_keys()
        return keys.get("public_key")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subscription management
# ---------------------------------------------------------------------------
def save_subscription(endpoint: str, p256dh: str, auth: str, user_agent: str = "") -> bool:
    """Store or update a browser push subscription."""
    ensure_schema()
    try:
        from core.database import db
        db.execute(
            """
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, user_agent, created_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                p256dh=excluded.p256dh,
                auth=excluded.auth,
                user_agent=excluded.user_agent,
                last_used_at=excluded.last_used_at
            """,
            (endpoint, p256dh, auth, user_agent, datetime.now().isoformat(), datetime.now().isoformat()),
        )
        return True
    except Exception as e:
        logger.debug(f"push_notifier.save_subscription: {e}")
        return False


def remove_subscription(endpoint: str) -> bool:
    """Remove a push subscription (user opted out)."""
    try:
        from core.database import db
        db.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
        return True
    except Exception as e:
        logger.debug(f"push_notifier.remove_subscription: {e}")
        return False


def get_all_subscriptions() -> list[dict]:
    ensure_schema()
    try:
        from core.database import db
        rows = db.query("SELECT * FROM push_subscriptions") or []
        return [dict(r) for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Send push notification
# ---------------------------------------------------------------------------
def send_push(title: str, body: str, url: str = "/", icon: str = "/static/logo-192.png") -> int:
    """
    Send a Web Push notification to all stored subscriptions.
    Returns number of successfully sent notifications.

    Stale subscriptions (410 Gone) are automatically removed.
    """
    if not _webpush_available:
        logger.debug("push_notifier: pywebpush not available, skipping push")
        return 0

    ensure_schema()
    subscriptions = get_all_subscriptions()
    if not subscriptions:
        return 0

    try:
        keys = _get_or_create_vapid_keys()
        if not keys:
            return 0
        vapid_private = keys["public_key"]  # pywebpush wants private
        vapid_private = keys.get("private_key", "")
        if not vapid_private:
            return 0
    except Exception as e:
        logger.debug(f"push_notifier.send_push key error: {e}")
        return 0

    payload = json.dumps({"title": title, "body": body, "url": url, "icon": icon})
    sent = 0
    stale = []

    for sub in subscriptions:
        try:
            subscription_info = {
                "endpoint": sub["endpoint"],
                "keys": {
                    "p256dh": sub["p256dh"],
                    "auth": sub["auth"],
                },
            }
            webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=vapid_private,
                vapid_claims={"sub": "mailto:admin@stockholm.local"},
            )
            sent += 1
            # Update last_used_at
            try:
                from core.database import db
                db.execute(
                    "UPDATE push_subscriptions SET last_used_at = ? WHERE endpoint = ?",
                    (datetime.now().isoformat(), sub["endpoint"]),
                )
            except Exception:
                pass
        except WebPushException as e:
            if e.response and e.response.status_code == 410:
                # Subscription expired — remove it
                stale.append(sub["endpoint"])
            else:
                logger.debug(f"push_notifier.send_push WebPushException: {e}")
        except Exception as e:
            logger.debug(f"push_notifier.send_push error for {sub['endpoint'][:40]}...: {e}")

    for endpoint in stale:
        remove_subscription(endpoint)
        logger.info(f"push_notifier: removed stale subscription {endpoint[:40]}...")

    if sent:
        logger.info(f"push_notifier: sent {sent}/{len(subscriptions)} push notifications")
    return sent


# ---------------------------------------------------------------------------
# Convenience wrappers for alert triggers
# ---------------------------------------------------------------------------
def push_signal_alert(ticker: str, signal: str, score: int) -> None:
    """Push notification for STRONG_BUY / STRONG_SELL signals."""
    if signal not in ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL"):
        return
    emoji = "🚀" if "BUY" in signal else "🔻"
    send_push(
        title=f"{emoji} {signal}: {ticker}",
        body=f"Score: {score}/10 — tap to view analysis",
        url=f"/stock/{ticker}",
    )


def push_geo_alert(ticker: str, geo_score: int, region: str) -> None:
    """Push notification for high geo-risk events (severity >= 8)."""
    if geo_score < 8:
        return
    send_push(
        title=f"⚠️ Geo Risk: {ticker}",
        body=f"Geo risk score {geo_score}/10 — {region}",
        url=f"/stock/{ticker}",
    )


def push_composite_signal(pattern_name: str, description: str) -> None:
    """Push notification for composite macro signals."""
    send_push(
        title=f"📊 Macro Signal: {pattern_name}",
        body=description[:100],
        url="/macro",
    )
