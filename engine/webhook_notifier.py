"""
Webhook Notifier — Telegram + Discord real-time alerts
Complements email notifications with instant messaging channels.
"""
import requests
import json
import logging
from datetime import datetime
from typing import Optional
from core.database import db

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send alerts via Telegram Bot API."""

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self._load_settings()

    def _load_settings(self):
        self.enabled = bool(db.get_setting("telegram_enabled"))
        self.token = db.get_setting("telegram_bot_token") or ""
        self.chat_id = db.get_setting("telegram_chat_id") or ""

    def reload(self):
        self._load_settings()

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        if not self.enabled or not self.is_configured():
            return False
        try:
            url = self.BASE_URL.format(token=self.token)
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok:
                logger.info(f"Telegram message sent to {self.chat_id}")
                return True
            else:
                logger.warning(f"Telegram API error: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def test(self, token: str = None, chat_id: str = None) -> tuple:
        """Send a test message. Returns (success: bool, message: str)."""
        tok = token or self.token
        cid = chat_id or self.chat_id
        if not tok or not cid:
            return False, "Missing token or chat_id"
        try:
            url = self.BASE_URL.format(token=tok)
            payload = {
                "chat_id": cid,
                "text": "✅ *AI Investment Monitor* connected successfully!\n\nYou'll receive trading alerts here.",
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok:
                return True, "Test message sent successfully"
            else:
                data = resp.json()
                return False, data.get("description", f"HTTP {resp.status_code}")
        except Exception as e:
            return False, str(e)


class DiscordNotifier:
    """Send alerts via Discord webhook."""

    def __init__(self):
        self._load_settings()

    def _load_settings(self):
        self.enabled = bool(db.get_setting("discord_enabled"))
        self.webhook_url = db.get_setting("discord_webhook_url") or ""

    def reload(self):
        self._load_settings()

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str) -> bool:
        if not self.enabled or not self.is_configured():
            return False
        try:
            payload = {"content": message, "username": "Investment Monitor"}
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            if resp.ok or resp.status_code == 204:
                logger.info("Discord message sent")
                return True
            else:
                logger.warning(f"Discord webhook error: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False

    def test(self, webhook_url: str = None) -> tuple:
        """Send a test message. Returns (success: bool, message: str)."""
        url = webhook_url or self.webhook_url
        if not url:
            return False, "Missing webhook URL"
        try:
            payload = {
                "content": "✅ **AI Investment Monitor** connected!\nYou'll receive trading alerts here.",
                "username": "Investment Monitor",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok or resp.status_code == 204:
                return True, "Test message sent successfully"
            else:
                return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return False, str(e)


class WebhookNotifier:
    """
    Facade over Telegram + Discord notifiers.
    Sends to all configured channels simultaneously.
    """

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.discord = DiscordNotifier()

    def reload(self):
        self.telegram.reload()
        self.discord.reload()

    def send_signal_alert(self, ticker: str, signal: str, confidence: int,
                          recommendation: str = "", extra: str = "") -> bool:
        """Format and send a trading signal alert."""
        emoji = {
            "STRONG_BUY": "🚀", "BUY": "📈",
            "STRONG_SELL": "🔴", "SELL": "📉", "HOLD": "⏸️",
        }.get(signal, "📊")

        conf_bar = "█" * (confidence // 10) + "░" * (10 - confidence // 10)

        msg = (
            f"{emoji} *{signal}* — `{ticker}`\n"
            f"Confidence: {confidence}% `{conf_bar}`\n"
        )
        if extra:
            msg += f"_{extra}_\n"
        if recommendation:
            # Truncate long recommendations
            short_rec = recommendation[:300] + "…" if len(recommendation) > 300 else recommendation
            msg += f"\n{short_rec}\n"
        msg += f"\n🕐 {datetime.now().strftime('%H:%M %d.%m.%Y')}"

        sent = False
        sent |= self.telegram.send(msg)
        sent |= self.discord.send(msg.replace("*", "**").replace("`", "`"))
        return sent

    def send_earnings_alert(self, ticker: str, alert_message: str) -> bool:
        """Send a pre-earnings positioning alert."""
        msg = f"📅 *Earnings Alert*\n{alert_message}"
        sent = False
        sent |= self.telegram.send(msg)
        sent |= self.discord.send(msg.replace("*", "**"))
        return sent

    def send_custom(self, message: str) -> bool:
        """Send a custom text message to all channels."""
        sent = False
        sent |= self.telegram.send(message)
        sent |= self.discord.send(message)
        return sent

    def any_configured(self) -> bool:
        return (
            (self.telegram.is_configured() and self.telegram.enabled) or
            (self.discord.is_configured() and self.discord.enabled)
        )


# Singleton
webhook_notifier = WebhookNotifier()
