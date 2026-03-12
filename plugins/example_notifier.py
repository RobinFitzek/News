"""
Example Plugin: Telegram Notifier
===================================
Sends Stockholm signals to a Telegram chat via the Bot API.

To use this plugin:
1. Create a Telegram bot via @BotFather — note down the bot token
2. Start a chat with your bot, then visit:
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   to find your chat_id
3. Install this file in Settings → Plugins
4. Configure the bot_token and chat_id in plugin settings
5. Enable the plugin

This is a NOTIFIER plugin — it fires automatically whenever Stockholm
produces a STRONG_BUY or STRONG_SELL signal.
"""

# ── Required plugin metadata ──────────────────────────────────────────────────

PLUGIN_NAME        = "Telegram Notifier"
PLUGIN_VERSION     = "1.0.0"
PLUGIN_TYPE        = "notifier"          # notifier | analyzer | screener | exporter
PLUGIN_DESCRIPTION = "Sends STRONG_BUY/STRONG_SELL signals to a Telegram chat"
PLUGIN_AUTHOR      = "Stockholm"

# ── Settings schema ───────────────────────────────────────────────────────────
# Each key here becomes a form field in the plugin's Settings dialog.
# Supported types: "string", "password", "number", "boolean"

PLUGIN_SETTINGS = {
    "bot_token": {
        "type":     "password",
        "label":    "Bot Token",
        "required": True,
        # e.g. 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
    },
    "chat_id": {
        "type":     "string",
        "label":    "Chat ID",
        "required": True,
        # e.g. -1001234567890 (group) or 987654321 (personal)
    },
    "min_confidence": {
        "type":    "number",
        "label":   "Minimum Confidence % to send",
        "default": 60,
    },
    "only_strong": {
        "type":    "boolean",
        "label":   "Only STRONG_BUY / STRONG_SELL (skip BUY/SELL)",
        "default": True,
    },
}


# ── Plugin entry point ────────────────────────────────────────────────────────

def run(context: dict, settings: dict) -> dict:
    """
    Called by the plugin manager when a signal fires.

    Parameters
    ----------
    context : dict
        For notifier plugins, contains:
          - ticker        (str)   e.g. "AAPL"
          - signal        (str)   e.g. "STRONG_BUY"
          - recommendation (str)  short summary from the AI analysis
          - confidence    (int)   0–100
          - risk_score    (int)   1–10

    settings : dict
        Values the user configured in the Settings UI.

    Returns
    -------
    dict  {"ok": bool, "message": str}
    """
    bot_token = (settings.get("bot_token") or "").strip()
    chat_id   = (settings.get("chat_id") or "").strip()

    if not bot_token or not chat_id:
        return {"ok": False, "message": "bot_token and chat_id are required"}

    ticker      = context.get("ticker", "?")
    signal      = context.get("signal", "?")
    confidence  = context.get("confidence", 0)
    risk_score  = context.get("risk_score", 5)
    recommendation = (context.get("recommendation") or "")[:300]

    # Apply filters
    min_confidence = float(settings.get("min_confidence") or 0)
    if confidence < min_confidence:
        return {"ok": True, "message": f"Skipped (confidence {confidence} < {min_confidence})"}

    only_strong = settings.get("only_strong", True)
    if only_strong and signal not in ("STRONG_BUY", "STRONG_SELL"):
        return {"ok": True, "message": f"Skipped (signal={signal}, only_strong=True)"}

    # Build message
    emoji = "🟢" if "BUY" in signal else "🔴"
    text = (
        f"{emoji} *{signal}* — {ticker}\n"
        f"Confidence: {confidence}%  |  Risk: {risk_score}/10\n\n"
        f"{recommendation}"
    )

    return _send_telegram(bot_token, chat_id, text)


def _send_telegram(bot_token: str, chat_id: str, text: str) -> dict:
    """POST a message to the Telegram Bot API."""
    import urllib.request
    import urllib.parse
    import json as _json

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = _json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = _json.loads(resp.read())
            if body.get("ok"):
                return {"ok": True, "message": f"Sent to Telegram chat {chat_id}"}
            return {"ok": False, "message": f"Telegram error: {body.get('description', 'unknown')}"}
    except Exception as e:
        return {"ok": False, "message": f"HTTP error: {e}"}
