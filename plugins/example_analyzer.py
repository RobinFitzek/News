"""
Example Plugin: Webhook Analyzer
==================================
Sends post-analysis data to a custom webhook endpoint (e.g. n8n, Make, Zapier).

This is an ANALYZER plugin — it runs after every Stage 3 AI analysis.
Use it to pipe analysis results into your own workflows.

Install, configure your webhook URL in Settings → Plugins, then enable.
"""

PLUGIN_NAME        = "Webhook Analyzer"
PLUGIN_VERSION     = "1.0.0"
PLUGIN_TYPE        = "analyzer"
PLUGIN_DESCRIPTION = "POSTs analysis results to a custom webhook URL"
PLUGIN_AUTHOR      = "Stockholm"

PLUGIN_SETTINGS = {
    "webhook_url": {
        "type":     "string",
        "label":    "Webhook URL",
        "required": True,
        # e.g. https://hooks.make.com/abc123 or https://n8n.example.com/webhook/xyz
    },
    "include_recommendation": {
        "type":    "boolean",
        "label":   "Include full recommendation text",
        "default": False,
    },
    "secret_header": {
        "type":  "password",
        "label": "X-Secret header value (optional)",
    },
}


def run(context: dict, settings: dict) -> dict:
    """
    Parameters
    ----------
    context : dict
        For analyzer plugins:
          - ticker          (str)
          - analysis_result (dict)  — signal, confidence, risk_score, bull_case, bear_case, …

    settings : dict
        User-configured values.
    """
    import json as _json
    import urllib.request

    webhook_url = (settings.get("webhook_url") or "").strip()
    if not webhook_url:
        return {"ok": False, "message": "webhook_url is required"}

    ticker          = context.get("ticker", "?")
    analysis        = context.get("analysis_result", {})
    include_full    = settings.get("include_recommendation", False)

    payload = {
        "ticker":     ticker,
        "signal":     analysis.get("signal"),
        "confidence": analysis.get("confidence"),
        "risk_score": analysis.get("risk_score"),
    }
    if include_full:
        payload["recommendation"] = analysis.get("recommendation", "")
        payload["bull_case"]      = analysis.get("bull_case", "")
        payload["bear_case"]      = analysis.get("bear_case", "")

    body = _json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    secret = (settings.get("secret_header") or "").strip()
    if secret:
        headers["X-Secret"] = secret

    req = urllib.request.Request(webhook_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"ok": True, "message": f"Delivered to webhook (status {resp.status})"}
    except Exception as e:
        return {"ok": False, "message": f"Webhook delivery failed: {e}"}
