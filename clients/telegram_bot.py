"""
Two-way Telegram Bot — #14
Polls for incoming commands and responds with analysis, watchlist, geo, and top picks.
Security: only responds to the configured telegram_chat_id.
"""
import threading
import time
import logging
import requests
from datetime import datetime
from typing import Optional

from core.database import db

logger = logging.getLogger(__name__)

POLL_TIMEOUT = 30          # long-poll timeout in seconds
POLL_SLEEP_ON_ERROR = 10   # seconds to back-off on errors

HELP_TEXT = (
    "📊 *News Bot Commands*\n\n"
    "/analyze <TICKER> — Full AI analysis of a stock\n"
    "/watchlist — Show your current watchlist with latest signals\n"
    "/toppicks — Top 5 stocks by composite score (today)\n"
    "/geo — Latest geopolitical risk summary\n"
    "/status — System health & last scan time\n"
    "/help — Show this message"
)


class TelegramBot:
    """Long-polling Telegram bot that handles inbound commands."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._offset = 0
        self._load_settings()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self):
        self.enabled = bool(db.get_setting("telegram_bot_enabled"))
        self.token = db.get_setting("telegram_bot_token") or ""
        self.allowed_chat_id = str(db.get_setting("telegram_chat_id") or "").strip()

    def _base(self) -> str:
        return self.BASE_URL.format(token=self.token)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start background polling thread (idempotent)."""
        self._load_settings()
        if not self.enabled or not self.token:
            logger.debug("Telegram bot disabled or no token — skipping start")
            return
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-bot")
        self._thread.start()
        logger.info("Telegram bot polling started")

    def stop(self):
        """Signal polling thread to stop."""
        self._stop_event.set()

    def restart(self):
        """Reload settings and restart the polling thread."""
        self.stop()
        if self._thread:
            self._thread.join(timeout=5)
        self._load_settings()
        self.start()

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        logger.info("Telegram bot poll loop running")
        while not self._stop_event.is_set():
            try:
                updates = self._get_updates()
                for update in updates:
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as e:
                logger.error(f"Telegram poll error: {e}")
                time.sleep(POLL_SLEEP_ON_ERROR)

    def _get_updates(self) -> list:
        try:
            resp = requests.get(
                f"{self._base()}/getUpdates",
                params={"offset": self._offset, "timeout": POLL_TIMEOUT},
                timeout=POLL_TIMEOUT + 5,
            )
            data = resp.json()
            if data.get("ok"):
                return data.get("result", [])
        except Exception as e:
            logger.debug(f"getUpdates failed: {e}")
        return []

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_update(self, update: dict):
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))

        # Security gate — only the configured chat may use the bot
        if self.allowed_chat_id and chat_id != self.allowed_chat_id:
            self._send(chat_id, "⛔ Unauthorized.")
            logger.warning(f"Telegram bot: rejected message from chat {chat_id}")
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        parts = text.split()
        cmd = parts[0].lower().lstrip("/")
        # Strip @botname suffix (e.g. /analyze@MyBot)
        cmd = cmd.split("@")[0]
        args = parts[1:]

        if cmd == "analyze":
            self._cmd_analyze(chat_id, args)
        elif cmd == "watchlist":
            self._cmd_watchlist(chat_id)
        elif cmd == "toppicks":
            self._cmd_toppicks(chat_id)
        elif cmd == "geo":
            self._cmd_geo(chat_id)
        elif cmd == "status":
            self._cmd_status(chat_id)
        elif cmd in ("help", "start"):
            self._send(chat_id, HELP_TEXT)
        else:
            self._send(chat_id, f"Unknown command `{cmd}`. Use /help for a list of commands.")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_analyze(self, chat_id: str, args: list):
        if not args:
            self._send(chat_id, "Usage: /analyze <TICKER>  e.g. /analyze AAPL")
            return
        ticker = args[0].upper()
        self._send(chat_id, f"⏳ Analyzing *{ticker}*… this may take ~30 s.")
        threading.Thread(target=self._do_analyze, args=(chat_id, ticker), daemon=True).start()

    def _do_analyze(self, chat_id: str, ticker: str):
        try:
            from engine.agents import InvestmentSwarm
            swarm = InvestmentSwarm()
            result = swarm.analyze_single_stock(ticker)
            if not result or result.get("error"):
                self._send(chat_id, f"❌ Analysis failed for {ticker}: {result.get('error','unknown error')}")
                return

            signal = result.get("recommendation") or result.get("signal") or "N/A"
            score = result.get("composite_score") or result.get("score") or 0
            summary = result.get("summary") or result.get("reason") or ""
            anomalies = result.get("anomalies") or []
            anomaly_text = ""
            if anomalies:
                items = [a.get("description", "") for a in anomalies[:3] if a.get("description")]
                if items:
                    anomaly_text = "\n⚠️ " + "\n⚠️ ".join(items)

            db.save_analysis(ticker, result)

            msg = (
                f"📊 *{ticker} Analysis*\n"
                f"Signal: *{signal}*  |  Score: {score}/100\n"
                f"{summary[:300]}"
                f"{anomaly_text}"
            )
            self._send(chat_id, msg)
        except Exception as e:
            logger.error(f"Telegram /analyze {ticker} failed: {e}")
            self._send(chat_id, f"❌ Error analyzing {ticker}: {e}")

    def _cmd_watchlist(self, chat_id: str):
        try:
            watchlist = db.get_watchlist()
            if not watchlist:
                self._send(chat_id, "Your watchlist is empty.")
                return

            tickers = [item["ticker"] for item in watchlist]
            lines = ["📋 *Watchlist* (latest signals)\n"]
            for ticker in tickers[:20]:
                analysis = db.get_latest_analysis(ticker)
                if analysis:
                    signal = analysis.get("recommendation") or analysis.get("signal") or "—"
                    score = analysis.get("composite_score") or analysis.get("score") or "—"
                    lines.append(f"• *{ticker}* — {signal} ({score})")
                else:
                    lines.append(f"• *{ticker}* — no analysis yet")

            if len(tickers) > 20:
                lines.append(f"_…and {len(tickers)-20} more_")

            self._send(chat_id, "\n".join(lines))
        except Exception as e:
            logger.error(f"Telegram /watchlist failed: {e}")
            self._send(chat_id, f"❌ Error: {e}")

    def _cmd_toppicks(self, chat_id: str):
        try:
            rows = db.execute(
                """SELECT ticker, recommendation, composite_score, created_at
                   FROM analyses
                   WHERE date(created_at) = date('now')
                   ORDER BY composite_score DESC
                   LIMIT 5""",
                fetch=True,
            )
            if not rows:
                self._send(chat_id, "No analyses found for today yet.")
                return

            lines = ["🏆 *Top Picks Today*\n"]
            for i, row in enumerate(rows, 1):
                ticker = row["ticker"]
                signal = row["recommendation"] or "—"
                score = row["composite_score"] or "—"
                lines.append(f"{i}. *{ticker}* — {signal} ({score})")
            self._send(chat_id, "\n".join(lines))
        except Exception as e:
            logger.error(f"Telegram /toppicks failed: {e}")
            self._send(chat_id, f"❌ Error: {e}")

    def _cmd_geo(self, chat_id: str):
        try:
            rows = db.execute(
                """SELECT country, event_summary, severity, created_at
                   FROM geopolitical_events
                   ORDER BY created_at DESC
                   LIMIT 5""",
                fetch=True,
            )
            if not rows:
                self._send(chat_id, "No recent geopolitical events found.")
                return

            lines = ["🌍 *Recent Geopolitical Events*\n"]
            for row in rows:
                country = row.get("country") or "Global"
                summary = (row.get("event_summary") or "")[:120]
                severity = row.get("severity") or "—"
                date_str = (row.get("created_at") or "")[:10]
                lines.append(f"• [{date_str}] *{country}* (sev: {severity})\n  {summary}")
            self._send(chat_id, "\n".join(lines))
        except Exception as e:
            logger.error(f"Telegram /geo failed: {e}")
            self._send(chat_id, f"❌ Error: {e}")

    def _cmd_status(self, chat_id: str):
        try:
            last_scan = db.execute(
                "SELECT MAX(created_at) as ts FROM analyses", fetch=True
            )
            last_ts = (last_scan[0]["ts"] or "never") if last_scan else "never"

            watchlist_count = len(db.get_watchlist() or [])
            total_analyses = db.execute("SELECT COUNT(*) as n FROM analyses", fetch=True)
            n = total_analyses[0]["n"] if total_analyses else 0

            system_paused = db.get_setting("system_paused_accuracy") or False
            paused_str = "⏸ Paused (accuracy mode)" if system_paused else "✅ Running"

            msg = (
                f"🖥 *System Status*\n"
                f"State: {paused_str}\n"
                f"Last analysis: {last_ts[:19]}\n"
                f"Watchlist: {watchlist_count} tickers\n"
                f"Total analyses: {n}"
            )
            self._send(chat_id, msg)
        except Exception as e:
            logger.error(f"Telegram /status failed: {e}")
            self._send(chat_id, f"❌ Error: {e}")

    # ------------------------------------------------------------------
    # Send helper
    # ------------------------------------------------------------------

    def _send(self, chat_id: str, text: str):
        if not self.token:
            return
        try:
            requests.post(
                f"{self._base()}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")


# Singleton
telegram_bot = TelegramBot()
