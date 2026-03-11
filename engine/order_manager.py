"""
Order Manager — Phase 6 of Auto-Trading Integration
Routes trade execution through the appropriate BrokerClient and keeps
auto_paper_trades in sync with broker positions.
"""
import logging
from datetime import datetime
from typing import Dict, Any

import yfinance as yf

from core.database import db
from clients.broker_client import get_broker_client

logger = logging.getLogger(__name__)


def _fetch_price(ticker: str) -> float:
    """Fetch latest close price via yfinance; raises on failure."""
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist["Close"].iloc[-1])


class OrderManager:
    """Orchestrates order lifecycle: entry → broker → DB → notification."""

    # ------------------------------------------------------------------ #
    #  Entry                                                               #
    # ------------------------------------------------------------------ #

    def execute_entry(self, ticker: str, direction: str, size_usd: float,
                      analysis_id: int = None) -> Dict[str, Any]:
        """
        Execute a new trade entry.

        1. Risk gate check
        2. Fetch current price
        3. Calculate share qty
        4. Place order via broker
        5. Write to auto_paper_trades
        6. Send notification

        Returns {success, trade_id, order_id, error}.
        """
        from engine.auto_paper_trader import auto_paper_trader

        # 1. Risk gate
        gate = auto_paper_trader._run_risk_gate(ticker, size_usd)
        if not gate["allowed"]:
            logger.warning(f"OrderManager: entry blocked for {ticker}: {gate['reason']}")
            try:
                db.execute(
                    """INSERT OR IGNORE INTO auto_paper_trades
                       (analysis_id, ticker, direction, entry_date, entry_price, status, blocked_reason, origin)
                       VALUES (?, ?, ?, ?, 0, 'blocked', ?, 'auto')""",
                    (analysis_id, ticker, direction, datetime.now().isoformat(), gate["reason"])
                )
            except Exception:
                pass
            return {"success": False, "error": gate["reason"], "trade_id": None, "order_id": None}

        # 2. Price
        try:
            price = _fetch_price(ticker)
        except Exception as e:
            return {"success": False, "error": f"Price fetch failed: {e}", "trade_id": None, "order_id": None}

        # 3. Qty
        qty = size_usd / price if price > 0 else 0

        # 4. Broker order
        broker = get_broker_client()
        side = "buy" if direction == "LONG" else "sell"
        order_result = broker.place_order(ticker, qty, side)
        order_id = order_result.get("order_id")
        order_status = order_result.get("status", "unknown")

        if order_result.get("status") == "error":
            logger.error(f"OrderManager: broker order failed for {ticker}: {order_result.get('error')}")
            return {
                "success": False,
                "error": order_result.get("error", "Broker order failed"),
                "trade_id": None,
                "order_id": None,
            }

        # 5. Write to DB — add origin column idempotently
        self._ensure_origin_column()
        try:
            db.execute(
                """INSERT INTO auto_paper_trades
                   (analysis_id, ticker, direction, entry_date, entry_price, status, origin)
                   VALUES (?, ?, ?, ?, ?, 'open', 'auto')""",
                (analysis_id, ticker, direction, datetime.now().isoformat(), price)
            )
            row = db.query_one(
                "SELECT id FROM auto_paper_trades WHERE ticker=? AND status='open' ORDER BY id DESC LIMIT 1",
                (ticker,)
            )
            trade_id = row["id"] if row else None
        except Exception as e:
            logger.error(f"OrderManager: DB write failed: {e}")
            return {"success": False, "error": str(e), "trade_id": None, "order_id": order_id}

        # 6. Notification
        try:
            from engine.webhook_notifier import webhook_notifier
            mode = (db.get_setting("auto_trade_mode") or "paper").upper()
            msg = (
                f"⚡ *Trade Executed: {ticker} {direction}*\n"
                f"Mode: {mode}  ·  Price: ${price:.2f}  ·  Size: ${size_usd:.0f}\n"
                f"Order: {order_id or 'paper'}  ·  Status: {order_status}"
            )
            webhook_notifier.send_custom(msg)
        except Exception:
            pass

        logger.info(f"OrderManager: entry executed — {direction} {ticker} @ ${price:.2f} (order {order_id})")
        return {"success": True, "trade_id": trade_id, "order_id": order_id, "price": price}

    # ------------------------------------------------------------------ #
    #  Exit                                                                #
    # ------------------------------------------------------------------ #

    def execute_exit(self, trade_id: int, reason: str = "manual") -> Dict[str, Any]:
        """
        Close an open position.

        1. Fetch trade from DB
        2. Fetch current price
        3. Place sell/cover order via broker
        4. Update DB with exit data
        5. Send notification

        Returns {success, pnl_pct, order_id, error}.
        """
        trade = db.query_one(
            "SELECT * FROM auto_paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
        )
        if not trade:
            return {"success": False, "error": f"Open trade {trade_id} not found"}

        ticker = trade["ticker"]
        direction = trade.get("direction", "LONG")
        entry_price = float(trade.get("entry_price") or 0)

        # Price
        try:
            exit_price = _fetch_price(ticker)
        except Exception as e:
            return {"success": False, "error": f"Price fetch failed: {e}"}

        # Broker order
        side = "sell" if direction == "LONG" else "buy"  # cover short with buy
        broker = get_broker_client()
        order_result = broker.place_order(ticker, 1, side)  # qty 1 placeholder for paper
        order_id = order_result.get("order_id")

        if order_result.get("status") == "error":
            return {"success": False, "error": order_result.get("error", "Broker order failed")}

        # PnL
        if entry_price > 0:
            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
        else:
            pnl_pct = 0.0

        # Update DB
        try:
            db.execute(
                """UPDATE auto_paper_trades
                   SET exit_date=?, exit_price=?, status='closed', close_reason=?, pnl_pct=?
                   WHERE id=?""",
                (datetime.now().isoformat(), exit_price, reason, pnl_pct, trade_id)
            )
        except Exception as e:
            logger.error(f"OrderManager: exit DB update failed: {e}")
            return {"success": False, "error": str(e)}

        # Notification
        try:
            from engine.webhook_notifier import webhook_notifier
            sign = "+" if pnl_pct >= 0 else ""
            emoji = "✅" if pnl_pct >= 0 else "🔴"
            msg = (
                f"{emoji} *Trade Closed: {ticker} {direction}*\n"
                f"Entry: ${entry_price:.2f}  ·  Exit: ${exit_price:.2f}\n"
                f"PnL: {sign}{pnl_pct*100:.2f}%  ·  Reason: {reason}"
            )
            webhook_notifier.send_custom(msg)
        except Exception:
            pass

        logger.info(f"OrderManager: exit executed — {ticker} @ ${exit_price:.2f} ({pnl_pct*100:+.2f}%)")
        return {"success": True, "pnl_pct": round(pnl_pct * 100, 2), "order_id": order_id, "exit_price": exit_price}

    # ------------------------------------------------------------------ #
    #  Broker sync                                                         #
    # ------------------------------------------------------------------ #

    def sync_broker_positions(self) -> int:
        """
        Pull live positions from broker and sync auto_paper_trades:
        - Upsert positions found at broker (origin='broker-sync')
        - Mark trades closed if they're absent at broker

        Returns count of upserted/updated rows.
        """
        mode = (db.get_setting("auto_trade_mode") or "paper").lower()
        if mode == "paper":
            logger.debug("OrderManager: sync skipped (paper mode)")
            return 0

        broker = get_broker_client()
        try:
            live = broker.get_positions()
        except Exception as e:
            logger.error(f"OrderManager: get_positions failed: {e}")
            return 0

        self._ensure_origin_column()
        live_tickers = {p["ticker"] for p in live}
        synced = 0

        # Upsert live positions
        for pos in live:
            ticker = pos["ticker"]
            existing = db.query_one(
                "SELECT id FROM auto_paper_trades WHERE ticker=? AND status='open'", (ticker,)
            )
            if existing:
                # Update with live values (if we have them)
                try:
                    db.execute(
                        "UPDATE auto_paper_trades SET origin='broker-sync' WHERE id=?",
                        (existing["id"],)
                    )
                    synced += 1
                except Exception:
                    pass
            else:
                # New position found at broker — insert it
                try:
                    direction = pos.get("side", "LONG")
                    price = pos.get("avg_entry_price", 0)
                    db.execute(
                        """INSERT OR IGNORE INTO auto_paper_trades
                           (ticker, direction, entry_date, entry_price, status, origin)
                           VALUES (?, ?, ?, ?, 'open', 'broker-sync')""",
                        (ticker, direction, datetime.now().isoformat(), price)
                    )
                    synced += 1
                except Exception as e:
                    logger.error(f"OrderManager: sync insert failed for {ticker}: {e}")

        # Mark closed if absent at broker
        open_trades = db.query(
            "SELECT id, ticker FROM auto_paper_trades WHERE status='open' AND origin='broker-sync'"
        ) or []
        for trade in open_trades:
            if trade["ticker"] not in live_tickers:
                try:
                    db.execute(
                        "UPDATE auto_paper_trades SET status='closed', close_reason='broker-sync', exit_date=? WHERE id=?",
                        (datetime.now().isoformat(), trade["id"])
                    )
                    synced += 1
                except Exception:
                    pass

        db.set_setting("broker_last_sync", datetime.now().isoformat())
        logger.info(f"OrderManager: broker sync complete — {synced} positions synced")
        return synced

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _ensure_origin_column(self):
        """Add origin column to auto_paper_trades if not present (idempotent)."""
        try:
            db.execute("ALTER TABLE auto_paper_trades ADD COLUMN origin TEXT DEFAULT 'auto'")
        except Exception:
            pass  # Column already exists


# Singleton
order_manager = OrderManager()
