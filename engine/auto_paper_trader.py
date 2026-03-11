"""
Automatic Paper Trading Validation
Simulates entering and exiting trades based on AI/Quant signals to build a verifiable track record.
Settings are read from the DB so the user can configure them via the Settings page.
"""
import secrets
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from core.database import db

logger = logging.getLogger(__name__)


class AutoPaperTrader:
    def __init__(self):
        self._init_tables()

    # ------------------------------------------------------------------ #
    #  DB init                                                             #
    # ------------------------------------------------------------------ #

    def _init_tables(self):
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS auto_paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_date TEXT,
                    entry_price REAL,
                    exit_date TEXT,
                    exit_price REAL,
                    status TEXT DEFAULT 'open',
                    close_reason TEXT,
                    pnl_pct REAL,
                    blocked_reason TEXT,
                    UNIQUE(analysis_id)
                )
            """)
            # Idempotent: add columns to existing DBs
            for col_sql in [
                "ALTER TABLE auto_paper_trades ADD COLUMN blocked_reason TEXT",
            ]:
                try:
                    db.execute(col_sql)
                except Exception:
                    pass

            db.execute("""
                CREATE TABLE IF NOT EXISTS auto_trade_pending (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL,
                    analysis_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    score INTEGER,
                    proposed_entry_price REAL,
                    proposed_shares REAL,
                    proposed_size_usd REAL,
                    risk_tp_price REAL,
                    risk_sl_price REAL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    decided_at TEXT
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create auto_paper_trader tables: {e}")

    # ------------------------------------------------------------------ #
    #  Settings helpers                                                    #
    # ------------------------------------------------------------------ #

    def _setting(self, key: str, default):
        val = db.get_setting(key)
        return val if val is not None else default

    def _get_config(self) -> Dict[str, Any]:
        return {
            "enabled":           self._setting("auto_trade_enabled", False),
            "mode":              self._setting("auto_trade_mode", "paper"),
            "signal_filter":     self._setting("auto_trade_signal_filter", "STRONG"),
            "take_profit_pct":   float(self._setting("auto_trade_take_profit_pct", 8.0)) / 100,
            "stop_loss_pct":     float(self._setting("auto_trade_stop_loss_pct", 4.0)) / 100,
            "max_days":          int(self._setting("auto_trade_max_days_open", 30)),
            "position_size_pct": float(self._setting("auto_trade_position_size_pct", 5.0)) / 100,
            "max_positions":     int(self._setting("auto_trade_max_open_positions", 10)),
            "require_confirm":   self._setting("auto_trade_require_confirm", True),
        }

    def _get_trust_config(self) -> Dict[str, Any]:
        return {
            "min_trades":   int(self._setting("auto_trade_min_trust_trades", 20)),
            "min_win_rate": float(self._setting("auto_trade_min_trust_win_rate", 55.0)),
        }

    # ------------------------------------------------------------------ #
    #  Risk Gate                                                           #
    # ------------------------------------------------------------------ #

    def _run_risk_gate(self, ticker: str, proposed_size_usd: float) -> Dict[str, Any]:
        """
        Pre-trade risk checks (fail-fast):
          1. Global portfolio loss gate
          2. Position concentration
          3. Sector concentration
          4. Duplicate open position
        Returns {allowed, reason, checks[]}.
        """
        checks = []

        # 1. Global portfolio loss gate
        try:
            from engine.portfolio_manager import portfolio_manager
            gate = portfolio_manager.get_risk_gate_status()
            ok = not gate.get("active", False)
            checks.append({
                "name": "global_loss_gate",
                "ok": ok,
                "detail": f"Portfolio loss: {gate.get('loss_pct', 0):.1f}% (limit {gate.get('threshold_pct', 10):.0f}%)"
            })
            if not ok:
                return {"allowed": False, "reason": f"Global risk gate active: {gate.get('reason', 'loss limit hit')}", "checks": checks}
        except Exception:
            checks.append({"name": "global_loss_gate", "ok": True, "detail": "skipped (no portfolio)"})

        # 2. Position concentration
        try:
            max_pos_pct = float(self._setting("portfolio_max_position_pct", 10.0)) / 100
            from engine.portfolio_manager import portfolio_manager
            rules = portfolio_manager.check_all_rules()
            portfolio_value = rules.get("portfolio_value", 0)
            if portfolio_value > 0:
                pos_pct = proposed_size_usd / portfolio_value
                ok = pos_pct <= max_pos_pct
                checks.append({
                    "name": "position_concentration",
                    "ok": ok,
                    "detail": f"{ticker} would be {pos_pct*100:.1f}% (limit {max_pos_pct*100:.0f}%)"
                })
                if not ok:
                    return {"allowed": False, "reason": f"Position {ticker} {pos_pct*100:.1f}% > limit {max_pos_pct*100:.0f}%", "checks": checks}
            else:
                checks.append({"name": "position_concentration", "ok": True, "detail": "skipped (empty portfolio)"})
        except Exception:
            checks.append({"name": "position_concentration", "ok": True, "detail": "skipped"})

        # 3. Sector concentration
        try:
            max_sector_pct = float(self._setting("portfolio_max_sector_pct", 30.0)) / 100
            stock_info = yf.Ticker(ticker).info or {}
            sector = stock_info.get("sector", "")
            if sector:
                from engine.portfolio_manager import portfolio_manager
                rules = portfolio_manager.check_all_rules()
                sector_weights: Dict[str, float] = {}
                for h in rules.get("holdings", []):
                    s = h.get("sector") or "Unknown"
                    sector_weights[s] = sector_weights.get(s, 0) + (h.get("position_pct", 0) / 100)
                current = sector_weights.get(sector, 0)
                ok = current <= max_sector_pct
                checks.append({
                    "name": "sector_concentration",
                    "ok": ok,
                    "detail": f"{sector} at {current*100:.1f}% (limit {max_sector_pct*100:.0f}%)"
                })
                if not ok:
                    return {"allowed": False, "reason": f"Sector {sector} {current*100:.1f}% > limit {max_sector_pct*100:.0f}%", "checks": checks}
            else:
                checks.append({"name": "sector_concentration", "ok": True, "detail": "sector unknown — skipped"})
        except Exception:
            checks.append({"name": "sector_concentration", "ok": True, "detail": "skipped"})

        # 4. Duplicate open position
        existing = db.query_one(
            "SELECT id FROM auto_paper_trades WHERE ticker = ? AND status = 'open'", (ticker,)
        )
        ok = existing is None
        checks.append({
            "name": "duplicate_position",
            "ok": ok,
            "detail": f"{ticker} already open" if not ok else f"No existing {ticker} position"
        })
        if not ok:
            return {"allowed": False, "reason": f"Already have open position in {ticker}", "checks": checks}

        return {"allowed": True, "reason": "all clear", "checks": checks}

    # ------------------------------------------------------------------ #
    #  Core entry / exit logic                                             #
    # ------------------------------------------------------------------ #

    def process_new_signals(self) -> int:
        """Find recent signals and enter (or queue) paper trades.

        Respects: enabled gate, signal_filter, max_open_positions, dedup,
        risk gate, position_size_pct, require_confirm.
        """
        cfg = self._get_config()
        if not cfg["enabled"]:
            return 0

        cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')

        if cfg["signal_filter"] == "STRONG":
            signals_clause = "('STRONG_BUY', 'STRONG_SELL')"
        else:
            signals_clause = "('STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL')"

        new_signals = db.query(f"""
            SELECT id, ticker, signal, score, timestamp
            FROM analysis_history
            WHERE signal IN {signals_clause}
              AND timestamp >= ?
              AND id NOT IN (
                  SELECT analysis_id FROM auto_paper_trades WHERE analysis_id IS NOT NULL
                  UNION
                  SELECT analysis_id FROM auto_trade_pending WHERE analysis_id IS NOT NULL
              )
        """, (cutoff_24h,))

        if not new_signals:
            return 0

        open_count_row = db.query_one(
            "SELECT COUNT(*) as c FROM auto_paper_trades WHERE status = 'open'"
        )
        open_count = open_count_row['c'] if open_count_row else 0

        count = 0
        for sig in new_signals:
            if open_count >= cfg["max_positions"]:
                logger.info(f"Auto-trade: max_positions ({cfg['max_positions']}) reached, skipping")
                break

            ticker = sig['ticker']
            direction = 'LONG' if sig['signal'] in ['STRONG_BUY', 'BUY'] else 'SHORT'

            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if hist.empty:
                    continue

                entry_price = float(hist['Close'].iloc[-1])
                portfolio_value = self._estimate_portfolio_value()
                position_usd = portfolio_value * cfg["position_size_pct"]
                shares = position_usd / entry_price if entry_price > 0 else 0

                gate = self._run_risk_gate(ticker, position_usd)
                if not gate["allowed"]:
                    logger.info(f"Auto-trade BLOCKED for {ticker}: {gate['reason']}")
                    try:
                        db.execute("""
                            INSERT OR IGNORE INTO auto_paper_trades
                            (analysis_id, ticker, direction, entry_date, entry_price, status, blocked_reason)
                            VALUES (?, ?, ?, ?, ?, 'blocked', ?)
                        """, (sig['id'], ticker, direction,
                              datetime.now().isoformat(), entry_price, gate['reason']))
                    except Exception:
                        pass
                    continue

                if cfg["require_confirm"]:
                    self._create_pending(sig, ticker, direction, entry_price, shares, position_usd, cfg)
                    count += 1
                    logger.info(f"Auto-trade PENDING confirmation: {direction} {ticker} @ ${entry_price:.2f}")
                else:
                    db.execute("""
                        INSERT INTO auto_paper_trades
                        (analysis_id, ticker, direction, entry_date, entry_price)
                        VALUES (?, ?, ?, ?, ?)
                    """, (sig['id'], ticker, direction, datetime.now().isoformat(), entry_price))
                    open_count += 1
                    count += 1
                    logger.info(f"Auto-trade ENTERED: {direction} {ticker} @ ${entry_price:.2f} (${position_usd:.0f})")

            except Exception as e:
                logger.error(f"Failed to process auto-trade for {ticker}: {e}")

        return count

    def _estimate_portfolio_value(self) -> float:
        """Get paper portfolio total value for position sizing; fallback 10k."""
        try:
            from engine.paper_trading import paper_trader
            summary = paper_trader.get_portfolio_summary()
            return float(summary.get("total_value", 10000))
        except Exception:
            return 10000.0

    def _create_pending(self, sig: dict, ticker: str, direction: str,
                         entry_price: float, shares: float, size_usd: float,
                         cfg: dict):
        token = secrets.token_hex(16)
        now = datetime.now()
        expires = now + timedelta(minutes=5)
        tp = entry_price * (1 + cfg["take_profit_pct"]) if direction == "LONG" else entry_price * (1 - cfg["take_profit_pct"])
        sl = entry_price * (1 - cfg["stop_loss_pct"]) if direction == "LONG" else entry_price * (1 + cfg["stop_loss_pct"])

        try:
            db.execute("""
                INSERT OR IGNORE INTO auto_trade_pending
                (token, analysis_id, ticker, direction, signal, score,
                 proposed_entry_price, proposed_shares, proposed_size_usd,
                 risk_tp_price, risk_sl_price, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (token, sig['id'], ticker, direction, sig['signal'], sig.get('score'),
                  entry_price, shares, size_usd, tp, sl,
                  now.isoformat(), expires.isoformat()))
        except Exception as e:
            logger.error(f"Failed to create pending confirmation: {e}")
            return

        self._notify_pending(token, ticker, direction, sig['signal'], sig.get('score'),
                              entry_price, size_usd, tp, sl)

    def _notify_pending(self, token: str, ticker: str, direction: str, signal: str,
                         score, entry_price: float, size_usd: float,
                         tp_price: float, sl_price: float):
        try:
            tp_pct = (tp_price / entry_price - 1) * 100
            sl_pct = (sl_price / entry_price - 1) * 100
            msg = (
                f"⚡ *Auto-Trade Proposal: {ticker} {direction}*\n"
                f"Signal: {signal}" + (f"  ·  Score: {score}/100" if score else "") + "\n"
                f"Entry: ${entry_price:.2f}  |  Size: ${size_usd:.0f}\n"
                f"Target: ${tp_price:.2f} ({tp_pct:+.1f}%)  ·  Stop: ${sl_price:.2f} ({sl_pct:+.1f}%)\n"
                f"_Expires in 5 minutes_"
            )
            from engine.webhook_notifier import webhook_notifier
            # Telegram: send with Approve / Skip inline keyboard
            buttons = [[
                {"text": "✅ Approve", "callback_data": f"at_approve:{token}"},
                {"text": "❌ Skip",    "callback_data": f"at_skip:{token}"},
            ]]
            sent = webhook_notifier.send_with_inline_keyboard(msg, buttons)
            # Fallback to plain message if keyboard send not supported/configured
            if not sent:
                webhook_notifier.send_custom(msg)
        except Exception:
            pass

        # Email: signed one-click confirm/skip links
        try:
            from core.notifications import notifications
            notifications.reload_settings()
            site_url = (db.get_setting("site_url") or "").rstrip("/")
            if not site_url:
                # Best-effort fallback using configured host/port
                from core.config import WEB_HOST, WEB_PORT
                host = WEB_HOST if WEB_HOST != "0.0.0.0" else "localhost"
                site_url = f"http://{host}:{WEB_PORT}"
            tp_pct = (tp_price / entry_price - 1) * 100
            sl_pct = (sl_price / entry_price - 1) * 100
            confirm_url = f"{site_url}/action/auto-trade/confirm/{token}"
            skip_url    = f"{site_url}/action/auto-trade/skip/{token}"
            subject = f"⚡ Auto-Trade Proposal: {ticker} {direction}"
            html = f"""
<html><body style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;background:#f9fafb;">
<div style="background:#111827;color:#fff;padding:20px;text-align:center;">
  <h2 style="margin:0;">⚡ Auto-Trade Proposal</h2>
  <h3 style="margin:8px 0 0;font-weight:400;">{ticker} {direction}</h3>
</div>
<div style="padding:20px;">
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <tr><td style="padding:8px 0;color:#6b7280;">Signal</td><td style="font-weight:600;">{signal}{(' · Score: ' + str(score) + '/100') if score else ''}</td></tr>
    <tr><td style="padding:8px 0;color:#6b7280;">Entry price</td><td style="font-family:monospace;">${entry_price:.2f}</td></tr>
    <tr><td style="padding:8px 0;color:#6b7280;">Position size</td><td style="font-family:monospace;">${size_usd:.0f}</td></tr>
    <tr><td style="padding:8px 0;color:#6b7280;">Take-profit</td><td style="color:#10b981;font-family:monospace;">${tp_price:.2f} ({tp_pct:+.1f}%)</td></tr>
    <tr><td style="padding:8px 0;color:#6b7280;">Stop-loss</td><td style="color:#ef4444;font-family:monospace;">${sl_price:.2f} ({sl_pct:+.1f}%)</td></tr>
  </table>
  <p style="color:#9ca3af;font-size:12px;margin:16px 0;">This proposal expires in 5 minutes.</p>
  <div style="text-align:center;margin:24px 0;display:flex;gap:12px;justify-content:center;">
    <a href="{confirm_url}" style="display:inline-block;padding:12px 28px;background:#10b981;color:#fff;text-decoration:none;border-radius:4px;font-weight:600;">✅ Approve</a>
    <a href="{skip_url}"    style="display:inline-block;padding:12px 28px;background:#6b7280;color:#fff;text-decoration:none;border-radius:4px;font-weight:600;">❌ Skip</a>
  </div>
  <p style="color:#9ca3af;font-size:11px;text-align:center;">AI Investment Monitor — Auto-Trading</p>
</div>
</body></html>"""
            notifications._send_email(subject, html)
        except Exception:
            pass

    def confirm_trade(self, token: str) -> Dict[str, Any]:
        """Approve a pending confirmation. Executes the entry if still valid."""
        pending = db.query_one(
            "SELECT * FROM auto_trade_pending WHERE token = ? AND status = 'pending'", (token,)
        )
        if not pending:
            return {"success": False, "error": "Token not found or already decided"}

        try:
            expires = datetime.fromisoformat(pending['expires_at'])
            if datetime.now() > expires:
                db.execute(
                    "UPDATE auto_trade_pending SET status='expired', decided_at=? WHERE token=?",
                    (datetime.now().isoformat(), token)
                )
                return {"success": False, "error": "Confirmation expired"}
        except Exception:
            pass

        try:
            db.execute("""
                INSERT OR IGNORE INTO auto_paper_trades
                (analysis_id, ticker, direction, entry_date, entry_price)
                VALUES (?, ?, ?, ?, ?)
            """, (pending['analysis_id'], pending['ticker'], pending['direction'],
                  datetime.now().isoformat(), pending['proposed_entry_price']))

            db.execute(
                "UPDATE auto_trade_pending SET status='approved', decided_at=? WHERE token=?",
                (datetime.now().isoformat(), token)
            )
            logger.info(f"Auto-trade CONFIRMED: {pending['direction']} {pending['ticker']}")
            return {"success": True, "ticker": pending['ticker'], "direction": pending['direction']}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def skip_trade(self, token: str) -> Dict[str, Any]:
        """Skip/decline a pending trade by token."""
        db.execute(
            "UPDATE auto_trade_pending SET status='skipped', decided_at=? WHERE token=? AND status='pending'",
            (datetime.now().isoformat(), token)
        )
        return {"success": True}

    def expire_pending(self):
        """Expire all pending confirmations that have passed their expiry time."""
        now = datetime.now().isoformat()
        db.execute(
            "UPDATE auto_trade_pending SET status='expired', decided_at=? WHERE status='pending' AND expires_at < ?",
            (now, now)
        )

    def check_open_positions(self) -> int:
        """Check open trades for exit conditions (TP / SL / time limit)."""
        open_trades = db.query("SELECT * FROM auto_paper_trades WHERE status = 'open'")
        if not open_trades:
            return 0

        cfg = self._get_config()
        count = 0
        now = datetime.now()

        for trade in open_trades:
            ticker = trade['ticker']
            entry_price = trade['entry_price']
            direction = trade['direction']

            try:
                try:
                    entry_dt = datetime.fromisoformat(trade['entry_date'])
                except (ValueError, TypeError):
                    entry_dt = datetime.strptime(trade['entry_date'], '%Y-%m-%d %H:%M:%S')

                days_open = (now - entry_dt).days

                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if hist.empty:
                    continue

                current_price = float(hist['Close'].iloc[-1])

                pnl_pct = (
                    (current_price - entry_price) / entry_price
                    if direction == 'LONG'
                    else (entry_price - current_price) / entry_price
                )

                close_reason = None
                if pnl_pct >= cfg["take_profit_pct"]:
                    close_reason = 'take_profit'
                elif pnl_pct <= -cfg["stop_loss_pct"]:
                    close_reason = 'stop_loss'
                elif days_open >= cfg["max_days"]:
                    close_reason = 'time_limit'

                if close_reason:
                    db.execute("""
                        UPDATE auto_paper_trades
                        SET exit_date = ?, exit_price = ?, status = 'closed',
                            close_reason = ?, pnl_pct = ?
                        WHERE id = ?
                    """, (now.isoformat(), current_price, close_reason, pnl_pct, trade['id']))
                    count += 1
                    logger.info(
                        f"Auto-trade EXIT: {direction} {ticker} @ ${current_price:.2f} "
                        f"PnL: {pnl_pct*100:+.2f}% ({close_reason})"
                    )
                    self._notify_close(ticker, direction, entry_price, current_price, pnl_pct, close_reason)

            except Exception as e:
                logger.error(f"Failed to check open position for {ticker}: {e}")

        return count

    def manual_close(self, trade_id: int) -> Dict[str, Any]:
        """Force-close a specific open position at current market price."""
        trade = db.query_one(
            "SELECT * FROM auto_paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
        )
        if not trade:
            return {"success": False, "error": "Trade not found or already closed"}

        try:
            hist = yf.Ticker(trade['ticker']).history(period="5d")
            if hist.empty:
                return {"success": False, "error": "Could not fetch current price"}

            current_price = float(hist['Close'].iloc[-1])
            entry_price = trade['entry_price']
            pnl_pct = (
                (current_price - entry_price) / entry_price
                if trade['direction'] == 'LONG'
                else (entry_price - current_price) / entry_price
            )

            db.execute("""
                UPDATE auto_paper_trades
                SET exit_date = ?, exit_price = ?, status = 'closed',
                    close_reason = 'manual', pnl_pct = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), current_price, pnl_pct, trade_id))

            logger.info(f"Auto-trade MANUAL CLOSE: {trade['ticker']} PnL {pnl_pct*100:+.2f}%")
            return {"success": True, "pnl_pct": round(pnl_pct * 100, 2), "exit_price": current_price}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _notify_close(self, ticker: str, direction: str, entry: float,
                       exit_price: float, pnl_pct: float, reason: str):
        try:
            from engine.webhook_notifier import webhook_notifier
            icons = {"take_profit": "🟢", "stop_loss": "🔴", "time_limit": "⏰", "manual": "✋"}
            emoji = icons.get(reason, "⚪")
            msg = (
                f"{emoji} *Auto-Trade Closed: {ticker}*\n"
                f"Direction: {direction}  ·  Entry: ${entry:.2f}  ·  Exit: ${exit_price:.2f}\n"
                f"*PnL: {pnl_pct*100:+.2f}%* ({reason})"
            )
            webhook_notifier.send_custom(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Queries / reporting                                                 #
    # ------------------------------------------------------------------ #

    def get_performance_summary(self) -> Dict[str, Any]:
        rows = db.query("""
            SELECT
                COUNT(*) as total_closed,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(pnl_pct) as total_pnl_pct
            FROM auto_paper_trades
            WHERE status = 'closed'
        """)

        summary = {
            "total_closed": 0,
            "win_rate_pct": 0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "open_positions": 0,
        }

        if rows and rows[0]['total_closed'] > 0:
            r = rows[0]
            summary["total_closed"] = r['total_closed']
            summary["win_rate_pct"] = round((r['winning_trades'] / r['total_closed']) * 100, 1)
            summary["avg_pnl_pct"] = round(r['avg_pnl_pct'] * 100, 2)
            summary["total_pnl_pct"] = round(r['total_pnl_pct'] * 100, 2)

        open_count = db.query_one(
            "SELECT COUNT(*) as c FROM auto_paper_trades WHERE status = 'open'"
        )
        if open_count:
            summary["open_positions"] = open_count['c']

        return summary

    def get_open_positions(self) -> List[Dict[str, Any]]:
        rows = db.query("""
            SELECT id, ticker, direction, entry_date, entry_price, origin
            FROM auto_paper_trades
            WHERE status = 'open'
            ORDER BY entry_date DESC
        """)
        return rows if rows else []

    def get_trade_log(self, limit: int = 50, page: int = 1) -> List[Dict[str, Any]]:
        """Paginated closed trades, newest first."""
        offset = (page - 1) * limit
        rows = db.query("""
            SELECT id, ticker, direction, entry_date, entry_price,
                   exit_date, exit_price, pnl_pct, close_reason, origin
            FROM auto_paper_trades
            WHERE status = 'closed'
            ORDER BY exit_date DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        return rows if rows else []

    def get_trade_log_count(self) -> int:
        row = db.query_one("SELECT COUNT(*) as c FROM auto_paper_trades WHERE status = 'closed'")
        return row['c'] if row else 0

    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        rows = db.query("""
            SELECT id, token, ticker, direction, signal, score,
                   proposed_entry_price, proposed_size_usd,
                   risk_tp_price, risk_sl_price, created_at, expires_at
            FROM auto_trade_pending
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """)
        return rows if rows else []

    def get_status(self) -> Dict[str, Any]:
        """Full status dict for the UI status card."""
        cfg = self._get_config()
        return {
            "enabled":       cfg["enabled"],
            "mode":          cfg["mode"],
            "require_confirm": cfg["require_confirm"],
            "performance":   self.get_performance_summary(),
            "trust_gate":    self.get_trust_gate(),
            "pending_count": len(self.get_pending_confirmations()),
        }

    def get_trust_gate(self) -> Dict[str, Any]:
        tc = self._get_trust_config()
        perf = self.get_performance_summary()
        closed = perf["total_closed"]
        win_rate = perf["win_rate_pct"]
        trusted = (
            closed >= tc["min_trades"]
            and win_rate >= tc["min_win_rate"]
            and perf["total_pnl_pct"] > 0
        )
        return {
            "trusted":          trusted,
            "closed":           closed,
            "win_rate_pct":     win_rate,
            "needed_trades":    tc["min_trades"],
            "needed_win_rate":  tc["min_win_rate"],
            "trades_remaining": max(0, tc["min_trades"] - closed),
        }

    def should_trust_signals(self) -> bool:
        return self.get_trust_gate()["trusted"]

    def get_risk_gate_status(self, ticker: str = "AAPL", proposed_size_usd: float = 500) -> Dict[str, Any]:
        """Public endpoint for the UI to check live risk gate state."""
        return self._run_risk_gate(ticker, proposed_size_usd)

    def get_blocked_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = db.query("""
            SELECT ticker, direction, entry_date, blocked_reason
            FROM auto_paper_trades
            WHERE status = 'blocked'
            ORDER BY entry_date DESC
            LIMIT ?
        """, (limit,))
        return rows if rows else []


# Singleton
auto_paper_trader = AutoPaperTrader()
