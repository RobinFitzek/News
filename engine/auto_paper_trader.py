"""
Automatic Paper Trading Validation
Simulates entering and exiting trades based on AI/Quant signals to build a verifiable track record.
"""
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from core.database import db

logger = logging.getLogger(__name__)

class AutoPaperTrader:
    def __init__(self):
        self._init_table()
        
    def _init_table(self):
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
                    UNIQUE(analysis_id)
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create auto_paper_trades table: {e}")

    def process_new_signals(self) -> int:
        """Find recent STRONG_BUY / STRONG_SELL signals and enter paper trades."""
        cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get actionable signals not yet in trades
        new_signals = db.query("""
            SELECT id, ticker, signal, timestamp 
            FROM analysis_history 
            WHERE signal IN ('STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL') 
              AND timestamp >= ?
              AND id NOT IN (SELECT analysis_id FROM auto_paper_trades)
        """, (cutoff_24h,))
        
        if not new_signals:
            return 0
            
        count = 0
        for sig in new_signals:
            ticker = sig['ticker']
            direction = 'LONG' if sig['signal'] in ['STRONG_BUY', 'BUY'] else 'SHORT'
            
            try:
                # Get current market price (or last close)
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if hist.empty:
                    continue
                
                entry_price = float(hist['Close'].iloc[-1])
                
                db.execute("""
                    INSERT INTO auto_paper_trades 
                    (analysis_id, ticker, direction, entry_date, entry_price) 
                    VALUES (?, ?, ?, ?, ?)
                """, (sig['id'], ticker, direction, datetime.now().isoformat(), entry_price))
                count += 1
                
                logger.info(f"Paper Trade ENTRANCE: {direction} {ticker} at ${entry_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to enter paper trade for {ticker}: {e}")
                
        return count

    def check_open_positions(self) -> int:
        """Check all open paper trades for exit conditions."""
        open_trades = db.query("SELECT * FROM auto_paper_trades WHERE status = 'open'")
        if not open_trades:
            return 0
            
        count = 0
        now = datetime.now()
        
        take_profit_pct = 0.08  # +8%
        stop_loss_pct = -0.04   # -4%
        max_days_open = 30      # Exit after 30 days
        
        for trade in open_trades:
            ticker = trade['ticker']
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            try:
                entry_date_str = trade['entry_date']
                try:
                    entry_dt = datetime.fromisoformat(entry_date_str)
                except ValueError:
                    entry_dt = datetime.strptime(entry_date_str, '%Y-%m-%d %H:%M:%S')
                
                days_open = (now - entry_dt).days
                
                # Fetch current price
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if hist.empty:
                    continue
                    
                current_price = float(hist['Close'].iloc[-1])
                
                # Calculate PnL (without leverage, assume 1x)
                if direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                close_reason = None
                if pnl_pct >= take_profit_pct:
                    close_reason = 'take_profit'
                elif pnl_pct <= stop_loss_pct:
                    close_reason = 'stop_loss'
                elif days_open >= max_days_open:
                    close_reason = 'time_limit'
                    
                if close_reason:
                    db.execute("""
                        UPDATE auto_paper_trades 
                        SET exit_date = ?, exit_price = ?, status = 'closed', close_reason = ?, pnl_pct = ?
                        WHERE id = ?
                    """, (now.isoformat(), current_price, close_reason, pnl_pct, trade['id']))
                    count += 1
                    
                    pnl_display = pnl_pct * 100
                    logger.info(f"Paper Trade EXIT: {direction} {ticker} at ${current_price:.2f}. PnL: {pnl_display:+.2f}% ({close_reason})")
                    
                    try:
                        from engine.webhook_notifier import webhook_notifier
                        emoji = "🟢" if pnl_pct > 0 else "🔴"
                        msg = f"{emoji} *Paper Trade Closed: {ticker}*\nDirection: {direction}\nEntry: ${entry_price:.2f}\nExit: ${current_price:.2f}\n*PnL: {pnl_display:+.2f}%* ({close_reason})"
                        webhook_notifier.send_custom(msg)
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Failed to check open position for {ticker}: {e}")
                
        return count

    def get_performance_summary(self) -> Dict[str, Any]:
        """Aggregate paper trading performance."""
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
            "open_positions": 0
        }
        
        if rows and rows[0]['total_closed'] > 0:
            r = rows[0]
            summary["total_closed"] = r['total_closed']
            summary["win_rate_pct"] = round((r['winning_trades'] / r['total_closed']) * 100, 1)
            summary["avg_pnl_pct"] = round(r['avg_pnl_pct'] * 100, 2)
            summary["total_pnl_pct"] = round(r['total_pnl_pct'] * 100, 2)
            
        open_count = db.query_one("SELECT COUNT(*) as c FROM auto_paper_trades WHERE status = 'open'")
        if open_count:
            summary["open_positions"] = open_count['c']
            
        return summary
        
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return a list of currently open paper trades."""
        rows = db.query("""
            SELECT id, ticker, direction, entry_date, entry_price 
            FROM auto_paper_trades 
            WHERE status = 'open' 
            ORDER BY entry_date DESC
        """)
        
        return rows
        
    def should_trust_signals(self) -> bool:
        """Simple heuristic: if win rate > 55% or avg PnL is solidly positive over > 20 trades, trust is high."""
        summary = self.get_performance_summary()
        if summary["total_closed"] < 20:
            return False
            
        if summary["win_rate_pct"] > 55.0 and summary["total_pnl_pct"] > 0:
            return True
            
        return False

# Singleton
auto_paper_trader = AutoPaperTrader()
