"""
Paper Trading Engine
Simulates trading based on system signals to validate performance over time.
Auto-executes trades when confidence exceeds threshold.
"""
import json
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CAPITAL = 10000.0
DEFAULT_POSITION_SIZE_PCT = 5.0  # 5% per trade
DEFAULT_MIN_CONFIDENCE = 70
DEFAULT_MAX_POSITIONS = 10


class PaperTrader:
    """
    Virtual portfolio that auto-executes signals.
    Tracks performance vs SPY benchmark.
    """
    
    def __init__(self):
        self._ensure_tables()
        self._settings_cache = None
        self._spy_cache = {}
    
    def _ensure_tables(self):
        """Create paper trading tables if they don't exist."""
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Paper trading settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                starting_capital REAL DEFAULT 10000.0,
                position_size_pct REAL DEFAULT 5.0,
                min_confidence INTEGER DEFAULT 70,
                max_positions INTEGER DEFAULT 10,
                auto_execute INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize settings if not exist
        cursor.execute('INSERT OR IGNORE INTO paper_settings (id) VALUES (1)')
        
        # Paper trades log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                signal_id INTEGER,
                signal_confidence INTEGER,
                pnl REAL,
                pnl_pct REAL,
                status TEXT DEFAULT 'OPEN',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Daily portfolio snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT UNIQUE NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                positions_json TEXT,
                spy_value REAL,
                spy_return_pct REAL,
                portfolio_return_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_snapshots_date ON paper_snapshots(snapshot_date)')

        # Migrations: add slippage/commission columns
        migrations = [
            ("paper_settings", "slippage_pct", "REAL DEFAULT 0.1"),
            ("paper_settings", "commission", "REAL DEFAULT 0.0"),
            ("paper_trades", "slippage_cost", "REAL DEFAULT 0"),
            ("paper_trades", "commission_cost", "REAL DEFAULT 0"),
        ]
        for table, col, col_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists

        conn.commit()
        conn.close()
    
    def get_settings(self) -> Dict:
        """Get paper trading settings."""
        if self._settings_cache:
            return self._settings_cache
        
        row = db.query_one('SELECT * FROM paper_settings WHERE id = 1')
        if row:
            self._settings_cache = {
                'starting_capital': row.get('starting_capital', DEFAULT_CAPITAL),
                'position_size_pct': row.get('position_size_pct', DEFAULT_POSITION_SIZE_PCT),
                'min_confidence': row.get('min_confidence', DEFAULT_MIN_CONFIDENCE),
                'max_positions': row.get('max_positions', DEFAULT_MAX_POSITIONS),
                'auto_execute': bool(row.get('auto_execute', 1)),
                'slippage_pct': row.get('slippage_pct', 0.1),
                'commission': row.get('commission', 0.0),
            }
        else:
            self._settings_cache = {
                'starting_capital': DEFAULT_CAPITAL,
                'position_size_pct': DEFAULT_POSITION_SIZE_PCT,
                'min_confidence': DEFAULT_MIN_CONFIDENCE,
                'max_positions': DEFAULT_MAX_POSITIONS,
                'auto_execute': True,
                'slippage_pct': 0.1,
                'commission': 0.0,
            }
        return self._settings_cache
    
    def update_settings(self, **kwargs) -> bool:
        """Update paper trading settings."""
        try:
            valid_keys = ['starting_capital', 'position_size_pct', 'min_confidence',
                          'max_positions', 'auto_execute', 'slippage_pct', 'commission']
            updates = {k: v for k, v in kwargs.items() if k in valid_keys}
            
            if not updates:
                return False
            
            set_clause = ', '.join([f'{k} = ?' for k in updates.keys()])
            values = list(updates.values()) + [datetime.now().isoformat()]
            
            db.execute(
                f'UPDATE paper_settings SET {set_clause}, updated_at = ? WHERE id = 1',
                tuple(values)
            )
            self._settings_cache = None  # Clear cache
            return True
        except Exception as e:
            logger.error(f"Error updating paper settings: {e}")
            return False
    
    def get_current_cash(self) -> float:
        """Calculate current available cash."""
        settings = self.get_settings()
        starting = settings['starting_capital']
        
        # Sum all closed trade P&L
        closed = db.query('''
            SELECT COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(SUM(entry_price * shares), 0) as total_bought,
                   COALESCE(SUM(exit_price * shares), 0) as total_sold
            FROM paper_trades WHERE status = 'CLOSED'
        ''')
        closed_pnl = closed[0].get('total_pnl', 0) if closed else 0
        
        # Sum open positions cost
        open_positions = db.query('''
            SELECT COALESCE(SUM(entry_price * shares), 0) as invested
            FROM paper_trades WHERE status = 'OPEN'
        ''')
        invested = open_positions[0].get('invested', 0) if open_positions else 0
        
        return starting + closed_pnl - invested
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return db.query('''
            SELECT * FROM paper_trades 
            WHERE status = 'OPEN' 
            ORDER BY entry_date DESC
        ''')
    
    def get_position_count(self) -> int:
        """Get count of open positions."""
        result = db.query_one('SELECT COUNT(*) as cnt FROM paper_trades WHERE status = ?', ('OPEN',))
        return result.get('cnt', 0) if result else 0
    
    def has_open_position(self, ticker: str) -> bool:
        """Check if ticker has an open position."""
        result = db.query_one(
            'SELECT id FROM paper_trades WHERE ticker = ? AND status = ?',
            (ticker.upper(), 'OPEN')
        )
        return result is not None
    
    def _get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current price via yfinance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Could not fetch price for {ticker}: {e}")
        return None
    
    def execute_signal(self, ticker: str, signal: str, confidence: int, 
                       current_price: float = None, signal_id: int = None) -> Dict:
        """
        Execute a trade based on signal.
        Returns dict with success status and message.
        """
        settings = self.get_settings()
        ticker = ticker.upper()
        
        # Check if auto-execute is enabled
        if not settings['auto_execute']:
            return {'success': False, 'message': 'Auto-execute disabled'}
        
        # Check confidence threshold
        if confidence < settings['min_confidence']:
            return {'success': False, 'message': f'Confidence {confidence} < minimum {settings["min_confidence"]}'}
        
        # Get current price if not provided
        if current_price is None:
            current_price = self._get_current_price(ticker)
            if current_price is None:
                return {'success': False, 'message': 'Could not fetch current price'}
        
        # Handle BUY signal
        if signal.upper() in ('BUY', 'STRONG BUY', 'OPPORTUNITY'):
            return self._execute_buy(ticker, current_price, confidence, signal_id)
        
        # Handle SELL signal
        elif signal.upper() in ('SELL', 'STRONG SELL', 'CAUTION'):
            return self._execute_sell(ticker, current_price, signal_id)
        
        # HOLD signals - do nothing
        return {'success': False, 'message': f'No action for signal: {signal}'}
    
    def _execute_buy(self, ticker: str, price: float, confidence: int, signal_id: int = None) -> Dict:
        """Execute a buy order with slippage and commission."""
        settings = self.get_settings()

        # Check max positions
        if self.get_position_count() >= settings['max_positions']:
            return {'success': False, 'message': f'Max positions ({settings["max_positions"]}) reached'}

        # Check if already holding
        if self.has_open_position(ticker):
            return {'success': False, 'message': f'Already holding {ticker}'}

        # Apply slippage: buy at slightly higher price
        slippage_pct = settings.get('slippage_pct', 0.1)
        adjusted_price = price * (1 + slippage_pct / 100)
        slippage_cost = (adjusted_price - price)  # Per share
        commission = settings.get('commission', 0.0)

        # Calculate shares to buy
        cash = self.get_current_cash()
        position_value = settings['starting_capital'] * (settings['position_size_pct'] / 100)

        if position_value > cash:
            position_value = cash * 0.95  # Use 95% of remaining cash

        # Account for commission in available capital
        position_value -= commission

        if position_value < 10:  # Minimum $10 position
            return {'success': False, 'message': 'Insufficient cash'}

        shares = round(position_value / adjusted_price, 4)
        total_slippage = round(slippage_cost * shares, 2)

        # Record the trade with costs
        db.execute('''
            INSERT INTO paper_trades
            (ticker, direction, shares, entry_price, entry_date, signal_id, signal_confidence, status,
             slippage_cost, commission_cost)
            VALUES (?, 'BUY', ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        ''', (ticker, shares, adjusted_price, datetime.now().isoformat(), signal_id, confidence,
              total_slippage, commission))

        logger.info(f"Paper trade: BUY {shares:.4f} {ticker} @ ${adjusted_price:.2f} (slippage: ${total_slippage:.2f}, commission: ${commission:.2f})")

        return {
            'success': True,
            'message': f'Bought {shares:.4f} {ticker} @ ${adjusted_price:.2f} (costs: ${total_slippage + commission:.2f})',
            'ticker': ticker,
            'shares': shares,
            'price': adjusted_price,
            'value': shares * adjusted_price,
            'slippage_cost': total_slippage,
            'commission': commission,
        }
    
    def _execute_sell(self, ticker: str, price: float, signal_id: int = None) -> Dict:
        """Execute a sell order with slippage and commission."""
        settings = self.get_settings()
        position = db.query_one(
            'SELECT * FROM paper_trades WHERE ticker = ? AND status = ?',
            (ticker, 'OPEN')
        )

        if not position:
            return {'success': False, 'message': f'No open position for {ticker}'}

        # Apply slippage: sell at slightly lower price
        slippage_pct = settings.get('slippage_pct', 0.1)
        adjusted_price = price * (1 - slippage_pct / 100)
        commission = settings.get('commission', 0.0)

        shares = position['shares']
        entry_price = position['entry_price']
        slippage_cost = round((price - adjusted_price) * shares, 2)

        # P&L includes both entry and exit costs
        entry_slippage = position.get('slippage_cost', 0) or 0
        entry_commission = position.get('commission_cost', 0) or 0
        gross_pnl = (adjusted_price - entry_price) * shares
        total_costs = slippage_cost + commission  # Exit costs only (entry already in entry_price)
        net_pnl = gross_pnl - commission  # slippage already reflected in adjusted_price
        pnl_pct = ((adjusted_price - entry_price) / entry_price) * 100

        # Update the trade
        db.execute('''
            UPDATE paper_trades
            SET exit_price = ?, exit_date = ?, pnl = ?, pnl_pct = ?, status = 'CLOSED',
                slippage_cost = COALESCE(slippage_cost, 0) + ?,
                commission_cost = COALESCE(commission_cost, 0) + ?
            WHERE id = ?
        ''', (adjusted_price, datetime.now().isoformat(), net_pnl, pnl_pct,
              slippage_cost, commission, position['id']))

        logger.info(f"Paper trade: SELL {shares:.4f} {ticker} @ ${adjusted_price:.2f} (net P&L: ${net_pnl:.2f})")

        return {
            'success': True,
            'message': f'Sold {shares:.4f} {ticker} @ ${adjusted_price:.2f} (net P&L: ${net_pnl:.2f})',
            'ticker': ticker,
            'shares': shares,
            'price': adjusted_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'slippage_cost': slippage_cost,
            'commission': commission,
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary with P&L."""
        settings = self.get_settings()
        starting = settings['starting_capital']
        
        # Get open positions with current values
        positions = self.get_open_positions()
        positions_value = 0
        positions_with_prices = []
        
        for pos in positions:
            current = self._get_current_price(pos['ticker'])
            if current:
                value = current * pos['shares']
                unrealized_pnl = (current - pos['entry_price']) * pos['shares']
                unrealized_pnl_pct = ((current - pos['entry_price']) / pos['entry_price']) * 100
                positions_value += value
                positions_with_prices.append({
                    **pos,
                    'current_price': current,
                    'current_value': value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                })
            else:
                # Use entry price if can't fetch current
                value = pos['entry_price'] * pos['shares']
                positions_value += value
                positions_with_prices.append({
                    **pos,
                    'current_price': pos['entry_price'],
                    'current_value': value,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0,
                })
        
        # Get closed trades stats
        closed_stats = db.query_one('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as realized_pnl,
                COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN pnl <= 0 THEN pnl END), 0) as avg_loss,
                COALESCE(SUM(slippage_cost), 0) as total_slippage,
                COALESCE(SUM(commission_cost), 0) as total_commission
            FROM paper_trades WHERE status = 'CLOSED'
        ''')
        
        wins = closed_stats.get('wins', 0) or 0
        losses = closed_stats.get('losses', 0) or 0
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        realized_pnl = closed_stats.get('realized_pnl', 0) or 0
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions_with_prices)
        
        cash = self.get_current_cash()
        total_value = cash + positions_value
        total_return = total_value - starting
        total_return_pct = ((total_value - starting) / starting) * 100
        
        # Get SPY return for comparison (from start date)
        first_trade = db.query_one('SELECT MIN(entry_date) as first FROM paper_trades')
        spy_return_pct = 0
        if first_trade and first_trade.get('first'):
            spy_return_pct = self._get_spy_return_since(first_trade['first'])
        
        total_slippage = round(closed_stats.get('total_slippage', 0) or 0, 2)
        total_commission = round(closed_stats.get('total_commission', 0) or 0, 2)
        total_costs = total_slippage + total_commission
        avg_win = round(closed_stats.get('avg_win', 0) or 0, 2)
        avg_loss = round(closed_stats.get('avg_loss', 0) or 0, 2)
        payoff_ratio = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0

        return {
            'starting_capital': starting,
            'total_value': round(total_value, 2),
            'cash': round(cash, 2),
            'positions_value': round(positions_value, 2),
            'realized_pnl': round(realized_pnl, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'spy_return_pct': round(spy_return_pct, 2),
            'alpha': round(total_return_pct - spy_return_pct, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'payoff_ratio': payoff_ratio,
            'total_slippage': total_slippage,
            'total_commission': total_commission,
            'total_costs': total_costs,
            'open_positions': len(positions),
            'positions': positions_with_prices,
        }
    
    def _get_spy_return_since(self, start_date_str: str) -> float:
        """Calculate SPY return since a date."""
        try:
            start_date = datetime.fromisoformat(start_date_str.split('T')[0])
            spy = yf.Ticker('SPY')
            hist = spy.history(start=start_date, end=datetime.now())
            if len(hist) >= 2:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                return ((end_price - start_price) / start_price) * 100
        except Exception as e:
            logger.warning(f"Could not calculate SPY return: {e}")
        return 0
    
    def get_equity_curve(self, days_back: int = 90) -> List[Dict]:
        """Get daily portfolio value for charting."""
        snapshots = db.query('''
            SELECT snapshot_date, total_value, spy_value, portfolio_return_pct, spy_return_pct
            FROM paper_snapshots
            WHERE snapshot_date >= date('now', ?)
            ORDER BY snapshot_date ASC
        ''', (f'-{days_back} days',))
        
        return snapshots
    
    def take_snapshot(self):
        """Take a daily snapshot of portfolio value."""
        today = date.today().isoformat()
        
        # Check if already snapped today
        existing = db.query_one('SELECT id FROM paper_snapshots WHERE snapshot_date = ?', (today,))
        if existing:
            return  # Already have today's snapshot
        
        summary = self.get_portfolio_summary()
        settings = self.get_settings()
        
        # Calculate SPY value (starting capital invested in SPY)
        try:
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1d')
            spy_price = hist['Close'].iloc[-1] if not hist.empty else None
        except:
            spy_price = None
        
        # Get first trade date to calculate SPY comparison
        first_trade = db.query_one('SELECT MIN(entry_date) as first FROM paper_trades')
        spy_value = settings['starting_capital']
        spy_return_pct = 0
        
        if first_trade and first_trade.get('first'):
            spy_return_pct = self._get_spy_return_since(first_trade['first'])
            spy_value = settings['starting_capital'] * (1 + spy_return_pct / 100)
        
        portfolio_return_pct = summary['total_return_pct']
        
        db.execute('''
            INSERT INTO paper_snapshots 
            (snapshot_date, total_value, cash, positions_value, positions_json, 
             spy_value, spy_return_pct, portfolio_return_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            summary['total_value'],
            summary['cash'],
            summary['positions_value'],
            json.dumps([{
                'ticker': p['ticker'],
                'shares': p['shares'],
                'value': p.get('current_value', 0)
            } for p in summary['positions']]),
            round(spy_value, 2),
            round(spy_return_pct, 2),
            round(portfolio_return_pct, 2)
        ))
        
        logger.info(f"Paper trading snapshot: ${summary['total_value']:.2f} ({portfolio_return_pct:+.2f}% vs SPY {spy_return_pct:+.2f}%)")
    
    def get_trade_log(self, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        return db.query('''
            SELECT * FROM paper_trades 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
    
    def get_risk_metrics(self) -> Dict:
        """Calculate risk-adjusted performance metrics from paper trading snapshots.

        Returns Sharpe, Sortino, Max Drawdown, Calmar ratio, and win/loss stats.
        """
        snapshots = db.query('''
            SELECT snapshot_date, total_value, portfolio_return_pct, spy_return_pct
            FROM paper_snapshots
            ORDER BY snapshot_date ASC
        ''')

        if len(snapshots) < 5:
            return {
                'sufficient_data': False,
                'total_snapshots': len(snapshots),
                'message': f'Need at least 5 daily snapshots, have {len(snapshots)}. Let the system run for a few more days.'
            }

        # Calculate daily returns from total_value series
        values = [s['total_value'] for s in snapshots]
        daily_returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                daily_returns.append((values[i] - values[i - 1]) / values[i - 1])

        if not daily_returns:
            return {'sufficient_data': False, 'message': 'Could not calculate daily returns'}

        mean_return = sum(daily_returns) / len(daily_returns)
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)) if len(daily_returns) > 1 else 0

        # Sharpe Ratio (annualized, risk-free rate ≈ 0 for simplicity)
        sharpe = round((mean_return / std_return) * math.sqrt(252), 2) if std_return > 0 else 0

        # Sortino Ratio (only downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        downside_std = math.sqrt(sum(r ** 2 for r in negative_returns) / len(daily_returns)) if negative_returns else 0
        sortino = round((mean_return / downside_std) * math.sqrt(252), 2) if downside_std > 0 else 0

        # Max Drawdown from equity curve
        try:
            from engine.drawdown_tracker import drawdown_tracker
            dd_snapshots = [{'date': s['snapshot_date'], 'value': s['total_value']} for s in snapshots]
            dd_analysis = drawdown_tracker.analyze_equity_curve(dd_snapshots)
            max_drawdown = dd_analysis.get('max_drawdown', {})
            current_drawdown = dd_analysis.get('current_drawdown', {})
            underwater = dd_analysis.get('underwater', {})
            reality_check = dd_analysis.get('reality_check', '')
        except Exception:
            max_drawdown = {'pct': 0}
            current_drawdown = {'pct': 0, 'in_drawdown': False}
            underwater = {}
            reality_check = ''

        max_dd_pct = max_drawdown.get('pct', 0)

        # Calmar Ratio (annualized return / max drawdown)
        settings = self.get_settings()
        starting = settings['starting_capital']
        current_value = values[-1] if values else starting
        total_return_pct = ((current_value - starting) / starting) * 100
        days = len(snapshots)
        annualized_return = total_return_pct * (365 / days) if days > 0 else 0
        calmar = round(annualized_return / abs(max_dd_pct), 2) if max_dd_pct != 0 else 0

        # Win/loss stats from closed trades
        summary = self.get_portfolio_summary()

        return {
            'sufficient_data': True,
            'total_snapshots': len(snapshots),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': round(max_dd_pct, 2),
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'underwater': underwater,
            'calmar_ratio': calmar,
            'annualized_return': round(annualized_return, 2),
            'win_rate': summary.get('win_rate', 0),
            'payoff_ratio': summary.get('payoff_ratio', 0),
            'avg_win': summary.get('avg_win', 0),
            'avg_loss': summary.get('avg_loss', 0),
            'total_costs': summary.get('total_costs', 0),
            'reality_check': reality_check,
        }

    def reset_portfolio(self) -> bool:
        """Reset paper trading portfolio (delete all trades and snapshots)."""
        try:
            db.execute('DELETE FROM paper_trades')
            db.execute('DELETE FROM paper_snapshots')
            logger.info("Paper trading portfolio reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting paper portfolio: {e}")
            return False


    def get_spy_correlation(self) -> Dict:
        """Calculate portfolio beta and alpha vs SPY.

        If beta ~ 1 and alpha ~ 0, the system just tracks the market.
        Positive alpha means genuine stock-picking skill.
        """
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT snapshot_date, total_value FROM paper_snapshots
            ORDER BY snapshot_date ASC
        """)
        snapshots = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if len(snapshots) < 10:
            return {
                'insufficient_data': True,
                'message': f'Need at least 10 portfolio snapshots, have {len(snapshots)}'
            }

        # Get SPY prices for the same period
        first_date = snapshots[0]['snapshot_date'][:10]
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(start=first_date)
            if spy_hist.empty or len(spy_hist) < 10:
                return {'insufficient_data': True, 'message': 'Could not fetch SPY data'}
        except Exception as e:
            return {'insufficient_data': True, 'message': f'SPY fetch error: {str(e)}'}

        # Build date-aligned return series
        spy_prices = {}
        for idx, row in spy_hist.iterrows():
            spy_prices[idx.strftime('%Y-%m-%d')] = float(row['Close'])

        portfolio_returns = []
        spy_returns = []

        for i in range(1, len(snapshots)):
            date_curr = snapshots[i]['snapshot_date'][:10]
            date_prev = snapshots[i - 1]['snapshot_date'][:10]
            val_curr = snapshots[i]['total_value']
            val_prev = snapshots[i - 1]['total_value']

            if val_prev <= 0:
                continue

            # Find closest SPY prices
            spy_curr = spy_prices.get(date_curr)
            spy_prev = spy_prices.get(date_prev)

            if spy_curr is None or spy_prev is None or spy_prev <= 0:
                continue

            port_ret = (val_curr - val_prev) / val_prev
            spy_ret = (spy_curr - spy_prev) / spy_prev

            portfolio_returns.append(port_ret)
            spy_returns.append(spy_ret)

        if len(portfolio_returns) < 5:
            return {
                'insufficient_data': True,
                'message': f'Only {len(portfolio_returns)} aligned data points, need at least 5'
            }

        n = len(portfolio_returns)

        # Beta = Cov(portfolio, spy) / Var(spy)
        mean_p = sum(portfolio_returns) / n
        mean_s = sum(spy_returns) / n

        cov_ps = sum((portfolio_returns[i] - mean_p) * (spy_returns[i] - mean_s) for i in range(n)) / n
        var_s = sum((spy_returns[i] - mean_s) ** 2 for i in range(n)) / n

        if var_s == 0:
            beta = 0.0
        else:
            beta = cov_ps / var_s

        # Alpha (annualized) = annualized_portfolio_return - beta * annualized_spy_return
        ann_p = mean_p * 252
        ann_s = mean_s * 252
        alpha = ann_p - beta * ann_s

        # Correlation coefficient
        std_p = math.sqrt(sum((r - mean_p) ** 2 for r in portfolio_returns) / n)
        std_s = math.sqrt(var_s)
        correlation = cov_ps / (std_p * std_s) if (std_p > 0 and std_s > 0) else 0

        # R-squared
        r_squared = correlation ** 2

        # Interpretation
        if abs(beta - 1.0) < 0.15 and abs(alpha) < 0.02:
            verdict = 'market_tracker'
            verdict_text = 'Portfolio closely tracks SPY. No real stock-picking alpha.'
        elif alpha > 0.05:
            verdict = 'alpha_positive'
            verdict_text = f'Positive alpha of {alpha * 100:.1f}% annualized. System adds value beyond market exposure.'
        elif alpha < -0.05:
            verdict = 'alpha_negative'
            verdict_text = f'Negative alpha of {alpha * 100:.1f}%. System underperforms its market risk. Consider SPY ETF.'
        else:
            verdict = 'inconclusive'
            verdict_text = 'Alpha is near zero. More data needed to determine if system adds value.'

        return {
            'insufficient_data': False,
            'beta': round(beta, 3),
            'alpha_annualized': round(alpha * 100, 2),
            'correlation': round(correlation, 3),
            'r_squared': round(r_squared, 3),
            'data_points': n,
            'annualized_portfolio_return': round(ann_p * 100, 2),
            'annualized_spy_return': round(ann_s * 100, 2),
            'verdict': verdict,
            'verdict_text': verdict_text,
        }


# Singleton instance
paper_trader = PaperTrader()
