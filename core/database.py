"""
SQLite Database Manager for Investment Monitor
Handles settings, watchlist, analysis history, and API keys securely.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.config import DB_PATH, DEFAULT_SETTINGS
from core.encryption import encryption

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Settings table (key-value store)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # API Keys (encrypted in production, plain for simplicity here)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                service TEXT PRIMARY KEY,
                api_key TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Watchlist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                name TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Analysis History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                news TEXT,
                fundamental TEXT,
                technical TEXT,
                recommendation TEXT,
                signal TEXT,
                confidence INTEGER,
                risk_score INTEGER DEFAULT 5
            )
        """)
        
        # Alerts sent
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                signal TEXT,
                message TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sent_to TEXT
            )
        """)
        
        # Scheduler runs log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tickers_scanned INTEGER,
                alerts_sent INTEGER,
                errors TEXT,
                duration_seconds REAL
            )
        """)
        
        # Investment Categories (Risk Classification)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investment_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                risk_level INTEGER DEFAULT 5,
                time_horizon TEXT DEFAULT 'medium',
                notes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Investment Strategies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investment_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                risk_tolerance TEXT DEFAULT 'medium',
                time_horizon TEXT DEFAULT 'long_term',
                asset_mix TEXT,
                scan_frequency TEXT DEFAULT 'daily',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cycle Schedules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycle_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_type TEXT NOT NULL,
                strategy_id INTEGER,
                tickers TEXT,
                last_run TIMESTAMP,
                next_run TIMESTAMP,
                status TEXT DEFAULT 'pending',
                results TEXT,
                FOREIGN KEY (strategy_id) REFERENCES investment_strategies(id)
            )
        """)
        
        # Discovered Stocks (Opportunities found by Discovery Engine)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,  -- 'VOLUME_SPIKE', 'RSI_OVERSOLD', 'BREAKOUT'
                confidence INTEGER,
                price REAL,
                found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'new'  -- 'new', 'analyzed', 'dismissed'
            )
        """)
        
        # Prompt Templates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                strategy_type TEXT DEFAULT 'balanced',
                category TEXT,
                template TEXT NOT NULL,
                output_format TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Initialize default strategies
        default_strategies = [
            ("conservative", "Konservative Strategie - Fokus auf Sicherheit und Dividenden", "low", "long_term", 
             '{"etf": 50, "blue_chip": 40, "growth": 10}', "weekly"),
            ("balanced", "Ausgewogene Strategie - Qualität zu fairem Preis", "medium", "medium_term",
             '{"etf": 30, "blue_chip": 40, "growth": 25, "startup": 5}', "daily"),
            ("aggressive", "Aggressive Strategie - Wachstum und Spekulation", "high", "short_term",
             '{"blue_chip": 20, "growth": 40, "startup": 30, "speculative": 10}', "daily"),
        ]
        for name, desc, risk, horizon, mix, freq in default_strategies:
            cursor.execute("""
                INSERT OR IGNORE INTO investment_strategies 
                (name, description, risk_tolerance, time_horizon, asset_mix, scan_frequency)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, desc, risk, horizon, mix, freq))
        
        # Initialize default settings
        for key, value in DEFAULT_SETTINGS.items():
            cursor.execute("""
                INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)
            """, (key, json.dumps(value)))
        
        # Initialize default watchlist
        default_watchlist = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("NVDA", "NVIDIA Corporation"),
            ("GOOGL", "Alphabet Inc."),
            ("TSLA", "Tesla Inc."),
        ]
        for ticker, name in default_watchlist:
            cursor.execute("""
                INSERT OR IGNORE INTO watchlist (ticker, name) VALUES (?, ?)
            """, (ticker, name))
        
        # Portfolio Tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                amount REAL NOT NULL,
                price REAL NOT NULL,
                date TEXT NOT NULL,
                fees REAL DEFAULT 0,
                notes TEXT,
                analysis_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                invested_value REAL NOT NULL,
                roi_percentage REAL,
                market_comparison REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        # Create default admin user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            # Default password: "changeme123"
            # This is the bcrypt hash of "changeme123"
            default_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5OMx/c5w.H6K6"
            cursor.execute("""
                INSERT INTO users (username, password_hash, email)
                VALUES ('admin', ?, 'admin@localhost')
            """, (default_hash,))
            print("⚠️  Default admin user created - Password: changeme123")
            print("⚠️  CHANGE THIS IMMEDIATELY AFTER FIRST LOGIN!")

        # Indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON portfolio_trades(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON portfolio_trades(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_date ON performance_snapshots(date)")
        
        conn.commit()
        conn.close()
    
    # === Settings ===
    def get_setting(self, key: str) -> Any:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            value = json.loads(row['value'])
            # Decrypt email password
            if key == 'email_smtp_password' and value:
                return encryption.decrypt(value)
            return value
        return DEFAULT_SETTINGS.get(key)
    
    def set_setting(self, key: str, value: Any):
        conn = self._get_conn()
        cursor = conn.cursor()
        # Encrypt email password
        if key == 'email_smtp_password' and value:
            value = encryption.encrypt(value)
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now()))
        conn.commit()
        conn.close()
    
    def get_all_settings(self) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        conn.close()
        return {row['key']: json.loads(row['value']) for row in rows}
    
    # === API Keys ===
    def get_api_key(self, service: str) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT api_key FROM api_keys WHERE service = ?", (service,))
        row = cursor.fetchone()
        conn.close()
        if row and row['api_key']:
            return encryption.decrypt(row['api_key'])
        return None
    
    def set_api_key(self, service: str, api_key: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        encrypted_key = encryption.encrypt(api_key)
        cursor.execute("""
            INSERT OR REPLACE INTO api_keys (service, api_key, updated_at)
            VALUES (?, ?, ?)
        """, (service, encrypted_key, datetime.now()))
        conn.commit()
        conn.close()
    
    # === Watchlist ===
    def get_watchlist(self, active_only: bool = True) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT * FROM watchlist WHERE is_active = 1 ORDER BY ticker")
        else:
            cursor.execute("SELECT * FROM watchlist ORDER BY ticker")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def add_to_watchlist(self, ticker: str, name: str = ""):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO watchlist (ticker, name, is_active) 
            VALUES (?, ?, 1)
        """, (ticker.upper(), name))
        conn.commit()
        conn.close()
    
    def remove_from_watchlist(self, ticker: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE watchlist SET is_active = 0 WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
        conn.close()
    
    # === Analysis History ===
    def save_analysis(self, ticker: str, results: Dict):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Extract signal from recommendation
        signal = "HOLD"
        confidence = 50
        rec = results.get('recommendation', '')
        if rec:
            if 'Strong Buy' in rec:
                signal = "STRONG_BUY"
                confidence = 90
            elif 'Strong Sell' in rec:
                signal = "STRONG_SELL"
                confidence = 90
            elif 'Buy' in rec:
                signal = "BUY"
                confidence = 70
            elif 'Sell' in rec:
                signal = "SELL"
                confidence = 70
        
        cursor.execute("""
            INSERT INTO analysis_history 
            (ticker, news, fundamental, technical, recommendation, signal, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            results.get('news', ''),
            results.get('fundamental', ''),
            results.get('technical', ''),
            results.get('recommendation', ''),
            signal,
            confidence
        ))
        conn.commit()
        analysis_id = cursor.lastrowid
        conn.close()
        return analysis_id, signal, confidence
    
    def get_analysis_history(self, ticker: str = None, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if ticker:
            cursor.execute("""
                SELECT * FROM analysis_history 
                WHERE ticker = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (ticker.upper(), limit))
        else:
            cursor.execute("""
                SELECT * FROM analysis_history 
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_latest_analysis(self, ticker: str) -> Optional[Dict]:
        history = self.get_analysis_history(ticker, limit=1)
        return history[0] if history else None
    
    # === Alerts ===
    def log_alert(self, ticker: str, signal: str, message: str, sent_to: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (ticker, signal, message, sent_to) 
            VALUES (?, ?, ?, ?)
        """, (ticker, signal, message, sent_to))
        conn.commit()
        conn.close()
    
    def get_alerts(self, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts ORDER BY sent_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # === Scheduler Log ===
    def log_scheduler_run(self, tickers_scanned: int, alerts_sent: int, 
                          errors: str = "", duration: float = 0):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scheduler_log (tickers_scanned, alerts_sent, errors, duration_seconds)
            VALUES (?, ?, ?, ?)
        """, (tickers_scanned, alerts_sent, errors, duration))
        conn.commit()
        conn.close()
    
    def get_scheduler_logs(self, limit: int = 20) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scheduler_log ORDER BY run_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # === Investment Strategies ===
    def get_strategies(self, active_only: bool = True) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT * FROM investment_strategies WHERE is_active = 1")
        else:
            cursor.execute("SELECT * FROM investment_strategies")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_strategy(self, name: str) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM investment_strategies WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def save_strategy(self, name: str, description: str, risk_tolerance: str,
                      time_horizon: str, asset_mix: str, scan_frequency: str = "daily"):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO investment_strategies 
            (name, description, risk_tolerance, time_horizon, asset_mix, scan_frequency, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (name, description, risk_tolerance, time_horizon, asset_mix, scan_frequency))
        conn.commit()
        conn.close()
    
    # === Investment Categories ===
    def get_category(self, ticker: str) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM investment_categories WHERE ticker = ?", (ticker.upper(),))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def set_category(self, ticker: str, category: str, risk_level: int = 5,
                     time_horizon: str = "medium", notes: str = ""):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO investment_categories 
            (ticker, category, risk_level, time_horizon, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker.upper(), category, risk_level, time_horizon, notes, datetime.now()))
        conn.commit()
        conn.close()
    
    def get_tickers_by_category(self, category: str) -> List[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM investment_categories WHERE category = ?", (category,))
        rows = cursor.fetchall()
        conn.close()
        return [row['ticker'] for row in rows]
    
    # === Cycle Schedules ===
    def get_cycle_schedule(self, cycle_type: str) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cycle_schedules WHERE cycle_type = ?", (cycle_type,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def save_cycle_result(self, cycle_type: str, strategy_id: int, tickers: str, results: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cycle_schedules 
            (cycle_type, strategy_id, tickers, last_run, status, results)
            VALUES (?, ?, ?, ?, 'completed', ?)
        """, (cycle_type, strategy_id, tickers, datetime.now(), results))
        conn.commit()
        conn.close()
    
    def get_cycle_history(self, cycle_type: str = None, limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if cycle_type:
            cursor.execute("""
                SELECT * FROM cycle_schedules 
                WHERE cycle_type = ? ORDER BY last_run DESC LIMIT ?
            """, (cycle_type, limit))
        else:
            cursor.execute("SELECT * FROM cycle_schedules ORDER BY last_run DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # === Prompt Templates ===
    def get_prompt_template(self, name: str) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT template FROM prompt_templates WHERE name = ? AND is_active = 1", (name,))
        row = cursor.fetchone()
        conn.close()
        return row['template'] if row else None
    
    def get_prompt_for_stage(self, stage: str, strategy_type: str = "balanced", 
                              category: str = None) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if category:
            cursor.execute("""
                SELECT template FROM prompt_templates 
                WHERE stage = ? AND strategy_type = ? AND category = ? AND is_active = 1
            """, (stage, strategy_type, category))
        else:
            cursor.execute("""
                SELECT template FROM prompt_templates 
                WHERE stage = ? AND strategy_type = ? AND is_active = 1
            """, (stage, strategy_type))
        row = cursor.fetchone()
        conn.close()
        return row['template'] if row else None
    
    def save_prompt_template(self, name: str, stage: str, template: str,
                              strategy_type: str = "balanced", category: str = None,
                              output_format: str = None):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO prompt_templates 
            (name, stage, strategy_type, category, template, output_format, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (name, stage, strategy_type, category, template, output_format))
        conn.commit()
        conn.close()
    
    # === Portfolio Management ===
    def add_trade(self, ticker: str, trade_type: str, amount: float, price: float, 
                  date: str = None, fees: float = 0, notes: str = "", analysis_id: int = None):
        """Record a portfolio trade (buy/sell)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if not date:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        cursor.execute("""
            INSERT INTO portfolio_trades 
            (ticker, type, amount, price, date, fees, notes, analysis_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker.upper(), trade_type.upper(), amount, price, date, fees, notes, analysis_id))
        
        conn.commit()
        conn.close()

    def get_trades(self, ticker: str = None) -> List[Dict]:
        """Get trading history"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if ticker:
            cursor.execute("SELECT * FROM portfolio_trades WHERE ticker = ? ORDER BY date DESC", (ticker.upper(),))
        else:
            cursor.execute("SELECT * FROM portfolio_trades ORDER BY date DESC")
            
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_portfolio_holdings(self) -> List[Dict]:
        """Calculate current portfolio holdings based on trades"""
        trades = self.get_trades()
        holdings = {}
        
        for trade in trades:
            ticker = trade['ticker']
            if ticker not in holdings:
                holdings[ticker] = {
                    'ticker': ticker,
                    'shares': 0,
                    'avg_price': 0.0,
                    'total_cost': 0.0,
                    'total_invested': 0.0,  # Sum of buys
                    'realized_gain': 0.0
                }
            
            h = holdings[ticker]
            amount = trade['amount']
            price = trade['price']
            fees = trade['fees']
            
            if trade['type'] == 'BUY':
                total_cost = (amount * price) + fees
                # Weighted average price
                new_shares = h['shares'] + amount
                if new_shares > 0:
                    h['avg_price'] = ((h['shares'] * h['avg_price']) + total_cost) / new_shares
                h['shares'] = new_shares
                h['total_cost'] += total_cost
                h['total_invested'] += total_cost
            
            elif trade['type'] == 'SELL':
                proceeds = (amount * price) - fees
                cost_basis = amount * h['avg_price']
                h['realized_gain'] += (proceeds - cost_basis)
                h['shares'] -= amount
                h['total_cost'] -= cost_basis
                
        # Filter out holdings with 0 shares (unless there's realized gain to show)
        active_holdings = [h for h in holdings.values() if h['shares'] > 0 or h['realized_gain'] != 0]
        return active_holdings

    def get_portfolio_summary(self) -> Dict:
        """Get overall portfolio statistics"""
        holdings = self.get_portfolio_holdings()
        
        total_invested = sum(h['total_cost'] for h in holdings if h['shares'] > 0)
        total_realized_gain = sum(h['realized_gain'] for h in holdings)
        
        # In a real app complexity, we'd fetch current prices here using yfinance.
        # For now, we return the base structure which the frontend or a separate service will enrich.
        
        return {
            'total_invested': total_invested,
            'realized_gains': total_realized_gain,
            'holdings_count': len([h for h in holdings if h['shares'] > 0]),
            'holdings': holdings
        }

    # === Top Picks / Performance Analytics ===
    def get_top_picks(self, min_predictions: int = 5, min_accuracy: float = 0.6, limit: int = 10) -> List[Dict]:
        """
        Get top-performing stocks based on prediction accuracy.
        Returns stocks ranked by a composite "pick score" that combines:
        - Accuracy (50% weight)
        - Average Confidence (30% weight)
        - Consistency (20% weight)
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Query to get prediction stats per ticker from prediction_outcomes table
        # Note: prediction_outcomes is created by learning_optimizer.py
        cursor.execute("""
            SELECT 
                ticker,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN accuracy_score > 0.5 THEN 1 ELSE 0 END) as accuracy,
                AVG(confidence) as avg_confidence,
                MAX(verified_at) as last_verified,
                SUM(CASE WHEN accuracy_score = 1.0 THEN 1 ELSE 0 END) as correct_count,
                SUM(CASE WHEN accuracy_score = 0.0 THEN 1 ELSE 0 END) as wrong_count
            FROM prediction_outcomes
            WHERE verified_at IS NOT NULL
            GROUP BY ticker
            HAVING COUNT(*) >= ?
        """, (min_predictions,))
        
        rows = cursor.fetchall()
        conn.close()
        
        picks = []
        for row in rows:
            accuracy = float(row['accuracy'])
            
            # Filter by minimum accuracy
            if accuracy < min_accuracy:
                continue
            
            avg_confidence = float(row['avg_confidence']) if row['avg_confidence'] else 50.0
            
            # Calculate consistency (inverse of variance approximation)
            # Higher consistency = more stable predictions
            total = row['total_predictions']
            correct = row['correct_count']
            if total > 1:
                # Simple consistency measure: how close to perfect is the ratio
                consistency = 1.0 - abs(2 * (correct / total) - 1)  # Peaks at 1 when 100% or 0%, low when ~50%
            else:
                consistency = 0.5
            
            # Calculate composite pick score
            pick_score = (accuracy * 0.5) + ((avg_confidence / 100) * 0.3) + (consistency * 0.2)
            
            # Determine trust tier
            if accuracy >= 0.8 and avg_confidence >= 75:
                trust_tier = "gold"
            elif accuracy >= 0.7 and avg_confidence >= 65:
                trust_tier = "silver"
            elif accuracy >= 0.6:
                trust_tier = "bronze"
            else:
                trust_tier = "none"
            
            picks.append({
                'ticker': row['ticker'],
                'total_predictions': row['total_predictions'],
                'accuracy': round(accuracy * 100, 1),
                'avg_confidence': round(avg_confidence, 1),
                'consistency': round(consistency * 100, 1),
                'pick_score': round(pick_score * 100, 1),
                'trust_tier': trust_tier,
                'correct_count': row['correct_count'],
                'wrong_count': row['wrong_count'],
                'last_verified': row['last_verified']
            })
        
        # Sort by pick_score descending
        picks.sort(key=lambda x: x['pick_score'], reverse=True)
        
        return picks[:limit]
    
    def get_ticker_trust_score(self, ticker: str) -> Optional[Dict]:
        """Get trust score and stats for a specific ticker"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ticker,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN accuracy_score > 0.5 THEN 1 ELSE 0 END) as accuracy,
                AVG(confidence) as avg_confidence
            FROM prediction_outcomes
            WHERE ticker = ? AND verified_at IS NOT NULL
            GROUP BY ticker
        """, (ticker.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row['total_predictions'] < 3:
            return None
        
        accuracy = float(row['accuracy'])
        avg_confidence = float(row['avg_confidence']) if row['avg_confidence'] else 50.0
        trust_score = (accuracy * 0.6) + ((avg_confidence / 100) * 0.4)
        
        return {
            'ticker': ticker.upper(),
            'total_predictions': row['total_predictions'],
            'accuracy': round(accuracy * 100, 1),
            'avg_confidence': round(avg_confidence, 1),
            'trust_score': round(trust_score * 100, 1),
            'is_trusted': accuracy >= 0.7 and avg_confidence >= 60
        }
    
    def get_recent_high_confidence_predictions(self, days: int = 7, min_confidence: int = 70) -> List[Dict]:
        """Get recent predictions with high confidence for trusted tickers"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ah.ticker,
                ah.signal,
                ah.confidence,
                ah.timestamp,
                ah.recommendation
            FROM analysis_history ah
            WHERE ah.confidence >= ?
                AND ah.timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY ah.timestamp DESC
            LIMIT 20
        """, (min_confidence, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_trusted_tickers(self, min_accuracy: float = 0.7) -> List[str]:
        """Get list of ticker symbols that have proven accuracy"""
        picks = self.get_top_picks(min_predictions=3, min_accuracy=min_accuracy, limit=100)
        return [pick['ticker'] for pick in picks]

    # === User Management (Authentication) ===
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def verify_user(self, username: str, password: str) -> bool:
        """Verify username and password"""
        from core.auth import auth_manager
        user = self.get_user(username)
        if not user or not user['is_active']:
            return False
        return auth_manager.verify_password(password, user['password_hash'])

    def update_password(self, username: str, new_password: str):
        """Update user password"""
        from core.auth import auth_manager
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET password_hash = ? WHERE username = ?
        """, (auth_manager.hash_password(new_password), username))
        conn.commit()
        conn.close()

    def update_last_login(self, username: str):
        """Update last login timestamp"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE username = ?
        """, (datetime.now(), username))
        conn.commit()
        conn.close()

# Singleton
db = Database()

