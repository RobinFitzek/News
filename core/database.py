"""
SQLite Database Manager for Investment Monitor
Handles settings, watchlist, analysis history, and API keys securely.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.config import DB_PATH, DEFAULT_SETTINGS
from core.encryption import encryption
import logging
import time
from contextlib import contextmanager

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds

        # Ensure database directory exists
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create database directory: {e}")
            raise

        self._init_db()

    def _get_conn(self, timeout: float = 10.0) -> sqlite3.Connection:
        """Get database connection with retry logic and proper configuration"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row

                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

                # Set journal mode to WAL for better concurrency
                conn.execute("PRAGMA journal_mode = WAL")

                # Set synchronous mode for better performance with WAL
                conn.execute("PRAGMA synchronous = NORMAL")

                return conn

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < max_attempts - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error("Database locked after all retry attempts")
                        raise
                else:
                    self.logger.error(f"Database connection error: {e}")
                    raise

            except Exception as e:
                self.logger.error(f"Unexpected error connecting to database: {e}", exc_info=True)
                raise

        raise sqlite3.OperationalError("Failed to connect to database after all retries")

    @contextmanager
    def _get_transaction(self):
        """Context manager for database transactions with automatic rollback on error"""
        conn = None
        try:
            conn = self._get_conn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
                self.logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
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
                risk_score INTEGER DEFAULT 5,
                bull_case TEXT,
                bear_case TEXT,
                sources TEXT,
                risk_profile TEXT,
                insider_sentiment TEXT
            )
        """)
        
        # Migrate analysis_history: add new columns if missing
        cursor.execute("PRAGMA table_info(analysis_history)")
        ah_cols = {row['name'] for row in cursor.fetchall()}
        ah_migrations = [
            ("bull_case", "TEXT"),
            ("bear_case", "TEXT"),
            ("sources", "TEXT"),
            ("risk_profile", "TEXT"),
            ("insider_sentiment", "TEXT"),
        ]
        for col_name, col_type in ah_migrations:
            if col_name not in ah_cols:
                try:
                    cursor.execute(f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass
        
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

        # Ensure alerts table supports deduplication fields used by alert_manager
        cursor.execute("PRAGMA table_info(alerts)")
        alert_cols = {row['name'] for row in cursor.fetchall()}
        alert_migrations = [
            ("alert_hash", "TEXT"),
            ("type", "TEXT"),
            ("severity", "TEXT"),
            ("timestamp", "TEXT DEFAULT CURRENT_TIMESTAMP"),
            ("acknowledged", "INTEGER DEFAULT 0"),
            ("metadata", "TEXT"),
        ]
        for col_name, col_type in alert_migrations:
            if col_name not in alert_cols:
                try:
                    cursor.execute(f"ALTER TABLE alerts ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass
        
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
                signal_type TEXT NOT NULL,
                confidence INTEGER,
                price REAL,
                found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'new',
                source TEXT DEFAULT 'auto',
                strategy TEXT,
                quant_score REAL,
                sector TEXT,
                market_cap REAL,
                promoted_at TIMESTAMP,
                dismissed_at TIMESTAMP,
                dismiss_reason TEXT
            )
        """)

        # Migrate discovered_stocks: add new columns if missing
        cursor.execute("PRAGMA table_info(discovered_stocks)")
        ds_cols = {row['name'] for row in cursor.fetchall()}
        ds_migrations = [
            ("source", "TEXT DEFAULT 'auto'"),
            ("strategy", "TEXT"),
            ("quant_score", "REAL"),
            ("sector", "TEXT"),
            ("market_cap", "REAL"),
            ("promoted_at", "TIMESTAMP"),
            ("dismissed_at", "TIMESTAMP"),
            ("dismiss_reason", "TEXT"),
        ]
        for col_name, col_type in ds_migrations:
            if col_name not in ds_cols:
                try:
                    cursor.execute(f"ALTER TABLE discovered_stocks ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        # Discovery Log (tracks each auto-discovery run)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovery_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_type TEXT NOT NULL,
                strategies_run TEXT,
                tickers_scanned INTEGER DEFAULT 0,
                discoveries_found INTEGER DEFAULT 0,
                promoted_count INTEGER DEFAULT 0,
                duration_seconds REAL,
                errors TEXT
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_risk_overrides (
                ticker TEXT PRIMARY KEY,
                stop_loss_pct REAL,
                max_position_pct REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                must_change_password BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        # Migration: add must_change_password for existing databases
        cursor.execute("PRAGMA table_info(users)")
        user_cols = {row['name'] for row in cursor.fetchall()}
        if 'must_change_password' not in user_cols:
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN DEFAULT 1")
            except Exception:
                pass

        # Create default admin user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            # Default password: "changeme123"
            # This is the bcrypt hash of "changeme123"
            default_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5OMx/c5w.H6K6"
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, must_change_password)
                VALUES ('admin', ?, 'admin@localhost', 1)
            """, (default_hash,))
            print("⚠️  Default admin user created - Password: changeme123")
            print("⚠️  CHANGE THIS IMMEDIATELY AFTER FIRST LOGIN!")

        # Persistent user sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)

        # Migration: optional metadata on sessions for device overview
        cursor.execute("PRAGMA table_info(user_sessions)")
        session_cols = {row['name'] for row in cursor.fetchall()}
        if 'ip_address' not in session_cols:
            try:
                cursor.execute("ALTER TABLE user_sessions ADD COLUMN ip_address TEXT")
            except Exception:
                pass
        if 'user_agent' not in session_cols:
            try:
                cursor.execute("ALTER TABLE user_sessions ADD COLUMN user_agent TEXT")
            except Exception:
                pass

        # Login failure tracking (backoff/lockout)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                ip_address TEXT,
                attempted_at TEXT NOT NULL
            )
        """)

        # Insider transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insider_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                insider_name TEXT,
                title TEXT,
                transaction_date TEXT,
                filing_date TEXT,
                transaction_type TEXT,
                transaction_code TEXT,
                shares REAL,
                price REAL,
                value REAL,
                significance_score INTEGER,
                form4_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, insider_name, transaction_date, shares, price)
            )
        """)

        # API Cost Logging (for adaptive budget tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cost_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                date TEXT NOT NULL,
                api TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0,
                month TEXT NOT NULL
            )
        """)

        # Handle legacy backtest_results schema (old version had different columns)
        cursor.execute("PRAGMA table_info(backtest_results)")
        _bt_cols = [row['name'] for row in cursor.fetchall()]
        if _bt_cols and 'run_id' not in _bt_cols:
            cursor.execute("ALTER TABLE backtest_results RENAME TO backtest_results_legacy")

        # Backtest Results (per-ticker per-date)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                test_date TEXT NOT NULL,
                signal TEXT NOT NULL,
                tech_score REAL,
                momentum_score REAL,
                composite_score REAL,
                forward_5d_return REAL,
                forward_20d_return REAL,
                hit INTEGER DEFAULT 0,
                benchmark_ticker TEXT,
                benchmark_return REAL,
                alpha REAL,
                forward_10d_return REAL,
                forward_40d_return REAL,
                forward_60d_return REAL,
                regime TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
            )
        """)

        # Backtest Runs (summary per run)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tickers_tested INTEGER DEFAULT 0,
                months_tested INTEGER DEFAULT 0,
                overall_accuracy REAL,
                avg_return_buy REAL,
                avg_return_sell REAL,
                best_weights TEXT,
                status TEXT DEFAULT 'running',
                progress_pct REAL DEFAULT 0,
                progress_msg TEXT DEFAULT '',
                error TEXT,
                sharpe_ratio REAL,
                max_drawdown REAL,
                profit_factor REAL,
                volatility REAL,
                win_loss_ratio REAL,
                risk_metrics TEXT,
                survivorship_warning TEXT,
                coverage_warning TEXT,
                model_alignment_pct REAL,
                expected_value_per_trade REAL,
                out_of_sample_accuracy REAL,
                portfolio_total_return REAL,
                portfolio_sharpe REAL,
                walk_forward_windows TEXT,
                weight_stability REAL
            )
        """)

        # Migrate existing tables: add walk-forward columns if missing
        for col, col_type in [('walk_forward_windows', 'TEXT'), ('weight_stability', 'REAL')]:
            try:
                cursor.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists

        # Indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_run ON backtest_results(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_ticker ON backtest_results(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON portfolio_trades(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON portfolio_trades(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_hash ON alerts(alert_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_exp ON user_sessions(expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_login_fail_user_time ON login_failures(username, attempted_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_login_fail_ip_time ON login_failures(ip_address, attempted_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_date ON performance_snapshots(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider_transactions(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insider_date ON insider_transactions(transaction_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_api_month ON api_cost_log(api, month)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_api_date ON api_cost_log(api, date)")

        # AI Cross-Check Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_crosscheck_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_id INTEGER,
                claims_found INTEGER DEFAULT 0,
                claims_verified INTEGER DEFAULT 0,
                accuracy REAL,
                trust_score REAL,
                details TEXT,
                warning TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crosscheck_ticker ON ai_crosscheck_log(ticker)")

        # Portfolio Benchmarks (daily snapshots for portfolio vs SPY comparison)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT UNIQUE NOT NULL,
                portfolio_value REAL,
                portfolio_return_pct REAL,
                spy_equivalent_value REAL,
                spy_return_pct REAL,
                alpha REAL,
                cash_invested REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_date ON portfolio_benchmarks(snapshot_date)")

        # Fundamental Snapshots (for 4-factor backtesting — Feature 5)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                pe_ratio REAL,
                pb_ratio REAL,
                roe REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                fcf_yield REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, snapshot_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_snap_ticker ON fundamental_snapshots(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_snap_date ON fundamental_snapshots(snapshot_date)")

        # Discovery indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_ticker ON discovered_stocks(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_status ON discovered_stocks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_found ON discovered_stocks(found_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovery_log_type ON discovery_log(run_type)")

        # Ticker Graveyard (survivorship bias tracking — Feature 7)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_graveyard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                last_seen TEXT,
                reason TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graveyard_ticker ON ticker_graveyard(ticker)")

        # === NEW TABLES ===

        # Financial statement cache (8-quarter trend data, DCF, key stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                UNIQUE(ticker, data_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_cache_ticker ON financial_cache(ticker)")

        # Discovery outcome tracking (performance at 30/60/90 days)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovery_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovery_id INTEGER UNIQUE,
                ticker TEXT NOT NULL,
                promoted_at TEXT,
                promoted_price REAL,
                price_30d REAL,
                price_60d REAL,
                price_90d REAL,
                return_30d REAL,
                return_60d REAL,
                return_90d REAL,
                strategy TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_disc_outcomes_ticker ON discovery_outcomes(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_disc_outcomes_strategy ON discovery_outcomes(strategy)")

        # Per-stock notes (free text)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_notes (
                ticker TEXT PRIMARY KEY,
                note_text TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade journal
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                shares REAL,
                trade_type TEXT DEFAULT 'LONG',
                system_signal TEXT,
                user_action TEXT,
                entry_reason TEXT,
                exit_reason TEXT,
                outcome_pct REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_ticker ON trade_journal(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_date ON trade_journal(entry_date)")

        # Watchlist tier column migration
        cursor.execute("PRAGMA table_info(watchlist)")
        wl_cols = {row['name'] for row in cursor.fetchall()}
        if 'tier' not in wl_cols:
            try:
                cursor.execute("ALTER TABLE watchlist ADD COLUMN tier TEXT DEFAULT 'core'")
            except Exception:
                pass
        if 'tags' not in wl_cols:
            try:
                cursor.execute("ALTER TABLE watchlist ADD COLUMN tags TEXT")
            except Exception:
                pass

        # Price alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                threshold REAL,
                direction TEXT,
                active INTEGER DEFAULT 1,
                triggered_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Webhook settings migration (ensure settings exist for new keys)
        webhook_defaults = [
            ('telegram_enabled', 'false'),
            ('telegram_bot_token', '""'),
            ('telegram_chat_id', '""'),
            ('discord_enabled', 'false'),
            ('discord_webhook_url', '""'),
        ]
        for wkey, wval in webhook_defaults:
            cursor.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (wkey, wval)
            )

        conn.commit()
        conn.close()
    
    # === Settings ===
    # === Helper Methods for New Modules ===
    
    def query(self, sql: str, params: tuple = ()) -> List[Dict]:
        """Execute SELECT query and return list of dicts"""
        try:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Query error: {e}", exc_info=True)
            return []
    
    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        """Execute SELECT query and return single dict or None"""
        try:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Query one error: {e}", exc_info=True)
            return None
    
    def execute(self, sql: str, params: tuple = ()):
        """Execute INSERT/UPDATE/DELETE query"""
        try:
            with self._get_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                return cursor
        except Exception as e:
            self.logger.error(f"Execute error: {e}", exc_info=True)
            raise
    
    # === Settings ===
    
    def get_setting(self, key: str) -> Any:
        """Get setting with error handling and default fallback"""
        if not key or not isinstance(key, str):
            self.logger.error("Invalid setting key")
            return None

        try:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cursor.fetchone()

                if row:
                    try:
                        value = json.loads(row['value'])

                        # Decrypt email password
                        if key == 'email_smtp_password' and value:
                            try:
                                return encryption.decrypt(value)
                            except Exception as e:
                                self.logger.error(f"Failed to decrypt {key}: {e}")
                                return None

                        return value

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON for setting {key}: {e}")
                        return DEFAULT_SETTINGS.get(key)

                return DEFAULT_SETTINGS.get(key)

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error retrieving setting {key}: {e}", exc_info=True)
            return DEFAULT_SETTINGS.get(key)
    
    def set_setting(self, key: str, value: Any):
        """Set setting with validation and error handling"""
        if not key or not isinstance(key, str):
            self.logger.error("Invalid setting key")
            raise ValueError("Setting key must be a non-empty string")

        try:
            # Encrypt email password
            if key == 'email_smtp_password' and value:
                try:
                    value = encryption.encrypt(value)
                except Exception as e:
                    self.logger.error(f"Failed to encrypt password: {e}")
                    raise

            # Serialize value
            try:
                json_value = json.dumps(value)
            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to serialize value for {key}: {e}")
                raise ValueError(f"Value for {key} cannot be serialized to JSON")

            # Save to database
            with self._get_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, json_value, datetime.now()))

            self.logger.info(f"Setting updated: {key}")

        except Exception as e:
            self.logger.error(f"Error setting {key}: {e}", exc_info=True)
            raise
    
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
        """Save analysis with comprehensive error handling and validation"""
        # Input validation
        if not ticker or not isinstance(ticker, str):
            self.logger.error("Invalid ticker provided")
            raise ValueError("Ticker must be a non-empty string")

        if not results or not isinstance(results, dict):
            self.logger.error("Invalid results provided")
            raise ValueError("Results must be a non-empty dictionary")

        ticker = ticker.upper().strip()

        # Extract signal — use quant screener signal if available, fallback to old parsing
        signal = results.get('signal', 'NEUTRAL')

        # Map new signal types to DB values
        signal_map = {
            'Opportunity': 'OPPORTUNITY',
            'Caution': 'CAUTION',
            'Neutral': 'NEUTRAL',
        }
        signal = signal_map.get(signal, signal)

        # Use composite score as strength (honest metric), fallback to score, then 50
        confidence = results.get('composite_score', results.get('score', 50))

        # Legacy fallback: parse old-style Buy/Sell from recommendation text
        rec = results.get('recommendation', '')
        if signal in ('NEUTRAL', 'HOLD') and rec:
            try:
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
            except Exception as e:
                self.logger.warning(f"Error parsing recommendation: {e}")

        # Use transaction context manager
        try:
            with self._get_transaction() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO analysis_history
                    (ticker, news, fundamental, technical, recommendation, signal, confidence, risk_score, bull_case, bear_case, sources, risk_profile, insider_sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    str(results.get('news', ''))[:10000],  # Limit length
                    str(results.get('fundamental', ''))[:10000],
                    str(results.get('technical', ''))[:10000],
                    str(results.get('recommendation', ''))[:10000],
                    signal,
                    confidence,
                    results.get('risk_score', 5),
                    str(results.get('bull_case', ''))[:5000],
                    str(results.get('bear_case', ''))[:5000],
                    str(results.get('sources', ''))[:5000],
                    str(results.get('risk_profile', 'Unknown'))[:100],
                    str(results.get('insider_sentiment', ''))[:1000]
                ))

                analysis_id = cursor.lastrowid

                self.logger.info(f"Analysis saved for {ticker}: ID={analysis_id}, Signal={signal}")
                return analysis_id, signal, confidence

        except sqlite3.IntegrityError as e:
            self.logger.error(f"Integrity constraint violated while saving analysis for {ticker}: {e}")
            raise

        except sqlite3.OperationalError as e:
            self.logger.error(f"Database operational error while saving analysis for {ticker}: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error saving analysis for {ticker}: {e}", exc_info=True)
            raise
    
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

    def get_ticker_risk_override(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get risk override for a ticker, if configured."""
        if not ticker:
            return None

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, stop_loss_pct, max_position_pct, updated_at FROM ticker_risk_overrides WHERE ticker = ?",
            (ticker.upper(),)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_ticker_risk_overrides(self) -> List[Dict[str, Any]]:
        """List all ticker risk overrides."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, stop_loss_pct, max_position_pct, updated_at FROM ticker_risk_overrides ORDER BY ticker"
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def set_ticker_risk_override(self, ticker: str,
                                 stop_loss_pct: Optional[float] = None,
                                 max_position_pct: Optional[float] = None):
        """Create or update per-ticker risk override. Removes override if both values are None."""
        if not ticker:
            raise ValueError("Ticker is required")

        ticker = ticker.upper().strip()

        if stop_loss_pct is None and max_position_pct is None:
            self.delete_ticker_risk_override(ticker)
            return

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ticker_risk_overrides (ticker, stop_loss_pct, max_position_pct, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                stop_loss_pct = excluded.stop_loss_pct,
                max_position_pct = excluded.max_position_pct,
                updated_at = excluded.updated_at
        """, (ticker, stop_loss_pct, max_position_pct, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def delete_ticker_risk_override(self, ticker: str):
        """Delete per-ticker risk override."""
        if not ticker:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ticker_risk_overrides WHERE ticker = ?", (ticker.upper().strip(),))
        conn.commit()
        conn.close()

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

    # === Signal P&L Summary ===
    def get_signal_pnl_summary(self) -> Dict:
        """Aggregate P&L stats from prediction_outcomes table."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Per-signal breakdown
            cursor.execute("""
                SELECT
                    signal,
                    COUNT(*) as total,
                    SUM(CASE WHEN accuracy_score >= 0.5 THEN 1 ELSE 0 END) as correct,
                    AVG((actual_price_after - actual_price_at_prediction)
                        / actual_price_at_prediction * 100) as avg_return_pct
                FROM prediction_outcomes
                WHERE verified_at IS NOT NULL AND actual_price_at_prediction > 0
                GROUP BY signal
            """)
            by_signal = [dict(row) for row in cursor.fetchall()]

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_verified,
                    SUM(CASE WHEN accuracy_score >= 0.5 THEN 1 ELSE 0 END) as total_correct,
                    AVG(CASE WHEN signal IN ('OPPORTUNITY', 'BUY', 'STRONG_BUY')
                        THEN (actual_price_after - actual_price_at_prediction)
                             / actual_price_at_prediction * 100 END) as avg_buy_return,
                    AVG(CASE WHEN signal IN ('CAUTION', 'SELL', 'STRONG_SELL')
                        THEN (actual_price_after - actual_price_at_prediction)
                             / actual_price_at_prediction * 100 END) as avg_sell_return
                FROM prediction_outcomes
                WHERE verified_at IS NOT NULL AND actual_price_at_prediction > 0
            """)
            overall = dict(cursor.fetchone())
            conn.close()

            total = overall['total_verified'] or 0
            correct = overall['total_correct'] or 0
            hit_rate = round(correct / total * 100, 1) if total > 0 else 0

            return {
                'total_verified': total,
                'hit_rate': hit_rate,
                'avg_buy_return': round(overall['avg_buy_return'], 2) if overall['avg_buy_return'] else 0,
                'avg_sell_return': round(overall['avg_sell_return'], 2) if overall['avg_sell_return'] else 0,
                'by_signal': by_signal,
            }
        except Exception as e:
            self.logger.error(f"Error getting signal P&L summary: {e}")
            return {'total_verified': 0, 'hit_rate': 0, 'avg_buy_return': 0, 'avg_sell_return': 0, 'by_signal': []}

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
            UPDATE users
            SET password_hash = ?, must_change_password = 0
            WHERE username = ?
        """, (auth_manager.hash_password(new_password), username))
        conn.commit()
        conn.close()

    def user_must_change_password(self, username: str) -> bool:
        """Return whether user is required to change password."""
        user = self.get_user(username)
        if not user:
            return False
        return bool(user.get('must_change_password', 0))

    def update_last_login(self, username: str):
        """Update last login timestamp"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE username = ?
        """, (datetime.now(), username))
        conn.commit()
        conn.close()

    def create_user_session(self, session_id: str, username: str,
                            created_at: str, last_activity: str, expires_at: str,
                            ip_address: str = None, user_agent: str = None):
        """Persist a user session."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_sessions
            (session_id, username, created_at, last_activity, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, username, created_at, last_activity, expires_at, ip_address, user_agent))
        conn.commit()
        conn.close()

    def get_user_session(self, session_id: str) -> Optional[Dict]:
        """Get a persisted user session by session id."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user_sessions(self, username: str) -> List[Dict[str, Any]]:
        """List active sessions for user."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, username, created_at, last_activity, expires_at, ip_address, user_agent
            FROM user_sessions
            WHERE username = ?
            ORDER BY last_activity DESC
        """, (username,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def touch_user_session(self, session_id: str, last_activity: str, expires_at: str):
        """Refresh session activity + expiry (sliding timeout)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_sessions
            SET last_activity = ?, expires_at = ?
            WHERE session_id = ?
        """, (last_activity, expires_at, session_id))
        conn.commit()
        conn.close()

    def delete_user_session(self, session_id: str):
        """Delete a session by id."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def delete_other_user_sessions(self, username: str, keep_session_id: str):
        """Delete all user sessions except current one."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM user_sessions
            WHERE username = ?
              AND session_id != ?
        """, (username, keep_session_id or ''))
        conn.commit()
        conn.close()

    def delete_user_session_for_user(self, username: str, session_id: str):
        """Delete a specific session belonging to a user."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM user_sessions
            WHERE username = ?
              AND session_id = ?
        """, (username, session_id))
        conn.commit()
        conn.close()

    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE expires_at <= ?", (datetime.now().isoformat(),))
        conn.commit()
        conn.close()

    def record_login_failure(self, username: str, ip_address: str):
        """Store failed login attempt for backoff policy."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO login_failures (username, ip_address, attempted_at)
            VALUES (?, ?, ?)
        """, ((username or '').strip().lower(), ip_address or '', datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def clear_login_failures(self, username: str):
        """Clear failures after successful login."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_failures WHERE username = ?", ((username or '').strip().lower(),))
        deleted = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        conn.commit()
        conn.close()
        return deleted

    def clear_login_failures_for_ip(self, ip_address: str):
        """Clear failures for a specific IP address."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_failures WHERE ip_address = ?", ((ip_address or '').strip(),))
        deleted = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        conn.commit()
        conn.close()
        return deleted

    def clear_recent_login_failures(self, hours: int = 24):
        """Clear login failure records in a recent time window."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_failures WHERE attempted_at >= ?", (cutoff,))
        deleted = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        conn.commit()
        conn.close()
        return deleted

    def cleanup_old_login_failures(self, days: int = 30):
        """Remove stale login failure records older than retention window."""
        cutoff = (datetime.now() - timedelta(days=max(1, days))).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_failures WHERE attempted_at < ?", (cutoff,))
        deleted = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        conn.commit()
        conn.close()
        return deleted

    def get_login_lockout_info(self, username: str, ip_address: str) -> Dict[str, Any]:
        """Check if username/IP is currently lockout-blocked."""
        max_attempts = int(self.get_setting('auth_max_failed_attempts') or 5)
        window_minutes = int(self.get_setting('auth_attempt_window_minutes') or 15)
        lockout_minutes = int(self.get_setting('auth_lockout_minutes') or 15)

        now = datetime.now()
        window_start = (now - timedelta(minutes=window_minutes)).isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as failures, MAX(attempted_at) as last_attempt
            FROM login_failures
            WHERE attempted_at >= ?
              AND (
                   username = ?
                   OR ip_address = ?
              )
        """, (window_start, (username or '').strip().lower(), ip_address or ''))
        row = cursor.fetchone()
        conn.close()

        failures = int(row['failures'] or 0) if row else 0
        last_attempt_raw = row['last_attempt'] if row else None

        if failures < max_attempts or not last_attempt_raw:
            return {
                'locked': False,
                'failures': failures,
                'remaining_seconds': 0,
                'max_attempts': max_attempts,
            }

        try:
            last_attempt = datetime.fromisoformat(last_attempt_raw)
        except Exception:
            return {
                'locked': False,
                'failures': failures,
                'remaining_seconds': 0,
                'max_attempts': max_attempts,
            }

        unlock_at = last_attempt + timedelta(minutes=lockout_minutes)
        if now >= unlock_at:
            return {
                'locked': False,
                'failures': failures,
                'remaining_seconds': 0,
                'max_attempts': max_attempts,
            }

        return {
            'locked': True,
            'failures': failures,
            'remaining_seconds': int((unlock_at - now).total_seconds()),
            'max_attempts': max_attempts,
        }

    def get_login_failures_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get aggregate login failure and lockout statistics for recent period."""
        max_attempts = int(self.get_setting('auth_max_failed_attempts') or 5)
        window_minutes = int(self.get_setting('auth_attempt_window_minutes') or 15)
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) as total
            FROM login_failures
            WHERE attempted_at >= ?
        """, (cutoff,))
        total_row = cursor.fetchone()
        total_failures = int(total_row['total']) if total_row and total_row['total'] is not None else 0

        cursor.execute("""
            SELECT username, COUNT(*) as failures
            FROM login_failures
            WHERE attempted_at >= ?
              AND username != ''
            GROUP BY username
            ORDER BY failures DESC
            LIMIT 10
        """, (cutoff,))
        by_user = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT ip_address, COUNT(*) as failures
            FROM login_failures
            WHERE attempted_at >= ?
              AND ip_address != ''
            GROUP BY ip_address
            ORDER BY failures DESC
            LIMIT 10
        """, (cutoff,))
        by_ip = [dict(row) for row in cursor.fetchall()]
        conn.close()

        estimated_locked_users = sum(1 for row in by_user if int(row.get('failures', 0)) >= max_attempts)

        return {
            'window_hours': hours,
            'window_minutes': window_minutes,
            'max_attempts': max_attempts,
            'total_failures': total_failures,
            'estimated_locked_users': estimated_locked_users,
            'by_user': by_user,
            'by_ip': by_ip,
        }

    def get_recent_login_failures(self, limit: int = 50, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent login failure events for audit display."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, ip_address, attempted_at
            FROM login_failures
            WHERE attempted_at >= ?
            ORDER BY attempted_at DESC
            LIMIT ?
        """, (cutoff, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # === Insider Transactions ===
    def save_insider_transaction(self, transaction: Dict):
        """Save an insider transaction to database"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO insider_transactions (
                    ticker, insider_name, title, transaction_date, filing_date,
                    transaction_type, transaction_code, shares, price, value,
                    significance_score, form4_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction['ticker'],
                transaction.get('insider_name'),
                transaction.get('title'),
                transaction.get('transaction_date'),
                transaction.get('filing_date'),
                transaction.get('transaction_type'),
                transaction.get('transaction_code'),
                transaction.get('shares'),
                transaction.get('price'),
                transaction.get('value'),
                transaction.get('significance_score'),
                transaction.get('form4_url')
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving insider transaction: {e}")

    def save_insider_transactions_bulk(self, transactions: List[Dict]):
        """Save multiple insider transactions at once"""
        for txn in transactions:
            self.save_insider_transaction(txn)

    def get_insider_transactions(self, ticker: str = None, days_back: int = 90) -> List[Dict]:
        """
        Get insider transactions from database

        Args:
            ticker: Optional ticker filter
            days_back: How many days of history to fetch

        Returns:
            List of insider transactions
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

            if ticker:
                cursor.execute("""
                    SELECT * FROM insider_transactions
                    WHERE ticker = ? AND transaction_date >= ?
                    ORDER BY transaction_date DESC, significance_score DESC
                """, (ticker, cutoff_date))
            else:
                cursor.execute("""
                    SELECT * FROM insider_transactions
                    WHERE transaction_date >= ?
                    ORDER BY transaction_date DESC, significance_score DESC
                """, (cutoff_date,))

            rows = cursor.fetchall()
            conn.close()

            transactions = []
            for row in rows:
                transactions.append({
                    'id': row['id'],
                    'ticker': row['ticker'],
                    'insider_name': row['insider_name'],
                    'title': row['title'],
                    'transaction_date': row['transaction_date'],
                    'filing_date': row['filing_date'],
                    'transaction_type': row['transaction_type'],
                    'transaction_code': row['transaction_code'],
                    'shares': row['shares'],
                    'price': row['price'],
                    'value': row['value'],
                    'significance_score': row['significance_score'],
                    'form4_url': row['form4_url']
                })

            return transactions

        except Exception as e:
            self.logger.error(f"Error fetching insider transactions: {e}")
            return []

    def get_top_insider_signals(self, limit: int = 10) -> List[Dict]:
        """Get top insider trading signals (high significance recent transactions)"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Get transactions from last 30 days with high significance
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

            cursor.execute("""
                SELECT
                    ticker,
                    COUNT(*) as transaction_count,
                    SUM(CASE WHEN transaction_type = 'Purchase' THEN value ELSE 0 END) as buy_value,
                    SUM(CASE WHEN transaction_type = 'Sale' THEN value ELSE 0 END) as sell_value,
                    MAX(significance_score) as max_significance,
                    MAX(transaction_date) as latest_date
                FROM insider_transactions
                WHERE transaction_date >= ? AND significance_score >= 70
                GROUP BY ticker
                HAVING transaction_count > 0
                ORDER BY max_significance DESC, transaction_count DESC
                LIMIT ?
            """, (cutoff_date, limit))

            rows = cursor.fetchall()
            conn.close()

            signals = []
            for row in rows:
                net_value = row['buy_value'] - row['sell_value']
                signal = 'BULLISH' if net_value > 0 else 'BEARISH' if net_value < 0 else 'NEUTRAL'

                signals.append({
                    'ticker': row['ticker'],
                    'transaction_count': row['transaction_count'],
                    'buy_value': row['buy_value'],
                    'sell_value': row['sell_value'],
                    'net_value': net_value,
                    'signal': signal,
                    'max_significance': row['max_significance'],
                    'latest_date': row['latest_date']
                })

            return signals

        except Exception as e:
            self.logger.error(f"Error fetching top insider signals: {e}")
            return []

    # === API Cost Tracking ===
    def log_api_cost(self, api: str, model: str, input_tokens: int,
                     output_tokens: int, estimated_cost: float, month: str, date: str):
        """Log an API request with estimated cost."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_cost_log (api, model, input_tokens, output_tokens,
                                          estimated_cost, month, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (api, model, input_tokens, output_tokens, estimated_cost, month, date))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error logging API cost: {e}")

    def get_api_spending(self, api: str, month: str) -> float:
        """Get total USD spending for an API in a given month."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(estimated_cost), 0) as total
                FROM api_cost_log WHERE api = ? AND month = ?
            """, (api, month))
            row = cursor.fetchone()
            conn.close()
            return float(row['total']) if row else 0.0
        except Exception as e:
            self.logger.error(f"Error getting API spending: {e}")
            return 0.0

    def get_api_spending_day(self, api: str, date: str) -> float:
        """Get total USD spending for an API on a given date."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(estimated_cost), 0) as total
                FROM api_cost_log WHERE api = ? AND date = ?
            """, (api, date))
            row = cursor.fetchone()
            conn.close()
            return float(row['total']) if row else 0.0
        except Exception as e:
            self.logger.error(f"Error getting API daily spending: {e}")
            return 0.0

    def get_api_request_count(self, api: str, date: str) -> int:
        """Get number of API requests on a given date."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM api_cost_log
                WHERE api = ? AND date = ?
            """, (api, date))
            row = cursor.fetchone()
            conn.close()
            return int(row['cnt']) if row else 0
        except Exception as e:
            self.logger.error(f"Error getting API request count: {e}")
            return 0

    # === Backtest ===
    def create_backtest_run(self, tickers_tested: int, months_tested: int) -> int:
        """Create a new backtest run and return its ID."""
        try:
            with self._get_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO backtest_runs (tickers_tested, months_tested, status)
                    VALUES (?, ?, 'running')
                """, (tickers_tested, months_tested))
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error creating backtest run: {e}")
            raise

    def update_backtest_run(self, run_id: int, **kwargs):
        """Update backtest run fields (progress_pct, progress_msg, status, etc.)."""
        allowed = {'overall_accuracy', 'avg_return_buy', 'avg_return_sell',
                    'best_weights', 'status', 'progress_pct', 'progress_msg', 'error',
                    'sharpe_ratio', 'max_drawdown', 'profit_factor', 'volatility',
                    'win_loss_ratio', 'risk_metrics', 'survivorship_warning',
                    'coverage_warning', 'model_alignment_pct',
                    'expected_value_per_trade', 'out_of_sample_accuracy',
                    'portfolio_total_return', 'portfolio_sharpe',
                    'walk_forward_windows', 'weight_stability'}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        try:
            with self._get_transaction() as conn:
                conn.execute(f"UPDATE backtest_runs SET {set_clause} WHERE id = ?", tuple(values))
        except Exception as e:
            self.logger.error(f"Error updating backtest run {run_id}: {e}")

    def save_backtest_result(self, run_id: int, result: Dict):
        """Save a single backtest result row."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    INSERT INTO backtest_results
                    (run_id, ticker, test_date, signal, tech_score, momentum_score,
                     composite_score, forward_5d_return, forward_20d_return, hit,
                     benchmark_ticker, benchmark_return, alpha,
                     forward_10d_return, forward_40d_return, forward_60d_return, regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    result['ticker'],
                    result['test_date'],
                    result['signal'],
                    result.get('tech_score'),
                    result.get('momentum_score'),
                    result.get('composite_score'),
                    result.get('forward_5d_return'),
                    result.get('forward_20d_return'),
                    result.get('hit', 0),
                    result.get('benchmark_ticker'),
                    result.get('benchmark_return'),
                    result.get('alpha'),
                    result.get('forward_10d_return'),
                    result.get('forward_40d_return'),
                    result.get('forward_60d_return'),
                    result.get('regime'),
                ))
        except Exception as e:
            self.logger.error(f"Error saving backtest result: {e}")

    def get_backtest_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent backtest runs."""
        return self.query(
            "SELECT * FROM backtest_runs ORDER BY run_date DESC LIMIT ?", (limit,)
        )

    def get_backtest_run(self, run_id: int) -> Optional[Dict]:
        """Get a single backtest run."""
        return self.query_one("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))

    def get_backtest_results(self, run_id: int) -> List[Dict]:
        """Get all results for a backtest run."""
        return self.query(
            "SELECT * FROM backtest_results WHERE run_id = ? ORDER BY ticker, test_date",
            (run_id,)
        )

    # === AI Cross-Check ===
    def save_crosscheck(self, ticker: str, analysis_id: int, result: Dict):
        """Save AI cross-check result."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    INSERT INTO ai_crosscheck_log
                    (ticker, analysis_id, claims_found, claims_verified,
                     accuracy, trust_score, details, warning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker.upper(),
                    analysis_id,
                    result.get('claims_found', 0),
                    result.get('claims_verified', 0),
                    result.get('accuracy'),
                    result.get('trust_score'),
                    json.dumps(result.get('details')) if result.get('details') else None,
                    result.get('warning'),
                ))
        except Exception as e:
            self.logger.error(f"Error saving crosscheck for {ticker}: {e}")

    def get_crosscheck_history(self, ticker: str = None, limit: int = 20) -> List[Dict]:
        """Get cross-check history, optionally filtered by ticker."""
        if ticker:
            return self.query(
                "SELECT * FROM ai_crosscheck_log WHERE ticker = ? ORDER BY checked_at DESC LIMIT ?",
                (ticker.upper(), limit)
            )
        return self.query(
            "SELECT * FROM ai_crosscheck_log ORDER BY checked_at DESC LIMIT ?",
            (limit,)
        )

    # === Auto-Discovery ===

    def save_discovery(self, ticker: str, signal_type: str, confidence: int,
                       price: float, strategy: str, sector: str = None,
                       market_cap: float = None, source: str = 'auto') -> Optional[int]:
        """Save a discovered stock opportunity."""
        try:
            with self._get_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO discovered_stocks
                    (ticker, signal_type, confidence, price, strategy, sector, market_cap, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker.upper(), signal_type, confidence, price, strategy, sector, market_cap, source))
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error saving discovery for {ticker}: {e}")
            return None

    def get_pending_discoveries(self, limit: int = 20) -> List[Dict]:
        """Get discoveries not yet screened (quant_score is NULL)."""
        return self.query("""
            SELECT * FROM discovered_stocks
            WHERE status = 'new' AND quant_score IS NULL
            ORDER BY found_at DESC LIMIT ?
        """, (limit,))

    def promote_discovery(self, ticker: str):
        """Mark a discovery as promoted and set promoted_at timestamp."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    UPDATE discovered_stocks
                    SET status = 'promoted', promoted_at = ?
                    WHERE ticker = ? AND status IN ('new', 'screened')
                """, (datetime.now().isoformat(), ticker.upper()))
        except Exception as e:
            self.logger.error(f"Error promoting discovery {ticker}: {e}")

    def dismiss_discovery(self, discovery_id: int, reason: str = ''):
        """Mark a discovery as dismissed."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    UPDATE discovered_stocks
                    SET status = 'dismissed', dismissed_at = ?, dismiss_reason = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), reason, discovery_id))
        except Exception as e:
            self.logger.error(f"Error dismissing discovery {discovery_id}: {e}")

    def update_discovery_score(self, ticker: str, quant_score: float):
        """Update quant score on a discovery and set status to screened."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    UPDATE discovered_stocks
                    SET quant_score = ?, status = 'screened'
                    WHERE ticker = ? AND status = 'new'
                """, (quant_score, ticker.upper()))
        except Exception as e:
            self.logger.error(f"Error updating discovery score for {ticker}: {e}")

    def get_discovery_stats(self) -> Dict:
        """Get discovery statistics by status and strategy."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as cnt
                FROM discovered_stocks
                GROUP BY status
            """)
            by_status = {row['status']: row['cnt'] for row in cursor.fetchall()}

            # Counts by strategy (last 7 days)
            cursor.execute("""
                SELECT strategy, COUNT(*) as cnt
                FROM discovered_stocks
                WHERE found_at >= datetime('now', '-7 days')
                GROUP BY strategy
            """)
            by_strategy = {row['strategy']: row['cnt'] for row in cursor.fetchall()}

            # Last 7 days total
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM discovered_stocks
                WHERE found_at >= datetime('now', '-7 days')
            """)
            week_total = cursor.fetchone()['cnt']

            # Promoted last 7 days
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM discovered_stocks
                WHERE status = 'promoted' AND promoted_at >= datetime('now', '-7 days')
            """)
            week_promoted = cursor.fetchone()['cnt']

            # Last run
            cursor.execute("""
                SELECT * FROM discovery_log ORDER BY run_at DESC LIMIT 1
            """)
            last_run_row = cursor.fetchone()
            last_run = dict(last_run_row) if last_run_row else None

            conn.close()

            return {
                'by_status': by_status,
                'by_strategy': by_strategy,
                'week_total': week_total,
                'week_promoted': week_promoted,
                'total_new': by_status.get('new', 0),
                'total_screened': by_status.get('screened', 0),
                'total_promoted': by_status.get('promoted', 0),
                'total_dismissed': by_status.get('dismissed', 0),
                'last_run': last_run,
            }
        except Exception as e:
            self.logger.error(f"Error getting discovery stats: {e}")
            return {
                'by_status': {}, 'by_strategy': {}, 'week_total': 0,
                'week_promoted': 0, 'total_new': 0, 'total_screened': 0,
                'total_promoted': 0, 'total_dismissed': 0, 'last_run': None,
            }

    def log_discovery_run(self, run_type: str, strategies: str, scanned: int,
                          found: int, promoted: int, duration: float, errors: str = ''):
        """Log a discovery run to the discovery_log table."""
        try:
            with self._get_transaction() as conn:
                conn.execute("""
                    INSERT INTO discovery_log
                    (run_type, strategies_run, tickers_scanned, discoveries_found,
                     promoted_count, duration_seconds, errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (run_type, strategies, scanned, found, promoted, duration, errors))
        except Exception as e:
            self.logger.error(f"Error logging discovery run: {e}")

    def get_recent_discoveries(self, days: int = 7, status: str = None) -> List[Dict]:
        """Get recent discoveries, optionally filtered by status."""
        if status:
            return self.query("""
                SELECT * FROM discovered_stocks
                WHERE found_at >= datetime('now', '-' || ? || ' days') AND status = ?
                ORDER BY found_at DESC
            """, (days, status))
        return self.query("""
            SELECT * FROM discovered_stocks
            WHERE found_at >= datetime('now', '-' || ? || ' days')
            ORDER BY found_at DESC
        """, (days,))

    def is_recently_discovered(self, ticker: str, days: int = 30) -> bool:
        """Check if a ticker was already discovered recently."""
        result = self.query_one("""
            SELECT COUNT(*) as cnt FROM discovered_stocks
            WHERE ticker = ? AND found_at >= datetime('now', '-' || ? || ' days')
        """, (ticker.upper(), days))
        return result['cnt'] > 0 if result else False

    def get_discovery_log(self, limit: int = 20) -> List[Dict]:
        """Get recent discovery run logs."""
        return self.query(
            "SELECT * FROM discovery_log ORDER BY run_at DESC LIMIT ?", (limit,)
        )

    def get_api_spending_breakdown(self, api: str, month: str) -> list:
        """Get spending breakdown by model for an API in a month."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model, COUNT(*) as requests,
                       SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output,
                       SUM(estimated_cost) as total_cost
                FROM api_cost_log
                WHERE api = ? AND month = ?
                GROUP BY model ORDER BY total_cost DESC
            """, (api, month))
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting API spending breakdown: {e}")
            return []


    # === Stock Notes ===
    def get_stock_note(self, ticker: str) -> Optional[str]:
        """Get free-text note for a ticker."""
        row = self.query_one("SELECT note_text FROM stock_notes WHERE ticker = ?", (ticker.upper(),))
        return row['note_text'] if row else None

    def save_stock_note(self, ticker: str, note_text: str):
        """Save or update free-text note for a ticker."""
        self.execute("""
            INSERT INTO stock_notes (ticker, note_text, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                note_text = excluded.note_text,
                updated_at = excluded.updated_at
        """, (ticker.upper(), note_text, datetime.now().isoformat()))

    # === Watchlist Tier ===
    def update_watchlist_tier(self, ticker: str, tier: str):
        """Update the tier for a watchlist entry."""
        self.execute(
            "UPDATE watchlist SET tier = ? WHERE ticker = ?",
            (tier, ticker.upper())
        )

    # === Trade Journal ===
    def add_journal_entry(self, entry: dict):
        """Add a new trade journal entry."""
        self.execute("""
            INSERT INTO trade_journal
            (ticker, entry_date, exit_date, entry_price, exit_price, shares,
             trade_type, system_signal, user_action, entry_reason, exit_reason,
             outcome_pct, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get('ticker', '').upper(),
            entry.get('entry_date'),
            entry.get('exit_date'),
            entry.get('entry_price'),
            entry.get('exit_price'),
            entry.get('shares'),
            entry.get('trade_type', 'LONG'),
            entry.get('system_signal'),
            entry.get('user_action'),
            entry.get('entry_reason'),
            entry.get('exit_reason'),
            entry.get('outcome_pct'),
            entry.get('notes'),
        ))

    def get_journal_entries(self, ticker: str = None, limit: int = 50) -> List[Dict]:
        """Get trade journal entries, optionally filtered by ticker."""
        if ticker:
            return self.query(
                "SELECT * FROM trade_journal WHERE ticker = ? ORDER BY entry_date DESC LIMIT ?",
                (ticker.upper(), limit)
            )
        return self.query(
            "SELECT * FROM trade_journal ORDER BY entry_date DESC LIMIT ?", (limit,)
        )

    def update_journal_entry(self, entry_id: int, exit_price: float, exit_date: str,
                             exit_reason: str = "", notes: str = ""):
        """Close out a journal entry with exit data."""
        entry = self.query_one(
            "SELECT entry_price, shares FROM trade_journal WHERE id = ?", (entry_id,)
        )
        outcome_pct = None
        if entry and entry.get('entry_price') and exit_price:
            direction = 1 if (entry.get('trade_type', 'LONG') == 'LONG') else -1
            outcome_pct = round(
                ((exit_price - entry['entry_price']) / entry['entry_price']) * 100 * direction, 2
            )
        self.execute("""
            UPDATE trade_journal SET
                exit_date = ?, exit_price = ?, exit_reason = ?,
                notes = COALESCE(NULLIF(?, ''), notes),
                outcome_pct = ?
            WHERE id = ?
        """, (exit_date, exit_price, exit_reason, notes, outcome_pct, entry_id))

    def delete_journal_entry(self, entry_id: int):
        """Delete a trade journal entry."""
        self.execute("DELETE FROM trade_journal WHERE id = ?", (entry_id,))

    # === Discoveries Bulk Operations ===
    def bulk_promote_discoveries(self, discovery_ids: List[int]):
        """Promote a batch of discoveries to watchlist."""
        for disc_id in discovery_ids:
            self.execute("""
                UPDATE discovered_stocks
                SET status = 'promoted', promoted_at = ?
                WHERE id = ? AND status IN ('new', 'screened')
            """, (datetime.now().isoformat(), disc_id))

    def bulk_dismiss_discoveries(self, discovery_ids: List[int], reason: str = "bulk_dismiss"):
        """Dismiss a batch of discoveries."""
        for disc_id in discovery_ids:
            self.execute("""
                UPDATE discovered_stocks
                SET status = 'dismissed', dismissed_at = ?, dismiss_reason = ?
                WHERE id = ? AND status IN ('new', 'screened')
            """, (datetime.now().isoformat(), reason, disc_id))


# Singleton
db = Database()

