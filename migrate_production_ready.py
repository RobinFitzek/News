#!/usr/bin/env python3
"""
Database migration script for production-ready improvements.
Adds earnings calendar, volume tracking, correlation matrices, dividend tracking,
cash positions, and alert acknowledgment features.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

DB_PATH = Path(__file__).parent / 'core' / 'data' / 'investment_monitor.db'


def migrate():
    """Apply all production-ready improvements to the database."""
    print("Starting production-ready migration...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Earnings calendar tracking
        print("\n1. Adding earnings calendar table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                earnings_date TEXT NOT NULL,
                estimate_eps REAL,
                fiscal_quarter TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, earnings_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_calendar(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(earnings_date)")
        print("✓ Earnings calendar table created")
        
        # 2. Dividend/Ex-date tracking
        print("\n2. Adding dividend calendar table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dividend_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                payment_date TEXT,
                dividend_amount REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, ex_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividend_ticker ON dividend_calendar(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividend_exdate ON dividend_calendar(ex_date)")
        print("✓ Dividend calendar table created")
        
        # 3. Volume metrics cache - stores volume analysis results
        print("\n3. Adding volume metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volume_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                volume REAL,
                avg_volume_20d REAL,
                volume_ratio REAL,
                vwap REAL,
                vwap_deviation_pct REAL,
                is_anomaly BOOLEAN DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume_ticker ON volume_metrics(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume_date ON volume_metrics(date)")
        print("✓ Volume metrics table created")
        
        # 4. Correlation matrix cache
        print("\n4. Adding correlation matrix table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlation_matrix (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker1 TEXT NOT NULL,
                ticker2 TEXT NOT NULL,
                correlation_90d REAL,
                correlation_30d REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker1, ticker2)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_ticker1 ON correlation_matrix(ticker1)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_ticker2 ON correlation_matrix(ticker2)")
        print("✓ Correlation matrix table created")
        
        # 5. Cash position tracking (extend portfolio)
        print("\n5. Adding cash positions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cash_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT DEFAULT 'default',
                balance REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insert default cash position if none exists
        cursor.execute("SELECT COUNT(*) FROM cash_positions")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO cash_positions (account_name, balance) VALUES ('default', 0.0)
            """)
        print("✓ Cash positions table created")
        
        # 6. Alert acknowledgment - add columns to alerts table
        print("\n6. Adding alert acknowledgment fields...")
        try:
            cursor.execute("ALTER TABLE alerts ADD COLUMN acknowledged BOOLEAN DEFAULT 0")
            cursor.execute("ALTER TABLE alerts ADD COLUMN acknowledged_at TIMESTAMP")
            cursor.execute("ALTER TABLE alerts ADD COLUMN priority INTEGER DEFAULT 5")
            cursor.execute("ALTER TABLE alerts ADD COLUMN alert_hash TEXT")
            print("✓ Alert acknowledgment fields added")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("✓ Alert fields already exist")
            else:
                raise
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_hash ON alerts(alert_hash)")
        
        # 7. Signal staleness tracking - add to analysis_history
        print("\n7. Adding signal staleness tracking...")
        try:
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN freshness_score INTEGER DEFAULT 100")
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN is_stale BOOLEAN DEFAULT 0")
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN pre_earnings_flag BOOLEAN DEFAULT 0")
            print("✓ Signal staleness fields added")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("✓ Staleness fields already exist")
            else:
                raise
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_freshness ON analysis_history(freshness_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_stale ON analysis_history(is_stale)")
        
        # 8. Tax lot tracking - extend portfolio_trades
        print("\n8. Adding tax lot tracking...")
        try:
            cursor.execute("ALTER TABLE portfolio_trades ADD COLUMN lot_id TEXT")
            cursor.execute("ALTER TABLE portfolio_trades ADD COLUMN is_closed BOOLEAN DEFAULT 0")
            cursor.execute("ALTER TABLE portfolio_trades ADD COLUMN closed_date TEXT")
            cursor.execute("ALTER TABLE portfolio_trades ADD COLUMN realized_pnl REAL")
            print("✓ Tax lot fields added")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("✓ Tax lot fields already exist")
            else:
                raise
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_lot ON portfolio_trades(lot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_closed ON portfolio_trades(is_closed)")
        
        # 9. Learning context - track why predictions failed
        print("\n9. Adding prediction failure context...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_signal TEXT,
                actual_outcome TEXT,
                failure_reason TEXT,
                context_headline TEXT,
                market_condition TEXT,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_failures_ticker ON prediction_failures(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_failures_date ON prediction_failures(prediction_date)")
        print("✓ Prediction failure context table created")
        
        # 10. Backtest results storage
        print("\n10. Adding backtest results table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                variant TEXT NOT NULL,
                threshold INTEGER,
                start_date TEXT,
                end_date TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                win_rate REAL,
                avg_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                vs_spy_alpha REAL,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_variant ON backtest_results(variant)")
        print("✓ Backtest results table created")
        
        conn.commit()
        print("\n✅ Migration completed successfully!")
        print(f"\nNew capabilities added:")
        print("  • Earnings calendar tracking")
        print("  • Volume confirmation metrics")
        print("  • Correlation awareness")
        print("  • Signal decay/staleness")
        print("  • Dividend/ex-date tracking")
        print("  • Cash position tracking")
        print("  • Alert deduplication & acknowledgment")
        print("  • Tax lot tracking")
        print("  • Prediction failure analysis")
        print("  • Backtest validation framework")
        
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        conn.close()


if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
