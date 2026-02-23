import sqlite3
import sys
import os

# Add current directory to path so we can import core modules
sys.path.append(os.getcwd())

from core.database import db

def migrate():
    print("Checking database schema...")
    conn = db._get_conn()
    cursor = conn.cursor()
    
    # 1. Check risk_score in analysis_history
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [row['name'] for row in cursor.fetchall()]
    
    if 'risk_score' not in columns:
        print("adding risk_score column implementation...")
        try:
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN risk_score INTEGER DEFAULT 5")
            print("✅ Added risk_score column to analysis_history")
        except Exception as e:
            print(f"❌ Error adding risk_score column: {e}")
    else:
        print("✅ risk_score column already exists")

    # 2. Check current_volume in volume_metrics
    try:
        cursor.execute("PRAGMA table_info(volume_metrics)")
        vm_columns = [row['name'] for row in cursor.fetchall()]
        
        if 'current_volume' not in vm_columns and vm_columns:
            print("Adding current_volume to volume_metrics...")
            cursor.execute("ALTER TABLE volume_metrics ADD COLUMN current_volume REAL DEFAULT 0")
            print("✅ Added current_volume column to volume_metrics")
        elif not vm_columns:
            print("⚠️ volume_metrics table looks empty or missing")
        else:
            print("✅ current_volume column already exists in volume_metrics")
    except Exception as e:
        print(f"❌ Error checking volume_metrics: {e}")

    # 3. Check days_until in earnings_calendar
    try:
        cursor.execute("PRAGMA table_info(earnings_calendar)")
        ec_columns = [row['name'] for row in cursor.fetchall()]
        
        if 'days_until' not in ec_columns and ec_columns:
            print("Adding days_until to earnings_calendar...")
            cursor.execute("ALTER TABLE earnings_calendar ADD COLUMN days_until INTEGER DEFAULT 0")
            print("✅ Added days_until column to earnings_calendar")
        elif not ec_columns:
            print("⚠️ earnings_calendar table looks empty or missing")
        else:
            print("✅ days_until column already exists in earnings_calendar")
    except Exception as e:
        print(f"❌ Error checking earnings_calendar: {e}")

    # 3b. Handle incompatible backtest_results schema (old version had different columns)
    try:
        cursor.execute("PRAGMA table_info(backtest_results)")
        bt_res_cols = [row['name'] for row in cursor.fetchall()]
        if bt_res_cols and 'run_id' not in bt_res_cols:
            print("Old backtest_results schema detected (no run_id). Renaming to backtest_results_legacy...")
            cursor.execute("ALTER TABLE backtest_results RENAME TO backtest_results_legacy")
            print("✅ Renamed old backtest_results → backtest_results_legacy")
            print("   New backtest_results table will be created on next startup.")
    except Exception as e:
        print(f"❌ Error checking backtest_results schema: {e}")

    # 4. Add risk metrics columns to backtest_runs
    try:
        cursor.execute("PRAGMA table_info(backtest_runs)")
        br_columns = [row['name'] for row in cursor.fetchall()]

        new_br_cols = {
            'sharpe_ratio': 'REAL',
            'max_drawdown': 'REAL',
            'profit_factor': 'REAL',
            'volatility': 'REAL',
            'win_loss_ratio': 'REAL',
            'risk_metrics': 'TEXT',
            'survivorship_warning': 'TEXT',
            'coverage_warning': 'TEXT',
            'model_alignment_pct': 'REAL',
        }
        for col, dtype in new_br_cols.items():
            if col not in br_columns:
                cursor.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} to backtest_runs")
            else:
                print(f"✅ {col} already exists in backtest_runs")
    except Exception as e:
        print(f"❌ Error migrating backtest_runs: {e}")

    # 5. Add benchmark columns to backtest_results
    try:
        cursor.execute("PRAGMA table_info(backtest_results)")
        bres_columns = [row['name'] for row in cursor.fetchall()]

        new_bres_cols = {
            'benchmark_ticker': 'TEXT',
            'benchmark_return': 'REAL',
            'alpha': 'REAL',
        }
        for col, dtype in new_bres_cols.items():
            if col not in bres_columns:
                cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} to backtest_results")
            else:
                print(f"✅ {col} already exists in backtest_results")
    except Exception as e:
        print(f"❌ Error migrating backtest_results: {e}")

    # 6. Add signal_type and verification_window_days to prediction_outcomes
    try:
        cursor.execute("PRAGMA table_info(prediction_outcomes)")
        po_columns = [row['name'] for row in cursor.fetchall()]

        new_po_cols = {
            'signal_type': 'TEXT',
            'verification_window_days': 'INTEGER',
        }
        for col, dtype in new_po_cols.items():
            if col not in po_columns:
                cursor.execute(f"ALTER TABLE prediction_outcomes ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} to prediction_outcomes")
            else:
                print(f"✅ {col} already exists in prediction_outcomes")
    except Exception as e:
        print(f"❌ Error migrating prediction_outcomes: {e}")

    # 7. Add forward return and regime columns to backtest_results
    try:
        cursor.execute("PRAGMA table_info(backtest_results)")
        bres_columns2 = [row['name'] for row in cursor.fetchall()]

        new_bres_cols2 = {
            'forward_10d_return': 'REAL',
            'forward_40d_return': 'REAL',
            'forward_60d_return': 'REAL',
            'regime': 'TEXT',
        }
        for col, dtype in new_bres_cols2.items():
            if col not in bres_columns2:
                cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} to backtest_results")
            else:
                print(f"✅ {col} already exists in backtest_results")
    except Exception as e:
        print(f"❌ Error migrating backtest_results (forward returns): {e}")

    # 8. Add new summary columns to backtest_runs
    try:
        cursor.execute("PRAGMA table_info(backtest_runs)")
        br_columns2 = [row['name'] for row in cursor.fetchall()]

        new_br_cols2 = {
            'expected_value_per_trade': 'REAL',
            'out_of_sample_accuracy': 'REAL',
            'portfolio_total_return': 'REAL',
            'portfolio_sharpe': 'REAL',
        }
        for col, dtype in new_br_cols2.items():
            if col not in br_columns2:
                cursor.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} to backtest_runs")
            else:
                print(f"✅ {col} already exists in backtest_runs")
    except Exception as e:
        print(f"❌ Error migrating backtest_runs (new summary cols): {e}")

    # 9. Create ai_crosscheck_log table
    try:
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
        print("✅ ai_crosscheck_log table ready")
    except Exception as e:
        print(f"❌ Error creating ai_crosscheck_log: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate()
