import sqlite3
from database import db

def migrate():
    print("Checking database schema...")
    conn = db._get_conn()
    cursor = conn.cursor()
    
    # Check if risk_score column exists in analysis_history
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [row['name'] for row in cursor.fetchall()]
    
    if 'risk_score' not in columns:
        print("adding risk_score column implementation...")
        try:
            cursor.execute("ALTER TABLE analysis_history ADD COLUMN risk_score INTEGER DEFAULT 5")
            conn.commit()
            print("✅ Added risk_score column to analysis_history")
        except Exception as e:
            print(f"❌ Error adding column: {e}")
    else:
        print("✅ risk_score column already exists")
        
    conn.close()

if __name__ == "__main__":
    migrate()
