#!/usr/bin/env python3
"""
AI Investment Monitor - Main Entry Point
Runs the web dashboard and scheduler for automated investment analysis.
"""
import signal
import sys
import threading
from datetime import datetime

from core.config import WEB_HOST, WEB_PORT
from core.database import db
from scheduler import scheduler

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     AI Investment Monitor                                    ║
║   ─────────────────────────────────────────────────────────  ║
║   Automatisiertes Investment-Analyse-System                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n\nShutting down...")
    scheduler.stop()
    print("Goodbye!")
    sys.exit(0)

def main():
    print_banner()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check API keys
    pplx_key = db.get_api_key("perplexity")
    gemini_key = db.get_api_key("gemini")
    
    print("Status Check:")
    print(f"   ├─ Perplexity API: {'Konfiguriert' if pplx_key else 'Nicht konfiguriert'}")
    print(f"   ├─ Gemini API: {'Konfiguriert' if gemini_key else 'Nicht konfiguriert'}")
    print(f"   ├─ Watchlist: {len(db.get_watchlist())} Aktien")
    print(f"   └─ Datenbank: {db.db_path}")
    print()
    
    # Start scheduler
    auto_start = db.get_setting("auto_start_scheduler") 
    if auto_start is None:
        auto_start = True  # Default to auto-start
    
    if auto_start and pplx_key and gemini_key:
        print("Starting scheduler...")
        scheduler.start()
    else:
        if not (pplx_key and gemini_key):
            print("(!) Scheduler nicht gestartet - API Keys fehlen")
            print("   Bitte konfigurieren unter: http://{}:{}/settings".format(WEB_HOST, WEB_PORT))
        print()
    
    # Start web server
    print(f"Starting web dashboard...")
    print(f"   └─ http://{WEB_HOST}:{WEB_PORT}")
    print()
    print("=" * 60)
    print("   Drücke Ctrl+C zum Beenden")
    print("=" * 60)
    print()
    
    # Delegate to the canonical uvicorn bootstrap in app.py so the HTTPS
    # branching logic lives in exactly one place.
    from app import run_server

    dev_mode = bool(db.get_setting("development_mode"))
    run_server(log_level="debug" if dev_mode else "warning")

if __name__ == "__main__":
    main()
