#!/usr/bin/env python3
"""Development Mode Toggle"""
import sys
from core.database import db

def enable():
    db.set_setting('development_mode', True)
    print("✅ Development mode ENABLED - Restart app for changes")

def disable():
    db.set_setting('development_mode', False)
    print("✅ Development mode DISABLED - Restart app for changes")

def status():
    dev = db.get_setting('development_mode')
    print(f"Development mode: {'🔧 ENABLED' if dev else '📊 DISABLED'}")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'enable': enable()
    elif cmd == 'disable': disable()
    else: status()
