#!/usr/bin/env python3
"""
Password Reset Tool for Investment Monitor

This script allows you to reset the password for any user directly from the server.
"""

import sqlite3
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.database import db
    from core.auth import auth_manager
except ImportError:
    print("Error: Could not import core modules. Make sure you are running this from the project root directory.", file=sys.stderr)
    sys.exit(1)

def reset_password(username: str, new_password: str):
    """Reset the password for a given user"""
    if len(new_password) < 8:
        print("Error: Password must be at least 8 characters long.")
        return False
        
    user = db.get_user(username)
    if not user:
        print(f"Error: User '{username}' not found.")
        return False
        
    print(f"Resetting password for user: {username}")
    try:
        db.update_password(username, new_password)
        print("✅ Password successfully reset.")
        return True
    except Exception as e:
        print(f"❌ Failed to reset password: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python reset_password.py <username> <new_password>")
        print("Example: python reset_password.py admin newSecurePass123")
        sys.exit(1)
        
    username = sys.argv[1]
    new_password = sys.argv[2]
    
    reset_password(username, new_password)

if __name__ == "__main__":
    main()
