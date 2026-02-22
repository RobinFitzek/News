"""
Session-based authentication for single-user deployment
Simple but secure authentication system for home network use.
"""
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
import secrets
from fastapi import Request, HTTPException, status
from core.database import db

class AuthManager:
    def __init__(self):
        self.session_timeout = timedelta(hours=24)

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        # Encode password to bytes and truncate to 72 bytes max (bcrypt limit)
        password_bytes = password.encode('utf-8')[:72]
        hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
        return hashed.decode('utf-8')

    def verify_password(self, plain: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        try:
            # Encode password to bytes and truncate to 72 bytes max (bcrypt limit)
            password_bytes = plain.encode('utf-8')[:72]
            hashed_bytes = hashed.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception as e:
            print(f"Password verification error: {e}")
            return False

    def create_session(self, username: str, ip_address: str = None, user_agent: str = None) -> str:
        """Create a new session and return session ID"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        db.create_user_session(
            session_id=session_id,
            username=username,
            created_at=now.isoformat(),
            last_activity=now.isoformat(),
            expires_at=(now + self.session_timeout).isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return username if valid"""
        if not session_id:
            return None

        # Best-effort cleanup of expired sessions
        try:
            db.cleanup_expired_sessions()
        except Exception:
            pass

        session = db.get_user_session(session_id)
        if not session:
            return None

        # Check expiry from persisted session
        try:
            expires_at = datetime.fromisoformat(session['expires_at'])
        except Exception:
            db.delete_user_session(session_id)
            return None

        if datetime.now() > expires_at:
            db.delete_user_session(session_id)
            return None

        # Update last activity
        now = datetime.now()
        db.touch_user_session(
            session_id=session_id,
            last_activity=now.isoformat(),
            expires_at=(now + self.session_timeout).isoformat(),
        )
        return session["username"]

    def destroy_session(self, session_id: str):
        """Destroy a session (logout)"""
        if session_id:
            db.delete_user_session(session_id)

    def get_current_user(self, request: Request) -> Optional[str]:
        """Get current authenticated user from request"""
        session_id = request.cookies.get("session_id")
        return self.validate_session(session_id)

auth_manager = AuthManager()
