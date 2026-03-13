"""
Session-based authentication for single-user deployment
Simple but secure authentication system for home network use.
"""
import bcrypt
import hashlib
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

    # === TOTP / 2FA (#33) ===

    def generate_totp_secret(self) -> str:
        """Generate a new TOTP secret key."""
        try:
            import pyotp
            return pyotp.random_base32()
        except ImportError:
            raise RuntimeError("pyotp is not installed. Run: pip install pyotp")

    def get_totp_uri(self, username: str, secret: str, issuer: str = "Stockholm") -> str:
        """Return a provisioning URI for QR code generation."""
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=issuer)

    def generate_qr_code_base64(self, uri: str) -> str:
        """Generate a QR code PNG as base64 string for embedding in HTML."""
        try:
            import qrcode
            import io
            import base64
            img = qrcode.make(uri)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except ImportError:
            raise RuntimeError("qrcode[pil] is not installed. Run: pip install 'qrcode[pil]'")

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify a TOTP code against the secret. Allows 1 window drift."""
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except Exception:
            return False

    def generate_backup_codes(self, count: int = 8) -> list:
        """Generate N one-time backup codes."""
        import secrets as _secrets
        return [_secrets.token_hex(4).upper() for _ in range(count)]

    def save_totp_for_user(self, username: str, secret: str, backup_codes: list):
        """Enable TOTP for a user in the database."""
        import json
        db.execute(
            "UPDATE users SET totp_secret = ?, totp_enabled = 1, backup_codes = ? WHERE username = ?",
            (secret, json.dumps(backup_codes), username)
        )

    def disable_totp_for_user(self, username: str):
        """Disable TOTP for a user."""
        db.execute(
            "UPDATE users SET totp_secret = NULL, totp_enabled = 0, backup_codes = NULL WHERE username = ?",
            (username,)
        )

    def get_user_totp_info(self, username: str) -> dict:
        """Return TOTP status for a user."""
        row = db.query_one(
            "SELECT totp_enabled, totp_secret, backup_codes FROM users WHERE username = ?",
            (username,)
        )
        if not row:
            return {'enabled': False}
        return {
            'enabled': bool(row.get('totp_enabled')),
            'has_secret': bool(row.get('totp_secret')),
        }

    # === Personal API Keys (#42) ===

    def generate_personal_api_key(self, label: str, scope: str = 'read') -> tuple:
        """Generate a new personal API key. Returns (raw_key, key_id).
        The raw key is shown ONCE — only the SHA-256 hash is stored."""
        raw_key = "sk_" + secrets.token_hex(32)  # 67 chars: sk_ + 64 hex
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = db.create_personal_api_key(key_hash=key_hash, label=label, scope=scope)
        return raw_key, key_id

    def validate_bearer_token(self, token: str) -> Optional[tuple]:
        """Validate a Bearer token. Returns (username, scope) or None."""
        if not token or not token.startswith("sk_"):
            return None
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        row = db.get_personal_api_key_by_hash(key_hash)
        if not row:
            return None
        db.touch_personal_api_key(key_hash)
        return ("admin", row["scope"])

    def use_backup_code(self, username: str, code: str) -> bool:
        """Try to use a backup code. Returns True and removes the code if valid."""
        import json
        row = db.query_one("SELECT backup_codes FROM users WHERE username = ?", (username,))
        if not row or not row.get('backup_codes'):
            return False
        try:
            codes = json.loads(row['backup_codes'])
        except Exception:
            return False
        code_upper = code.upper().strip()
        if code_upper in codes:
            codes.remove(code_upper)
            db.execute(
                "UPDATE users SET backup_codes = ? WHERE username = ?",
                (json.dumps(codes), username)
            )
            return True
        return False

auth_manager = AuthManager()
