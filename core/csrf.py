"""CSRF Protection"""
from itsdangerous import URLSafeTimedSerializer, BadSignature
from fastapi import Request, HTTPException, status, Form
import secrets
import os
from pathlib import Path

class CSRFProtection:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or self._get_or_create_secret()
        self.serializer = URLSafeTimedSerializer(self.secret_key)
        self.token_timeout = 3600  # 1 hour
    
    def _get_or_create_secret(self) -> str:
        """Get stable secret key from file or environment"""
        # Try environment variable first
        secret = os.getenv("CSRF_SECRET_KEY")
        if secret:
            return secret
        
        # Try secret file
        secret_file = Path(__file__).parent.parent / "data" / ".csrf_secret"
        if secret_file.exists():
            with open(secret_file, "r") as f:
                return f.read().strip()
        
        # Generate new secret
        secret = secrets.token_urlsafe(32)
        secret_file.parent.mkdir(exist_ok=True)
        with open(secret_file, "w") as f:
            f.write(secret)
        os.chmod(secret_file, 0o600)
        
        return secret

    def generate_token(self, session_id: str = None) -> str:
        data = session_id or secrets.token_urlsafe(16)
        return self.serializer.dumps(data, salt="csrf-token")

    def validate_token(self, token: str) -> bool:
        try:
            self.serializer.loads(token, salt="csrf-token", max_age=self.token_timeout)
            return True
        except BadSignature:
            return False

    def get_token(self, request: Request) -> str:
        session_id = request.cookies.get("session_id", "")
        return self.generate_token(session_id)

    def verify_token(self, request: Request, token: str = Form(None)):
        if not token or not self.validate_token(token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing CSRF token"
            )

csrf = CSRFProtection()
