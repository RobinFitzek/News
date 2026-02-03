"""CSRF Protection"""
from itsdangerous import URLSafeTimedSerializer, BadSignature
from fastapi import Request, HTTPException, status, Form
import secrets

class CSRFProtection:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.serializer = URLSafeTimedSerializer(self.secret_key)
        self.token_timeout = 3600  # 1 hour

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
