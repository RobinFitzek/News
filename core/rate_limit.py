"""Rate limiting for DoS protection"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

def get_client_identifier(request: Request) -> str:
    """Use session ID if authenticated, otherwise IP"""
    session_id = request.cookies.get("session_id")
    if session_id:
        return f"session:{session_id}"
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=["100/hour"],  # Default limit for all endpoints
    storage_uri="memory://",
)
