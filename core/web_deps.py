"""
Shared web dependencies for FastAPI routers.

Routers live under ``routers/`` and do ``from core.web_deps import *`` to pull
in every commonly used helper: the ``templates`` instance, the three auth
dependency functions (``require_auth``, ``require_auth_basic``,
``require_api_key_or_session``), and the module-level singletons the original
monolithic ``app.py`` kept at module scope.

Keeping this module as the single facade lets ``app.py`` shrink to a thin
bootstrap that just wires the routers into the FastAPI instance.
"""

from pathlib import Path
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Form, HTTPException, Depends, status
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse,
    FileResponse,
)
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.config import TEMPLATES_DIR
from core.database import db
from core.notifications import notifications
from core.auth import auth_manager
from core.csrf import csrf
from core.rate_limit import limiter
from core.audit_log import audit_log
from core.plugin_manager import plugin_manager
from core.budget_tracker import budget_tracker
from scheduler import scheduler
from engine.agents import swarm
from engine.learning_optimizer import learning_optimizer
from engine.staleness_tracker import staleness_tracker
from engine.ai_crosscheck import ai_crosscheck
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
from clients.custom_provider_client import custom_provider_client
from clients.provider_registry import (
    provider_registry,
    PROVIDER_SHORTCUTS,
    STAGE_INFO,
)

# Project root — routers live one level deep, so they cannot rely on
# ``Path(__file__).parent`` to locate files like the React SPA index.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Shared Jinja2 templates instance (autoescape for XSS protection).
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)
templates = Jinja2Templates(env=jinja_env)


def require_auth_basic(request: Request) -> str:
    """Require an active session. Redirect to ``/login`` if missing."""
    username = auth_manager.get_current_user(request)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    return username


def require_auth(request: Request) -> str:
    """Require auth and enforce the first-login password-change flow."""
    username = require_auth_basic(request)
    if db.user_must_change_password(username):
        path = request.url.path
        if path not in ("/change-password", "/logout"):
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": "/change-password"},
            )
    return username


def require_api_key_or_session(request: Request) -> str:
    """
    Accept either a ``Bearer`` personal API token or an active session cookie.

    Used on every ``/api/*`` endpoint so external tools can authenticate with a
    token while browser clients continue to rely on the session cookie.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        result = auth_manager.validate_bearer_token(token)
        if result:
            return result[0]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )
    return require_auth_basic(request)


__all__ = [
    # stdlib re-exports used by route bodies
    "Path",
    "PROJECT_ROOT",
    "datetime",
    "timedelta",
    # FastAPI
    "APIRouter",
    "Request",
    "Form",
    "HTTPException",
    "Depends",
    "status",
    "HTMLResponse",
    "RedirectResponse",
    "JSONResponse",
    "FileResponse",
    # Core singletons
    "db",
    "notifications",
    "auth_manager",
    "csrf",
    "limiter",
    "audit_log",
    "plugin_manager",
    "budget_tracker",
    "scheduler",
    "swarm",
    "learning_optimizer",
    "staleness_tracker",
    "ai_crosscheck",
    "pplx_client",
    "gemini_client",
    "custom_provider_client",
    "provider_registry",
    "PROVIDER_SHORTCUTS",
    "STAGE_INFO",
    # Templates + auth dependencies
    "templates",
    "require_auth",
    "require_auth_basic",
    "require_api_key_or_session",
]
