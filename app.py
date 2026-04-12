"""
FastAPI Web Dashboard for Investment Monitor.

Thin bootstrap: configure the FastAPI application, middleware, rate limiter,
static files and then wire up all feature routers from :mod:`routers`. Every
route handler lives under ``routers/`` so this module stays small and
navigable — historically this file held ~6400 lines of routes which made the
dashboard almost impossible to read or modify safely.
"""

# Initialise logging first so every imported module can use it.
from logging_config import setup_logging
setup_logging()

from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
import uvicorn

from core.config import WEB_HOST, WEB_PORT, ENABLE_HTTPS
from core.csrf import csrf
from core.rate_limit import limiter

from routers import (
    alerts,
    analysis,
    api_auth,
    auth,
    auto_trade,
    backtest,
    broker,
    dark_pool,
    dashboard,
    discovery,
    exports,
    fundamentals,
    insider,
    logs,
    macro,
    market_data,
    models,
    notifications as notifications_router,
    paper_trading,
    portfolio,
    reports,
    scheduler_api,
    sentiment,
    settings as settings_router,
    signals,
    watchlist,
)


app = FastAPI(title="AI Investment Monitor", version="1.0.0")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
app.state.limiter = limiter


async def _custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Redirect back with a friendly error instead of a raw JSON 429 page."""
    referer = request.headers.get("referer", "/")
    parsed = urlparse(referer)
    safe_back = parsed.path or "/"
    msg = "Too+many+requests+%E2%80%94+please+wait+a+moment+before+trying+again"
    sep = "&" if "?" in safe_back else "?"
    return RedirectResponse(url=f"{safe_back}{sep}error={msg}", status_code=303)


app.add_exception_handler(RateLimitExceeded, _custom_rate_limit_handler)


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_csrf_token(request: Request, call_next):
    """Expose a CSRF token on ``request.state`` so templates can render it."""
    request.state.csrf_token = csrf.get_token(request)
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add baseline security headers to every response."""
    response = await call_next(request)

    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if ENABLE_HTTPS:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


# ---------------------------------------------------------------------------
# Wire up feature routers
# ---------------------------------------------------------------------------
for _module in (
    auth,
    api_auth,
    dashboard,
    watchlist,
    settings_router,
    auto_trade,
    broker,
    analysis,
    discovery,
    dark_pool,
    insider,
    portfolio,
    backtest,
    paper_trading,
    macro,
    market_data,
    fundamentals,
    sentiment,
    models,
    alerts,
    signals,
    notifications_router,
    scheduler_api,
    logs,
    reports,
    exports,
):
    app.include_router(_module.router)


# ---------------------------------------------------------------------------
# SPA catch-all — must be registered AFTER every router.
# ---------------------------------------------------------------------------
_REACT_INDEX = Path(__file__).parent / "static" / "react" / "index.html"


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_spa(full_path: str):
    """Serve the React SPA for any non-API, non-static route."""
    # Block server-side API / auth endpoints — they must 404 cleanly.
    if full_path.startswith("api/") or full_path == "logout":
        raise HTTPException(status_code=404)

    # Serve static files directly (mount may not win over path routes).
    if full_path.startswith("static/"):
        file_path = Path(__file__).parent / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        raise HTTPException(status_code=404)

    if _REACT_INDEX.exists():
        return FileResponse(str(_REACT_INDEX))

    raise HTTPException(
        status_code=404,
        detail="React build not found. Run: cd frontend && npm run build",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_server(*, log_level: str = "warning") -> None:
    """Run the web server using uvicorn.

    :param log_level: uvicorn log level. Callers can pass ``"debug"`` when
        dev mode is enabled without having to duplicate the HTTPS branch.
    """
    from core.config import CERT_FILE, KEY_FILE

    if ENABLE_HTTPS:
        if not CERT_FILE.exists() or not KEY_FILE.exists():
            print("[ERROR] HTTPS enabled but certificates not found!")
            print(f"   Expected: {CERT_FILE} and {KEY_FILE}")
            return
        print(f"[HTTPS] Server starting on https://{WEB_HOST}:{WEB_PORT}")
        uvicorn.run(
            app,
            host=WEB_HOST,
            port=WEB_PORT,
            ssl_certfile=str(CERT_FILE),
            ssl_keyfile=str(KEY_FILE),
            log_level=log_level,
        )
    else:
        print(f"[WARNING] HTTP server starting on http://{WEB_HOST}:{WEB_PORT}")
        print("[WARNING] Enable HTTPS in .env for secure connections!")
        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level=log_level)


if __name__ == "__main__":
    run_server()
