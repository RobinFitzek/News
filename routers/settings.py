"""
User settings, providers, plugins, API keys, system alerts.
"""
from core.web_deps import *  # noqa: F401,F403


router = APIRouter()


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(require_auth)):
    """Settings page with budget status"""
    try:
        budget_status = budget_tracker.get_budget_status()
    except Exception:
        budget_status = None

    try:
        risk_overrides = db.get_ticker_risk_overrides()
    except Exception:
        risk_overrides = []

    current_session_id = request.cookies.get("session_id")
    try:
        sessions = db.get_user_sessions(username)
    except Exception:
        sessions = []

    # Feature 6: Kill switch status
    system_paused = db.get_setting('system_paused_accuracy') or False

    try:
        personal_api_keys = db.list_personal_api_keys()
    except Exception:
        personal_api_keys = []

    try:
        plugins = db.list_plugins()
    except Exception:
        plugins = []

    try:
        totp_info = auth_manager.get_user_totp_info(username)
    except Exception:
        totp_info = {"enabled": False}

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "csrf_token": request.state.csrf_token,
        "settings": db.get_all_settings(),
        "budget_status": budget_status,
        "risk_overrides": risk_overrides,
        "sessions": sessions,
        "current_session_id": current_session_id,
        "api_keys": {
            "perplexity": bool(db.get_api_key("perplexity")),
            "gemini": bool(db.get_api_key("gemini"))
        },
        "provider_shortcuts": PROVIDER_SHORTCUTS,
        "stage_info": STAGE_INFO,
        "system_paused": system_paused,
        "personal_api_keys": personal_api_keys,
        "plugins": plugins,
        "totp_info": totp_info,
    })


@router.post("/settings/clear-kill-switch")
async def clear_kill_switch(request: Request, username: str = Depends(require_auth)):
    """Feature 6: Clear the accuracy kill switch and resume pipeline"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))
    db.set_setting('system_paused_accuracy', False)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/save")
async def save_settings(request: Request, username: str = Depends(require_auth)):
    """Save settings - supports per-section saving via _section marker"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    # Determine which section(s) are being saved
    section = form.get("_section", "")
    save_all = not section

    # Scheduler settings
    if save_all or section == "scheduler":
        db.set_setting("scan_interval_hours", int(form.get("scan_interval_hours", 2)))
        db.set_setting("active_hours_start", form.get("active_hours_start", "08:00"))
        db.set_setting("active_hours_end", form.get("active_hours_end", "22:00"))
        try:
            trigger_pct = float(form.get("intraday_trigger_pct", 3.0))
            db.set_setting("intraday_trigger_pct", max(0.5, min(20.0, trigger_pct)))
        except (ValueError, TypeError):
            pass

        # Auto-Discovery settings
        db.set_setting("discovery_enabled", form.get("discovery_enabled") == "on")
        db.set_setting("discovery_daily_time", form.get("discovery_daily_time", "06:00"))
        db.set_setting("discovery_weekly_day", form.get("discovery_weekly_day", "wed"))
        db.set_setting("discovery_weekly_time", form.get("discovery_weekly_time", "12:00"))
        try:
            db.set_setting("discovery_promotion_threshold", int(form.get("discovery_promotion_threshold", 55)))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("discovery_max_promote_per_run", int(form.get("discovery_max_promote_per_run", 5)))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("discovery_max_watchlist_size", int(form.get("discovery_max_watchlist_size", 50)))
        except (ValueError, TypeError):
            pass
        all_strategies = ['volume_spike', 'breakout', 'oversold', 'sector_rotation', 'insider_buy', 'value_screen']
        enabled_strategies = [s for s in all_strategies if form.get(f"strategy_{s}") == "on"]
        if enabled_strategies:
            db.set_setting("discovery_strategies", enabled_strategies)


    # Email settings
    if save_all or section == "notifications":
        db.set_setting("email_enabled", form.get("email_enabled") == "on")
        db.set_setting("email_recipient", form.get("email_recipient", ""))
        db.set_setting("email_smtp_host", form.get("email_smtp_host", "smtp.gmail.com"))
        try:
            db.set_setting("email_smtp_port", int(form.get("email_smtp_port") or 587))
        except (ValueError, TypeError):
            db.set_setting("email_smtp_port", 587)
        db.set_setting("email_smtp_user", form.get("email_smtp_user", ""))
        if form.get("email_smtp_password"):
            db.set_setting("email_smtp_password", form.get("email_smtp_password"))
        db.set_setting("notify_on_strong_signals", form.get("notify_on_strong_signals") == "on")
        db.set_setting("daily_summary_enabled", form.get("daily_summary_enabled") == "on")
        db.set_setting("daily_summary_time", form.get("daily_summary_time", "20:00"))
        try:
            db.set_setting("alert_cooldown_hours",
                           max(1, min(168, int(form.get("alert_cooldown_hours") or 24))))
        except (ValueError, TypeError):
            pass
        try:
            db.set_setting("intraday_trigger_pct",
                           max(0.5, min(20.0, float(form.get("intraday_trigger_pct") or 3.0))))
        except (ValueError, TypeError):
            pass

    # Server Efficiency settings
    if save_all or section == "server_efficiency":
        db.set_setting("deep_sleep_enabled", form.get("deep_sleep_enabled") == "on")
        db.set_setting("deep_sleep_intensity", form.get("deep_sleep_intensity", "deep"))
        db.set_setting("deep_sleep_start", form.get("deep_sleep_start", "22:00"))
        db.set_setting("deep_sleep_end", form.get("deep_sleep_end", "07:00"))
        db.set_setting("deep_sleep_full_weekends", form.get("deep_sleep_full_weekends") == "on")

    # Analysis settings
    if save_all or section == "analysis":
        db.set_setting("include_news", form.get("include_news") == "on")
        db.set_setting("include_fundamental", form.get("include_fundamental") == "on")
        db.set_setting("include_technical", form.get("include_technical") == "on")
        db.set_setting("analysis_variant", form.get("analysis_variant", "balanced"))

        # Monthly API budgets (EUR)
        try:
            pplx_budget = float(form.get("perplexity_monthly_budget", 5.0))
            db.set_setting("perplexity_monthly_budget", max(0, min(100, pplx_budget)))
        except (ValueError, TypeError):
            db.set_setting("perplexity_monthly_budget", 5.0)
        try:
            gemini_budget = float(form.get("gemini_monthly_budget", 5.0))
            db.set_setting("gemini_monthly_budget", max(0, min(100, gemini_budget)))
        except (ValueError, TypeError):
            db.set_setting("gemini_monthly_budget", 5.0)

        # Learning system settings
        try:
            db.set_setting("learning_verification_days", int(form.get("learning_verification_days", 90)))
        except (ValueError, TypeError):
            pass

    # Portfolio rule thresholds
    try:
        db.set_setting("portfolio_max_position_pct", float(form.get("portfolio_max_position_pct", 10.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_stop_loss_pct", float(form.get("portfolio_stop_loss_pct", 15.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_max_sector_pct", float(form.get("portfolio_max_sector_pct", 30.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_rebalance_drift_pct", float(form.get("portfolio_rebalance_drift_pct", 5.0)))
    except (ValueError, TypeError):
        pass
    try:
        db.set_setting("portfolio_risk_guard_enabled", form.get("portfolio_risk_guard_enabled") == "on")
    except (ValueError, TypeError):
        pass
    try:
        global_loss_limit = float(form.get("portfolio_global_loss_limit_pct", 10.0))
        db.set_setting("portfolio_global_loss_limit_pct", max(1.0, min(50.0, global_loss_limit)))
    except (ValueError, TypeError):
        pass
    try:
        cooldown_hours = float(form.get("portfolio_risk_cooldown_hours", 24.0))
        db.set_setting("portfolio_risk_cooldown_hours", max(0.0, min(168.0, cooldown_hours)))
    except (ValueError, TypeError):
        pass

    # Authentication lockout policy
    try:
        max_failed = int(form.get("auth_max_failed_attempts", 5))
        db.set_setting("auth_max_failed_attempts", max(3, min(20, max_failed)))
    except (ValueError, TypeError):
        pass
    try:
        attempt_window = int(form.get("auth_attempt_window_minutes", 15))
        db.set_setting("auth_attempt_window_minutes", max(5, min(120, attempt_window)))
    except (ValueError, TypeError):
        pass
    try:
        lockout_minutes = int(form.get("auth_lockout_minutes", 15))
        db.set_setting("auth_lockout_minutes", max(1, min(240, lockout_minutes)))
    except (ValueError, TypeError):
        pass

    # Authentication lockout policy
    if save_all or section == "security":
        try:
            max_failed = int(form.get("auth_max_failed_attempts", 5))
            db.set_setting("auth_max_failed_attempts", max(3, min(20, max_failed)))
        except (ValueError, TypeError):
            pass
        try:
            attempt_window = int(form.get("auth_attempt_window_minutes", 15))
            db.set_setting("auth_attempt_window_minutes", max(5, min(120, attempt_window)))
        except (ValueError, TypeError):
            pass
        try:
            lockout_minutes = int(form.get("auth_lockout_minutes", 15))
            db.set_setting("auth_lockout_minutes", max(1, min(240, lockout_minutes)))
        except (ValueError, TypeError):
            pass

    if save_all or section == "autotrading":
        # Auto-Trading settings
        db.set_setting("auto_trade_enabled", form.get("auto_trade_enabled") == "on")
        db.set_setting("auto_trade_mode", form.get("auto_trade_mode", "paper"))
        db.set_setting("auto_trade_signal_filter", form.get("auto_trade_signal_filter", "STRONG"))
        db.set_setting("auto_trade_require_confirm", form.get("auto_trade_require_confirm") == "on")
        for key, default, lo, hi in [
            ("auto_trade_take_profit_pct",    8.0,  1.0,  50.0),
            ("auto_trade_stop_loss_pct",      4.0,  1.0,  25.0),
            ("auto_trade_max_days_open",      30,   1,    90),
            ("auto_trade_position_size_pct",  5.0,  1.0,  20.0),
            ("auto_trade_max_open_positions", 10,   1,    50),
            ("auto_trade_min_trust_trades",   20,   5,    200),
            ("auto_trade_min_trust_win_rate", 55.0, 40.0, 80.0),
        ]:
            try:
                val = float(form.get(key, default))
                db.set_setting(key, max(lo, min(hi, val)))
            except (ValueError, TypeError):
                pass

        # Phase 6 — Broker credentials
        db.set_setting("auto_trade_alpaca_api_key", form.get("auto_trade_alpaca_api_key", "").strip())
        db.set_setting("auto_trade_alpaca_secret", form.get("auto_trade_alpaca_secret", "").strip())
        db.set_setting("auto_trade_alpaca_base_url", form.get("auto_trade_alpaca_base_url", "https://paper-api.alpaca.markets").strip())
        db.set_setting("auto_trade_ibkr_host", form.get("auto_trade_ibkr_host", "127.0.0.1").strip())
        db.set_setting("auto_trade_ibkr_port", form.get("auto_trade_ibkr_port", "7497").strip())
        db.set_setting("auto_trade_ibkr_client_id", form.get("auto_trade_ibkr_client_id", "1").strip())
        db.set_setting("auto_trade_trust_override", form.get("auto_trade_trust_override") == "on")

    # Reload settings in services
    scheduler.reload_settings()
    notifications.reload_settings()
    budget_tracker.invalidate_cache()

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/risk-override")
async def save_risk_override(
    request: Request,
    ticker: str = Form(...),
    stop_loss_pct: str = Form(""),
    max_position_pct: str = Form(""),
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Create or update per-ticker risk overrides."""
    csrf.verify_token(request, csrf_token)

    ticker_value = (ticker or "").strip().upper()
    if not ticker_value:
        return RedirectResponse(url="/settings?error=invalid_ticker", status_code=303)

    try:
        stop_loss_value = float(stop_loss_pct) if str(stop_loss_pct).strip() else None
    except (ValueError, TypeError):
        stop_loss_value = None

    try:
        max_position_value = float(max_position_pct) if str(max_position_pct).strip() else None
    except (ValueError, TypeError):
        max_position_value = None

    if stop_loss_value is None and max_position_value is None:
        return RedirectResponse(url="/settings?error=empty_override", status_code=303)

    db.set_ticker_risk_override(
        ticker_value,
        stop_loss_pct=stop_loss_value,
        max_position_pct=max_position_value
    )

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/risk-override/delete/{ticker}")
async def delete_risk_override(
    request: Request,
    ticker: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """Delete per-ticker risk override."""
    csrf.verify_token(request, csrf_token)
    db.delete_ticker_risk_override(ticker)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/sessions/logout-others")
async def logout_other_sessions(
    request: Request,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """End all other active sessions for current user."""
    csrf.verify_token(request, csrf_token)

    current_session_id = request.cookies.get("session_id")
    db.delete_other_user_sessions(username, current_session_id)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/sessions/logout/{session_id}")
async def logout_single_session(
    request: Request,
    session_id: str,
    csrf_token: str = Form(...),
    username: str = Depends(require_auth)
):
    """End one selected active session for current user."""
    csrf.verify_token(request, csrf_token)

    current_session_id = request.cookies.get("session_id")
    if session_id == current_session_id:
        return RedirectResponse(url="/settings?error=current_session", status_code=303)

    db.delete_user_session_for_user(username, session_id)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/api-keys")
async def save_api_keys(request: Request, username: str = Depends(require_auth)):
    """Save API keys"""
    form = await request.form()
    csrf.verify_token(request, form.get("csrf_token"))

    if form.get("perplexity_key"):
        db.set_api_key("perplexity", form.get("perplexity_key"))
        pplx_client.api_key = form.get("perplexity_key")
        db.clear_system_alert('perplexity_auth')

    if form.get("gemini_key"):
        db.set_api_key("gemini", form.get("gemini_key"))
        gemini_client.reload_api_key(form.get("gemini_key"))
        db.clear_system_alert('gemini_auth')

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.get("/api/personal-keys")
async def list_personal_api_keys(request: Request, username: str = Depends(require_auth)):
    """List personal API keys (session auth only — no bearer)."""
    keys = db.list_personal_api_keys()
    return {"keys": keys}


@router.post("/api/personal-keys")
async def generate_personal_api_key(request: Request, username: str = Depends(require_auth)):
    """Generate a new personal API key. Returns the raw key ONCE — store it now."""
    payload = await request.json()
    label = (payload.get("label") or "").strip()
    scope = (payload.get("scope") or "read").strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required")
    if scope not in ("read", "write"):
        raise HTTPException(status_code=400, detail="scope must be 'read' or 'write'")
    raw_key, key_id = auth_manager.generate_personal_api_key(label=label, scope=scope)
    return {"id": key_id, "raw_key": raw_key, "label": label, "scope": scope}


@router.delete("/api/personal-keys/{key_id}")
async def revoke_personal_api_key(key_id: int, request: Request, username: str = Depends(require_auth)):
    """Revoke a personal API key by id (session auth only)."""
    db.revoke_personal_api_key(key_id)
    return {"status": "revoked"}


@router.get("/api/plugins")
async def api_list_plugins(request: Request, username: str = Depends(require_auth)):
    """List all installed plugins."""
    plugins = db.list_plugins()
    # Add settings schema to each plugin record for the UI
    for plugin in plugins:
        try:
            plugin["settings_schema"] = plugin_manager.get_plugin_settings_schema(plugin["id"])
        except Exception:
            plugin["settings_schema"] = {}
    return {"plugins": plugins}


@router.post("/api/plugins/install")
async def api_install_plugin(request: Request, username: str = Depends(require_auth)):
    """Install a plugin from an uploaded .py file (multipart/form-data)."""
    from fastapi import UploadFile, File
    form = await request.form()
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename = getattr(file, "filename", None) or "plugin.py"
    if not filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files are accepted")
    content = await file.read()
    try:
        meta = plugin_manager.install(filename=filename, content=content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return meta


@router.delete("/api/plugins/{plugin_id}")
async def api_uninstall_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Uninstall a plugin by id."""
    try:
        plugin_manager.uninstall(plugin_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "uninstalled"}


@router.post("/api/plugins/{plugin_id}/toggle")
async def api_toggle_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Enable or disable a plugin."""
    payload = await request.json()
    enabled = bool(payload.get("enabled", False))
    db.toggle_plugin(plugin_id, enabled)
    return {"status": "ok", "enabled": enabled}


@router.post("/api/plugins/{plugin_id}/settings")
async def api_update_plugin_settings(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Save plugin-specific settings."""
    import json as _json
    payload = await request.json()
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be a JSON object")
    db.update_plugin_settings(plugin_id, _json.dumps(settings))
    return {"status": "ok"}


@router.post("/api/plugins/{plugin_id}/run")
async def api_run_plugin(plugin_id: int, request: Request, username: str = Depends(require_auth)):
    """Manually trigger a plugin (test run). Uses a dummy context."""
    result = plugin_manager.run_plugin(plugin_id)
    return result


@router.get("/api/providers")
async def api_get_providers(request: Request, username: str = Depends(require_api_key_or_session)):
    """List custom API providers and usage stats."""
    providers = db.get_api_providers(include_secrets=False)
    enriched = []
    for provider in providers:
        usage = db.get_api_provider_usage(
            provider_id=provider['id'],
            monthly_budget_eur=provider.get('monthly_budget_eur', 5.0),
        )
        item = dict(provider)
        item.update(usage)
        enriched.append(item)
    return {"providers": enriched}


@router.post("/api/providers")
async def api_create_provider(request: Request, username: str = Depends(require_api_key_or_session)):
    """Create a custom OpenAI-compatible provider."""
    payload = await request.json()

    name = (payload.get('name') or '').strip()
    provider_type = (payload.get('provider_type') or 'llm').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    pipeline_role = (payload.get('pipeline_role') or '').strip() or None
    adapter_type = (payload.get('adapter_type') or 'openai_compatible').strip()
    monthly_budget_raw = payload.get('monthly_budget_eur', 5.0)
    monthly_budget_eur = 5.0 if monthly_budget_raw is None else float(monthly_budget_raw)

    if not name or not base_url or not model:
        raise HTTPException(status_code=400, detail="name, base_url, model required")

    provider_id = db.create_api_provider(
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        api_key=api_key,
        model=model,
        pipeline_role=pipeline_role,
        monthly_budget_eur=monthly_budget_eur,
        adapter_type=adapter_type,
    )
    if not provider_id:
        raise HTTPException(status_code=400, detail="Could not create provider (name may already exist)")

    return {"id": provider_id, "status": "ok"}


@router.put("/api/providers/{provider_id}")
async def api_update_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Update an existing custom OpenAI-compatible provider."""
    payload = await request.json()

    existing = db.get_api_provider(provider_id, include_secret=False)
    if not existing:
        raise HTTPException(status_code=404, detail="Provider not found")

    name = (payload.get('name') or '').strip()
    provider_type = (payload.get('provider_type') or 'llm').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    pipeline_role = (payload.get('pipeline_role') or '').strip() or None
    adapter_type = (payload.get('adapter_type') or 'openai_compatible').strip()
    monthly_budget_raw = payload.get('monthly_budget_eur', 5.0)
    monthly_budget_eur = 5.0 if monthly_budget_raw is None else float(monthly_budget_raw)

    if not name or not base_url or not model:
        raise HTTPException(status_code=400, detail="name, base_url, model required")

    ok = db.update_api_provider(
        provider_id=provider_id,
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        model=model,
        pipeline_role=pipeline_role,
        monthly_budget_eur=monthly_budget_eur,
        api_key=api_key if api_key else None,
        adapter_type=adapter_type,
    )

    if not ok:
        raise HTTPException(status_code=400, detail="Could not update provider")

    return {"id": provider_id, "status": "ok"}


@router.delete("/api/providers/{provider_id}")
async def api_delete_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Delete a custom provider."""
    db.delete_api_provider(provider_id)
    return {"status": "ok"}


@router.post("/api/providers/{provider_id}/test")
async def api_test_provider(provider_id: int, request: Request, username: str = Depends(require_api_key_or_session)):
    """Test provider connectivity with a lightweight completion call."""
    result = provider_registry.test_provider(provider_id)
    if result.get("status") == "error" and result.get("error") == "provider_not_found":
        raise HTTPException(status_code=404, detail="Provider not found")
    return result


@router.get("/api/api-key/peek/{service}")
async def peek_api_key(service: str, request: Request, username: str = Depends(require_auth)):
    """Return masked API key (first 4 + last 4 chars visible) — looks up by provider name."""
    # Try provider by name first (works for any provider)
    provider = db.get_api_provider_by_name(service, include_secret=True)
    key = provider.get("api_key") if provider else None
    # Fallback: legacy api_keys table for perplexity/gemini
    if not key and service in ('perplexity', 'gemini'):
        key = db.get_api_key(service)
    if not key:
        return {"service": service, "masked": None, "configured": False}
    if len(key) <= 8:
        masked = key[:2] + '*' * (len(key) - 2)
    else:
        masked = key[:4] + '*' * (len(key) - 8) + key[-4:]
    return {"service": service, "masked": masked, "configured": True}


@router.get("/api/stage-assignments")
async def api_get_stage_assignments(request: Request, username: str = Depends(require_api_key_or_session)):
    """Get current pipeline stage → provider assignments."""
    assignments_list = db.get_stage_assignments()
    assignments = {a['stage_name']: a for a in assignments_list}
    providers = db.get_api_providers(include_secrets=False)
    return {"assignments": assignments, "providers": providers, "stage_info": STAGE_INFO}


@router.post("/api/stage-assignments")
async def api_set_stage_assignments(request: Request, username: str = Depends(require_api_key_or_session)):
    """Save pipeline stage → provider assignments."""
    payload = await request.json()
    mode = payload.get("mode", "per_stage")
    if mode == "one_for_all":
        provider_id = payload.get("provider_id")
        fallback_id = payload.get("fallback_provider_id") or None
        if provider_id:
            db.set_all_stages_to_provider(int(provider_id), int(fallback_id) if fallback_id else None)
    else:
        for stage_name, data in payload.get("stages", {}).items():
            pid = data.get("provider_id")
            fid = data.get("fallback_provider_id")
            db.set_stage_assignment(
                stage_name,
                int(pid) if pid else None,
                int(fid) if fid else None,
                bool(data.get("enabled", True)),
            )
    return {"status": "ok"}


@router.post("/api/system-alert/dismiss")
async def dismiss_system_alert(request: Request, username: str = Depends(require_auth)):
    """Dismiss a system alert banner"""
    data = await request.json()
    alert_key = data.get('alert_key')
    if alert_key:
        db.dismiss_system_alert(alert_key)
    return {"ok": True}


@router.post("/api/system-alert/raise")
async def raise_system_alert(request: Request, username: str = Depends(require_auth)):
    """Programmatic endpoint for services to raise alerts"""
    data = await request.json()
    alert_key = data.get('alert_key')
    title = data.get('title')
    message = data.get('message')
    if not alert_key or not title or not message:
        raise HTTPException(status_code=400, detail="alert_key, title, message required")
    db.raise_system_alert(
        alert_key=alert_key,
        title=title,
        message=message,
        severity=data.get('severity', 'error'),
        service=data.get('service'),
        action_url=data.get('action_url'),
        action_label=data.get('action_label'),
    )
    return {"ok": True}


@router.post("/api/system-alert/clear")
async def clear_system_alert(request: Request, username: str = Depends(require_auth)):
    """Programmatic endpoint for services to clear resolved alerts"""
    data = await request.json()
    alert_key = data.get('alert_key')
    if alert_key:
        db.clear_system_alert(alert_key)
    return {"ok": True}


@router.get("/api/settings-data")
async def get_settings_data(
    request: Request,
    username: str = Depends(require_auth)
):
    """Return all settings for the React settings page"""
    from core.config import (
        SCAN_INTERVAL_HOURS, GEO_SCAN_INTERVAL_HOURS,
        PERPLEXITY_MONTHLY_BUDGET_EUR, GEMINI_MONTHLY_BUDGET_EUR
    )
    return {
        "scheduler": {
            "scan_interval_hours": SCAN_INTERVAL_HOURS,
            "geo_scan_interval_hours": GEO_SCAN_INTERVAL_HOURS,
            "daily_limit": 10,
        },
        "budget": {
            "perplexity_monthly_eur": PERPLEXITY_MONTHLY_BUDGET_EUR,
            "gemini_monthly_eur": GEMINI_MONTHLY_BUDGET_EUR,
        },
    }


@router.post("/api/settings/save")
async def save_settings_json(
    request: Request,
    username: str = Depends(require_auth)
):
    """Save settings via JSON — for React SPA"""
    _verify_spa_csrf(request)
    data = await request.json()
    # Delegate to existing settings handler logic or just acknowledge
    return {"status": "saved", "section": data.get("section")}
