# Startup & Entrypoint Guide

This document explains every startup/entrypoint file in the repository, which one is the **correct primary startup**, and how the files relate to each other.

---

## ✅ Correct / Primary Startup

```bash
python main.py
```

Or using the convenience shell script:

```bash
./start.sh
```

`main.py` is the **official entry point**. It:

1. Prints a startup banner
2. Checks that required API keys (Perplexity, Gemini) are configured
3. Auto-starts the **background scheduler** (if keys are present and `auto_start_scheduler` is enabled)
4. Starts the **FastAPI web dashboard** via `uvicorn` on the configured host/port (default: `http://localhost:8443`)
5. Handles `SIGINT`/`SIGTERM` for clean shutdown

> First-time setup: run `./setup.sh` before `python main.py`. The setup script creates a virtual environment, installs dependencies, and initialises the SQLite database.

---

## What the Application Does

**Stockholm — AI Investment Monitor** is a self-hosted, autonomous investment intelligence platform. It runs a three-stage analysis pipeline:

```
Stage 1 — Quant Screen        (zero API cost, computed from yfinance)
          ↓ top candidates
Stage 2 — Market Intelligence  (Perplexity AI: real-time news, geopolitical scan)
          ↓ enriched candidates
Stage 3 — Research Synthesis   (Gemini: Bull/Bear/Risk score, source citations)
          ↓
       SQLite DB → Web Dashboard → Email / Telegram / Webhook Alerts
```

The web dashboard runs on a single FastAPI application (`app.py`) and exposes 131 REST endpoints plus 26 HTML views (dark theme, Jinja2 templates). All configuration (API keys, scan intervals, budgets, strategies, alerts) is managed through the **Settings** page in the dashboard — no manual file editing required after initial setup.

---

## All Entrypoint & Startup Files

### 1. `main.py` — **Primary entry point** ✅

| Property | Value |
|---|---|
| **Run with** | `python main.py` |
| **What it starts** | Scheduler + web dashboard |
| **Port** | Configured in `core/config.py` (default `8443`) |
| **Use for** | Normal use, production, daily operation |

This is the file you should always use to start the application. `app.py` is imported here and handed to `uvicorn`; you do **not** need to touch `app.py` directly.

---

### 2. `start.sh` — **Convenience wrapper** ✅

```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py
```

Activates the virtual environment and calls `python main.py`. Functionally identical to running `main.py` directly. Safe to use and keep.

---

### 3. `app.py` — **FastAPI application module** ⚠️ (not meant to be run directly in production)

| Property | Value |
|---|---|
| **Run with** | `python app.py` (works, but not recommended) |
| **What it starts** | Web dashboard only (no scheduler) |
| **Port** | Same as `main.py` |
| **Use for** | Quick local UI testing without the scheduler overhead |

`app.py` contains the entire FastAPI application (4,600+ lines, 131 endpoints). It **can** be executed directly (useful during development to test just the UI), but this skips the scheduler and the status/banner checks. In normal operation `main.py` imports and runs it — you should not run `app.py` standalone in production.

| Feature | `python main.py` | `python app.py` |
|---|---|---|
| Startup banner & status check | ✅ | ❌ |
| Auto-starts scheduler | ✅ | ❌ |
| Starts web dashboard | ✅ | ✅ |
| HTTPS support | ✅ | ✅ |
| Recommended for production | ✅ | ❌ |

---

### 4. `scheduler.py` — **Background job daemon** (library, not a standalone startup)

`scheduler.py` is imported and managed by `main.py`. It runs 14+ background jobs (stock scans, geopolitical scans, discovery, paper-trade entries/exits, ML retraining, health checks). You do **not** run this file directly — `main.py` calls `scheduler.start()` automatically.

---

### 5. `cli.py` — **Headless CLI interface** (optional, separate tool)

```bash
python cli.py analyze AAPL
python cli.py scan
python cli.py watchlist --json
python cli.py geo
python cli.py autostatus
```

A separate command-line interface that bypasses the web UI entirely. Useful for scripting, CI pipelines, or running a quick on-demand analysis without opening a browser. It does **not** start a web server. This is intentional and should be kept.

---

### 6. `setup.sh` — **One-time installation script**

```bash
./setup.sh
```

Run once after cloning. Creates the virtual environment, installs `requirements.txt`, creates required directories, secures `.env` permissions, and initialises the database. Not a recurring startup command.

---

### 7. `dev_mode.py` — **Development mode toggle**

```bash
python dev_mode.py enable   # turns on verbose/debug logging
python dev_mode.py disable
python dev_mode.py status
```

A small utility that flips the `development_mode` flag in the database. Restart the app after toggling. Not a startup file.

---

### 8. `migrate_db.py` / `migrate_production_ready.py` — **Database migration scripts**

One-time or manual migration utilities. Run when upgrading from an older version of the app. Not startup files.

---

### 9. `reset_password.py` — **Password reset utility**

Emergency utility to reset the dashboard login password from the command line. Not a startup file.

---

### 10. `systemd/investment-monitor.service` — **Auto-start on boot**

A systemd unit file for running the application as a system service. Install with:

```bash
sudo cp systemd/investment-monitor.service /etc/systemd/system/
sudo systemctl enable investment-monitor
sudo systemctl start investment-monitor
```

This also delegates to `python main.py` internally.

---

## Summary: Which Files to Keep

| File | Keep? | Notes |
|---|---|---|
| `main.py` | ✅ Keep | Primary entry point — use this |
| `start.sh` | ✅ Keep | Convenience wrapper around `main.py` |
| `app.py` | ✅ Keep | Core application module — imported by `main.py` |
| `scheduler.py` | ✅ Keep | Background jobs — imported by `main.py` |
| `cli.py` | ✅ Keep | Useful headless interface |
| `setup.sh` | ✅ Keep | One-time setup |
| `dev_mode.py` | ✅ Keep | Useful dev utility |
| `migrate_db.py` | ✅ Keep | Needed for version upgrades |
| `migrate_production_ready.py` | ✅ Keep | Needed for version upgrades |
| `reset_password.py` | ✅ Keep | Emergency utility |
| `systemd/investment-monitor.service` | ✅ Keep | Production auto-start |

**There are no duplicate UIs and no redundant startup files.** Each file has a distinct, well-defined purpose. The only confusion is that `app.py` *can* be started directly, but it is primarily a library module — `main.py` is the correct and complete way to start the application.

---

## Two UI Branches (for the developer's reference)

If you are comparing branches:

| Branch | Settings page layout |
|---|---|
| `main` / `develop` | Sidebar + panel navigation, settings search bar |
| `feature/develop` (older) | Single vertical scroll page, more settings sections |

These are **different branches of the same app**, not two separate running UIs. Both branches use the same `main.py` → `app.py` → `uvicorn` startup chain.

---

## Quick-Reference

```bash
# First-time setup
./setup.sh

# Start the application (recommended)
python main.py
# or
./start.sh

# Start web UI only (development, no scheduler)
python app.py

# CLI usage (no web server)
python cli.py analyze AAPL
python cli.py scan --json

# Toggle debug logging
python dev_mode.py enable
python dev_mode.py status

# Auto-start on boot (Linux/systemd)
sudo cp systemd/investment-monitor.service /etc/systemd/system/
sudo systemctl enable --now investment-monitor
```

Dashboard URL: **http://localhost:8443** (or the host/port configured in `core/config.py`)
