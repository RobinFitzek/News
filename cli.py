#!/usr/bin/env python3
"""
Stockholm AI Investment Monitor — Headless CLI
Interact with the engine directly without starting the web server.

Usage:
    python cli.py analyze AAPL [--strategy balanced] [--json]
    python cli.py scan [--json]
    python cli.py watchlist [--json]
    python cli.py geo [--json]
    python cli.py autostatus [--json]
"""
import argparse
import json
import sys
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "STRONG_BUY":  "\033[92m",   # bright green
    "BUY":         "\033[32m",   # green
    "HOLD":        "\033[33m",   # yellow
    "SELL":        "\033[31m",   # red
    "STRONG_SELL": "\033[91m",   # bright red
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _signal_label(signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, "")
    return f"{color}{signal}{RESET}" if sys.stdout.isatty() else signal


def _header(title: str):
    width = 60
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * width}{RESET}")


def _row(label: str, value, width: int = 22):
    label_str = f"{label}:".ljust(width)
    print(f"  {DIM}{label_str}{RESET}{value}")


def _output(data, use_json: bool):
    if use_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        # Caller is responsible for human-readable output before calling _output
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_analyze(ticker: str, strategy: str, use_json: bool):
    """Run full 3-stage analysis on a single ticker."""
    ticker = ticker.upper()
    if not use_json:
        print(f"\nAnalyzing {BOLD}{ticker}{RESET} (strategy: {strategy}) …")
        print(f"{DIM}This may take 30–90 seconds depending on API availability.{RESET}\n")

    try:
        from engine.agents import InvestmentSwarm
        swarm = InvestmentSwarm()
        result = swarm.analyze_single_stock(ticker, strategy=strategy)
    except Exception as e:
        sys.exit(f"Error running analysis: {e}")

    if use_json:
        print(json.dumps(result, indent=2, default=str))
        return

    if not result:
        sys.exit("Analysis returned no result.")

    rec = result.get("recommendation", {})
    signal  = rec.get("signal", result.get("signal", "—"))
    score   = rec.get("score", result.get("score", "—"))
    summary = rec.get("summary", result.get("summary", ""))
    risks   = rec.get("risks", result.get("risks", []))
    cats    = rec.get("catalysts", result.get("catalysts", []))

    _header(f"Analysis: {ticker}")
    _row("Signal",   _signal_label(signal))
    _row("Score",    f"{score}/100" if score != "—" else "—")
    _row("Strategy", strategy)

    if summary:
        print(f"\n  {BOLD}Summary{RESET}")
        # Word-wrap at 70 chars
        words = summary.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 70:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"    {l}")

    if risks:
        print(f"\n  {BOLD}Risks{RESET}")
        for r in (risks if isinstance(risks, list) else [risks]):
            print(f"    • {r}")

    if cats:
        print(f"\n  {BOLD}Catalysts{RESET}")
        for c in (cats if isinstance(cats, list) else [cats]):
            print(f"    • {c}")

    print()


def cmd_scan(use_json: bool):
    """Run the full daily pipeline scan."""
    if not use_json:
        print(f"\n{BOLD}Running daily scan pipeline…{RESET}")
        print(f"{DIM}This may take several minutes.{RESET}\n")

    try:
        from engine.pipeline import pipeline
        result = pipeline.run_daily_cycle()
    except Exception as e:
        sys.exit(f"Scan failed: {e}")

    if use_json:
        print(json.dumps(result, indent=2, default=str))
        return

    analyzed = result.get("analyzed", 0)
    strong   = result.get("strong_signals", [])
    errors   = result.get("errors", 0)

    _header("Scan Complete")
    _row("Analyzed",       analyzed)
    _row("Errors",         errors)
    _row("Strong signals", len(strong))

    if strong:
        print(f"\n  {BOLD}Strong Signals{RESET}")
        for s in strong:
            ticker = s.get("ticker", "?")
            sig    = s.get("signal", "")
            score  = s.get("score", "")
            print(f"    {_signal_label(sig):20s}  {ticker:<8}  score {score}")
    print()


def cmd_watchlist(use_json: bool):
    """List watchlist tickers with latest signal."""
    try:
        from core.database import db
        rows = db.query("""
            SELECT w.ticker, w.name,
                   ah.signal AS last_signal,
                   ah.score  AS last_score,
                   ah.timestamp AS last_analyzed
            FROM watchlist w
            LEFT JOIN analysis_history ah
                ON ah.id = (
                    SELECT id FROM analysis_history
                    WHERE ticker = w.ticker
                    ORDER BY timestamp DESC LIMIT 1
                )
            ORDER BY w.ticker
        """)
    except Exception as e:
        sys.exit(f"DB error: {e}")

    rows = rows or []

    if use_json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header(f"Watchlist ({len(rows)} tickers)")
    fmt = "  {:<8}  {:<20}  {:<14}  {:>5}  {}"
    print(fmt.format("Ticker", "Name", "Signal", "Score", "Last Analyzed"))
    print("  " + "─" * 58)
    for r in rows:
        ticker   = r.get("ticker", "")
        name     = (r.get("name") or "")[:20]
        signal   = r.get("last_signal") or "—"
        score    = r.get("last_score")
        analyzed = r.get("last_analyzed") or "never"
        if analyzed and analyzed != "never":
            try:
                analyzed = analyzed[:10]  # date only
            except Exception:
                pass
        score_str = f"{score}/100" if score is not None else "—"
        print(fmt.format(ticker, name, _signal_label(signal), score_str, analyzed))
    print()


def cmd_geo(use_json: bool):
    """Show recent geopolitical events."""
    try:
        from core.database import db
        rows = db.query("""
            SELECT id, timestamp, severity_avg, raw_summary
            FROM geopolitical_events
            ORDER BY timestamp DESC
            LIMIT 10
        """)
    except Exception as e:
        sys.exit(f"DB error: {e}")

    rows = rows or []

    if use_json:
        print(json.dumps(rows, indent=2, default=str))
        return

    _header(f"Geopolitical Events (last {len(rows)})")
    for r in rows:
        ts       = (r.get("timestamp") or "")[:16]
        severity = r.get("severity_avg")
        sev_str  = f"{severity:.1f}/10" if severity is not None else "—"
        summary  = (r.get("raw_summary") or "")[:120].replace("\n", " ")
        if severity is not None:
            if severity >= 8:
                sev_display = f"\033[91m{sev_str}{RESET}" if sys.stdout.isatty() else sev_str
            elif severity >= 5:
                sev_display = f"\033[33m{sev_str}{RESET}" if sys.stdout.isatty() else sev_str
            else:
                sev_display = sev_str
        else:
            sev_display = sev_str
        print(f"\n  {DIM}{ts}{RESET}  Severity: {sev_display}")
        print(f"  {summary}{'…' if len(r.get('raw_summary','')) > 120 else ''}")
    print()


def cmd_autostatus(use_json: bool):
    """Show auto-trading summary: open/closed/win-rate/PnL."""
    try:
        from core.database import db

        open_row = db.query_one(
            "SELECT COUNT(*) as c FROM auto_paper_trades WHERE status='open'"
        )
        closed_row = db.query_one(
            "SELECT COUNT(*) as c FROM auto_paper_trades WHERE status='closed'"
        )
        perf_row = db.query_one("""
            SELECT
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                COUNT(*) as total,
                AVG(pnl_pct) as avg_pnl,
                SUM(pnl_pct) as total_pnl
            FROM auto_paper_trades
            WHERE status='closed' AND pnl_pct IS NOT NULL
        """)
        mode    = db.get_setting("auto_trade_mode") or "paper"
        enabled = db.get_setting("auto_trade_enabled")
    except Exception as e:
        sys.exit(f"DB error: {e}")

    open_count   = (open_row or {}).get("c", 0)
    closed_count = (closed_row or {}).get("c", 0)
    wins         = (perf_row or {}).get("wins") or 0
    total        = (perf_row or {}).get("total") or 0
    avg_pnl      = (perf_row or {}).get("avg_pnl") or 0.0
    total_pnl    = (perf_row or {}).get("total_pnl") or 0.0
    win_rate     = round(wins / total * 100, 1) if total > 0 else 0.0

    data = {
        "enabled":      bool(enabled),
        "mode":         mode,
        "open":         open_count,
        "closed":       closed_count,
        "wins":         wins,
        "win_rate_pct": win_rate,
        "avg_pnl_pct":  round(float(avg_pnl) * 100, 2),
        "total_pnl_pct": round(float(total_pnl) * 100, 2),
    }

    if use_json:
        print(json.dumps(data, indent=2))
        return

    _header("Auto-Trade Status")
    _row("Enabled",    "Yes" if enabled else "No")
    _row("Mode",       mode.upper())
    _row("Open trades", open_count)
    _row("Closed trades", closed_count)
    _row("Win rate",   f"{win_rate}%  ({wins}W / {total - wins}L)")
    _row("Avg PnL",    f"{data['avg_pnl_pct']:+.2f}%")
    _row("Total PnL",  f"{data['total_pnl_pct']:+.2f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Stockholm AI Investment Monitor — headless CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  analyze TICKER   Full 3-stage analysis for a single ticker
  scan             Run the daily pipeline scan (all watchlist tickers)
  watchlist        List watchlist tickers with latest signal
  geo              Show recent geopolitical events
  autostatus       Auto-trade summary (open/closed/win-rate/PnL)

Examples:
  python cli.py analyze AAPL
  python cli.py analyze MSFT --strategy defensive --json
  python cli.py watchlist
  python cli.py geo --json
  python cli.py autostatus
        """,
    )

    parser.add_argument("--json", action="store_true", help="Output as JSON")
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single ticker")
    p_analyze.add_argument("ticker", help="Ticker symbol (e.g. AAPL)")
    p_analyze.add_argument(
        "--strategy",
        choices=["balanced", "defensive", "aggressive"],
        default="balanced",
        help="Analysis strategy (default: balanced)",
    )

    # scan
    subparsers.add_parser("scan", help="Run the full daily pipeline scan")

    # watchlist
    subparsers.add_parser("watchlist", help="List watchlist tickers with latest signals")

    # geo
    subparsers.add_parser("geo", help="Show recent geopolitical events")

    # autostatus
    subparsers.add_parser("autostatus", help="Show auto-trading summary")

    args = parser.parse_args()
    use_json = args.json

    if args.command == "analyze":
        cmd_analyze(args.ticker, args.strategy, use_json)
    elif args.command == "scan":
        cmd_scan(use_json)
    elif args.command == "watchlist":
        cmd_watchlist(use_json)
    elif args.command == "geo":
        cmd_geo(use_json)
    elif args.command == "autostatus":
        cmd_autostatus(use_json)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
