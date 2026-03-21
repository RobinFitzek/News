"""
Macro Event Tracker — FOMC, ECB, rate decisions + live market data.
Hardcoded 2026 calendar, yfinance daily snapshots, DB storage.
Injected into Stage 3 prompt.
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)

# 2026 FOMC meeting dates (rate decision day = 2nd day)
FOMC_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-09",
]

# 2026 ECB Governing Council meeting dates
ECB_2026 = [
    "2026-01-30", "2026-03-05", "2026-04-16",
    "2026-06-04", "2026-07-23", "2026-09-10",
    "2026-10-22", "2026-12-03",
]

# Rate-sensitive sectors/tickers (extend as needed)
RATE_SENSITIVE_SECTORS = ['financials', 'real_estate', 'utilities', 'reits']
RATE_SENSITIVE_KEYWORDS = ['bank', 'reit', 'mortgage', 'insurance', 'utility', 'bond', 'finance']


class MacroTracker:
    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Return rate events in the next N days."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        events = []
        for d in FOMC_2026:
            dt = date.fromisoformat(d)
            if today <= dt <= cutoff:
                days_until = (dt - today).days
                events.append({'type': 'FOMC', 'date': d, 'days_until': days_until,
                                'description': 'Federal Reserve rate decision'})
        for d in ECB_2026:
            dt = date.fromisoformat(d)
            if today <= dt <= cutoff:
                days_until = (dt - today).days
                events.append({'type': 'ECB', 'date': d, 'days_until': days_until,
                                'description': 'ECB Governing Council rate decision'})
        events.sort(key=lambda x: x['date'])
        return events

    def get_next_event(self) -> Optional[Dict]:
        """Return the single nearest upcoming event."""
        events = self.get_upcoming_events(days_ahead=90)
        return events[0] if events else None

    def build_macro_context_block(self) -> str:
        """Return a formatted string for injection into Stage 3 prompt."""
        events = self.get_upcoming_events(days_ahead=14)
        if not events:
            return ""
        lines = ["UPCOMING CENTRAL BANK EVENTS:"]
        for e in events:
            lines.append(f"  - {e['type']} rate decision: {e['date']} (in {e['days_until']} days)")
        lines.append("Consider rate sensitivity in your analysis. Tighten confidence for rate-sensitive sectors if event is within 48h.")
        return "\n".join(lines)

    def is_rate_sensitive(self, ticker: str, name: str = "") -> bool:
        """Heuristic: is this ticker likely rate-sensitive?"""
        text = (ticker + " " + name).lower()
        return any(kw in text for kw in RATE_SENSITIVE_KEYWORDS)

    def check_portfolio_rate_exposure(self) -> List[Dict]:
        """Return portfolio holdings that are rate-sensitive and have upcoming events."""
        events = self.get_upcoming_events(days_ahead=48 // 24 + 1)
        if not events:
            return []
        try:
            from engine.portfolio_manager import portfolio_manager
            holdings = portfolio_manager.get_portfolio_holdings()
            exposed = []
            for h in holdings:
                ticker = h.get('ticker', '')
                name = h.get('name', '')
                if self.is_rate_sensitive(ticker, name):
                    exposed.append({'ticker': ticker, 'name': name,
                                    'events': events})
            return exposed
        except Exception as e:
            logger.warning(f"Could not check portfolio rate exposure: {e}")
            return []


    # ── yfinance snapshot (#22) ──────────────────────────────────────────────

    def fetch_and_store_snapshot(self) -> Optional[Dict]:
        """
        Fetch key macro tickers via yfinance, compute derived metrics,
        and upsert today's row into macro_snapshots.
        Returns the snapshot dict or None on failure.
        """
        try:
            import yfinance as yf
            tickers_needed = ["^TNX", "^FVX", "^IRX", "^VIX", "HYG", "LQD", "DX-Y.NYB"]
            data = {}
            for sym in tickers_needed:
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period="2d")
                    if not hist.empty:
                        data[sym] = float(hist['Close'].iloc[-1])
                    else:
                        data[sym] = None
                except Exception:
                    data[sym] = None

            yield_10y = data.get("^TNX")          # 10-year Treasury yield (%)
            yield_5y  = data.get("^FVX")          # 5-year (proxy for 2y when ^FVX available)
            yield_3m  = data.get("^IRX")          # 3-month T-bill
            vix       = data.get("^VIX")
            hyg       = data.get("HYG")
            lqd       = data.get("LQD")
            dxy       = data.get("DX-Y.NYB")

            # Spread: 10Y minus 3M (standard recession signal)
            spread = None
            if yield_10y is not None and yield_3m is not None:
                spread = round(yield_10y - yield_3m, 3)

            # Credit spread proxy: HYG / LQD ratio (higher = tighter credit)
            credit_spread = None
            if hyg is not None and lqd is not None and lqd > 0:
                credit_spread = round(hyg / lqd, 4)

            # Regime label based on yield-curve slope
            regime = "Normal"
            if spread is not None:
                if spread < 0:
                    regime = "Inverted (Recession Risk)"
                elif spread < 0.3:
                    regime = "Flat (Caution)"
                elif spread > 1.5:
                    regime = "Steep (Growth)"

            today_str = date.today().isoformat()
            db.save_macro_snapshot(
                date=today_str,
                yield_10y=yield_10y,
                yield_2y=yield_5y,   # using 5Y as proxy — closest free yfinance equivalent
                yield_3m=yield_3m,
                spread_2y10y=spread,
                vix=vix,
                hyg_price=hyg,
                lqd_price=lqd,
                credit_spread=credit_spread,
                dxy=dxy,
                regime_label=regime,
            )
            snapshot = {
                'date': today_str,
                'yield_10y': yield_10y,
                'yield_2y': yield_5y,
                'yield_3m': yield_3m,
                'spread_2y10y': spread,
                'vix': vix,
                'hyg_price': hyg,
                'lqd_price': lqd,
                'credit_spread': credit_spread,
                'dxy': dxy,
                'regime_label': regime,
            }
            logger.info(f"Macro snapshot stored: spread={spread}, vix={vix}, regime={regime}")
            return snapshot
        except Exception as e:
            logger.error(f"fetch_and_store_snapshot failed: {e}", exc_info=True)
            return None

    def get_macro_snapshots(self, days: int = 90) -> List[Dict]:
        """Return last N days of stored macro snapshots, oldest first (for charts)."""
        rows = db.get_macro_snapshots(days)
        return list(reversed(rows))  # oldest → newest for chart rendering

    def get_latest_snapshot(self) -> Optional[Dict]:
        """Return the most recent stored macro snapshot."""
        return db.get_latest_macro_snapshot()

    def build_macro_context_block(self) -> str:
        """Return a formatted string for injection into Stage 3 prompt."""
        events = self.get_upcoming_events(days_ahead=14)
        lines = []

        # Add live macro data if available
        snap = self.get_latest_snapshot()
        if snap:
            lines.append(f"MACRO ENVIRONMENT (as of {snap.get('date', 'today')}):")
            if snap.get('yield_10y'):
                lines.append(f"  - 10Y Treasury: {snap['yield_10y']:.2f}%")
            if snap.get('spread_2y10y') is not None:
                lines.append(f"  - Yield curve (10Y-3M spread): {snap['spread_2y10y']:+.2f}% → {snap.get('regime_label', '')}")
            if snap.get('vix'):
                lines.append(f"  - VIX: {snap['vix']:.1f}")
            if snap.get('dxy'):
                lines.append(f"  - DXY (USD Index): {snap['dxy']:.1f}")
            lines.append("")

        if events:
            lines.append("UPCOMING CENTRAL BANK EVENTS:")
            for e in events:
                lines.append(f"  - {e['type']} rate decision: {e['date']} (in {e['days_until']} days)")
            lines.append("Consider rate sensitivity in your analysis. Tighten confidence for rate-sensitive sectors if event is within 48h.")

        return "\n".join(lines) if lines else ""


macro_tracker = MacroTracker()
