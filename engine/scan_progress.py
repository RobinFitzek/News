"""
Thread-safe scan progress tracker for real-time UI feedback.
"""
import threading
from datetime import datetime


class ScanProgress:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = self._default_state()

    def _default_state(self):
        return {
            "active": False,
            "stage": None,
            "stage_label": "",
            "started_at": None,
            "progress_pct": 0,
            "current_ticker": None,
            "tickers_total": 0,
            "tickers_done": 0,
            "message": "",
            "error": None,
        }

    def start(self):
        with self._lock:
            self._state = {
                "active": True,
                "stage": "discovery",
                "stage_label": "Auto-Discovery",
                "started_at": datetime.now().isoformat(),
                "progress_pct": 5,
                "current_ticker": None,
                "tickers_total": 0,
                "tickers_done": 0,
                "message": "Running auto-discovery...",
                "error": None,
            }

    def set_stage1(self, total_tickers):
        with self._lock:
            self._state.update({
                "stage": "stage1",
                "stage_label": "Quant Screening",
                "progress_pct": 15,
                "tickers_total": total_tickers,
                "tickers_done": 0,
                "message": f"Screening {total_tickers} tickers...",
            })

    def complete_stage1(self, candidates_count):
        with self._lock:
            self._state.update({
                "progress_pct": 30,
                "message": f"Quant screen done -- {candidates_count} candidates",
            })

    def set_stage2(self, total, provider_label=None):
        stage_label = "News & Market Intel"
        if provider_label:
            stage_label = f"News & Market Intel · {provider_label}"
        with self._lock:
            self._state.update({
                "stage": "stage2",
                "stage_label": stage_label,
                "progress_pct": 35,
                "tickers_total": total,
                "tickers_done": 0,
                "message": f"Deep analysis on {total} candidates...",
            })

    def update_stage2(self, ticker, done, total, provider_label=None):
        with self._lock:
            base, span = 35, 30
            pct = base + int(span * done / max(total, 1))
            provider_suffix = f" via {provider_label}" if provider_label else ""
            self._state.update({
                "current_ticker": ticker,
                "tickers_done": done,
                "progress_pct": pct,
                "message": f"Analyzing {ticker} ({done}/{total}){provider_suffix}",
            })

    def set_stage3(self, total, provider_label=None):
        stage_label = "Final Synthesis"
        if provider_label:
            stage_label = f"Final Synthesis · {provider_label}"
        with self._lock:
            self._state.update({
                "stage": "stage3",
                "stage_label": stage_label,
                "progress_pct": 70,
                "tickers_total": total,
                "tickers_done": 0,
                "message": f"Synthesizing {total} finalists...",
            })

    def update_stage3(self, ticker, done, total, provider_label=None):
        with self._lock:
            base, span = 70, 25
            pct = base + int(span * done / max(total, 1))
            provider_suffix = f" via {provider_label}" if provider_label else ""
            self._state.update({
                "current_ticker": ticker,
                "tickers_done": done,
                "progress_pct": pct,
                "message": f"Synthesizing {ticker} ({done}/{total}){provider_suffix}",
            })

    def complete(self, summary=""):
        with self._lock:
            self._state.update({
                "active": False,
                "stage": "complete",
                "stage_label": "Complete",
                "progress_pct": 100,
                "current_ticker": None,
                "message": summary or "Scan complete",
            })

    def fail(self, error):
        with self._lock:
            self._state.update({
                "active": False,
                "stage": "error",
                "stage_label": "Error",
                "error": str(error),
                "message": f"Scan failed: {error}",
            })

    def get_state(self):
        with self._lock:
            return dict(self._state)


scan_progress = ScanProgress()
