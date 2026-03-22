"""
Economic Moat Scorer (items #45 / #54)

Scores a stock's competitive durability using quantitative financial heuristics —
no LLM required, no API cost. Based on 4 measurable factors:

1. P/E Stability  — low 3-year standard deviation of P/E ratio (mean-reversion = durable earnings)
2. Gross Margin Consistency — high gross margin with low inter-year variance
3. Free Cash Flow Trend  — positive FCF with year-over-year growth
4. Dividend History  — consistent dividend payments signal mature, predictable business

Scores each factor 0–25, combines to 0–100 composite moat_score.

Data source: yfinance (financials, income_stmt, cash_flow, dividends).
Cache: 24 hours per ticker.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TTL = timedelta(hours=24)


def _get_cached(ticker: str) -> Optional[dict]:
    entry = _CACHE.get(ticker)
    if entry and datetime.now() - entry["ts"] < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(ticker: str, data: dict) -> None:
    _CACHE[ticker] = {"data": data, "ts": datetime.now()}


# ---------------------------------------------------------------------------
# Individual factor scorers (each returns 0–25)
# ---------------------------------------------------------------------------

def _score_pe_stability(ticker_obj: yf.Ticker) -> tuple[float, str]:
    """P/E stability over 3 years (lower stddev = more stable = higher score)."""
    try:
        hist = ticker_obj.history(period="3y")
        info = ticker_obj.info or {}
        trailing_eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
        if not trailing_eps or trailing_eps <= 0 or hist.empty:
            return 12.5, "No EPS data"  # neutral

        # Approximate annual P/E from year-end prices ÷ trailing EPS
        year_end_prices = hist["Close"].resample("YE").last().dropna()
        if len(year_end_prices) < 2:
            return 12.5, "Insufficient price history"

        pe_series = year_end_prices / trailing_eps
        pe_std = float(pe_series.std())

        # Normalise: std < 2 → 25pts, std > 15 → 0pts
        score = max(0.0, min(25.0, 25.0 - (pe_std - 2.0) * (25.0 / 13.0)))
        note = f"P/E 3yr std={pe_std:.1f}"
        return round(score, 1), note
    except Exception as e:
        return 12.5, f"Error: {e}"


def _score_margin_consistency(ticker_obj: yf.Ticker) -> tuple[float, str]:
    """Gross margin: high + consistent (low YoY variance)."""
    try:
        inc = ticker_obj.income_stmt
        if inc is None or inc.empty:
            return 12.5, "No income statement"

        # Gross profit / Revenue
        gross_profit_row = next((r for r in inc.index if "Gross" in str(r) and "Profit" in str(r)), None)
        revenue_row = next((r for r in inc.index if "Total Revenue" in str(r) or "Revenue" == str(r)), None)

        if gross_profit_row is None or revenue_row is None:
            return 12.5, "No margin data"

        gp = inc.loc[gross_profit_row].dropna()
        rev = inc.loc[revenue_row].dropna()
        margins = (gp / rev).dropna()

        if len(margins) < 2:
            return 12.5, "Insufficient margin history"

        avg_margin = float(margins.mean())
        margin_std = float(margins.std())

        # High avg margin (>40% = max) + low std (<5% = stable)
        margin_score = min(12.5, avg_margin * 31.25)   # 40% → 12.5pts
        stability_score = max(0.0, 12.5 - margin_std * 250)  # std 0% → 12.5pts, 5% → 0pts
        score = margin_score + stability_score
        note = f"Avg gross margin={avg_margin:.1%}, std={margin_std:.1%}"
        return round(score, 1), note
    except Exception as e:
        return 12.5, f"Error: {e}"


def _score_fcf_trend(ticker_obj: yf.Ticker) -> tuple[float, str]:
    """Positive and growing free cash flow."""
    try:
        cf = ticker_obj.cash_flow
        if cf is None or cf.empty:
            return 12.5, "No cash flow data"

        fcf_row = next((r for r in cf.index if "Free Cash Flow" in str(r)), None)
        if fcf_row is None:
            # Approximate: Operating CF - CapEx
            ocf_row = next((r for r in cf.index if "Operating" in str(r) and "Cash" in str(r)), None)
            capex_row = next((r for r in cf.index if "Capital Expenditure" in str(r) or "Capex" in str(r).title()), None)
            if ocf_row and capex_row:
                fcf_series = (cf.loc[ocf_row] + cf.loc[capex_row]).dropna()
            else:
                return 12.5, "No FCF data"
        else:
            fcf_series = cf.loc[fcf_row].dropna()

        if len(fcf_series) < 2:
            return 12.5, "Insufficient FCF history"

        fcf_vals = fcf_series.values[::-1]  # oldest → newest
        positive_count = sum(1 for v in fcf_vals if float(v) > 0)
        positive_ratio = positive_count / len(fcf_vals)

        # Trend: is latest > earliest?
        growing = float(fcf_vals[-1]) > float(fcf_vals[0])

        score = positive_ratio * 20.0 + (5.0 if growing else 0.0)
        note = f"FCF positive {positive_count}/{len(fcf_vals)} years, trend={'up' if growing else 'down'}"
        return round(min(25.0, score), 1), note
    except Exception as e:
        return 12.5, f"Error: {e}"


def _score_dividend_history(ticker_obj: yf.Ticker) -> tuple[float, str]:
    """Consistent dividend history signals mature, predictable cash generation."""
    try:
        divs = ticker_obj.dividends
        if divs is None or divs.empty:
            return 0.0, "No dividends paid"

        # Count distinct years with dividends in last 5 years
        cutoff = datetime.now() - timedelta(days=5 * 365)
        recent = divs[divs.index.tz_localize(None) >= cutoff] if divs.index.tzinfo is not None else divs[divs.index >= cutoff]
        years_with_divs = len(set(recent.index.year)) if not recent.empty else 0

        # 5/5 years → 25pts
        score = min(25.0, years_with_divs * 5.0)
        note = f"Dividends in {years_with_divs}/5 recent years"
        return round(score, 1), note
    except Exception as e:
        return 0.0, f"Error: {e}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def moat_score(ticker: str) -> dict:
    """
    Calculate the composite moat score (0–100) for a ticker.

    Returns:
        {
            "ticker": str,
            "moat_score": float,  # 0–100 composite
            "grade": str,         # "Strong" / "Moderate" / "Weak" / "None"
            "factors": {
                "pe_stability":        {"score": float, "note": str},
                "margin_consistency":  {"score": float, "note": str},
                "fcf_trend":           {"score": float, "note": str},
                "dividend_history":    {"score": float, "note": str},
            },
            "scored_at": str,
            "error": str or None,
        }
    """
    cached = _get_cached(ticker)
    if cached:
        return cached

    ticker_obj = yf.Ticker(ticker.upper())

    pe_score, pe_note = _score_pe_stability(ticker_obj)
    margin_score, margin_note = _score_margin_consistency(ticker_obj)
    fcf_score, fcf_note = _score_fcf_trend(ticker_obj)
    div_score, div_note = _score_dividend_history(ticker_obj)

    composite = pe_score + margin_score + fcf_score + div_score

    if composite >= 70:
        grade = "Strong"
    elif composite >= 45:
        grade = "Moderate"
    elif composite >= 20:
        grade = "Weak"
    else:
        grade = "None"

    result = {
        "ticker": ticker.upper(),
        "moat_score": round(composite, 1),
        "grade": grade,
        "factors": {
            "pe_stability":       {"score": pe_score,     "note": pe_note},
            "margin_consistency": {"score": margin_score, "note": margin_note},
            "fcf_trend":          {"score": fcf_score,    "note": fcf_note},
            "dividend_history":   {"score": div_score,    "note": div_note},
        },
        "scored_at": datetime.now().isoformat(),
        "error": None,
    }

    _set_cached(ticker, result)
    return result


def batch_moat_scores(tickers: list[str]) -> list[dict]:
    """Score multiple tickers, returning sorted list (highest moat first)."""
    results = []
    for t in tickers:
        try:
            results.append(moat_score(t))
        except Exception as e:
            results.append({"ticker": t, "moat_score": 0.0, "grade": "Unknown", "error": str(e)})
    return sorted(results, key=lambda x: x.get("moat_score", 0), reverse=True)
