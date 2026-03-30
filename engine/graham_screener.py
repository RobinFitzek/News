"""
Graham Intrinsic Value Screener
Implements Benjamin Graham's formula: V = EPS * (8.5 + 2g) * 4.4 / Y
where g = estimated growth rate, Y = AAA corporate bond yield.

Buy condition: market_price <= intrinsic_value * (1 - discount_factor)
discount_factor: 0.0 = buy at or below IV, 0.2 = buy at 20% margin of safety, etc.
"""
import logging
import requests
import yfinance as yf
import numpy as np
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from core.database import db

logger = logging.getLogger(__name__)

# Fallback AAA yield if FRED is unavailable
_FALLBACK_AAA_YIELD = 4.8  # percent, reasonable 2026 estimate

# FRED CSV endpoint for Moody's AAA Corporate Bond Yield (no API key required)
_FRED_AAA_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DAAA"


class GrahamScreener:
    """
    Value investing screen using Graham's intrinsic value formula.
    Screens hundreds of stocks and identifies those trading below IV.
    """

    def __init__(self):
        self._aaa_cache: Optional[Tuple[float, datetime]] = None  # (yield, fetched_at)

    # ── AAA Bond Yield ─────────────────────────────────────────────────────────

    def fetch_aaa_yield(self) -> float:
        """
        Fetch current Moody's AAA corporate bond yield from FRED.
        Returns value in percent (e.g. 4.75 means 4.75%).
        Caches for 24h; falls back to hardcoded estimate on failure.
        """
        # Return cached if fresh (< 24h)
        if self._aaa_cache:
            val, fetched = self._aaa_cache
            if (datetime.now() - fetched).total_seconds() < 86400:
                return val

        try:
            resp = requests.get(_FRED_AAA_URL, timeout=10)
            resp.raise_for_status()
            lines = resp.text.strip().splitlines()
            # Format: DATE,VALUE  — last line is most recent
            for line in reversed(lines):
                if line.startswith("DATE"):
                    continue
                parts = line.split(",")
                if len(parts) == 2 and parts[1] != ".":
                    val = float(parts[1])
                    self._aaa_cache = (val, datetime.now())
                    logger.info(f"AAA bond yield fetched from FRED: {val}%")
                    return val
        except Exception as e:
            logger.warning(f"FRED AAA fetch failed: {e} — using fallback {_FALLBACK_AAA_YIELD}%")

        self._aaa_cache = (_FALLBACK_AAA_YIELD, datetime.now())
        return _FALLBACK_AAA_YIELD

    # ── EPS & Growth ───────────────────────────────────────────────────────────

    def get_trailing_eps_and_growth(self, ticker: str) -> Dict:
        """
        Returns trailing 12-month EPS and estimated growth rate from EPS history.
        Growth rate = CAGR over available quarterly EPS data.
        """
        try:
            stock = yf.Ticker(ticker)
            inc = stock.quarterly_income_stmt
            info = stock.info or {}

            shares = info.get("sharesOutstanding")
            book_value_per_share = info.get("bookValue")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

            ttm_eps = None
            growth_rate = None
            eps_history = []

            if inc is not None and not (hasattr(inc, "empty") and inc.empty):
                cols = list(inc.columns)[:8]  # up to 8 quarters, newest first
                for col in cols:
                    try:
                        ni = inc.get(col, {}).get("Net Income")
                        if ni is not None and shares and shares > 0:
                            eps_q = float(ni) / float(shares)
                            eps_history.append(eps_q)
                    except Exception:
                        continue

            # Trailing 12m EPS = sum of last 4 quarterly EPS
            if len(eps_history) >= 4:
                ttm_eps = sum(eps_history[:4])
            elif info.get("trailingEps"):
                ttm_eps = float(info["trailingEps"])
            elif info.get("epsTrailingTwelveMonths"):
                ttm_eps = float(info["epsTrailingTwelveMonths"])

            # Growth rate from EPS history (CAGR from oldest to most recent quarterly)
            if len(eps_history) >= 4:
                # Use annualized growth: compare first 4 vs last 4 quarters when available
                recent_ttm = sum(eps_history[:4])
                if len(eps_history) >= 8:
                    older_ttm = sum(eps_history[4:8])
                    years = 1.0  # 4 quarters earlier = ~1 year
                else:
                    older_ttm = eps_history[-1] * 4  # approximate
                    years = (len(eps_history) - 1) / 4.0

                if older_ttm and older_ttm > 0 and recent_ttm and years > 0:
                    growth_rate = (recent_ttm / older_ttm) ** (1 / years) - 1
                    # Cap growth rate: Graham suggests 0-20% is reasonable
                    growth_rate = max(-0.5, min(growth_rate, 0.20))

            # Fallback to analyst estimate
            if growth_rate is None:
                growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.07
                growth_rate = float(growth_rate)
                growth_rate = max(-0.5, min(growth_rate, 0.20))

            return {
                "ttm_eps": round(ttm_eps, 4) if ttm_eps is not None else None,
                "growth_rate": round(growth_rate * 100, 2),  # as percent
                "growth_rate_decimal": round(growth_rate, 4),
                "book_value_per_share": round(float(book_value_per_share), 2) if book_value_per_share else None,
                "current_price": round(float(current_price), 2) if current_price else None,
                "eps_history_quarters": len(eps_history),
            }

        except Exception as e:
            logger.error(f"EPS/growth fetch failed for {ticker}: {e}")
            return {"ttm_eps": None, "growth_rate": None, "growth_rate_decimal": None,
                    "book_value_per_share": None, "current_price": None, "eps_history_quarters": 0}

    # ── Graham Formula ─────────────────────────────────────────────────────────

    def calculate_intrinsic_value(self, eps: float, growth_rate_pct: float,
                                   aaa_yield: float) -> Optional[float]:
        """
        Graham intrinsic value: V = EPS * (8.5 + 2g) * 4.4 / Y
        eps: trailing 12-month EPS
        growth_rate_pct: expected 7–10yr growth rate in percent (e.g. 7.5 for 7.5%)
        aaa_yield: current AAA bond yield in percent (e.g. 4.8 for 4.8%)
        Returns intrinsic value per share, or None if inputs are invalid.
        Note: Can return negative values for companies with negative EPS.
        """
        if eps is None or aaa_yield is None or aaa_yield <= 0:
            return None
        try:
            iv = eps * (8.5 + 2 * growth_rate_pct) * (4.4 / aaa_yield)
            return round(iv, 2)
        except Exception:
            return None

    def screen_ticker(self, ticker: str, discount_factor: float = 0.0,
                      aaa_yield: Optional[float] = None) -> Dict:
        """
        Screen a single ticker for Graham value.
        discount_factor: 0.0 = buy at IV, 0.2 = buy at 20% below IV (margin of safety).
        Returns dict with buy_signal, intrinsic_value, current_price, upside_pct.
        """
        if aaa_yield is None:
            aaa_yield = self.fetch_aaa_yield()

        fundamentals = self.get_trailing_eps_and_growth(ticker)
        eps = fundamentals["ttm_eps"]
        growth = fundamentals["growth_rate"]  # in percent
        current_price = fundamentals["current_price"]

        if eps is None or growth is None:
            return {
                "ticker": ticker, "buy_signal": False, "reason": "Insufficient EPS data",
                "intrinsic_value": None, "current_price": current_price,
                "upside_pct": None, "aaa_yield": aaa_yield,
                **fundamentals,
            }

        iv = self.calculate_intrinsic_value(eps, growth, aaa_yield)

        if iv is None:
            return {
                "ticker": ticker, "buy_signal": False, "reason": "Could not calculate IV",
                "intrinsic_value": None, "current_price": current_price,
                "upside_pct": None, "aaa_yield": aaa_yield,
                **fundamentals,
            }

        buy_threshold = iv * (1 - discount_factor)
        buy_signal = False
        reason = ""

        if iv <= 0:
            reason = "Negative intrinsic value (negative EPS)"
        elif current_price is None:
            reason = "No current price available"
        elif current_price <= buy_threshold:
            buy_signal = True
            reason = f"Price {current_price:.2f} ≤ IV threshold {buy_threshold:.2f} ({discount_factor*100:.0f}% margin)"
        else:
            reason = f"Price {current_price:.2f} > IV threshold {buy_threshold:.2f}"

        upside_pct = None
        if current_price and iv and current_price > 0:
            upside_pct = round((iv - current_price) / current_price * 100, 1)

        return {
            "ticker": ticker,
            "buy_signal": buy_signal,
            "reason": reason,
            "intrinsic_value": iv,
            "buy_threshold": round(buy_threshold, 2),
            "current_price": current_price,
            "upside_pct": upside_pct,
            "aaa_yield": aaa_yield,
            "discount_factor": discount_factor,
            **fundamentals,
        }

    def screen_watchlist(self, tickers: List[str], discount_factor: float = 0.2,
                          max_positions: int = 50) -> Dict:
        """
        Screen multiple tickers and return ranked buy candidates.
        Only includes tickers with calculable intrinsic value (for fair benchmarking).
        """
        aaa_yield = self.fetch_aaa_yield()
        results = []
        iv_calculable = []

        for ticker in tickers:
            try:
                result = self.screen_ticker(ticker, discount_factor, aaa_yield)
                results.append(result)
                if result["intrinsic_value"] is not None and result["intrinsic_value"] > 0:
                    iv_calculable.append(result)
            except Exception as e:
                logger.warning(f"Graham screen failed for {ticker}: {e}")

        # Rank by upside (highest first)
        buy_candidates = sorted(
            [r for r in iv_calculable if r["buy_signal"]],
            key=lambda x: x["upside_pct"] or 0, reverse=True
        )[:max_positions]

        return {
            "screened_at": datetime.now().isoformat(),
            "aaa_yield": aaa_yield,
            "discount_factor": discount_factor,
            "total_screened": len(tickers),
            "iv_calculable": len(iv_calculable),
            "buy_candidates": len(buy_candidates),
            "max_positions": max_positions,
            "results": results,
            "buy_list": buy_candidates,
        }

    # ── Backtest ───────────────────────────────────────────────────────────────

    def backtest(self, tickers: List[str], discount_factor: float = 0.2,
                 max_positions: int = 50, holding_days: int = 252) -> Dict:
        """
        Simple Graham backtest: buy when price <= IV*(1-df), hold for holding_days,
        measure actual forward return. Uses yfinance historical prices.
        Benchmark: equal-weight of ONLY tickers with calculable intrinsic value.
        """
        import pandas as pd
        aaa_yield = self.fetch_aaa_yield()
        logger.info(f"Graham backtest: {len(tickers)} tickers, df={discount_factor}, max={max_positions}")

        trade_results = []
        benchmark_tickers = []
        end = date.today()
        start = end - timedelta(days=holding_days + 90)

        for ticker in tickers:
            try:
                result = self.screen_ticker(ticker, discount_factor, aaa_yield)
                if result["intrinsic_value"] is None or result["intrinsic_value"] <= 0:
                    continue  # skip: can't calculate IV, exclude from benchmark too

                benchmark_tickers.append(ticker)

                if not result["buy_signal"] or not result["current_price"]:
                    continue

                # Fetch historical prices for forward return
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start.isoformat(), end=end.isoformat())
                if hist.empty or len(hist) < holding_days // 5:
                    continue

                entry_price = float(hist["Close"].iloc[-holding_days // 5 - 1]) if len(hist) > holding_days // 5 else float(hist["Close"].iloc[0])
                exit_price = float(hist["Close"].iloc[-1])
                forward_return = round((exit_price - entry_price) / entry_price * 100, 2)

                trade_results.append({
                    "ticker": ticker,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "forward_return_pct": forward_return,
                    "intrinsic_value": result["intrinsic_value"],
                    "upside_pct": result["upside_pct"],
                    "growth_rate": result["growth_rate"],
                })

            except Exception as e:
                logger.debug(f"Graham backtest skip {ticker}: {e}")

        # Fair benchmark: equal-weight all IV-calculable tickers
        benchmark_return = None
        if benchmark_tickers:
            bench_returns = []
            for ticker in benchmark_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start.isoformat(), end=end.isoformat())
                    if not hist.empty and len(hist) > holding_days // 5:
                        ep = float(hist["Close"].iloc[-holding_days // 5 - 1])
                        xp = float(hist["Close"].iloc[-1])
                        bench_returns.append((xp - ep) / ep * 100)
                except Exception:
                    pass
            if bench_returns:
                benchmark_return = round(float(np.mean(bench_returns)), 2)

        if trade_results:
            returns = [t["forward_return_pct"] for t in trade_results]
            avg_return = round(float(np.mean(returns)), 2)
            win_rate = round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
        else:
            avg_return = None
            win_rate = None

        return {
            "backtest_date": datetime.now().isoformat(),
            "discount_factor": discount_factor,
            "max_positions": max_positions,
            "holding_days": holding_days,
            "aaa_yield": aaa_yield,
            "trades": len(trade_results),
            "iv_calculable_tickers": len(benchmark_tickers),
            "avg_forward_return_pct": avg_return,
            "win_rate_pct": win_rate,
            "benchmark_return_pct": benchmark_return,
            "alpha_vs_benchmark": round(avg_return - benchmark_return, 2) if avg_return is not None and benchmark_return is not None else None,
            "trade_results": trade_results[:max_positions],
        }


graham_screener = GrahamScreener()
