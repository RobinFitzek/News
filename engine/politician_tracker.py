"""
Politician / Senate Trade Tracker
Fetches Senate financial disclosure data and builds per-stock, per-date features
for use in the LSTM stock predictor.

Data source: Senate Stock Watcher public dataset
  https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json

Features extracted per stock per date (9 total):
  1. total_trades        — total political trades that day
  2. buy_count           — number of buy transactions
  3. sell_count          — number of sell transactions
  4. exchange_count      — number of exchanges
  5. options_count       — options trades (vs regular stock)
  6. bond_other_count    — bonds or other securities
  7. unique_senators     — distinct senators filing for this stock that day
  8. money_range_mid     — midpoint of declared $ range
  9. log_money_mid       — log-scale of midpoint (prevents large values dominating)
"""
import logging
import requests
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from core.database import db

logger = logging.getLogger(__name__)

_SENATE_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
    "aggregate/all_transactions.json"
)
_CACHE_HOURS = 24  # refresh daily

# Approximate midpoints for Senate $ range labels (in dollars)
_AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000":     8000,
    "$15,001 - $50,000":    32500,
    "$50,001 - $100,000":   75000,
    "$100,001 - $250,000":  175000,
    "$250,001 - $500,000":  375000,
    "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000,
    "$5,000,001 - $25,000,000": 15000000,
    "$25,000,001 - $50,000,000": 37500000,
    "$50,000,001 +":        75000000,
}


class PoliticianTracker:
    """
    Processes Senate financial disclosures and produces ticker-level trade features
    for ML model input.
    """

    def __init__(self):
        self._raw_cache: Optional[List[Dict]] = None
        self._cache_time: Optional[datetime] = None
        self._by_ticker: Optional[Dict[str, pd.DataFrame]] = None

    # ── Fetch & Cache ──────────────────────────────────────────────────────────

    def fetch_senate_trades(self) -> List[Dict]:
        """
        Fetch all Senate stock transactions from the public dataset.
        Returns list of raw transaction dicts.
        """
        if (self._raw_cache is not None and self._cache_time is not None and
                (datetime.now() - self._cache_time).total_seconds() < _CACHE_HOURS * 3600):
            return self._raw_cache

        try:
            resp = requests.get(_SENATE_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Senate trades: loaded {len(data)} records")
            self._raw_cache = data
            self._cache_time = datetime.now()
            self._by_ticker = None  # invalidate derived cache
            return data
        except Exception as e:
            logger.error(f"Senate trades fetch failed: {e}")
            return self._raw_cache or []

    # ── Feature Extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_amount_midpoint(amount_str: str) -> float:
        """Convert a Senate amount range string to a numeric midpoint."""
        if not amount_str:
            return 0.0
        # Direct lookup
        for key, mid in _AMOUNT_MIDPOINTS.items():
            if key.lower() in amount_str.lower() or amount_str.strip() in key:
                return float(mid)
        # Fallback: extract first number
        import re
        nums = re.findall(r"[\d,]+", amount_str.replace("$", ""))
        if nums:
            return float(nums[0].replace(",", ""))
        return 0.0

    @staticmethod
    def _log_money(amount: float) -> float:
        """Log-scale transformation to prevent large dollar amounts dominating model weights."""
        if amount <= 0:
            return 0.0
        return math.log1p(amount)  # log(1 + x) — handles 0 gracefully

    def _build_by_ticker(self) -> Dict[str, pd.DataFrame]:
        """Reorganize raw Senate data into a dict: ticker → DataFrame of trades."""
        if self._by_ticker is not None:
            return self._by_ticker

        trades = self.fetch_senate_trades()
        if not trades:
            self._by_ticker = {}
            return {}

        rows = []
        for t in trades:
            ticker = t.get("ticker", "").strip().upper()
            if not ticker or ticker in ("--", "N/A", ""):
                continue

            try:
                trade_date_str = t.get("transaction_date") or t.get("disclosure_date") or ""
                trade_date = None
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        trade_date = datetime.strptime(trade_date_str.strip(), fmt).date()
                        break
                    except ValueError:
                        continue
                if trade_date is None:
                    continue

                tx_type = (t.get("type") or t.get("transaction_type") or "").lower()
                asset_type = (t.get("asset_type") or "").lower()
                amount_str = t.get("amount") or ""
                senator = (t.get("senator") or t.get("first_name", "") + " " + t.get("last_name", "")).strip()

                amount_mid = PoliticianTracker._parse_amount_midpoint(amount_str)

                rows.append({
                    "ticker": ticker,
                    "date": trade_date,
                    "tx_type": tx_type,
                    "asset_type": asset_type,
                    "amount_mid": amount_mid,
                    "senator": senator,
                    "is_buy": int("purchase" in tx_type or "buy" in tx_type),
                    "is_sell": int("sale" in tx_type or "sell" in tx_type),
                    "is_exchange": int("exchange" in tx_type or "partial" in tx_type),
                    "is_option": int("option" in asset_type or "call" in asset_type or "put" in asset_type),
                    "is_bond_other": int("bond" in asset_type or "other" in asset_type or "fund" in asset_type),
                })
            except Exception:
                continue

        if not rows:
            self._by_ticker = {}
            return {}

        df_all = pd.DataFrame(rows)
        by_ticker = {}
        for ticker, grp in df_all.groupby("ticker"):
            by_ticker[ticker] = grp.reset_index(drop=True)

        self._by_ticker = by_ticker
        logger.info(f"Senate trades indexed: {len(by_ticker)} unique tickers")
        return by_ticker

    def get_features_for_date(self, ticker: str, trade_date: date) -> Dict:
        """
        Return the 9 political trade features for a specific ticker on a specific date.
        Uses the exact disclosure date (Senate Stock Act requires 45-day filing window).
        """
        by_ticker = self._build_by_ticker()
        empty = {
            "pol_total_trades": 0,
            "pol_buy_count": 0,
            "pol_sell_count": 0,
            "pol_exchange_count": 0,
            "pol_options_count": 0,
            "pol_bond_other_count": 0,
            "pol_unique_senators": 0,
            "pol_money_range_mid": 0.0,
            "pol_log_money_mid": 0.0,
        }

        ticker = ticker.upper()
        if ticker not in by_ticker:
            return empty

        df = by_ticker[ticker]
        day_df = df[df["date"] == trade_date]
        if day_df.empty:
            return empty

        total_mid = day_df["amount_mid"].sum()
        return {
            "pol_total_trades":    int(len(day_df)),
            "pol_buy_count":       int(day_df["is_buy"].sum()),
            "pol_sell_count":      int(day_df["is_sell"].sum()),
            "pol_exchange_count":  int(day_df["is_exchange"].sum()),
            "pol_options_count":   int(day_df["is_option"].sum()),
            "pol_bond_other_count": int(day_df["is_bond_other"].sum()),
            "pol_unique_senators": int(day_df["senator"].nunique()),
            "pol_money_range_mid": round(float(total_mid), 2),
            "pol_log_money_mid":   round(self._log_money(total_mid), 4),
        }

    def get_time_series_features(self, ticker: str,
                                  start_date: Optional[date] = None,
                                  end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Build a complete time-series of political trade features for a ticker.
        Fills zero for dates with no trades.
        Returns DataFrame with date column + 9 feature columns.
        """
        by_ticker = self._build_by_ticker()
        ticker = ticker.upper()

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=730)

        date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # business days
        rows = []

        if ticker in by_ticker:
            df = by_ticker[ticker].copy()
            df["date"] = pd.to_datetime(df["date"])
            df_grouped = df.groupby("date")

            for dt in date_range:
                d = dt.date()
                grp = df[df["date"].dt.date == d]
                if grp.empty:
                    rows.append({"date": d, "pol_total_trades": 0, "pol_buy_count": 0,
                                 "pol_sell_count": 0, "pol_exchange_count": 0,
                                 "pol_options_count": 0, "pol_bond_other_count": 0,
                                 "pol_unique_senators": 0, "pol_money_range_mid": 0.0,
                                 "pol_log_money_mid": 0.0})
                else:
                    total_mid = grp["amount_mid"].sum()
                    rows.append({
                        "date": d,
                        "pol_total_trades":    int(len(grp)),
                        "pol_buy_count":       int(grp["is_buy"].sum()),
                        "pol_sell_count":      int(grp["is_sell"].sum()),
                        "pol_exchange_count":  int(grp["is_exchange"].sum()),
                        "pol_options_count":   int(grp["is_option"].sum()),
                        "pol_bond_other_count": int(grp["is_bond_other"].sum()),
                        "pol_unique_senators": int(grp["senator"].nunique()),
                        "pol_money_range_mid": round(float(total_mid), 2),
                        "pol_log_money_mid":   round(self._log_money(total_mid), 4),
                    })
        else:
            # No trades for this ticker — all zeros
            for dt in date_range:
                rows.append({"date": dt.date(), "pol_total_trades": 0, "pol_buy_count": 0,
                             "pol_sell_count": 0, "pol_exchange_count": 0,
                             "pol_options_count": 0, "pol_bond_other_count": 0,
                             "pol_unique_senators": 0, "pol_money_range_mid": 0.0,
                             "pol_log_money_mid": 0.0})

        return pd.DataFrame(rows)

    def get_recent_trades(self, ticker: str = None, days: int = 30) -> List[Dict]:
        """Return raw recent trades, optionally filtered by ticker."""
        by_ticker = self._build_by_ticker()
        cutoff = date.today() - timedelta(days=days)
        result = []

        tickers_to_check = [ticker.upper()] if ticker else list(by_ticker.keys())
        for t in tickers_to_check:
            if t not in by_ticker:
                continue
            df = by_ticker[t]
            recent = df[df["date"] >= cutoff]
            for _, row in recent.iterrows():
                result.append({
                    "ticker": t,
                    "date": str(row["date"]),
                    "tx_type": row["tx_type"],
                    "asset_type": row["asset_type"],
                    "senator": row["senator"],
                    "amount_mid": row["amount_mid"],
                    "is_buy": bool(row["is_buy"]),
                    "is_sell": bool(row["is_sell"]),
                })

        result.sort(key=lambda x: x["date"], reverse=True)
        return result[:500]

    def get_top_traded_tickers(self, days: int = 90, top_n: int = 20) -> List[Dict]:
        """Return most actively traded tickers by politicians recently."""
        by_ticker = self._build_by_ticker()
        cutoff = date.today() - timedelta(days=days)
        counts = []

        for ticker, df in by_ticker.items():
            recent = df[df["date"] >= cutoff]
            if not recent.empty:
                counts.append({
                    "ticker": ticker,
                    "total_trades": len(recent),
                    "buy_count": int(recent["is_buy"].sum()),
                    "sell_count": int(recent["is_sell"].sum()),
                    "unique_senators": int(recent["senator"].nunique()),
                    "total_volume_mid": round(float(recent["amount_mid"].sum()), 0),
                })

        counts.sort(key=lambda x: x["total_trades"], reverse=True)
        return counts[:top_n]


politician_tracker = PoliticianTracker()
