"""
Fear & Greed Tracker
- CNN Fear & Greed Index (historical from 2011)
- VIX rolling averages (10/20/30-day) as model features
- Per-stock Fear & Greed sensitivity factor (60-day rolling correlation)

The sensitivity factor captures how each stock reacts to fear/greed shifts:
  positive value → correlated with greed (risk-on, e.g. Tesla)
  negative value → anti-correlated (defensive, e.g. Dollarama/Walmart)
"""
import logging
import requests
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from core.database import db

logger = logging.getLogger(__name__)

# CNN Fear & Greed historical data endpoint
_CNN_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
# Alternate: last N days
_CNN_FG_HISTORY_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date}"

_FG_CACHE_HOURS = 6  # refresh every 6 hours


class FearGreedTracker:
    """
    Provides CNN Fear & Greed Index and VIX rolling features for ML models.
    """

    def __init__(self):
        self._fg_cache: Optional[Dict] = None
        self._fg_cache_time: Optional[datetime] = None
        self._vix_cache: Optional[pd.DataFrame] = None
        self._vix_cache_date: Optional[date] = None

    # ── CNN Fear & Greed ───────────────────────────────────────────────────────

    def fetch_fear_greed_history(self) -> pd.DataFrame:
        """
        Fetch full historical Fear & Greed index from CNN (data from ~2011).
        Returns DataFrame with columns: date (datetime), fg_value (0-100).
        Caches for 6h.
        """
        if (self._fg_cache is not None and self._fg_cache_time is not None and
                (datetime.now() - self._fg_cache_time).total_seconds() < _FG_CACHE_HOURS * 3600):
            return self._fg_cache

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; InvestmentBot/1.0)",
                "Accept": "application/json",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
            }
            resp = requests.get(_CNN_FG_URL, headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            # Extract historical data points
            fg_data = payload.get("fear_and_greed_historical", {}).get("data", [])
            if not fg_data:
                # Try alternate key
                fg_data = payload.get("fear_and_greed", {}).get("data", [])

            if not fg_data:
                logger.warning("CNN Fear & Greed: empty data response")
                return self._get_fallback_fg_df()

            rows = []
            for point in fg_data:
                try:
                    # CNN returns timestamps in milliseconds
                    ts = point.get("x") or point.get("timestamp")
                    val = point.get("y") or point.get("value") or point.get("score")
                    if ts is not None and val is not None:
                        if isinstance(ts, (int, float)) and ts > 1e10:
                            dt = pd.to_datetime(ts, unit="ms")
                        else:
                            dt = pd.to_datetime(ts)
                        rows.append({"date": dt.normalize(), "fg_value": float(val)})
                except Exception:
                    continue

            if not rows:
                return self._get_fallback_fg_df()

            df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
            self._fg_cache = df
            self._fg_cache_time = datetime.now()
            logger.info(f"Fear & Greed: loaded {len(df)} historical data points")
            return df

        except Exception as e:
            logger.warning(f"CNN Fear & Greed fetch failed: {e}")
            return self._get_fallback_fg_df()

    def _get_fallback_fg_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with correct schema on fetch failure."""
        return pd.DataFrame(columns=["date", "fg_value"])

    def get_current_fear_greed(self) -> Optional[float]:
        """Return the most recent Fear & Greed value (0=Extreme Fear, 100=Extreme Greed)."""
        df = self.fetch_fear_greed_history()
        if df.empty:
            return None
        return float(df["fg_value"].iloc[-1])

    def get_fg_label(self, value: float) -> str:
        """Convert numeric F&G value to label."""
        if value <= 20:   return "Extreme Fear"
        if value <= 40:   return "Fear"
        if value <= 60:   return "Neutral"
        if value <= 80:   return "Greed"
        return "Extreme Greed"

    # ── VIX Rolling Averages ───────────────────────────────────────────────────

    def get_vix_history(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Fetch VIX history from yfinance and compute 10/20/30-day rolling averages.
        Returns DataFrame: date, vix, vix_ma10, vix_ma20, vix_ma30.
        """
        today = date.today()
        if (self._vix_cache is not None and self._vix_cache_date == today):
            return self._vix_cache

        try:
            fetch_start = (datetime.now() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
            ticker = yf.Ticker("^VIX")
            hist = ticker.history(start=fetch_start, end=today.isoformat())
            if hist.empty:
                return pd.DataFrame(columns=["date", "vix", "vix_ma10", "vix_ma20", "vix_ma30"])

            df = hist[["Close"]].copy()
            df.index = pd.to_datetime(df.index).normalize()
            df = df.rename(columns={"Close": "vix"})
            df["vix_ma10"] = df["vix"].rolling(10).mean()
            df["vix_ma20"] = df["vix"].rolling(20).mean()
            df["vix_ma30"] = df["vix"].rolling(30).mean()
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            df = df.sort_values("date").tail(lookback_days).reset_index(drop=True)

            self._vix_cache = df
            self._vix_cache_date = today
            return df

        except Exception as e:
            logger.error(f"VIX history fetch failed: {e}")
            return pd.DataFrame(columns=["date", "vix", "vix_ma10", "vix_ma20", "vix_ma30"])

    def get_latest_vix_features(self) -> Dict:
        """Return today's VIX + rolling averages as a flat dict for ML features."""
        df = self.get_vix_history(lookback_days=60)
        if df.empty:
            return {"vix": None, "vix_ma10": None, "vix_ma20": None, "vix_ma30": None}
        row = df.iloc[-1]
        return {
            "vix": _safe_float(row.get("vix")),
            "vix_ma10": _safe_float(row.get("vix_ma10")),
            "vix_ma20": _safe_float(row.get("vix_ma20")),
            "vix_ma30": _safe_float(row.get("vix_ma30")),
        }

    # ── Fear & Greed Sensitivity Factor ───────────────────────────────────────

    def get_fg_sensitivity_factor(self, ticker: str, lookback_days: int = 60) -> Optional[float]:
        """
        Calculate the 60-day rolling correlation between a stock's price and the
        CNN Fear & Greed index.

        Positive value: stock is correlated with greed (risk-on, drops when fear rises).
        Negative value: stock is anti-correlated (defensive, rises when fear rises).

        Returns correlation coefficient in [-1, 1], or None if insufficient data.
        """
        try:
            fg_df = self.fetch_fear_greed_history()
            if fg_df.empty:
                return None

            fetch_start = (datetime.now() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=fetch_start)
            if hist.empty:
                return None

            price_df = hist[["Close"]].copy()
            price_df.index = pd.to_datetime(price_df.index).normalize()
            price_df = price_df.rename(columns={"Close": "price"}).reset_index()
            price_df = price_df.rename(columns={"index": "date", "Date": "date"})

            # Align by date
            fg_df["date"] = pd.to_datetime(fg_df["date"]).dt.normalize()
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

            merged = pd.merge(price_df, fg_df, on="date", how="inner")
            if len(merged) < 20:
                return None

            # Correlation over the full lookback window
            corr = merged["price"].corr(merged["fg_value"])
            if np.isnan(corr):
                return None
            return round(float(corr), 4)

        except Exception as e:
            logger.warning(f"F&G sensitivity factor failed for {ticker}: {e}")
            return None

    def get_features_for_ticker(self, ticker: str, lookback_days: int = 60) -> Dict:
        """
        Return all Fear & Greed + VIX features for a ticker as a flat dict.
        Used as input features for the LSTM model.
        """
        fg_current = self.get_current_fear_greed()
        vix_features = self.get_latest_vix_features()
        fg_sensitivity = self.get_fg_sensitivity_factor(ticker, lookback_days)

        return {
            "fg_value": fg_current,
            "fg_label": self.get_fg_label(fg_current) if fg_current is not None else None,
            "fg_sensitivity_60d": fg_sensitivity,
            **vix_features,
        }

    def get_historical_features(self, ticker: str, start_date: str = None,
                                 end_date: str = None) -> pd.DataFrame:
        """
        Build a time-series feature DataFrame for ML training:
        date, fg_value, fg_sensitivity_60d, vix, vix_ma10, vix_ma20, vix_ma30.
        """
        fg_df = self.fetch_fear_greed_history()
        vix_df = self.get_vix_history(lookback_days=730)  # ~2 years

        if fg_df.empty:
            return pd.DataFrame()

        # Merge F&G and VIX on date
        fg_df["date"] = pd.to_datetime(fg_df["date"]).dt.normalize()
        if not vix_df.empty:
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.normalize()
            merged = pd.merge(fg_df, vix_df, on="date", how="left")
        else:
            merged = fg_df.copy()

        # Add rolling sensitivity factor (computed over past 60 days at each point)
        try:
            fetch_start = merged["date"].min().strftime("%Y-%m-%d")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=fetch_start, end=date.today().isoformat())
            if not hist.empty:
                price_df = hist[["Close"]].copy()
                price_df.index = pd.to_datetime(price_df.index).normalize()
                price_df = price_df.rename(columns={"Close": "price"}).reset_index()
                price_df = price_df.rename(columns={"index": "date", "Date": "date"})
                price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

                base = pd.merge(merged, price_df, on="date", how="left")
                # Rolling 60-day correlation
                base["fg_sensitivity_60d"] = base["price"].rolling(60).corr(base["fg_value"])
                merged = base.drop(columns=["price"], errors="ignore")
        except Exception as e:
            logger.warning(f"Historical sensitivity calc failed for {ticker}: {e}")
            merged["fg_sensitivity_60d"] = None

        if start_date:
            merged = merged[merged["date"] >= pd.to_datetime(start_date)]
        if end_date:
            merged = merged[merged["date"] <= pd.to_datetime(end_date)]

        return merged.reset_index(drop=True)


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


fear_greed_tracker = FearGreedTracker()
