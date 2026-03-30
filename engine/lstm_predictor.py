"""
LSTM Stock Predictor
Long Short-Term Memory neural network for stock return prediction.

Input features (per timestep):
  Fundamentals : pe_ratio, pb_ratio, eps_ttm, revenue_growth, roe, debt_equity
  Technical    : rsi_14, sma_cross (price/SMA50), bb_position, momentum_20d
  Sentiment    : vader_score, insider_buying (bool)
  VIX          : vix, vix_ma10, vix_ma20, vix_ma30
  Fear & Greed : fg_value, fg_sensitivity_60d
  Politician   : pol_total_trades, pol_buy_count, pol_sell_count,
                 pol_exchange_count, pol_options_count, pol_bond_other_count,
                 pol_unique_senators, pol_log_money_mid

Output: probability that next-period return will be "unusually high" (regression).
Buy signal fires when model confidence >= 50%.

Requires: torch (pip install torch --index-url https://download.pytorch.org/whl/cpu)
"""
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ── PyTorch imports (optional dependency) ─────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. LSTM predictor will run in stub mode. "
                   "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")

_MODEL_DIR = Path(__file__).parent.parent / "data" / "lstm_models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    # Fundamentals
    "pe_ratio", "pb_ratio", "eps_ttm", "revenue_growth", "roe", "debt_equity",
    # Technical
    "rsi_14", "sma_cross", "bb_position", "momentum_20d",
    # Sentiment
    "vader_score", "insider_buying",
    # VIX
    "vix", "vix_ma10", "vix_ma20", "vix_ma30",
    # Fear & Greed
    "fg_value", "fg_sensitivity_60d",
    # Politician (9 features)
    "pol_total_trades", "pol_buy_count", "pol_sell_count",
    "pol_exchange_count", "pol_options_count", "pol_bond_other_count",
    "pol_unique_senators", "pol_log_money_mid",
]

N_FEATURES = len(FEATURE_NAMES)
LOOKBACK = 30          # days of history fed into LSTM
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BUY_CONFIDENCE_THRESHOLD = 0.50  # model must be ≥50% confident to trigger buy


# ── Model Architecture ─────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        Two-layer LSTM → Linear → Sigmoid.
        Input : (batch, LOOKBACK, N_FEATURES)
        Output: (batch, 1) — probability of unusually high return
        """

        def __init__(self, n_features: int = N_FEATURES, hidden_size: int = HIDDEN_SIZE,
                     num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last = lstm_out[:, -1, :]   # take last timestep
            out = self.dropout(last)
            out = self.fc(out)
            return self.sigmoid(out).squeeze(-1)

    class StockDataset(Dataset):
        """Sliding-window dataset: (X: lookback × features, y: binary label)."""

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# ── Feature Builder ────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Orchestrates data prep, LSTM training, prediction, and trade logging.
    """

    def __init__(self):
        self.model: Optional["LSTMModel"] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._trade_log: List[Dict] = []

    # ── Feature Collection ─────────────────────────────────────────────────────

    def _fetch_price_features(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Build daily feature DataFrame for a single ticker over a date range."""
        import yfinance as yf
        from engine.fear_greed_tracker import fear_greed_tracker
        from engine.politician_tracker import politician_tracker

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)
        if hist.empty or len(hist) < LOOKBACK + 5:
            return pd.DataFrame()

        df = hist[["Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.rename(columns={"Close": "close"})

        # ── Technical ──────────────────────────────────────────────────
        # RSI-14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)

        # SMA cross: price / SMA50 − 1
        sma50 = df["close"].rolling(50).mean()
        df["sma_cross"] = df["close"] / sma50.replace(0, np.nan) - 1

        # Bollinger Band position: (price - lower) / (upper - lower)
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        lower = sma20 - 2 * std20
        upper = sma20 + 2 * std20
        df["bb_position"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)

        # 20-day momentum
        df["momentum_20d"] = df["close"].pct_change(20)

        # ── Fundamentals (static, filled forward) ─────────────────────
        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        df["pe_ratio"]        = _clip(info.get("trailingPE"), -50, 500)
        df["pb_ratio"]        = _clip(info.get("priceToBook"), -10, 100)
        df["eps_ttm"]         = _clip(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"), -100, 1000)
        df["revenue_growth"]  = _clip(info.get("revenueGrowth"), -1, 5)
        df["roe"]             = _clip(info.get("returnOnEquity"), -2, 5)
        df["debt_equity"]     = _clip(info.get("debtToEquity"), 0, 20)
        df["vader_score"]     = 0.0   # placeholder; update from sentiment engine if available
        df["insider_buying"]  = 0.0   # placeholder; update from insider_tracker if available

        # ── Fear & Greed + VIX ────────────────────────────────────────
        try:
            fg_hist = fear_greed_tracker.get_historical_features(ticker, start_date=start, end_date=end)
            if not fg_hist.empty:
                fg_hist["date"] = pd.to_datetime(fg_hist["date"]).dt.normalize()
                df_reset = df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
                df_reset["date"] = pd.to_datetime(df_reset.get("Date", df_reset.index)).dt.normalize()
                merged = df_reset.merge(
                    fg_hist[["date", "fg_value", "fg_sensitivity_60d",
                              "vix", "vix_ma10", "vix_ma20", "vix_ma30"]],
                    on="date", how="left"
                )
                for col in ["fg_value", "fg_sensitivity_60d", "vix", "vix_ma10", "vix_ma20", "vix_ma30"]:
                    df[col] = merged[col].values if col in merged.columns else 0.0
            else:
                for col in ["fg_value", "fg_sensitivity_60d", "vix", "vix_ma10", "vix_ma20", "vix_ma30"]:
                    df[col] = 0.0
        except Exception as e:
            logger.warning(f"F&G features failed for {ticker}: {e}")
            for col in ["fg_value", "fg_sensitivity_60d", "vix", "vix_ma10", "vix_ma20", "vix_ma30"]:
                df[col] = 0.0

        # ── Politician Features ───────────────────────────────────────
        try:
            pol_ts = politician_tracker.get_time_series_features(
                ticker, start_date=pd.to_datetime(start).date(), end_date=pd.to_datetime(end).date()
            )
            if not pol_ts.empty:
                pol_ts["date"] = pd.to_datetime(pol_ts["date"]).dt.normalize()
                df_idx = df.copy()
                df_idx.index = pd.to_datetime(df_idx.index).normalize()
                pol_map = pol_ts.set_index("date")
                for col in ["pol_total_trades", "pol_buy_count", "pol_sell_count",
                            "pol_exchange_count", "pol_options_count", "pol_bond_other_count",
                            "pol_unique_senators", "pol_log_money_mid"]:
                    df[col] = df_idx.index.map(lambda d: pol_map.at[d, col] if d in pol_map.index else 0.0)
            else:
                for col in ["pol_total_trades", "pol_buy_count", "pol_sell_count",
                            "pol_exchange_count", "pol_options_count", "pol_bond_other_count",
                            "pol_unique_senators", "pol_log_money_mid"]:
                    df[col] = 0.0
        except Exception as e:
            logger.warning(f"Politician features failed for {ticker}: {e}")
            for col in ["pol_total_trades", "pol_buy_count", "pol_sell_count",
                        "pol_exchange_count", "pol_options_count", "pol_bond_other_count",
                        "pol_unique_senators", "pol_log_money_mid"]:
                df[col] = 0.0

        # Fill NaN with 0
        df = df.ffill().fillna(0)
        return df

    def _build_windows(self, df: pd.DataFrame, target_col: str = "close",
                       high_return_pct: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding LOOKBACK-day windows and binary labels.
        Label = 1 if next-period return (20 trading days) >= high_return_pct.
        """
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        missing = set(FEATURE_NAMES) - set(feature_cols)
        if missing:
            for m in missing:
                df[m] = 0.0

        values = df[FEATURE_NAMES].values.astype(np.float32)
        prices = df[target_col].values

        X, y = [], []
        for i in range(LOOKBACK, len(values) - 20):
            window = values[i - LOOKBACK:i]
            fwd_return = (prices[i + 20] - prices[i]) / (prices[i] + 1e-8)
            label = 1.0 if fwd_return >= high_return_pct else 0.0
            X.append(window)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Z-score normalization. Fit on training data, transform on test data."""
        X_flat = X.reshape(-1, N_FEATURES)
        if fit:
            self.scaler_mean = X_flat.mean(axis=0)
            self.scaler_std = X_flat.std(axis=0) + 1e-8
        if self.scaler_mean is None:
            return X
        normed = (X_flat - self.scaler_mean) / self.scaler_std
        return normed.reshape(X.shape)

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, tickers: List[str], epochs: int = 20, batch_size: int = 64,
              lr: float = 1e-3, high_return_pct: float = 0.05,
              years_back: int = 3) -> Dict:
        """
        Train the LSTM on historical data for the given ticker universe.
        Saves model weights to data/lstm_models/.

        Returns training summary with loss history and final accuracy.
        """
        if not _TORCH_AVAILABLE:
            return {"error": "PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu"}

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=years_back * 365)).isoformat()

        logger.info(f"LSTM training: {len(tickers)} tickers, {years_back}yr data, {epochs} epochs")

        all_X, all_y = [], []
        for ticker in tickers:
            try:
                df = self._fetch_price_features(ticker, start, end)
                if df.empty:
                    continue
                X, y = self._build_windows(df, high_return_pct=high_return_pct)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    logger.debug(f"  {ticker}: {len(X)} windows")
            except Exception as e:
                logger.warning(f"  {ticker} skipped: {e}")

        if not all_X:
            return {"error": "No training data collected"}

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        # Shuffle
        idx = np.random.permutation(len(X_all))
        X_all, y_all = X_all[idx], y_all[idx]

        # Normalize
        X_all = self._normalize(X_all, fit=True)

        # Train/val split (80/20)
        split = int(0.8 * len(X_all))
        X_train, X_val = X_all[:split], X_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]

        device = torch.device("cpu")
        model = LSTMModel(n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Weight positive class to handle imbalance (typically 20-30% positive labels)
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = StockDataset(X_train, y_train)
        val_ds = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        history = []
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                # Use raw logits for BCEWithLogitsLoss (undo sigmoid in forward)
                lstm_out, _ = model.lstm(xb)
                last = lstm_out[:, -1, :]
                last = model.dropout(last)
                logits = model.fc(last).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = nn.BCELoss()(preds, yb)
                    val_loss += loss.item() * len(xb)
                    preds_bin = (preds >= BUY_CONFIDENCE_THRESHOLD).float()
                    correct += (preds_bin == yb).sum().item()
                    total += len(yb)
            val_loss /= max(len(val_ds), 1)
            val_acc = correct / max(total, 1)

            history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                             "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)})
            logger.info(f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), _MODEL_DIR / "best_model.pt")

        # Save final model and scaler
        torch.save(model.state_dict(), _MODEL_DIR / "latest_model.pt")
        np.save(_MODEL_DIR / "scaler_mean.npy", self.scaler_mean)
        np.save(_MODEL_DIR / "scaler_std.npy", self.scaler_std)
        self.model = model

        return {
            "status": "trained",
            "tickers": len(tickers),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "history": history,
        }

    def load_model(self) -> bool:
        """Load saved model weights. Returns True on success."""
        if not _TORCH_AVAILABLE:
            return False
        model_path = _MODEL_DIR / "best_model.pt"
        mean_path = _MODEL_DIR / "scaler_mean.npy"
        std_path = _MODEL_DIR / "scaler_std.npy"

        if not model_path.exists():
            logger.warning("No trained LSTM model found. Train first with /api/lstm/train")
            return False

        try:
            self.model = LSTMModel(n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                                   num_layers=NUM_LAYERS, dropout=DROPOUT)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            if mean_path.exists():
                self.scaler_mean = np.load(str(mean_path))
                self.scaler_std = np.load(str(std_path))
            logger.info("LSTM model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"LSTM model load failed: {e}")
            return False

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, ticker: str) -> Dict:
        """
        Predict probability of high return for a single ticker.
        Returns confidence (0-1), buy_signal (bool), and feature snapshot.
        """
        if not _TORCH_AVAILABLE:
            return {"error": "PyTorch not installed", "buy_signal": False, "confidence": None}

        if self.model is None:
            if not self.load_model():
                return {"error": "No trained model", "buy_signal": False, "confidence": None}

        try:
            end = date.today().isoformat()
            start = (date.today() - timedelta(days=LOOKBACK + 60)).isoformat()
            df = self._fetch_price_features(ticker, start, end)
            if df.empty or len(df) < LOOKBACK:
                return {"error": "Insufficient data", "buy_signal": False, "confidence": None}

            # Take last LOOKBACK rows
            feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
            for m in set(FEATURE_NAMES) - set(feat_cols):
                df[m] = 0.0
            window = df[FEATURE_NAMES].tail(LOOKBACK).values.astype(np.float32)

            if self.scaler_mean is not None:
                window = (window - self.scaler_mean) / self.scaler_std

            x = torch.FloatTensor(window).unsqueeze(0)  # (1, LOOKBACK, N_FEATURES)
            self.model.eval()
            with torch.no_grad():
                confidence = float(self.model(x).item())

            buy_signal = confidence >= BUY_CONFIDENCE_THRESHOLD

            return {
                "ticker": ticker,
                "confidence": round(confidence, 4),
                "buy_signal": buy_signal,
                "threshold": BUY_CONFIDENCE_THRESHOLD,
                "predicted_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"LSTM predict failed for {ticker}: {e}")
            return {"error": str(e), "buy_signal": False, "confidence": None}

    def get_buy_signals(self, tickers: List[str]) -> List[Dict]:
        """
        Batch prediction across all tickers.
        Returns only those with buy_signal=True, sorted by confidence desc.
        """
        results = []
        for ticker in tickers:
            result = self.predict(ticker)
            if result.get("buy_signal"):
                result["ticker"] = ticker
                results.append(result)
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results

    # ── Trade History Log ──────────────────────────────────────────────────────

    def log_trade(self, ticker: str, expected_return: float, hold_days: int,
                  confidence: float, actual_return: Optional[float] = None) -> Dict:
        """
        Record a trade entry. Actual return is filled in after hold_days.
        Persistent via SQLite.
        """
        entry = {
            "ticker": ticker,
            "entered_at": datetime.now().isoformat(),
            "expected_return_pct": round(expected_return * 100, 2),
            "hold_days": hold_days,
            "confidence": round(confidence, 4),
            "actual_return_pct": round(actual_return * 100, 2) if actual_return is not None else None,
            "verified": actual_return is not None,
        }
        try:
            db.execute("""
                INSERT INTO lstm_trade_log
                  (ticker, entered_at, expected_return_pct, hold_days, confidence,
                   actual_return_pct, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, entry["entered_at"], entry["expected_return_pct"],
                  hold_days, entry["confidence"], entry.get("actual_return_pct"), entry["verified"]))
        except Exception as e:
            logger.warning(f"Trade log DB write failed: {e} — keeping in-memory only")
            self._trade_log.append(entry)
        return entry

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Return recent LSTM trade log entries."""
        try:
            rows = db.query("""
                SELECT * FROM lstm_trade_log ORDER BY entered_at DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in rows]
        except Exception:
            return self._trade_log[-limit:]

    def get_performance_metrics(self) -> Dict:
        """
        Compute CAGR, max drawdown, completed trades count, and win rate
        from the trade history log.
        """
        trades = self.get_trade_history(limit=10000)
        completed = [t for t in trades if t.get("verified") and t.get("actual_return_pct") is not None]

        if not completed:
            return {
                "completed_trades": 0,
                "win_rate_pct": None,
                "avg_return_pct": None,
                "cagr_pct": None,
                "max_drawdown_pct": None,
            }

        returns = [t["actual_return_pct"] / 100.0 for t in completed]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
        avg_return = float(np.mean(returns)) * 100

        # Equity curve for CAGR and drawdown
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity = np.array(equity)

        # CAGR: annualized from first to last trade
        if len(completed) >= 2:
            try:
                first_dt = datetime.fromisoformat(completed[-1]["entered_at"])
                last_dt = datetime.fromisoformat(completed[0]["entered_at"])
                years = max((last_dt - first_dt).days / 365.25, 0.1)
                cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
                cagr_pct = round(cagr * 100, 2)
            except Exception:
                cagr_pct = None
        else:
            cagr_pct = None

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_drawdown_pct = round(float(np.min(drawdowns)) * 100, 2)

        return {
            "completed_trades": len(completed),
            "win_rate_pct": round(win_rate, 1),
            "avg_return_pct": round(avg_return, 2),
            "cagr_pct": cagr_pct,
            "max_drawdown_pct": max_drawdown_pct,
        }


def _clip(val, lo, hi):
    """Safe float clip with None-handling."""
    if val is None:
        return 0.0
    try:
        return float(np.clip(float(val), lo, hi))
    except (TypeError, ValueError):
        return 0.0


lstm_predictor = LSTMPredictor()
