"""
Broker Client Abstraction — Phase 6 of Auto-Trading Integration
Provides a uniform interface over paper, Alpaca, and IBKR execution.
"""
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import requests

from core.database import db

logger = logging.getLogger(__name__)


class BrokerClient(ABC):
    """Abstract base class for all broker integrations."""

    @abstractmethod
    def place_order(self, ticker: str, qty: float, side: str,
                    order_type: str = 'market') -> dict:
        """Place an order. Returns order dict with at least {order_id, status}."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""

    @abstractmethod
    def get_positions(self) -> list:
        """Return list of open positions as dicts."""

    @abstractmethod
    def get_account(self) -> dict:
        """Return account info: {equity, buying_power, cash, currency}."""


# ---------------------------------------------------------------------------
# Alpaca (pure requests — no alpaca-trade-api package required)
# ---------------------------------------------------------------------------

class AlpacaBrokerClient(BrokerClient):
    """Alpaca Markets REST API client using plain requests."""

    def __init__(self):
        self.api_key = db.get_setting("auto_trade_alpaca_api_key") or ""
        self.secret = db.get_setting("auto_trade_alpaca_secret") or ""
        self.base_url = (
            db.get_setting("auto_trade_alpaca_base_url")
            or "https://paper-api.alpaca.markets"
        ).rstrip("/")

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/v2{path}",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/v2{path}",
            headers=self._headers(),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> bool:
        resp = requests.delete(
            f"{self.base_url}/v2{path}",
            headers=self._headers(),
            timeout=10,
        )
        return resp.ok

    def place_order(self, ticker: str, qty: float, side: str,
                    order_type: str = 'market') -> dict:
        body = {
            "symbol": ticker,
            "qty": str(int(max(1, qty))),
            "side": side.lower(),       # "buy" or "sell"
            "type": order_type,
            "time_in_force": "day",
        }
        try:
            data = self._post("/orders", body)
            return {"order_id": data.get("id"), "status": data.get("status"), "raw": data}
        except Exception as e:
            logger.error(f"Alpaca place_order failed: {e}")
            return {"order_id": None, "status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        try:
            return self._delete(f"/orders/{order_id}")
        except Exception as e:
            logger.error(f"Alpaca cancel_order failed: {e}")
            return False

    def get_positions(self) -> list:
        try:
            positions = self._get("/positions")
            return [
                {
                    "ticker": p.get("symbol"),
                    "qty": float(p.get("qty", 0)),
                    "side": "LONG" if float(p.get("qty", 0)) > 0 else "SHORT",
                    "market_value": float(p.get("market_value", 0)),
                    "avg_entry_price": float(p.get("avg_entry_price", 0)),
                    "unrealized_pl": float(p.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(p.get("unrealized_plpc", 0)) * 100,
                    "origin": "broker-sync",
                }
                for p in (positions if isinstance(positions, list) else [])
            ]
        except Exception as e:
            logger.error(f"Alpaca get_positions failed: {e}")
            return []

    def get_account(self) -> dict:
        try:
            data = self._get("/account")
            return {
                "equity": float(data.get("equity", 0)),
                "buying_power": float(data.get("buying_power", 0)),
                "cash": float(data.get("cash", 0)),
                "currency": data.get("currency", "USD"),
                "broker": "alpaca",
            }
        except Exception as e:
            logger.error(f"Alpaca get_account failed: {e}")
            return {"equity": 0, "buying_power": 0, "cash": 0, "currency": "USD", "broker": "alpaca", "error": str(e)}


# ---------------------------------------------------------------------------
# IBKR via ib_insync (optional dependency — guarded import)
# ---------------------------------------------------------------------------

class IBKRBrokerClient(BrokerClient):
    """Interactive Brokers client via ib_insync."""

    def __init__(self):
        self.host = db.get_setting("auto_trade_ibkr_host") or "127.0.0.1"
        self.port = int(db.get_setting("auto_trade_ibkr_port") or 7497)
        self.client_id = int(db.get_setting("auto_trade_ibkr_client_id") or 1)
        self._ib = None

    def _connect(self):
        if self._ib and self._ib.isConnected():
            return self._ib
        try:
            import ib_insync
        except ImportError:
            raise RuntimeError(
                "ib_insync is not installed. Run: pip install ib_insync>=0.9.86"
            )
        ib = ib_insync.IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        self._ib = ib
        return ib

    def place_order(self, ticker: str, qty: float, side: str,
                    order_type: str = 'market') -> dict:
        try:
            import ib_insync
            ib = self._connect()
            contract = ib_insync.Stock(ticker, "SMART", "USD")
            action = "BUY" if side.upper() in ("BUY", "LONG") else "SELL"
            order = ib_insync.MarketOrder(action, int(max(1, qty)))
            trade = ib.placeOrder(contract, order)
            ib_insync.util.sleep(1)
            return {
                "order_id": str(trade.order.orderId),
                "status": trade.orderStatus.status,
                "raw": {},
            }
        except Exception as e:
            logger.error(f"IBKR place_order failed: {e}")
            return {"order_id": None, "status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        try:
            import ib_insync
            ib = self._connect()
            for trade in ib.trades():
                if str(trade.order.orderId) == str(order_id):
                    ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error(f"IBKR cancel_order failed: {e}")
            return False

    def get_positions(self) -> list:
        try:
            ib = self._connect()
            result = []
            for pos in ib.positions():
                result.append({
                    "ticker": pos.contract.symbol,
                    "qty": pos.position,
                    "side": "LONG" if pos.position > 0 else "SHORT",
                    "market_value": pos.position * pos.avgCost,
                    "avg_entry_price": pos.avgCost,
                    "unrealized_pl": 0,
                    "unrealized_plpc": 0,
                    "origin": "broker-sync",
                })
            return result
        except Exception as e:
            logger.error(f"IBKR get_positions failed: {e}")
            return []

    def get_account(self) -> dict:
        try:
            ib = self._connect()
            vals = {v.tag: v.value for v in ib.accountValues() if v.currency == "USD"}
            return {
                "equity": float(vals.get("NetLiquidation", 0) or 0),
                "buying_power": float(vals.get("BuyingPower", 0) or 0),
                "cash": float(vals.get("CashBalance", 0) or 0),
                "currency": "USD",
                "broker": "ibkr",
            }
        except Exception as e:
            logger.error(f"IBKR get_account failed: {e}")
            return {"equity": 0, "buying_power": 0, "cash": 0, "currency": "USD", "broker": "ibkr", "error": str(e)}


# ---------------------------------------------------------------------------
# Paper (delegates to auto_paper_trades table)
# ---------------------------------------------------------------------------

class PaperBrokerClient(BrokerClient):
    """Paper broker — reads/writes auto_paper_trades, no real execution."""

    def place_order(self, ticker: str, qty: float, side: str,
                    order_type: str = 'market') -> dict:
        # In paper mode, order is already written by OrderManager before calling place_order.
        # Return a synthetic filled order.
        order_id = f"paper-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        return {"order_id": order_id, "status": "filled", "broker": "paper"}

    def cancel_order(self, order_id: str) -> bool:
        return True  # Paper orders are always cancellable

    def get_positions(self) -> list:
        rows = db.query(
            "SELECT * FROM auto_paper_trades WHERE status = 'open' ORDER BY entry_date DESC"
        )
        result = []
        for r in (rows or []):
            result.append({
                "ticker": r.get("ticker"),
                "qty": 0,  # paper doesn't track shares
                "side": r.get("direction", "LONG"),
                "market_value": 0,
                "avg_entry_price": r.get("entry_price", 0),
                "unrealized_pl": 0,
                "unrealized_plpc": 0,
                "origin": r.get("origin", "auto"),
                "trade_id": r.get("id"),
            })
        return result

    def get_account(self) -> dict:
        try:
            from engine.paper_trading import paper_trader
            summary = paper_trader.get_portfolio_summary()
            equity = float(summary.get("total_value", 10000))
            cash = float(summary.get("cash", equity))
        except Exception:
            equity = 10000.0
            cash = 10000.0
        return {
            "equity": equity,
            "buying_power": cash,
            "cash": cash,
            "currency": "USD",
            "broker": "paper",
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_broker_client() -> BrokerClient:
    """Return the appropriate broker client based on current auto_trade_mode setting."""
    mode = (db.get_setting("auto_trade_mode") or "paper").lower()
    if mode == "alpaca":
        return AlpacaBrokerClient()
    if mode == "ibkr":
        return IBKRBrokerClient()
    return PaperBrokerClient()
