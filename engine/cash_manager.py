"""
Cash Position Management
Tracks cash alongside stock positions for accurate portfolio calculations.
Position sizing alerts should consider total portfolio (stocks + cash), not just stock value.
"""
from datetime import datetime
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class CashManager:
    """Manage cash positions in portfolio."""

    def add_cash(self, amount: float, description: str = "Cash deposit", date: str = None):
        """Add cash to portfolio."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            db.execute("""
                INSERT INTO cash_positions (amount, description, date, timestamp)
                VALUES (?, ?, ?, ?)
            """, (amount, description, date, datetime.now().isoformat()))
            
            logger.info(f"Added ${amount:.2f} cash: {description}")
            return True
        except Exception as e:
            logger.error(f"Error adding cash: {e}")
            return False

    def withdraw_cash(self, amount: float, description: str = "Cash withdrawal", date: str = None):
        """Remove cash from portfolio (negative amount)."""
        return self.add_cash(-amount, description, date)

    def get_total_cash(self) -> float:
        """Get current total cash balance."""
        try:
            result = db.query_one("SELECT SUM(amount) as total FROM cash_positions")
            return float(result['total']) if result and result['total'] else 0.0
        except Exception as e:
            logger.error(f"Error getting total cash: {e}")
            return 0.0

    def get_cash_history(self, limit: int = 50) -> List[Dict]:
        """Get recent cash transactions."""
        try:
            transactions = db.query("""
                SELECT id, amount, description, date, timestamp
                FROM cash_positions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return transactions
        except Exception as e:
            logger.error(f"Error getting cash history: {e}")
            return []

    def get_portfolio_total(self) -> Dict:
        """Get total portfolio value (stocks + cash)."""
        try:
            # Get stock value
            holdings = db.get_portfolio_holdings()
            
            from engine.portfolio_manager import portfolio_manager
            enriched = portfolio_manager._enrich_with_prices(
                [h for h in holdings if h['shares'] > 0]
            )
            stock_value = sum(h.get('current_value', 0) for h in enriched)
            
            # Get cash value
            cash_value = self.get_total_cash()
            
            total = stock_value + cash_value
            
            return {
                'stock_value': round(stock_value, 2),
                'cash_value': round(cash_value, 2),
                'total_value': round(total, 2),
                'cash_percentage': round((cash_value / total * 100), 1) if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Error getting portfolio total: {e}")
            return {
                'stock_value': 0,
                'cash_value': 0,
                'total_value': 0,
                'cash_percentage': 0,
            }

    def auto_record_trade_cash_flow(self, ticker: str, shares: float, price: float, 
                                   trade_type: str, date: str):
        """Automatically record cash flow from trades."""
        try:
            if trade_type.upper() == 'BUY':
                # Buying reduces cash
                cash_change = -(shares * price)
                description = f"BUY {shares} shares of {ticker} @ ${price:.2f}"
            elif trade_type.upper() == 'SELL':
                # Selling increases cash
                cash_change = shares * price
                description = f"SELL {shares} shares of {ticker} @ ${price:.2f}"
            else:
                logger.warning(f"Unknown trade type: {trade_type}")
                return False

            return self.add_cash(cash_change, description, date)
            
        except Exception as e:
            logger.error(f"Error recording trade cash flow: {e}")
            return False

    def get_cash_allocation_recommendation(self) -> Dict:
        """
        Recommend cash allocation based on portfolio size and strategy.
        General rule: keep 5-20% in cash for opportunities.
        """
        try:
            portfolio = self.get_portfolio_total()
            cash_pct = portfolio['cash_percentage']
            
            if cash_pct < 5:
                recommendation = {
                    'status': 'CRITICAL',
                    'message': f"Only {cash_pct:.1f}% cash — consider selling positions to increase liquidity",
                    'target_pct': 10,
                    'amount_needed': portfolio['total_value'] * 0.10 - portfolio['cash_value'],
                }
            elif cash_pct < 10:
                recommendation = {
                    'status': 'WARNING',
                    'message': f"{cash_pct:.1f}% cash — slightly low for capturing opportunities",
                    'target_pct': 15,
                    'amount_needed': portfolio['total_value'] * 0.15 - portfolio['cash_value'],
                }
            elif cash_pct > 30:
                recommendation = {
                    'status': 'WARNING',
                    'message': f"{cash_pct:.1f}% cash — too conservative, consider deploying capital",
                    'target_pct': 20,
                    'amount_to_invest': portfolio['cash_value'] - (portfolio['total_value'] * 0.20),
                }
            else:
                recommendation = {
                    'status': 'GOOD',
                    'message': f"{cash_pct:.1f}% cash — healthy allocation",
                    'target_pct': 15,
                }

            return recommendation

        except Exception as e:
            logger.error(f"Error generating cash recommendation: {e}")
            return {'status': 'ERROR', 'message': 'Could not calculate recommendation'}


# Singleton
cash_manager = CashManager()
