"""
Portfolio vs SPY Benchmark
Compares actual portfolio performance against a hypothetical SPY-only strategy.
For each BUY trade, calculates what SPY would have returned if same $ invested at trade date.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class PortfolioBenchmark:
    """Calculates portfolio vs SPY benchmark comparison."""

    def __init__(self, cache_minutes: int = 30):
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=cache_minutes)

    def calculate_portfolio_vs_spy(self) -> Dict:
        """Compare portfolio performance against SPY benchmark.

        For each BUY trade: calculate what SPY would have returned if
        the same dollar amount had been invested in SPY on that date.
        """
        if self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache

        holdings = db.get_portfolio_holdings()
        trades = db.get_trades()

        if not trades:
            return {
                'portfolio_value': 0, 'portfolio_return_pct': 0,
                'spy_equivalent_value': 0, 'spy_return_pct': 0,
                'alpha': 0, 'unrealized_pnl': 0,
                'holdings': [], 'message': 'No trades recorded',
            }

        # Get current prices for all held tickers
        active_holdings = [h for h in holdings if h['shares'] > 0]
        current_prices = {}
        for h in active_holdings:
            try:
                ticker_obj = yf.Ticker(h['ticker'])
                info = ticker_obj.info
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                if price:
                    current_prices[h['ticker']] = float(price)
            except Exception as e:
                logger.warning(f"Could not fetch price for {h['ticker']}: {e}")

        # Calculate portfolio current value
        portfolio_value = 0.0
        total_invested = 0.0
        enriched_holdings = []

        for h in active_holdings:
            price = current_prices.get(h['ticker'])
            if price and h['shares'] > 0:
                market_value = h['shares'] * price
                cost_basis = h['total_cost']
                unrealized = market_value - cost_basis
                portfolio_value += market_value
                total_invested += cost_basis
                enriched_holdings.append({
                    'ticker': h['ticker'],
                    'shares': h['shares'],
                    'avg_price': round(h['avg_price'], 2),
                    'current_price': round(price, 2),
                    'market_value': round(market_value, 2),
                    'cost_basis': round(cost_basis, 2),
                    'unrealized_pnl': round(unrealized, 2),
                    'return_pct': round((unrealized / cost_basis) * 100, 2) if cost_basis > 0 else 0,
                })

        # SPY benchmark: for each BUY trade, what would SPY have returned?
        spy_equivalent_value = 0.0
        spy_total_invested = 0.0

        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='5y')
            spy_current = float(spy_hist['Close'].iloc[-1]) if not spy_hist.empty else None
        except Exception as e:
            logger.warning(f"Failed to fetch SPY history: {e}")
            spy_hist = None
            spy_current = None

        if spy_current and spy_hist is not None and not spy_hist.empty:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            for trade in buy_trades:
                try:
                    trade_date = trade['date'][:10]
                    trade_cost = (trade['amount'] * trade['price']) + trade['fees']

                    # Find SPY price on trade date
                    spy_prices_on_date = spy_hist.loc[spy_hist.index.strftime('%Y-%m-%d') == trade_date, 'Close']
                    if spy_prices_on_date.empty:
                        # Find nearest date
                        idx = spy_hist.index.get_indexer(
                            [datetime.strptime(trade_date, '%Y-%m-%d')],
                            method='nearest'
                        )
                        if idx[0] >= 0:
                            spy_price_then = float(spy_hist['Close'].iloc[idx[0]])
                        else:
                            continue
                    else:
                        spy_price_then = float(spy_prices_on_date.iloc[0])

                    if spy_price_then > 0:
                        spy_shares = trade_cost / spy_price_then
                        spy_equivalent_value += spy_shares * spy_current
                        spy_total_invested += trade_cost
                except Exception as e:
                    logger.debug(f"Skipping trade for SPY calc: {e}")
                    continue

        # Handle SELL trades for SPY comparison (reduce proportionally)
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        for trade in sell_trades:
            try:
                trade_proceeds = (trade['amount'] * trade['price']) - trade['fees']
                if spy_total_invested > 0:
                    sell_ratio = trade_proceeds / spy_total_invested
                    spy_equivalent_value -= spy_equivalent_value * min(sell_ratio, 1.0)
                    spy_total_invested -= trade_proceeds
            except Exception:
                continue

        portfolio_return_pct = ((portfolio_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
        spy_return_pct = ((spy_equivalent_value - spy_total_invested) / spy_total_invested * 100) if spy_total_invested > 0 else 0
        alpha = portfolio_return_pct - spy_return_pct

        result = {
            'portfolio_value': round(portfolio_value, 2),
            'total_invested': round(total_invested, 2),
            'portfolio_return_pct': round(portfolio_return_pct, 2),
            'spy_equivalent_value': round(spy_equivalent_value, 2),
            'spy_return_pct': round(spy_return_pct, 2),
            'alpha': round(alpha, 2),
            'unrealized_pnl': round(portfolio_value - total_invested, 2),
            'holdings': enriched_holdings,
            'updated_at': datetime.now().isoformat(),
        }

        self._cache = result
        self._cache_time = datetime.now()

        # Persist snapshot
        try:
            self._save_snapshot(result)
        except Exception as e:
            logger.warning(f"Failed to save benchmark snapshot: {e}")

        return result

    def _save_snapshot(self, data: Dict):
        """Save benchmark snapshot to DB for historical tracking."""
        today = datetime.now().strftime('%Y-%m-%d')
        db.execute(
            """INSERT OR REPLACE INTO portfolio_benchmarks
               (snapshot_date, portfolio_value, portfolio_return_pct,
                spy_equivalent_value, spy_return_pct, alpha, cash_invested)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (today, data['portfolio_value'], data['portfolio_return_pct'],
             data['spy_equivalent_value'], data['spy_return_pct'],
             data['alpha'], data['total_invested'])
        )

    def get_time_series(self, days: int = 90) -> List[Dict]:
        """Get historical benchmark snapshots for charting."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return db.query(
            "SELECT * FROM portfolio_benchmarks WHERE snapshot_date >= ? ORDER BY snapshot_date",
            (cutoff,)
        )

    def invalidate_cache(self):
        self._cache = None
        self._cache_time = None


# Singleton
portfolio_benchmark = PortfolioBenchmark()
