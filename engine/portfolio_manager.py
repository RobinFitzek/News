"""
Portfolio Management Rules
Rule-based portfolio monitoring — position sizing, stop-losses, sector concentration,
benchmark tracking, and rebalancing suggestions. No AI opinions.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Rule-based portfolio monitoring."""

    def __init__(self):
        self._price_cache = {}
        self._sector_cache = {}
        self._cache_duration = timedelta(minutes=15)

    def check_all_rules(self) -> Dict:
        """Run all portfolio rules, return alerts and status."""
        holdings = db.get_portfolio_holdings()
        active_holdings = [h for h in holdings if h['shares'] > 0]

        if not active_holdings:
            return {
                'alerts': [],
                'benchmark': None,
                'rebalancing': [],
                'portfolio_value': 0,
                'message': 'No active holdings'
            }

        # Enrich with current prices
        enriched = self._enrich_with_prices(active_holdings)
        total_value = sum(h.get('current_value', 0) for h in enriched)

        if total_value == 0:
            return {
                'alerts': [],
                'benchmark': None,
                'rebalancing': [],
                'portfolio_value': 0,
                'message': 'Could not determine portfolio value'
            }

        # Load configurable thresholds
        max_position_pct = float(db.get_setting('portfolio_max_position_pct') or 10.0)
        stop_loss_pct = float(db.get_setting('portfolio_stop_loss_pct') or 15.0)
        max_sector_pct = float(db.get_setting('portfolio_max_sector_pct') or 30.0)

        # Run all checks
        alerts = []
        alerts.extend(self._check_position_size(enriched, total_value, max_position_pct))
        alerts.extend(self._check_stop_loss(enriched, stop_loss_pct))
        alerts.extend(self._check_sector_concentration(enriched, total_value, max_sector_pct))

        benchmark = self._track_benchmark(enriched)
        rebalancing = self._suggest_rebalancing(enriched, total_value)

        return {
            'alerts': alerts,
            'benchmark': benchmark,
            'rebalancing': rebalancing,
            'portfolio_value': round(total_value, 2),
            'holdings': enriched,
        }

    def _enrich_with_prices(self, holdings: List[Dict]) -> List[Dict]:
        """Add current price, value, sector, and P&L to each holding."""
        for h in holdings:
            ticker = h['ticker']
            price_data = self._get_current_price(ticker)

            if price_data:
                h['current_price'] = price_data['price']
                h['current_value'] = h['shares'] * price_data['price']
                h['sector'] = price_data.get('sector', 'Unknown')

                if h['avg_price'] > 0:
                    h['pnl_pct'] = round(((price_data['price'] - h['avg_price']) / h['avg_price']) * 100, 2)
                    h['pnl_value'] = round((price_data['price'] - h['avg_price']) * h['shares'], 2)
                else:
                    h['pnl_pct'] = 0.0
                    h['pnl_value'] = 0.0
            else:
                h['current_price'] = h.get('avg_price', 0)
                h['current_value'] = h['shares'] * h.get('avg_price', 0)
                h['sector'] = 'Unknown'
                h['pnl_pct'] = 0.0
                h['pnl_value'] = 0.0

        return holdings

    def _check_position_size(self, holdings: List[Dict], total_value: float,
                              threshold: float) -> List[Dict]:
        """Alert if any position exceeds threshold % of portfolio."""
        alerts = []
        for h in holdings:
            pct = (h.get('current_value', 0) / total_value * 100) if total_value > 0 else 0
            h['position_pct'] = round(pct, 1)

            if pct > threshold:
                alerts.append({
                    'type': 'POSITION_SIZE',
                    'ticker': h['ticker'],
                    'current_pct': round(pct, 1),
                    'threshold': threshold,
                    'severity': 'CRITICAL' if pct > threshold * 1.5 else 'WARNING',
                    'message': f"{h['ticker']} is {pct:.1f}% of portfolio (limit: {threshold}%)"
                })
        return alerts

    def _check_stop_loss(self, holdings: List[Dict], threshold: float) -> List[Dict]:
        """Alert if any position is down more than threshold % from entry."""
        alerts = []
        for h in holdings:
            pnl = h.get('pnl_pct', 0)
            if pnl < -threshold:
                alerts.append({
                    'type': 'STOP_LOSS',
                    'ticker': h['ticker'],
                    'entry_price': h.get('avg_price', 0),
                    'current_price': h.get('current_price', 0),
                    'loss_pct': pnl,
                    'threshold': threshold,
                    'severity': 'CRITICAL' if pnl < -threshold * 1.5 else 'WARNING',
                    'message': f"{h['ticker']} is down {pnl:.1f}% from entry (stop-loss: -{threshold}%)"
                })
        return alerts

    def _check_sector_concentration(self, holdings: List[Dict], total_value: float,
                                      threshold: float) -> List[Dict]:
        """Alert if any sector exceeds threshold % of portfolio."""
        sector_values = {}
        for h in holdings:
            sector = h.get('sector', 'Unknown')
            sector_values[sector] = sector_values.get(sector, 0) + h.get('current_value', 0)

        alerts = []
        for sector, value in sector_values.items():
            pct = (value / total_value * 100) if total_value > 0 else 0
            if pct > threshold and sector != 'Unknown':
                tickers = [h['ticker'] for h in holdings if h.get('sector') == sector]
                alerts.append({
                    'type': 'SECTOR_CONCENTRATION',
                    'sector': sector,
                    'current_pct': round(pct, 1),
                    'threshold': threshold,
                    'tickers': tickers,
                    'severity': 'WARNING',
                    'message': f"{sector} sector is {pct:.1f}% of portfolio (limit: {threshold}%) — tickers: {', '.join(tickers)}"
                })
        return alerts

    def _track_benchmark(self, holdings: List[Dict]) -> Optional[Dict]:
        """Compare portfolio return vs SPY over the holding period."""
        try:
            # Find the earliest trade date
            trades = db.get_trades()
            if not trades:
                return None

            buy_trades = [t for t in trades if t['type'] == 'BUY']
            if not buy_trades:
                return None

            # Parse dates
            dates = []
            for t in buy_trades:
                try:
                    dates.append(datetime.strptime(t['date'], '%Y-%m-%d'))
                except (ValueError, TypeError):
                    continue

            if not dates:
                return None

            earliest = min(dates)
            days = (datetime.now() - earliest).days
            if days < 7:
                return None

            # Calculate portfolio return
            total_invested = sum(h.get('total_invested', 0) for h in holdings if h['shares'] > 0)
            total_current = sum(h.get('current_value', 0) for h in holdings)

            if total_invested <= 0:
                return None

            portfolio_return = ((total_current - total_invested) / total_invested) * 100

            # Get SPY return over same period
            period = f"{min(days + 5, 365)}d"
            spy_hist = yf.Ticker('SPY').history(period=period)
            if spy_hist.empty or len(spy_hist) < 2:
                return None

            # Match to earliest trade date
            spy_start = float(spy_hist['Close'].iloc[0])
            spy_end = float(spy_hist['Close'].iloc[-1])
            spy_return = ((spy_end - spy_start) / spy_start) * 100

            alpha = portfolio_return - spy_return

            return {
                'portfolio_return': round(portfolio_return, 2),
                'spy_return': round(spy_return, 2),
                'alpha': round(alpha, 2),
                'period_days': days,
                'total_invested': round(total_invested, 2),
                'total_current': round(total_current, 2),
                'message': (f"Portfolio: {portfolio_return:+.1f}% vs SPY: {spy_return:+.1f}% "
                            f"(Alpha: {alpha:+.1f}%) over {days} days")
            }

        except Exception as e:
            logger.error(f"Error tracking benchmark: {e}")
            return None

    def _suggest_rebalancing(self, holdings: List[Dict], total_value: float) -> List[Dict]:
        """Suggest rebalancing when positions drift from target allocation."""
        drift_threshold = float(db.get_setting('portfolio_rebalance_drift_pct') or 5.0)

        # Get target allocation from active strategy
        try:
            from core.config import STRATEGY_PRESETS
            variant = db.get_setting('analysis_variant') or 'balanced'
            target_mix = STRATEGY_PRESETS.get(variant, {}).get('asset_mix', {})
        except Exception:
            target_mix = {}

        if not target_mix:
            return []

        # Calculate current allocation by sector grouping to asset category
        # Simple mapping: use sector -> category heuristic
        category_values = {}
        for h in holdings:
            category = self._sector_to_category(h.get('sector', 'Unknown'))
            category_values[category] = category_values.get(category, 0) + h.get('current_value', 0)

        suggestions = []
        for category, target_pct in target_mix.items():
            current_value = category_values.get(category, 0)
            current_pct = (current_value / total_value * 100) if total_value > 0 else 0
            diff = current_pct - target_pct

            if abs(diff) > drift_threshold:
                action = 'reduce' if diff > 0 else 'increase'
                suggestions.append({
                    'category': category,
                    'current_pct': round(current_pct, 1),
                    'target_pct': target_pct,
                    'drift': round(diff, 1),
                    'action': action,
                    'message': f"{action.capitalize()} {category}: {current_pct:.1f}% -> {target_pct}% (drift: {diff:+.1f}%)"
                })

        return suggestions

    @staticmethod
    def _sector_to_category(sector: str) -> str:
        """Map yfinance sector to asset category."""
        sector_lower = sector.lower()
        if sector_lower in ('unknown', ''):
            return 'growth'
        # ETFs don't have sectors typically
        if any(x in sector_lower for x in ['utilities', 'real estate', 'consumer defensive']):
            return 'blue_chip'
        if any(x in sector_lower for x in ['technology', 'communication']):
            return 'growth'
        if any(x in sector_lower for x in ['healthcare', 'financial', 'industrials', 'energy']):
            return 'blue_chip'
        if any(x in sector_lower for x in ['consumer cyclical', 'basic materials']):
            return 'growth'
        return 'growth'

    def _get_current_price(self, ticker: str) -> Optional[Dict]:
        """Get current price with caching."""
        if ticker in self._price_cache:
            entry = self._price_cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            info = yf.Ticker(ticker).info
            data = {
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'sector': info.get('sector', 'Unknown'),
            }
            if data['price'] and data['price'] > 0:
                self._price_cache[ticker] = {'data': data, 'timestamp': datetime.now()}
                return data
        except Exception as e:
            logger.warning(f"Could not fetch price for {ticker}: {e}")

        return None


# Singleton
portfolio_manager = PortfolioManager()
