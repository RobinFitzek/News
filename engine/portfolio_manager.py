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
        
        # NEW: Add cash position to total value
        cash_positions = db.query("SELECT SUM(amount) as total FROM cash_positions WHERE amount > 0")
        cash_total = cash_positions[0]['total'] if cash_positions and cash_positions[0]['total'] else 0
        
        stock_value = sum(h.get('current_value', 0) for h in enriched)
        total_value = stock_value + cash_total

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
        
        # NEW: Check correlation risk
        try:
            from engine.correlation_analyzer import correlation_analyzer
            correlation_alerts = correlation_analyzer.generate_correlation_alerts(enriched, threshold=0.75, combined_limit=15.0)
            alerts.extend(correlation_alerts)
            diversification_score = correlation_analyzer.get_diversification_score(enriched)
        except Exception as e:
            logger.warning(f"Could not check correlations: {e}")
            diversification_score = None

        benchmark = self._track_benchmark(enriched)
        rebalancing = self._suggest_rebalancing(enriched, total_value)
        risk_gate = self.get_risk_gate_status()

        return {
            'alerts': alerts,
            'benchmark': benchmark,
            'rebalancing': rebalancing,
            'portfolio_value': round(total_value, 2),
            'stock_value': round(stock_value, 2),
            'cash_value': round(cash_total, 2),
            'diversification_score': diversification_score,
            'risk_gate': risk_gate,
            'holdings': enriched,
        }

    def get_risk_gate_status(self) -> Dict:
        """Global risk gate based on current portfolio loss and cooldown period."""
        enabled = bool(db.get_setting('portfolio_risk_guard_enabled'))
        threshold = float(db.get_setting('portfolio_global_loss_limit_pct') or 10.0)
        cooldown_hours = float(db.get_setting('portfolio_risk_cooldown_hours') or 24)
        triggered_at_str = db.get_setting('portfolio_risk_guard_triggered_at')

        if not enabled:
            return {
                'active': False,
                'enabled': False,
                'loss_pct': 0.0,
                'threshold_pct': threshold,
                'cooldown_hours': cooldown_hours,
                'triggered_at': triggered_at_str,
                'reason': 'Risk guard disabled'
            }

        holdings = [h for h in db.get_portfolio_holdings() if h.get('shares', 0) > 0]
        if not holdings:
            return {
                'active': False,
                'enabled': True,
                'loss_pct': 0.0,
                'threshold_pct': threshold,
                'cooldown_hours': cooldown_hours,
                'triggered_at': triggered_at_str,
                'reason': 'No active holdings'
            }

        enriched = self._enrich_with_prices(holdings)
        total_invested = sum(h.get('total_invested', 0) for h in enriched if h.get('shares', 0) > 0)
        current_value = sum(h.get('current_value', 0) for h in enriched)
        loss_pct = 0.0
        if total_invested > 0:
            loss_pct = ((current_value - total_invested) / total_invested) * 100

        now = datetime.now()
        active = False
        reason = 'Risk guard inactive'

        if loss_pct <= -abs(threshold):
            active = True
            reason = f'Global loss limit hit ({loss_pct:.1f}% <= -{threshold:.1f}%)'
            if not triggered_at_str:
                db.set_setting('portfolio_risk_guard_triggered_at', now.isoformat())
                triggered_at_str = now.isoformat()
        elif triggered_at_str:
            try:
                triggered_at = datetime.fromisoformat(triggered_at_str)
                if now < triggered_at + timedelta(hours=cooldown_hours):
                    active = True
                    reason = 'Risk guard cooldown active'
                else:
                    db.set_setting('portfolio_risk_guard_triggered_at', None)
                    triggered_at_str = None
            except Exception:
                db.set_setting('portfolio_risk_guard_triggered_at', None)
                triggered_at_str = None

        return {
            'active': active,
            'enabled': True,
            'loss_pct': round(loss_pct, 2),
            'threshold_pct': threshold,
            'cooldown_hours': cooldown_hours,
            'triggered_at': triggered_at_str,
            'reason': reason,
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
            override = db.get_ticker_risk_override(h['ticker'])
            effective_threshold = threshold
            if override and override.get('max_position_pct') is not None:
                effective_threshold = float(override['max_position_pct'])

            pct = (h.get('current_value', 0) / total_value * 100) if total_value > 0 else 0
            h['position_pct'] = round(pct, 1)
            h['max_position_pct'] = round(effective_threshold, 2)

            if pct > effective_threshold:
                alerts.append({
                    'type': 'POSITION_SIZE',
                    'ticker': h['ticker'],
                    'current_pct': round(pct, 1),
                    'threshold': effective_threshold,
                    'severity': 'CRITICAL' if pct > effective_threshold * 1.5 else 'WARNING',
                    'message': f"{h['ticker']} is {pct:.1f}% of portfolio (limit: {effective_threshold}%)"
                })
        return alerts

    def _check_stop_loss(self, holdings: List[Dict], threshold: float) -> List[Dict]:
        """Alert if any position is down more than threshold % from entry."""
        alerts = []
        
        # NEW: Import dividend tracker for ex-date awareness
        try:
            from engine.dividend_tracker import dividend_tracker
        except Exception:
            dividend_tracker = None
        
        for h in holdings:
            pnl = h.get('pnl_pct', 0)
            ticker = h['ticker']

            override = db.get_ticker_risk_override(ticker)
            base_threshold = threshold
            if override and override.get('stop_loss_pct') is not None:
                base_threshold = float(override['stop_loss_pct'])
            
            # NEW: Adjust threshold for upcoming dividends
            adjusted_threshold = base_threshold
            if dividend_tracker:
                adjusted_threshold = dividend_tracker.adjust_stop_loss_for_dividend(ticker, base_threshold)
                
                # Check if this is a false alarm due to ex-dividend
                if pnl < -base_threshold and dividend_tracker.check_stop_loss_false_alarm(ticker, pnl):
                    # Skip alert - it's just ex-dividend drop
                    logger.info(f"Skipping stop-loss alert for {ticker} — likely ex-dividend adjustment")
                    continue
            
            if pnl < -adjusted_threshold:
                alerts.append({
                    'type': 'STOP_LOSS',
                    'ticker': ticker,
                    'entry_price': h.get('avg_price', 0),
                    'current_price': h.get('current_price', 0),
                    'loss_pct': pnl,
                    'threshold': adjusted_threshold,
                    'severity': 'CRITICAL' if pnl < -adjusted_threshold * 1.5 else 'WARNING',
                    'message': f"{ticker} is down {pnl:.1f}% from entry (stop-loss: -{adjusted_threshold:.1f}%)"
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

    def get_rebalancing_plan(self) -> List[Dict]:
        """
        Generate a concrete rebalancing execution plan with specific share counts.
        Returns ordered list of actions: sell X shares of TICKER / add $Y to TICKER.
        """
        active_holdings = db.get_active_holdings()
        if not active_holdings:
            return []

        # Enrich holdings with current prices
        enriched = []
        total_value = 0.0
        for h in active_holdings:
            price_data = self._get_current_price(h['ticker'])
            if price_data and price_data['price'] and h.get('shares'):
                current_value = price_data['price'] * h['shares']
                enriched.append({
                    **h,
                    'current_price': price_data['price'],
                    'current_value': current_value,
                    'sector': price_data.get('sector', h.get('sector', 'Unknown')),
                    'category': self._sector_to_category(price_data.get('sector', 'Unknown')),
                })
                total_value += current_value

        if total_value == 0 or not enriched:
            return []

        drift_threshold = float(db.get_setting('portfolio_rebalance_drift_pct') or 5.0)

        try:
            from core.config import STRATEGY_PRESETS
            variant = db.get_setting('analysis_variant') or 'balanced'
            target_mix = STRATEGY_PRESETS.get(variant, {}).get('asset_mix', {})
        except Exception:
            target_mix = {}

        if not target_mix:
            return []

        # Category-level drift
        category_values = {}
        category_holdings = {}
        for h in enriched:
            cat = h['category']
            category_values[cat] = category_values.get(cat, 0) + h['current_value']
            category_holdings.setdefault(cat, []).append(h)

        actions = []
        for category, target_pct in target_mix.items():
            current_value = category_values.get(category, 0)
            current_pct = (current_value / total_value * 100) if total_value > 0 else 0
            diff = current_pct - target_pct

            if abs(diff) <= drift_threshold:
                continue

            target_value = total_value * target_pct / 100
            delta_dollars = abs(current_value - target_value)

            if diff > 0:
                # Overweight — reduce: sell proportionally from largest positions in category
                holdings_in_cat = sorted(
                    category_holdings.get(category, []),
                    key=lambda x: x['current_value'], reverse=True
                )
                remaining = delta_dollars
                for h in holdings_in_cat:
                    if remaining <= 0:
                        break
                    sell_value = min(remaining, h['current_value'] * (diff / current_pct) if current_pct else 0)
                    sell_value = max(sell_value, 0)
                    if sell_value < 10:
                        continue
                    shares_to_sell = int(sell_value / h['current_price']) if h['current_price'] else 0
                    if shares_to_sell < 1:
                        continue
                    actual_value = shares_to_sell * h['current_price']
                    actions.append({
                        'action': 'sell',
                        'ticker': h['ticker'],
                        'shares': shares_to_sell,
                        'price': round(h['current_price'], 2),
                        'value': round(actual_value, 2),
                        'reason': f"{category} overweight {diff:+.1f}% (target {target_pct}%)",
                    })
                    remaining -= actual_value
            else:
                # Underweight — add capital
                actions.append({
                    'action': 'buy',
                    'ticker': None,
                    'category': category,
                    'shares': None,
                    'price': None,
                    'value': round(delta_dollars, 2),
                    'reason': f"{category} underweight {diff:+.1f}% (target {target_pct}%)",
                })

        return actions

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
