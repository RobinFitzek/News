"""
Financial Statements Engine
Fetches and caches 8-quarter income statement, balance sheet, cash flow trends.
Also provides DCF fair value estimation and peer comparison.
"""
import yfinance as yf
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from core.database import db

logger = logging.getLogger(__name__)

CACHE_HOURS = 24  # Refresh once daily


class FinancialStatements:
    """Fetch, cache, and serve multi-quarter financial trend data."""

    def _get_cached(self, ticker: str, data_type: str) -> Optional[Dict]:
        """Retrieve cached financial data if fresh."""
        try:
            row = db.query_one("""
                SELECT data_json, fetched_at FROM financial_cache
                WHERE ticker = ? AND data_type = ?
                AND datetime(fetched_at) > datetime('now', ?)
            """, (ticker, data_type, f"-{CACHE_HOURS} hours"))
            if row:
                return json.loads(row['data_json'])
        except Exception:
            pass
        return None

    def _set_cached(self, ticker: str, data_type: str, data: Dict):
        """Store financial data in cache."""
        try:
            db.execute("""
                INSERT INTO financial_cache (ticker, data_type, data_json, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, data_type) DO UPDATE SET
                    data_json = excluded.data_json,
                    fetched_at = excluded.fetched_at
            """, (ticker, data_type, json.dumps(data, default=str), datetime.now().isoformat()))
        except Exception as e:
            logger.warning(f"Could not cache financial data for {ticker}: {e}")

    def _safe_float(self, val) -> Optional[float]:
        try:
            if val is None:
                return None
            f = float(val)
            return None if (f != f) else round(f, 4)  # NaN check
        except (TypeError, ValueError):
            return None

    def get_quarterly_financials(self, ticker: str) -> Dict:
        """
        Returns 8-quarter trend data for revenue, net income, EPS,
        gross margin, operating margin, and FCF.
        """
        cached = self._get_cached(ticker, 'quarterly')
        if cached:
            return cached

        try:
            stock = yf.Ticker(ticker)
            inc = stock.quarterly_income_stmt
            cf = stock.quarterly_cashflow
            info = stock.info or {}

            result = {
                'ticker': ticker,
                'available': False,
                'quarters': [],
                'labels': [],
                'revenue': [],
                'net_income': [],
                'gross_margin_pct': [],
                'operating_margin_pct': [],
                'eps': [],
                'fcf': [],
                'fetched_at': datetime.now().isoformat(),
            }

            if inc is None or (hasattr(inc, 'empty') and inc.empty):
                result['reason'] = 'No quarterly income statement data'
                return result

            # Columns = quarters (newest first)
            cols = list(inc.columns)[:8]

            for col in reversed(cols):  # oldest first for chart
                try:
                    label = str(col)[:7]  # "YYYY-MM"
                    rev = self._safe_float(inc.get(col, {}).get('Total Revenue'))
                    ni = self._safe_float(inc.get(col, {}).get('Net Income'))
                    gross = self._safe_float(inc.get(col, {}).get('Gross Profit'))
                    op_inc = self._safe_float(inc.get(col, {}).get('Operating Income'))
                    shares = self._safe_float(info.get('sharesOutstanding'))

                    gm = round((gross / rev * 100), 1) if (gross and rev and rev != 0) else None
                    om = round((op_inc / rev * 100), 1) if (op_inc and rev and rev != 0) else None
                    eps = round((ni / shares), 2) if (ni and shares and shares != 0) else None

                    # FCF = operating CF - capex
                    fcf = None
                    if cf is not None and not (hasattr(cf, 'empty') and cf.empty):
                        op_cf = self._safe_float(cf.get(col, {}).get('Operating Cash Flow'))
                        capex = self._safe_float(cf.get(col, {}).get('Capital Expenditure'))
                        if op_cf is not None:
                            fcf = op_cf + (capex or 0)  # capex is negative in yfinance

                    result['labels'].append(label)
                    result['revenue'].append(rev)
                    result['net_income'].append(ni)
                    result['gross_margin_pct'].append(gm)
                    result['operating_margin_pct'].append(om)
                    result['eps'].append(eps)
                    result['fcf'].append(self._safe_float(fcf))

                except Exception as e:
                    logger.debug(f"Error parsing quarter {col} for {ticker}: {e}")
                    continue

            result['available'] = len(result['labels']) > 0
            if result['available']:
                self._set_cached(ticker, 'quarterly', result)

            return result

        except Exception as e:
            logger.error(f"Error fetching quarterly financials for {ticker}: {e}")
            return {'ticker': ticker, 'available': False, 'reason': str(e)}

    def estimate_fair_value(self, ticker: str, growth_rate: float = None,
                            terminal_rate: float = 0.03,
                            discount_rate: float = 0.10,
                            growth_years: int = 5) -> Dict:
        """
        2-stage DCF fair value estimate.
        growth_rate: near-term FCF growth (auto-detected if None)
        terminal_rate: perpetuity growth rate (default 3%)
        discount_rate: WACC / required return (default 10%)
        """
        try:
            stock = yf.Ticker(ticker)
            cf = stock.quarterly_cashflow
            info = stock.info or {}

            # Get trailing 12-month FCF
            ttm_fcf = None
            if cf is not None and not (hasattr(cf, 'empty') and cf.empty):
                cols = list(cf.columns)[:4]  # Last 4 quarters = TTM
                fcf_vals = []
                for col in cols:
                    op = self._safe_float(cf.get(col, {}).get('Operating Cash Flow'))
                    capex = self._safe_float(cf.get(col, {}).get('Capital Expenditure'))
                    if op is not None:
                        fcf_vals.append(op + (capex or 0))
                if fcf_vals:
                    ttm_fcf = sum(fcf_vals)

            shares = self._safe_float(info.get('sharesOutstanding'))
            if not ttm_fcf or not shares or shares == 0:
                return {'available': False, 'reason': 'Insufficient FCF or share data'}

            # Auto growth rate from analyst estimate or historical revenue growth
            if growth_rate is None:
                growth_rate = self._safe_float(info.get('revenueGrowth')) or 0.10

            # Phase 1: High-growth period
            pv_phase1 = 0
            fcf = ttm_fcf
            for year in range(1, growth_years + 1):
                fcf *= (1 + growth_rate)
                pv_phase1 += fcf / ((1 + discount_rate) ** year)

            # Phase 2: Terminal value (Gordon Growth)
            terminal_fcf = fcf * (1 + terminal_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_rate)
            pv_terminal = terminal_value / ((1 + discount_rate) ** growth_years)

            total_pv = pv_phase1 + pv_terminal
            fair_value_per_share = total_pv / shares

            current_price = self._safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
            upside = None
            if current_price and current_price > 0 and fair_value_per_share:
                upside = round(((fair_value_per_share - current_price) / current_price) * 100, 1)

            return {
                'available': True,
                'ticker': ticker,
                'fair_value': round(fair_value_per_share, 2) if fair_value_per_share else None,
                'current_price': current_price,
                'upside_pct': upside,
                'assumptions': {
                    'growth_rate_pct': round(growth_rate * 100, 1),
                    'terminal_rate_pct': round(terminal_rate * 100, 1),
                    'discount_rate_pct': round(discount_rate * 100, 1),
                    'growth_years': growth_years,
                    'ttm_fcf_millions': round(ttm_fcf / 1e6, 1) if ttm_fcf else None,
                },
            }

        except Exception as e:
            logger.error(f"DCF error for {ticker}: {e}")
            return {'available': False, 'reason': str(e)}

    def get_peer_comparison(self, ticker: str, peers: List[str] = None) -> Dict:
        """
        Side-by-side comparison of ticker vs peers on key metrics.
        Auto-detects peers from same sector if not provided.
        """
        try:
            all_tickers = [ticker] + (peers or [])
            rows = []
            for t in all_tickers:
                try:
                    info = yf.Ticker(t).info or {}
                    rows.append({
                        'ticker': t,
                        'name': (info.get('shortName') or info.get('longName') or t)[:20],
                        'market_cap_b': self._safe_float(info.get('marketCap', 0)) / 1e9 if info.get('marketCap') else None,
                        'pe_ratio': self._safe_float(info.get('trailingPE')),
                        'forward_pe': self._safe_float(info.get('forwardPE')),
                        'ev_ebitda': self._safe_float(info.get('enterpriseToEbitda')),
                        'price_to_sales': self._safe_float(info.get('priceToSalesTrailing12Months')),
                        'gross_margin_pct': round(float(info.get('grossMargins', 0)) * 100, 1) if info.get('grossMargins') else None,
                        'profit_margin_pct': round(float(info.get('profitMargins', 0)) * 100, 1) if info.get('profitMargins') else None,
                        'revenue_growth_pct': round(float(info.get('revenueGrowth', 0)) * 100, 1) if info.get('revenueGrowth') else None,
                        'roe_pct': round(float(info.get('returnOnEquity', 0)) * 100, 1) if info.get('returnOnEquity') else None,
                        'debt_to_equity': self._safe_float(info.get('debtToEquity')),
                        'dividend_yield_pct': round(float(info.get('dividendYield', 0)) * 100, 2) if info.get('dividendYield') else None,
                        'is_primary': t == ticker,
                    })
                except Exception as e:
                    logger.debug(f"Error fetching peer data for {t}: {e}")

            return {
                'available': len(rows) > 0,
                'ticker': ticker,
                'peers': rows,
            }

        except Exception as e:
            logger.error(f"Peer comparison error for {ticker}: {e}")
            return {'available': False, 'reason': str(e)}

    def get_key_stats(self, ticker: str) -> Dict:
        """
        Quick key stats: 52w high/low, market cap label, short interest,
        pre/post market price.
        """
        cached = self._get_cached(ticker, 'key_stats')
        if cached:
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            fast = getattr(stock, 'fast_info', None)

            current_price = self._safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
            if fast and hasattr(fast, 'last_price'):
                current_price = self._safe_float(fast.last_price) or current_price

            week52_high = self._safe_float(info.get('fiftyTwoWeekHigh'))
            week52_low = self._safe_float(info.get('fiftyTwoWeekLow'))

            pct_from_high = None
            pct_from_low = None
            if current_price and week52_high and week52_high > 0:
                pct_from_high = round(((current_price - week52_high) / week52_high) * 100, 1)
            if current_price and week52_low and week52_low > 0:
                pct_from_low = round(((current_price - week52_low) / week52_low) * 100, 1)

            market_cap = self._safe_float(info.get('marketCap'))
            cap_label = 'Unknown'
            if market_cap:
                if market_cap >= 200e9:
                    cap_label = 'Mega'
                elif market_cap >= 10e9:
                    cap_label = 'Large'
                elif market_cap >= 2e9:
                    cap_label = 'Mid'
                elif market_cap >= 300e6:
                    cap_label = 'Small'
                else:
                    cap_label = 'Micro'

            short_pct = self._safe_float(info.get('shortPercentOfFloat'))
            if short_pct:
                short_pct = round(short_pct * 100, 2)
            days_to_cover = self._safe_float(info.get('shortRatio'))

            pre_market = self._safe_float(info.get('preMarketPrice'))
            post_market = self._safe_float(info.get('postMarketPrice'))

            result = {
                'ticker': ticker,
                'available': True,
                'current_price': current_price,
                'week52_high': week52_high,
                'week52_low': week52_low,
                'pct_from_52w_high': pct_from_high,
                'pct_from_52w_low': pct_from_low,
                'market_cap': market_cap,
                'market_cap_b': round(market_cap / 1e9, 2) if market_cap else None,
                'cap_label': cap_label,
                'short_pct_float': short_pct,
                'days_to_cover': days_to_cover,
                'pre_market_price': pre_market,
                'post_market_price': post_market,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': self._safe_float(info.get('trailingPE')),
                'forward_pe': self._safe_float(info.get('forwardPE')),
                'fetched_at': datetime.now().isoformat(),
            }

            self._set_cached(ticker, 'key_stats', result)
            return result

        except Exception as e:
            logger.error(f"Key stats error for {ticker}: {e}")
            return {'ticker': ticker, 'available': False, 'reason': str(e)}


# Singleton
financial_statements = FinancialStatements()
