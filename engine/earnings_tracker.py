"""
Earnings Intelligence Engine
Tracks upcoming earnings dates, EPS/revenue estimates, beat/miss history,
and generates pre-earnings positioning alerts with historical context.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging
import json

logger = logging.getLogger(__name__)


class EarningsTracker:
    """Track earnings dates, EPS estimates, and surprise history."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(hours=12)  # Refresh twice daily

    def get_earnings_info(self, ticker: str) -> Optional[Dict]:
        """Get next earnings date and days until earnings."""
        # Check cache first
        if ticker in self._cache:
            entry = self._cache[ticker]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar

            logger.debug(f"Fetching earnings calendar for {ticker}, type: {type(calendar)}")

            if calendar is None:
                logger.debug(f"No earnings calendar for {ticker}")
                return None

            # Handle both dict and DataFrame formats from yfinance
            earnings_date = None

            if isinstance(calendar, dict):
                if 'Earnings Date' in calendar:
                    earnings_date = calendar['Earnings Date']
                    if isinstance(earnings_date, list) and earnings_date:
                        earnings_date = earnings_date[0]
            elif hasattr(calendar, 'empty') and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                elif hasattr(calendar, 'values') and len(calendar.values) > 0:
                    earnings_date = calendar.values[0][0]
            else:
                logger.debug(f"Unexpected calendar format for {ticker}: {type(calendar)}")
                return None

            if earnings_date is None:
                return None

            if hasattr(earnings_date, 'to_pydatetime'):
                earnings_date = earnings_date.to_pydatetime()
            elif not isinstance(earnings_date, datetime):
                earnings_date = datetime.fromisoformat(str(earnings_date))

            days_until = (earnings_date - datetime.now()).days

            # Get EPS/revenue estimates from calendar dict
            eps_estimate = None
            revenue_estimate = None
            if isinstance(calendar, dict):
                eps_estimate = calendar.get('Earnings Average') or calendar.get('EPS Estimate')
                revenue_estimate = calendar.get('Revenue Average') or calendar.get('Revenue Estimate')
                # Convert to float safely
                try:
                    eps_estimate = float(eps_estimate) if eps_estimate is not None else None
                except (TypeError, ValueError):
                    eps_estimate = None
                try:
                    revenue_estimate = float(revenue_estimate) if revenue_estimate is not None else None
                except (TypeError, ValueError):
                    revenue_estimate = None

            data = {
                'ticker': ticker,
                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'is_imminent': 0 <= days_until <= 14,
                'is_within_week': 0 <= days_until <= 7,
                'eps_estimate': eps_estimate,
                'revenue_estimate': revenue_estimate,
                'fetched_at': datetime.now().isoformat(),
            }

            # Cache result
            self._cache[ticker] = {'data': data, 'timestamp': datetime.now()}
            self._store_earnings_date(data)

            return data

        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return None

    def get_beat_history(self, ticker: str, quarters: int = 8) -> Dict:
        """
        Get historical EPS surprise data for the last N quarters.
        Returns avg beat %, beat streak, and per-quarter data.
        """
        try:
            stock = yf.Ticker(ticker)
            # Try earnings_history first (newer yfinance)
            history = stock.earnings_history
            if history is None or (hasattr(history, 'empty') and history.empty):
                # Fallback: quarterly_earnings
                history = stock.quarterly_earnings
            
            if history is None or (hasattr(history, 'empty') and history.empty):
                return {'available': False, 'reason': 'No earnings history data'}

            # Normalize columns
            history = history.reset_index() if hasattr(history, 'reset_index') else history
            
            quarters_data = []
            beat_count = 0
            total_surprise_pct = 0.0

            # Parse available rows
            for _, row in history.iterrows():
                try:
                    row_dict = dict(row)
                    # yfinance column names vary; try multiple common names
                    eps_actual = (row_dict.get('epsActual') or row_dict.get('Reported EPS') 
                                  or row_dict.get('actual'))
                    eps_estimate = (row_dict.get('epsEstimate') or row_dict.get('EPS Estimate') 
                                    or row_dict.get('estimate'))
                    surprise_pct = row_dict.get('epsSurprisePct') or row_dict.get('surprisePercent')
                    quarter_date = (row_dict.get('Earnings Date') or row_dict.get('quarterDate') 
                                    or row_dict.get('Date'))

                    if eps_actual is None and eps_estimate is None:
                        continue

                    # Compute surprise if not provided
                    if surprise_pct is None and eps_actual is not None and eps_estimate is not None:
                        try:
                            eps_estimate_f = float(eps_estimate)
                            eps_actual_f = float(eps_actual)
                            if eps_estimate_f != 0:
                                surprise_pct = ((eps_actual_f - eps_estimate_f) / abs(eps_estimate_f)) * 100
                        except (TypeError, ValueError):
                            surprise_pct = None

                    beat = False
                    if surprise_pct is not None:
                        try:
                            surprise_pct = float(surprise_pct)
                            beat = surprise_pct > 0
                            if beat:
                                beat_count += 1
                            total_surprise_pct += surprise_pct
                        except (TypeError, ValueError):
                            surprise_pct = None

                    quarters_data.append({
                        'date': str(quarter_date) if quarter_date else 'Unknown',
                        'eps_actual': float(eps_actual) if eps_actual is not None else None,
                        'eps_estimate': float(eps_estimate) if eps_estimate is not None else None,
                        'surprise_pct': round(surprise_pct, 2) if surprise_pct is not None else None,
                        'beat': beat,
                    })
                except Exception:
                    continue

            # Limit to last N quarters
            quarters_data = quarters_data[:quarters]

            if not quarters_data:
                return {'available': False, 'reason': 'No parseable quarterly data'}

            total = len(quarters_data)
            beat_rate = (beat_count / total * 100) if total > 0 else 0
            avg_surprise = (total_surprise_pct / total) if total > 0 else 0

            # Compute current beat streak (most recent quarters)
            streak = 0
            for q in quarters_data:
                if q.get('beat'):
                    streak += 1
                else:
                    break

            return {
                'available': True,
                'quarters': quarters_data,
                'total_quarters': total,
                'beat_count': beat_count,
                'beat_rate_pct': round(beat_rate, 1),
                'avg_surprise_pct': round(avg_surprise, 2),
                'current_beat_streak': streak,
            }

        except Exception as e:
            logger.error(f"Error fetching beat history for {ticker}: {e}")
            return {'available': False, 'reason': str(e)}

    def generate_positioning_alert(self, ticker: str) -> Optional[str]:
        """
        Generate a pre-earnings positioning alert message.
        E.g.: "NVDA reports in 3 days — last 8Q beat by avg 12.4%, 6Q streak"
        """
        info = self.get_earnings_info(ticker)
        if not info or not info.get('is_imminent'):
            return None

        days = info['days_until']
        beat_history = self.get_beat_history(ticker)

        parts = [f"📅 **{ticker}** reports in {days} day{'s' if days != 1 else ''}"]

        if beat_history.get('available'):
            avg = beat_history['avg_surprise_pct']
            rate = beat_history['beat_rate_pct']
            streak = beat_history['current_beat_streak']
            total = beat_history['total_quarters']

            direction = "beat" if avg > 0 else "missed"
            parts.append(f"(last {total}Q: {direction} by avg {abs(avg):.1f}%)")

            if streak >= 3:
                parts.append(f"🔥 {streak}Q beat streak")
            elif rate < 40:
                parts.append(f"⚠️ Only {rate:.0f}% beat rate — volatile")

        if info.get('eps_estimate') is not None:
            parts.append(f"| EPS est: ${info['eps_estimate']:.2f}")

        return " ".join(parts)

    def get_earnings_dashboard_data(self, tickers: List[str]) -> List[Dict]:
        """
        Get upcoming earnings for multiple tickers, sorted by proximity.
        Returns list of dicts ready for dashboard/alert display.
        """
        results = []
        for ticker in tickers:
            info = self.get_earnings_info(ticker)
            if not info:
                continue
            days = info.get('days_until', 999)
            if days < 0:  # Already passed
                continue
            
            beat_data = {}
            if info.get('is_imminent'):
                beat_data = self.get_beat_history(ticker, quarters=8)

            results.append({
                **info,
                'beat_history': beat_data,
                'alert_message': self.generate_positioning_alert(ticker),
            })

        return sorted(results, key=lambda x: x.get('days_until', 999))

    def _store_earnings_date(self, data: Dict):
        """Store earnings date in database."""
        try:
            db.execute("""
                INSERT INTO earnings_calendar (ticker, earnings_date, days_until, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    earnings_date = excluded.earnings_date,
                    days_until = excluded.days_until,
                    fetched_at = excluded.fetched_at
            """, (data['ticker'], data['earnings_date'], data['days_until'], data['fetched_at']))
        except Exception as e:
            logger.warning(f"Could not store earnings date: {e}")

    def check_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Check earnings for multiple tickers. Returns dict of ticker -> earnings_info."""
        results = {}
        for ticker in tickers:
            info = self.get_earnings_info(ticker)
            if info:
                results[ticker] = info
        return results

    def flag_pre_earnings_risk(self, ticker: str, analysis_data: Dict) -> Dict:
        """Add earnings warning flags to analysis data with beat history context."""
        earnings = self.get_earnings_info(ticker)

        if not earnings:
            analysis_data['earnings_risk'] = 'unknown'
            return analysis_data

        if earnings['is_within_week']:
            analysis_data['earnings_risk'] = 'high'
            
            # Enrich warning with beat history
            alert_msg = self.generate_positioning_alert(ticker)
            if alert_msg:
                base_warn = alert_msg
            else:
                base_warn = f"(Earnings) EARNINGS IN {earnings['days_until']} DAYS — High uncertainty"
            
            analysis_data['warnings'] = analysis_data.get('warnings', []) + [
                base_warn,
                "(Notice) Signal confidence reduced 40% due to earnings proximity"
            ]
            if 'confidence' in analysis_data:
                analysis_data['confidence'] = int(analysis_data['confidence'] * 0.6)

        elif earnings['is_imminent']:
            analysis_data['earnings_risk'] = 'moderate'
            analysis_data['warnings'] = analysis_data.get('warnings', []) + [
                f"(Earnings) Earnings in {earnings['days_until']} days — Increased volatility expected"
            ]
            if 'confidence' in analysis_data:
                analysis_data['confidence'] = int(analysis_data['confidence'] * 0.85)
        else:
            analysis_data['earnings_risk'] = 'low'

        analysis_data['next_earnings_date'] = earnings['earnings_date']
        analysis_data['days_until_earnings'] = earnings['days_until']

        return analysis_data

    def track_post_earnings_drift(self, ticker: str, earnings_date: datetime) -> Optional[Dict]:
        """Track stock movement in 5 days after earnings (for learning)."""
        try:
            end_date = earnings_date + timedelta(days=7)
            hist = yf.Ticker(ticker).history(start=earnings_date, end=end_date)

            if hist.empty or len(hist) < 2:
                return None

            open_price = float(hist['Open'].iloc[0])
            close_price = float(hist['Close'].iloc[-1])

            drift_pct = ((close_price - open_price) / open_price) * 100

            return {
                'ticker': ticker,
                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                'open_price': round(open_price, 2),
                'close_price_5d': round(close_price, 2),
                'drift_pct': round(drift_pct, 2),
                'direction': 'up' if drift_pct > 0 else 'down',
            }

        except Exception as e:
            logger.error(f"Error tracking post-earnings drift for {ticker}: {e}")
            return None

    def get_cached_earnings(self, ticker: str) -> Optional[Dict]:
        """Get earnings info from database cache (no API call)."""
        try:
            result = db.query_one("""
                SELECT ticker, earnings_date, days_until, fetched_at
                FROM earnings_calendar
                WHERE ticker = ?
                AND datetime(fetched_at) > datetime('now', '-12 hours')
            """, (ticker,))

            if result:
                return {
                    'ticker': result['ticker'],
                    'earnings_date': result['earnings_date'],
                    'days_until': result['days_until'],
                    'is_imminent': 0 <= result['days_until'] <= 14,
                    'is_within_week': 0 <= result['days_until'] <= 7,
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached earnings: {e}")
            return None


# Singleton
earnings_tracker = EarningsTracker()
