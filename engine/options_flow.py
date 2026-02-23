"""
Options Flow Analyzer
Computes put/call ratios, IV rank approximation, and detects unusual
options volume from yfinance option chain data.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)


class OptionsFlow:
    """Analyze options flow for sentiment and unusual activity signals."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=15)

    def get_options_summary(self, ticker: str) -> Optional[Dict]:
        """Get a complete options summary: P/C ratio, IV rank, and volume stats."""
        cache_key = ticker.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(cache_key)
            expiry_dates = stock.options

            if not expiry_dates:
                logger.debug(f"No options data available for {ticker}")
                return None

            # Use the nearest expiry for current sentiment
            nearest_expiry = expiry_dates[0]
            chain = stock.option_chain(nearest_expiry)

            calls = chain.calls
            puts = chain.puts

            if calls.empty and puts.empty:
                return None

            # Volume and open interest totals
            call_volume = int(calls['volume'].sum()) if 'volume' in calls.columns else 0
            put_volume = int(puts['volume'].sum()) if 'volume' in puts.columns else 0
            call_oi = int(calls['openInterest'].sum()) if 'openInterest' in calls.columns else 0
            put_oi = int(puts['openInterest'].sum()) if 'openInterest' in puts.columns else 0

            total_volume = call_volume + put_volume
            pc_ratio_volume = round(put_volume / call_volume, 3) if call_volume > 0 else None
            pc_ratio_oi = round(put_oi / call_oi, 3) if call_oi > 0 else None

            # IV rank approximation from current chain implied volatilities
            iv_rank = self._compute_iv_rank(calls, puts)

            # Sentiment interpretation
            sentiment = 'neutral'
            if pc_ratio_volume is not None:
                if pc_ratio_volume > 1.2:
                    sentiment = 'bearish'
                elif pc_ratio_volume < 0.7:
                    sentiment = 'bullish'

            data = {
                'ticker': cache_key,
                'expiry': nearest_expiry,
                'total_expiries': len(expiry_dates),
                'call_volume': call_volume,
                'put_volume': put_volume,
                'total_volume': total_volume,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'pc_ratio_volume': pc_ratio_volume,
                'pc_ratio_oi': pc_ratio_oi,
                'iv_rank': iv_rank,
                'sentiment': sentiment,
                'fetched_at': datetime.now().isoformat(),
            }

            self._cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data

        except Exception as e:
            logger.error(f"Error fetching options summary for {ticker}: {e}")
            return None

    def get_put_call_ratio(self, ticker: str) -> Optional[Dict]:
        """Get put/call ratio by volume and open interest for the nearest expiry."""
        summary = self.get_options_summary(ticker)
        if not summary:
            return None

        return {
            'ticker': summary['ticker'],
            'pc_ratio_volume': summary['pc_ratio_volume'],
            'pc_ratio_oi': summary['pc_ratio_oi'],
            'sentiment': summary['sentiment'],
            'call_volume': summary['call_volume'],
            'put_volume': summary['put_volume'],
        }

    def detect_unusual_activity(self, ticker: str) -> Optional[Dict]:
        """
        Detect unusual options activity by comparing current volume
        to open interest. High volume/OI ratios suggest new positioning.
        """
        try:
            stock = yf.Ticker(ticker.upper())
            expiry_dates = stock.options

            if not expiry_dates:
                return None

            nearest_expiry = expiry_dates[0]
            chain = stock.option_chain(nearest_expiry)

            unusual_calls = []
            unusual_puts = []

            # Flag strikes where volume > 2x open interest (unusual activity)
            if not chain.calls.empty and 'volume' in chain.calls.columns and 'openInterest' in chain.calls.columns:
                for _, row in chain.calls.iterrows():
                    vol = row.get('volume', 0) or 0
                    oi = row.get('openInterest', 0) or 0
                    if oi > 0 and vol > 2 * oi and vol >= 100:
                        unusual_calls.append({
                            'strike': float(row['strike']),
                            'volume': int(vol),
                            'open_interest': int(oi),
                            'vol_oi_ratio': round(vol / oi, 2),
                            'implied_volatility': round(float(row.get('impliedVolatility', 0)) * 100, 1),
                        })

            if not chain.puts.empty and 'volume' in chain.puts.columns and 'openInterest' in chain.puts.columns:
                for _, row in chain.puts.iterrows():
                    vol = row.get('volume', 0) or 0
                    oi = row.get('openInterest', 0) or 0
                    if oi > 0 and vol > 2 * oi and vol >= 100:
                        unusual_puts.append({
                            'strike': float(row['strike']),
                            'volume': int(vol),
                            'open_interest': int(oi),
                            'vol_oi_ratio': round(vol / oi, 2),
                            'implied_volatility': round(float(row.get('impliedVolatility', 0)) * 100, 1),
                        })

            # Sort by volume/OI ratio descending
            unusual_calls.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
            unusual_puts.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)

            has_unusual = len(unusual_calls) > 0 or len(unusual_puts) > 0

            return {
                'ticker': ticker.upper(),
                'expiry': nearest_expiry,
                'has_unusual_activity': has_unusual,
                'unusual_calls': unusual_calls[:5],  # Top 5
                'unusual_puts': unusual_puts[:5],
                'total_unusual_call_strikes': len(unusual_calls),
                'total_unusual_put_strikes': len(unusual_puts),
                'bias': 'bullish' if len(unusual_calls) > len(unusual_puts) else
                        'bearish' if len(unusual_puts) > len(unusual_calls) else 'neutral',
            }

        except Exception as e:
            logger.error(f"Error detecting unusual options activity for {ticker}: {e}")
            return None

    def _compute_iv_rank(self, calls, puts) -> Optional[float]:
        """
        Approximate IV rank from current chain implied volatilities.
        Uses the median IV of ATM options as current IV, then ranks it
        against the spread of IVs across strikes as a proxy.
        """
        try:
            all_ivs = []
            if not calls.empty and 'impliedVolatility' in calls.columns:
                all_ivs.extend(calls['impliedVolatility'].dropna().tolist())
            if not puts.empty and 'impliedVolatility' in puts.columns:
                all_ivs.extend(puts['impliedVolatility'].dropna().tolist())

            if len(all_ivs) < 3:
                return None

            all_ivs = [iv for iv in all_ivs if iv > 0]
            if not all_ivs:
                return None

            current_iv = float(np.median(all_ivs))
            iv_min = min(all_ivs)
            iv_max = max(all_ivs)

            if iv_max == iv_min:
                return 50.0

            # IV rank = (current - min) / (max - min) * 100
            iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
            return round(iv_rank, 1)

        except Exception as e:
            logger.warning(f"Could not compute IV rank: {e}")
            return None


# Singleton
options_flow = OptionsFlow()
