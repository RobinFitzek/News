"""
Market Regime Detection
Classifies current market as Bull, Bear, or Choppy based on SPY SMAs, VIX, and 10Y yield.
Used by quant screener for confidence adjustment and by dashboard for macro awareness.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegime:
    """Detects current market regime from SPY, VIX, and Treasury yield data."""

    def __init__(self, cache_minutes: int = 60):
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=cache_minutes)

    def get_current_regime(self) -> Dict:
        """Return current market regime with supporting data.

        Returns dict with keys:
            regime: 'bull' | 'bear' | 'choppy'
            spy_price, sma50, sma200, vix, ten_year_yield
        """
        if self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache

        result = {
            'regime': 'choppy',
            'spy_price': None,
            'sma50': None,
            'sma200': None,
            'vix': None,
            'ten_year_yield': None,
            'updated_at': datetime.now().isoformat(),
        }

        try:
            # SPY price and SMAs
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1y')
            if not hist.empty and len(hist) >= 200:
                close = hist['Close']
                result['spy_price'] = round(float(close.iloc[-1]), 2)
                result['sma50'] = round(float(close.rolling(50).mean().iloc[-1]), 2)
                result['sma200'] = round(float(close.rolling(200).mean().iloc[-1]), 2)

                # Regime classification
                price = result['spy_price']
                sma50 = result['sma50']
                sma200 = result['sma200']

                if price > sma200 and sma50 > sma200:
                    result['regime'] = 'bull'
                elif price < sma200 and sma50 < sma200:
                    result['regime'] = 'bear'
                else:
                    result['regime'] = 'choppy'
            elif not hist.empty:
                close = hist['Close']
                result['spy_price'] = round(float(close.iloc[-1]), 2)
                if len(hist) >= 50:
                    result['sma50'] = round(float(close.rolling(50).mean().iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Failed to fetch SPY data: {e}")

        # VIX
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='5d')
            if not vix_hist.empty:
                result['vix'] = round(float(vix_hist['Close'].iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")

        # 10-Year Treasury Yield
        try:
            tnx = yf.Ticker('^TNX')
            tnx_hist = tnx.history(period='5d')
            if not tnx_hist.empty:
                result['ten_year_yield'] = round(float(tnx_hist['Close'].iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Failed to fetch 10Y yield: {e}")

        self._cache = result
        self._cache_time = datetime.now()
        return result

    def get_confidence_adjustment(self, signal: str, regime: str) -> int:
        """Return confidence adjustment points based on regime vs signal direction.

        Bear market penalises buy signals, bull market penalises sell signals.
        """
        if regime == 'bear' and signal in ('Opportunity', 'BUY', 'STRONG_BUY'):
            return -15
        if regime == 'bull' and signal in ('Caution', 'SELL', 'STRONG_SELL'):
            return +15
        return 0

    def get_regime_weight_adjustments(self, regime: str = None) -> Dict:
        """Return weight adjustment multipliers for each scoring factor based on regime.

        In bull markets: boost momentum and technical.
        In bear markets: boost quality and valuation (defensive).
        In choppy markets: keep balanced.

        Returns dict with keys: valuation, technical, momentum, quality
        Each value is a multiplier (1.0 = no change, >1 = boost, <1 = reduce).
        """
        if regime is None:
            regime = self.get_current_regime().get('regime', 'choppy')

        adjustments = {
            'bull': {
                'valuation': 0.85,
                'technical': 1.15,
                'momentum': 1.20,
                'quality': 0.80,
            },
            'bear': {
                'valuation': 1.15,
                'technical': 0.90,
                'momentum': 0.75,
                'quality': 1.20,
            },
            'choppy': {
                'valuation': 1.0,
                'technical': 1.0,
                'momentum': 1.0,
                'quality': 1.0,
            },
        }

        return adjustments.get(regime, adjustments['choppy'])

    def invalidate_cache(self):
        self._cache = None
        self._cache_time = None


# Singleton
market_regime = MarketRegime()
