"""
Relative Strength Ranking
Ranks watchlist stocks by 3/6/12-month return vs SPY.
Momentum works — surface it systematically.
"""
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db

logger = logging.getLogger(__name__)


class RSRanking:
    """Rank stocks by relative strength vs benchmark."""

    BENCHMARK = "SPY"

    def _get_return(self, ticker: str, months: int) -> Optional[float]:
        """Get N-month total return for a ticker."""
        try:
            end = datetime.now()
            start = end - timedelta(days=months * 30)
            hist = yf.Ticker(ticker).history(start=start, end=end)
            if hist.empty or len(hist) < 5:
                return None
            first = float(hist['Close'].iloc[0])
            last = float(hist['Close'].iloc[-1])
            return round(((last - first) / first) * 100, 2) if first else None
        except Exception as e:
            logger.debug(f"RS error for {ticker}: {e}")
            return None

    def rank_tickers(self, tickers: List[str]) -> List[Dict]:
        """
        Rank tickers by relative strength vs SPY at 3/6/12 months.
        Returns list sorted by composite RS score (descending = strongest).
        """
        # Fetch SPY benchmark returns
        spy_3m = self._get_return(self.BENCHMARK, 3) or 0
        spy_6m = self._get_return(self.BENCHMARK, 6) or 0
        spy_12m = self._get_return(self.BENCHMARK, 12) or 0

        results = []
        for ticker in tickers:
            try:
                ret_3m = self._get_return(ticker, 3)
                ret_6m = self._get_return(ticker, 6)
                ret_12m = self._get_return(ticker, 12)

                rs_3m = round(ret_3m - spy_3m, 2) if ret_3m is not None else None
                rs_6m = round(ret_6m - spy_6m, 2) if ret_6m is not None else None
                rs_12m = round(ret_12m - spy_12m, 2) if ret_12m is not None else None

                # Composite RS score: weighted avg (3m=50%, 6m=30%, 12m=20%)
                parts = []
                weights = []
                for rs, w in [(rs_3m, 0.5), (rs_6m, 0.3), (rs_12m, 0.2)]:
                    if rs is not None:
                        parts.append(rs * w)
                        weights.append(w)
                composite = round(sum(parts) / sum(weights), 2) if weights else None

                results.append({
                    'ticker': ticker,
                    'return_3m': ret_3m,
                    'return_6m': ret_6m,
                    'return_12m': ret_12m,
                    'rs_3m': rs_3m,
                    'rs_6m': rs_6m,
                    'rs_12m': rs_12m,
                    'rs_composite': composite,
                    'spy_3m': spy_3m,
                    'spy_6m': spy_6m,
                    'spy_12m': spy_12m,
                })
            except Exception as e:
                logger.debug(f"Error ranking {ticker}: {e}")

        # Sort by composite RS (descending = strongest momentum)
        results.sort(key=lambda x: x.get('rs_composite') or -999, reverse=True)

        # Add rank position
        for i, r in enumerate(results):
            r['rank'] = i + 1
            r['rank_label'] = _rank_label(i + 1, len(results))

        return results


def _rank_label(rank: int, total: int) -> str:
    pct = rank / total * 100
    if pct <= 20:
        return 'Top 20%'
    elif pct <= 40:
        return 'Top 40%'
    elif pct <= 60:
        return 'Middle'
    elif pct <= 80:
        return 'Bottom 40%'
    else:
        return 'Bottom 20%'


# Singleton
rs_ranking = RSRanking()
