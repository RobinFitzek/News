"""
Sentiment Analyzer
Aggregates analyst ratings, news sentiment, and detects contrarian signals
from yfinance data. No paid API calls — uses freely available recommendations
and news metadata.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)

# Rating classification buckets
BULLISH_RATINGS = {'buy', 'strong_buy', 'overweight', 'outperform', 'positive', 'accumulate'}
BEARISH_RATINGS = {'sell', 'strong_sell', 'underweight', 'underperform', 'negative', 'reduce'}
NEUTRAL_RATINGS = {'hold', 'neutral', 'equal-weight', 'market perform', 'sector perform', 'peer perform'}


class SentimentAnalyzer:
    """Aggregate analyst and news sentiment for investment signals."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(minutes=30)

    def get_sentiment_summary(self, ticker: str) -> Dict:
        """
        Full sentiment overview for a ticker.
        Combines analyst consensus, news tone, and contrarian signals.
        """
        cache_key = f"summary_{ticker.upper()}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        analyst = self.get_analyst_consensus(ticker)
        contrarian = self.detect_contrarian_signal(ticker)

        # News headline count (basic tone proxy)
        news_data = self._get_news_summary(ticker)

        overall_score = 50  # Neutral baseline
        if analyst.get('consensus_score') is not None:
            overall_score = analyst['consensus_score']

        # Contrarian adjustments
        if contrarian.get('signal'):
            if contrarian['signal'] == 'contrarian_bullish':
                overall_score = max(overall_score, 60)
            elif contrarian['signal'] == 'contrarian_bearish':
                overall_score = min(overall_score, 40)

        sentiment_label = 'bullish' if overall_score >= 60 else 'bearish' if overall_score <= 40 else 'neutral'

        result = {
            'ticker': ticker.upper(),
            'overall_score': overall_score,
            'sentiment': sentiment_label,
            'analyst': analyst,
            'contrarian': contrarian,
            'news': news_data,
            'updated': datetime.now().isoformat(),
        }

        self._cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
        return result

    def get_analyst_consensus(self, ticker: str) -> Dict:
        """
        Parse yfinance recommendations into a consensus score.
        Returns score 0-100 (0=strong sell, 100=strong buy).
        """
        try:
            stock = yf.Ticker(ticker)
            recs = stock.recommendations

            if recs is None or (hasattr(recs, 'empty') and recs.empty):
                return {'consensus_score': None, 'total_ratings': 0, 'breakdown': {}, 'message': 'No analyst data'}

            # Take last 3 months of ratings
            recent = recs.tail(20)

            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            total = 0

            for _, row in recent.iterrows():
                grade = str(row.get('To Grade', row.get('toGrade', ''))).strip().lower()
                if not grade:
                    continue

                total += 1
                if grade in BULLISH_RATINGS or any(b in grade for b in ['buy', 'outperform', 'overweight']):
                    bullish_count += 1
                elif grade in BEARISH_RATINGS or any(b in grade for b in ['sell', 'underperform', 'underweight']):
                    bearish_count += 1
                else:
                    neutral_count += 1

            if total == 0:
                return {'consensus_score': None, 'total_ratings': 0, 'breakdown': {}, 'message': 'No parseable ratings'}

            # Score: 100 = all bullish, 0 = all bearish, 50 = neutral
            consensus_score = int(((bullish_count * 1.0 + neutral_count * 0.5) / total) * 100)

            return {
                'consensus_score': consensus_score,
                'total_ratings': total,
                'breakdown': {
                    'bullish': bullish_count,
                    'neutral': neutral_count,
                    'bearish': bearish_count,
                },
                'message': f"{bullish_count} buy, {neutral_count} hold, {bearish_count} sell from {total} recent ratings",
            }

        except Exception as e:
            logger.warning(f"Analyst consensus failed for {ticker}: {e}")
            return {'consensus_score': None, 'total_ratings': 0, 'breakdown': {}, 'message': str(e)}

    def detect_contrarian_signal(self, ticker: str) -> Dict:
        """
        Detect extreme consensus that historically reverts.
        Extreme bullish (>90% buy) = contrarian bearish.
        Extreme bearish (>70% sell) = contrarian bullish.
        """
        analyst = self.get_analyst_consensus(ticker)

        if analyst['total_ratings'] < 5:
            return {'signal': None, 'message': 'Insufficient ratings for contrarian analysis'}

        score = analyst.get('consensus_score')
        if score is None:
            return {'signal': None, 'message': 'No consensus score'}

        breakdown = analyst.get('breakdown', {})
        total = analyst['total_ratings']
        bullish_pct = (breakdown.get('bullish', 0) / total) * 100 if total > 0 else 0
        bearish_pct = (breakdown.get('bearish', 0) / total) * 100 if total > 0 else 0

        if bullish_pct >= 90:
            return {
                'signal': 'contrarian_bearish',
                'strength': min(100, int(bullish_pct)),
                'message': (
                    f"Extreme bullish consensus ({bullish_pct:.0f}% buy). "
                    f"Historically, near-unanimous buy ratings often precede pullbacks."
                ),
            }
        elif bearish_pct >= 70:
            return {
                'signal': 'contrarian_bullish',
                'strength': min(100, int(bearish_pct)),
                'message': (
                    f"Extreme bearish consensus ({bearish_pct:.0f}% sell). "
                    f"Deep pessimism can signal a sentiment-driven bottom."
                ),
            }

        return {'signal': None, 'message': 'No extreme consensus detected'}

    def _get_news_summary(self, ticker: str) -> Dict:
        """
        Summarize available news metadata from yfinance.
        Returns headline count and recency — no NLP scoring.
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return {'headline_count': 0, 'message': 'No recent news'}

            count = len(news)
            titles = [item.get('title', '') for item in news[:5]]

            return {
                'headline_count': count,
                'recent_headlines': titles,
                'message': f"{count} news items available",
            }

        except Exception as e:
            logger.debug(f"News fetch failed for {ticker}: {e}")
            return {'headline_count': 0, 'message': str(e)}


# Singleton
sentiment_analyzer = SentimentAnalyzer()
