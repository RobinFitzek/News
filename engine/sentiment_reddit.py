"""
Reddit Sentiment Analysis
Fetches and analyzes sentiment from Reddit posts and comments for watchlist tickers.
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RedditSentimentAnalyzer:
    """Analyze sentiment from Reddit posts and comments."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(hours=6)

    def fetch_reddit_posts(self, ticker: str, subreddits: List[str] = None) -> Optional[List[Dict]]:
        """
        Fetch recent posts mentioning the ticker from specified subreddits.
        Uses Reddit API (requires user-agent header).
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks']

        ticker = ticker.upper()
        cache_key = f"reddit_{ticker}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        headers = {
            'User-Agent': 'StockholmMonitor/1.0 (research@example.com)',
        }

        posts = []
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&sort=new&limit=10"
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        posts.append({
                            'title': post_data.get('title', ''),
                            'url': post_data.get('url', ''),
                            'score': post_data.get('score', 0),
                            'num_comments': post_data.get('num_comments', 0),
                            'created_utc': post_data.get('created_utc', 0),
                            'subreddit': subreddit,
                        })
            except Exception as e:
                logger.warning(f"Error fetching Reddit posts for {ticker} in r/{subreddit}: {e}")

        self._cache[cache_key] = {'data': posts, 'timestamp': datetime.now()}
        return posts

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a given text using a simple heuristic.
        Returns a sentiment score between -1.0 (negative) and +1.0 (positive).
        """
        if not text:
            return 0.0

        # Simple keyword-based sentiment analysis
        positive_words = ['bullish', 'buy', 'long', 'up', 'gain', 'profit', 'positive', 'good', 'great', 'strong']
        negative_words = ['bearish', 'sell', 'short', 'down', 'loss', 'negative', 'bad', 'weak', 'crash', 'drop']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count + negative_count == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment))

    def get_reddit_sentiment_score(self, ticker: str) -> Dict:
        """
        Get the overall sentiment score for a ticker from Reddit.
        Returns a dict with sentiment score and post details.
        """
        posts = self.fetch_reddit_posts(ticker)
        if not posts:
            return {'ticker': ticker, 'sentiment_score': 0.0, 'posts': []}

        total_sentiment = 0.0
        post_details = []
        for post in posts:
            sentiment = self.analyze_sentiment(post['title'])
            total_sentiment += sentiment
            post_details.append({
                'title': post['title'],
                'url': post['url'],
                'sentiment': sentiment,
                'score': post['score'],
                'num_comments': post['num_comments'],
                'subreddit': post['subreddit'],
            })

        avg_sentiment = total_sentiment / len(posts) if posts else 0.0
        return {
            'ticker': ticker,
            'sentiment_score': round(avg_sentiment, 2),
            'posts': post_details,
        }


# Singleton instance
reddit_sentiment_analyzer = RedditSentimentAnalyzer()
