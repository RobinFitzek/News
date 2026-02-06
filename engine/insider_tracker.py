"""
Insider Trading Tracker Engine
Analyzes insider transactions with focus on voluntary purchases and cluster buying.
Presents facts, not AI opinions.
"""
from clients.sec_edgar_client import sec_client
from clients.perplexity_client import pplx_client
from core.database import db
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging


# Insider seniority rankings
INSIDER_RANKS = {
    'ceo': 5, 'chief executive': 5, 'president': 5,
    'cfo': 5, 'chief financial': 5,
    'coo': 4, 'chief operating': 4, 'cto': 4, 'chief technology': 4,
    'vp': 3, 'vice president': 3, 'svp': 3, 'evp': 3,
    'director': 2,
    '10%': 1, 'owner': 1,
}


class InsiderTracker:
    """Tracks and analyzes insider trading activity — facts only, no AI interpretation."""

    def __init__(self):
        self.sec = sec_client
        self.logger = logging.getLogger(__name__)

    def scan_watchlist_insiders(self, days_back: int = 90) -> List[Dict]:
        """Scan entire watchlist for insider activity."""
        watchlist = db.get_watchlist(active_only=True)
        results = []

        print(f"\n  Scanning {len(watchlist)} watchlist stocks for insider activity...")

        for item in watchlist:
            ticker = item['ticker']

            try:
                print(f"  Checking {ticker}...", end=' ')
                summary = self.sec.get_insider_summary(ticker, days_back)

                if summary['transactions_count'] > 0:
                    # Enrich with voluntary purchase analysis
                    summary['voluntary_purchases'] = self._count_voluntary_purchases(
                        summary.get('recent_transactions', [])
                    )
                    summary['insider_rank'] = self._get_highest_rank(
                        summary.get('recent_transactions', [])
                    )
                    results.append(summary)
                    signal_emoji = self._get_signal_emoji(summary['net_signal'])
                    vol_count = summary['voluntary_purchases']
                    print(f"{signal_emoji} {summary['net_signal']} ({summary['transactions_count']} txns, {vol_count} voluntary buys)")
                else:
                    print("No activity")

            except Exception as e:
                self.logger.error(f"Error scanning {ticker}: {e}")
                print(f"Error")

        results.sort(key=lambda x: abs(x['signal_score']), reverse=True)

        print(f"\n  Found insider activity in {len(results)} stocks")
        return results

    def get_insider_analysis(self, ticker: str, days_back: int = 180) -> Dict:
        """
        Comprehensive insider analysis for a single stock.
        Returns structured facts — no AI interpretation.
        """
        print(f"\n  Analyzing insider activity for {ticker}...")

        transactions = self.sec.get_insider_transactions(ticker, days_back)
        summary = self.sec.get_insider_summary(ticker, days_back)

        if not transactions:
            return {
                'ticker': ticker,
                'has_activity': False,
                'message': 'No insider transactions found in the specified period'
            }

        # Separate voluntary purchases from noise
        voluntary_buys = [t for t in transactions if self._is_voluntary_purchase(t)]
        noise_txns = [t for t in transactions if not self._is_voluntary_purchase(t)]

        # Detect patterns
        patterns = self._detect_patterns(transactions)

        # Calculate relative trade sizes
        for txn in voluntary_buys:
            txn['relative_size'] = self._calculate_relative_size(txn, transactions)
            txn['insider_rank'] = self._rank_insider(txn.get('title', ''))

        # Build structured fact summary (replaces Gemini AI interpretation)
        fact_summary = self._build_fact_summary(ticker, summary, patterns, voluntary_buys)

        # Get news context from Perplexity if available (factual, not interpretive)
        news_context = None
        if pplx_client.is_configured():
            usage = pplx_client.get_usage()
            if usage['remaining'] > 0:
                news_context = self._get_perplexity_context(ticker, summary)

        return {
            'ticker': ticker,
            'has_activity': True,
            'summary': summary,
            'transactions': transactions,
            'voluntary_purchases': voluntary_buys,
            'noise_transactions': len(noise_txns),
            'patterns': patterns,
            'fact_summary': fact_summary,
            'news_context': news_context,
            'timestamp': datetime.now().isoformat()
        }

    @staticmethod
    def _is_voluntary_purchase(transaction: Dict) -> bool:
        """Only open market purchases (code 'P') are true conviction signals."""
        return transaction.get('transaction_code') == 'P'

    @staticmethod
    def _rank_insider(title: str) -> int:
        """Rank insider by seniority. Higher = more significant."""
        title_lower = title.lower()
        for keyword, rank in INSIDER_RANKS.items():
            if keyword in title_lower:
                return rank
        return 1

    def _calculate_relative_size(self, transaction: Dict, all_transactions: List[Dict]) -> float:
        """Compare this transaction's value vs the insider's average trade size."""
        insider_name = transaction.get('insider_name', '')
        insider_txns = [t for t in all_transactions
                        if t.get('insider_name') == insider_name and t.get('value', 0) > 0]

        if len(insider_txns) <= 1:
            return 1.0  # No history to compare

        avg_value = sum(t['value'] for t in insider_txns) / len(insider_txns)
        if avg_value == 0:
            return 1.0

        return round(transaction.get('value', 0) / avg_value, 1)

    def _count_voluntary_purchases(self, transactions: List[Dict]) -> int:
        """Count voluntary open-market purchases."""
        return sum(1 for t in transactions if self._is_voluntary_purchase(t))

    def _get_highest_rank(self, transactions: List[Dict]) -> int:
        """Get the highest insider rank from transactions."""
        if not transactions:
            return 0
        return max(self._rank_insider(t.get('title', '')) for t in transactions)

    def _detect_patterns(self, transactions: List[Dict]) -> Dict:
        """
        Detect patterns in insider transactions.
        - Cluster buying: multiple unique insiders buying within 14 days
        - Large positions, executive buying, consistent direction
        """
        if not transactions:
            return {}

        patterns = {
            'cluster_buying': False,
            'cluster_selling': False,
            'cluster_buy_count': 0,
            'executive_buying': False,
            'unusual_size': False,
            'consistent_direction': False,
        }

        # Cluster detection: unique insiders within 14-day window
        recent_14d = [t for t in transactions
                      if (datetime.now() - datetime.fromisoformat(t['transaction_date'])).days <= 14]

        # Only count voluntary purchases for cluster buying
        voluntary_recent = [t for t in recent_14d if self._is_voluntary_purchase(t)]
        unique_buyers = set(t['insider_name'] for t in voluntary_recent)

        if len(unique_buyers) >= 2:
            patterns['cluster_buying'] = True
            patterns['cluster_buy_count'] = len(unique_buyers)

        # Cluster selling (any sales in 14 days)
        sales_recent = [t for t in recent_14d if t['transaction_type'] == 'Sale']
        unique_sellers = set(t['insider_name'] for t in sales_recent)
        if len(unique_sellers) >= 3:
            patterns['cluster_selling'] = True

        # Executive buying (CEO/CFO voluntary purchases)
        exec_purchases = [t for t in voluntary_recent
                          if self._rank_insider(t.get('title', '')) >= 4]
        if exec_purchases:
            patterns['executive_buying'] = True

        # Unusual size (any voluntary purchase > $1M)
        large_purchases = [t for t in voluntary_recent if t.get('value', 0) > 1_000_000]
        if large_purchases:
            patterns['unusual_size'] = True

        # Consistent direction (80%+ same direction, voluntary only)
        all_voluntary = [t for t in transactions if self._is_voluntary_purchase(t)]
        if len(all_voluntary) >= 3 and len(transactions) >= 3:
            purchase_ratio = len(all_voluntary) / len(transactions)
            if purchase_ratio > 0.8:
                patterns['consistent_direction'] = True

        return patterns

    def _build_fact_summary(self, ticker: str, summary: Dict, patterns: Dict,
                            voluntary_buys: List[Dict]) -> str:
        """Build a structured fact summary — no opinions, just data."""
        lines = []
        lines.append(f"Insider Activity for {ticker} (last {summary.get('days_analyzed', 90)} days):")
        lines.append(f"  Total transactions: {summary['transactions_count']} "
                      f"({summary.get('purchases_count', 0)} purchases, {summary.get('sales_count', 0)} sales)")
        lines.append(f"  Voluntary open-market purchases: {len(voluntary_buys)}")
        lines.append(f"  Net value: ${summary.get('net_value', 0):,.0f}")

        if patterns.get('cluster_buying'):
            lines.append(f"  CLUSTER BUYING: {patterns['cluster_buy_count']} unique insiders bought within 14 days")

        if patterns.get('executive_buying'):
            exec_buys = [t for t in voluntary_buys if self._rank_insider(t.get('title', '')) >= 4]
            names = [t.get('insider_name', 'Unknown') for t in exec_buys]
            lines.append(f"  EXECUTIVE BUYING: {', '.join(names)}")

        if patterns.get('unusual_size'):
            large = [t for t in voluntary_buys if t.get('value', 0) > 1_000_000]
            for t in large:
                lines.append(f"  LARGE PURCHASE: {t.get('insider_name', 'Unknown')} "
                              f"({t.get('title', 'Unknown')}) — ${t.get('value', 0):,.0f}")

        if voluntary_buys:
            lines.append("\n  Top voluntary purchases:")
            for t in sorted(voluntary_buys, key=lambda x: x.get('value', 0), reverse=True)[:5]:
                rel = t.get('relative_size', 1.0)
                rel_text = f" ({rel}x avg)" if rel > 1.5 else ""
                lines.append(f"    {t.get('insider_name', 'Unknown')} ({t.get('title', '')}) — "
                              f"${t.get('value', 0):,.0f}{rel_text} on {t.get('transaction_date', 'N/A')[:10]}")

        return '\n'.join(lines)

    def _get_perplexity_context(self, ticker: str, summary: Dict) -> Optional[str]:
        """Use Perplexity for factual context on insider activity."""
        try:
            signal = summary['net_signal']
            action = 'buying' if 'BULLISH' in signal else 'selling' if 'BEARISH' in signal else 'trading'

            system_prompt = "Du bist ein Finanz-Researcher. Berichte nur Fakten, keine Meinungen."

            query = (f"Was sind die aktuellen Nachrichten und Events fuer {ticker}? "
                     f"Insiders haben kuerzlich {action} Aktivitaet gezeigt. "
                     f"Gibt es Earnings, regulatorische Aenderungen oder andere Katalysatoren?")

            result = pplx_client._call_api(system_prompt, query, recency="week")
            return result

        except Exception as e:
            self.logger.error(f"Error getting Perplexity context: {e}")
            return None

    def _get_signal_emoji(self, signal: str) -> str:
        emoji_map = {
            'BULLISH': '+',
            'SLIGHTLY_BULLISH': '~',
            'NEUTRAL': '-',
            'SLIGHTLY_BEARISH': '~',
            'BEARISH': '!'
        }
        return emoji_map.get(signal, '-')

    def get_top_insider_signals(self, limit: int = 10, min_significance: int = 70) -> List[Dict]:
        """Get top insider trading signals — filtered for voluntary purchases only."""
        results = self.scan_watchlist_insiders(days_back=90)

        top_signals = []
        for result in results:
            if abs(result['signal_score']) < 30:
                continue

            most_recent = result['most_significant']
            txn_date = datetime.fromisoformat(most_recent['transaction_date'])
            days_old = (datetime.now() - txn_date).days

            if days_old > 90:
                continue

            # Only include if there are voluntary purchases
            if result.get('voluntary_purchases', 0) > 0 and most_recent['significance_score'] >= min_significance:
                top_signals.append(result)

        top_signals.sort(key=lambda x: (
            abs(x['signal_score']) + x['most_significant']['significance_score'] +
            x.get('voluntary_purchases', 0) * 10 +
            x.get('insider_rank', 0) * 5
        ), reverse=True)

        return top_signals[:limit]

    def generate_insider_alert(self, ticker: str, transaction: Dict) -> Dict:
        """Generate an alert for a significant insider transaction."""
        is_voluntary = self._is_voluntary_purchase(transaction)
        rank = self._rank_insider(transaction.get('title', ''))

        return {
            'type': 'INSIDER_ALERT',
            'ticker': ticker,
            'title': f"Insider {transaction['transaction_type']}: {ticker}",
            'message': (f"{transaction['insider_name']} ({transaction['title']}) "
                        f"{transaction['transaction_type'].lower()}d "
                        f"{transaction['shares']:,.0f} shares @ ${transaction['price']:.2f}"),
            'value': transaction['value'],
            'significance': transaction['significance_score'],
            'is_voluntary_purchase': is_voluntary,
            'insider_rank': rank,
            'timestamp': transaction['transaction_date'],
            'priority': 'HIGH' if (is_voluntary and rank >= 4) else
                        'MEDIUM' if is_voluntary else 'LOW'
        }


# Singleton
insider_tracker = InsiderTracker()
