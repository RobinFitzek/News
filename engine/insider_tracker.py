"""
Insider Trading Tracker Engine
Analyzes insider transactions and generates trading signals
"""
from clients.sec_edgar_client import sec_client
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
from core.database import db
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging


class InsiderTracker:
    """Tracks and analyzes insider trading activity"""

    def __init__(self):
        self.sec = sec_client
        self.logger = logging.getLogger(__name__)

    def scan_watchlist_insiders(self, days_back: int = 90) -> List[Dict]:
        """
        Scan entire watchlist for insider activity

        Returns:
            List of stocks with notable insider activity
        """
        watchlist = db.get_watchlist(active_only=True)
        results = []

        print(f"\n🔍 Scanning {len(watchlist)} watchlist stocks for insider activity...")

        for item in watchlist:
            ticker = item['ticker']

            try:
                print(f"  Checking {ticker}...", end=' ')
                summary = self.sec.get_insider_summary(ticker, days_back)

                # Only include if there's actual activity
                if summary['transactions_count'] > 0:
                    results.append(summary)
                    signal_emoji = self._get_signal_emoji(summary['net_signal'])
                    print(f"{signal_emoji} {summary['net_signal']} ({summary['transactions_count']} txns)")
                else:
                    print("No activity")

            except Exception as e:
                self.logger.error(f"Error scanning {ticker}: {e}")
                print(f"❌ Error")

        # Sort by signal strength
        results.sort(key=lambda x: abs(x['signal_score']), reverse=True)

        print(f"\n✅ Found insider activity in {len(results)} stocks")
        return results

    def get_insider_analysis(self, ticker: str, days_back: int = 180) -> Dict:
        """
        Get comprehensive insider analysis for a single stock

        Includes:
        - Raw transactions from SEC
        - Signal calculation
        - AI-powered context from Perplexity (optional)
        - Gemini analysis of patterns
        """
        print(f"\n🔍 Analyzing insider activity for {ticker}...")

        # Get SEC data
        transactions = self.sec.get_insider_transactions(ticker, days_back)
        summary = self.sec.get_insider_summary(ticker, days_back)

        if not transactions:
            return {
                'ticker': ticker,
                'has_activity': False,
                'message': 'No insider transactions found in the specified period'
            }

        # Analyze patterns
        patterns = self._detect_patterns(transactions)

        # Get AI context if Perplexity available
        ai_context = None
        if pplx_client.is_configured():
            usage = pplx_client.get_usage()
            if usage['remaining'] > 0:
                ai_context = self._get_perplexity_context(ticker, summary)

        # Get Gemini interpretation
        gemini_analysis = self._get_gemini_analysis(ticker, summary, patterns)

        return {
            'ticker': ticker,
            'has_activity': True,
            'summary': summary,
            'transactions': transactions,
            'patterns': patterns,
            'ai_context': ai_context,
            'gemini_analysis': gemini_analysis,
            'timestamp': datetime.now().isoformat()
        }

    def _detect_patterns(self, transactions: List[Dict]) -> Dict:
        """
        Detect patterns in insider transactions

        Patterns:
        - Cluster buying/selling (multiple insiders at once)
        - Unusual timing (before earnings, etc.)
        - Large positions
        - Consistent direction
        """
        if not transactions:
            return {}

        patterns = {
            'cluster_buying': False,
            'cluster_selling': False,
            'executive_buying': False,
            'unusual_size': False,
            'consistent_direction': False
        }

        # Cluster detection (3+ transactions within 30 days)
        recent_30d = [t for t in transactions
                     if (datetime.now() - datetime.fromisoformat(t['transaction_date'])).days <= 30]

        purchases = [t for t in recent_30d if t['transaction_type'] == 'Purchase']
        sales = [t for t in recent_30d if t['transaction_type'] == 'Sale']

        if len(purchases) >= 3:
            patterns['cluster_buying'] = True

        if len(sales) >= 3:
            patterns['cluster_selling'] = True

        # Executive buying (CEO/CFO)
        exec_purchases = [t for t in purchases
                         if any(x in t['title'].lower() for x in ['ceo', 'cfo', 'president'])]
        if exec_purchases:
            patterns['executive_buying'] = True

        # Unusual size (>$1M)
        large_purchases = [t for t in purchases if t['value'] > 1_000_000]
        if large_purchases:
            patterns['unusual_size'] = True

        # Consistent direction (80%+ same direction)
        if len(transactions) >= 3:
            purchase_ratio = len(purchases) / len(transactions)
            if purchase_ratio > 0.8 or purchase_ratio < 0.2:
                patterns['consistent_direction'] = True

        return patterns

    def _get_perplexity_context(self, ticker: str, summary: Dict) -> Optional[str]:
        """
        Use Perplexity to get context on insider activity

        Ask: "Why are insiders buying/selling [ticker] recently?"
        """
        try:
            signal = summary['net_signal']
            action = 'buying' if 'BULLISH' in signal else 'selling' if 'BEARISH' in signal else 'trading'

            system_prompt = f"""Du bist ein Insider-Trading-Analyst.
Erkläre den Kontext von Insider-Transaktionen basierend auf aktuellen News und Unternehmensentwicklungen.

Format:
CONTEXT: [Warum handeln Insider jetzt?]
CATALYST: [Gibt es bevorstehende Events?]
INTERPRETATION: [Bullish/Bearish/Neutral und warum]
"""

            query = f"""Insider Trading Kontext für {ticker}:
- Insiders haben in den letzten 90 Tagen {summary['transactions_count']} Transaktionen getätigt
- Net Signal: {signal}
- Gesamtwert: ${summary.get('net_value', 0):,.0f}

Frage: Warum {action} Insiders von {ticker} gerade jetzt?
Gibt es aktuelle News, Earnings, oder andere Katalysatoren die dies erklären?"""

            result = pplx_client._call_api(system_prompt, query, recency="week")
            return result

        except Exception as e:
            self.logger.error(f"Error getting Perplexity context: {e}")
            return None

    def _get_gemini_analysis(self, ticker: str, summary: Dict, patterns: Dict) -> str:
        """
        Use Gemini to interpret insider activity

        Provides quick, actionable analysis
        """
        try:
            pattern_desc = self._describe_patterns(patterns)

            prompt = f"""Analysiere diese Insider-Trading-Daten für {ticker}:

Signal: {summary['net_signal']} (Score: {summary['signal_score']})
Transaktionen: {summary['transactions_count']} (Käufe: {summary['purchases_count']}, Verkäufe: {summary['sales_count']})
Netto-Wert: ${summary.get('net_value', 0):,.0f}
Muster: {pattern_desc}

Wichtigste Transaktion:
- {summary['most_significant']['insider_name']} ({summary['most_significant']['title']})
- {summary['most_significant']['transaction_type']}: {summary['most_significant']['shares']:,.0f} Aktien @ ${summary['most_significant']['price']:.2f}
- Wert: ${summary['most_significant']['value']:,.0f}
- Datum: {summary['most_significant']['transaction_date']}

Gib eine kurze, prägnante Interpretation (2-3 Sätze):
1. Was bedeutet dies für die Aktie?
2. Wie stark ist das Signal?
3. Was sollten Investoren beachten?

Halte es kurz und actionable."""

            analysis = gemini_client.generate(prompt, tier='flash')
            return analysis

        except Exception as e:
            self.logger.error(f"Error getting Gemini analysis: {e}")
            return "Analyse nicht verfügbar"

    def _describe_patterns(self, patterns: Dict) -> str:
        """Convert patterns dict to readable string"""
        active_patterns = [k.replace('_', ' ').title() for k, v in patterns.items() if v]
        return ', '.join(active_patterns) if active_patterns else 'Keine besonderen Muster'

    def _get_signal_emoji(self, signal: str) -> str:
        """Get emoji for signal strength"""
        emoji_map = {
            'BULLISH': '🟢',
            'SLIGHTLY_BULLISH': '🟡',
            'NEUTRAL': '⚪',
            'SLIGHTLY_BEARISH': '🟠',
            'BEARISH': '🔴'
        }
        return emoji_map.get(signal, '⚪')

    def get_top_insider_signals(self, limit: int = 10, min_significance: int = 70) -> List[Dict]:
        """
        Get top insider trading signals across watchlist

        Filters for:
        - High significance scores
        - Recent activity (last 30 days)
        - Strong directional signals
        """
        results = self.scan_watchlist_insiders(days_back=90)

        # Filter and score
        top_signals = []
        for result in results:
            # Must have strong signal
            if abs(result['signal_score']) < 30:
                continue

            # Must have recent activity
            most_recent = result['most_significant']
            txn_date = datetime.fromisoformat(most_recent['transaction_date'])
            days_old = (datetime.now() - txn_date).days

            if days_old > 90:
                continue

            # Check significance
            if most_recent['significance_score'] >= min_significance:
                top_signals.append(result)

        # Sort by combined score
        top_signals.sort(key=lambda x: (
            abs(x['signal_score']) + x['most_significant']['significance_score']
        ), reverse=True)

        return top_signals[:limit]

    def generate_insider_alert(self, ticker: str, transaction: Dict) -> Dict:
        """
        Generate an alert for a significant insider transaction

        Used for real-time monitoring
        """
        return {
            'type': 'INSIDER_ALERT',
            'ticker': ticker,
            'title': f"Insider {transaction['transaction_type']}: {ticker}",
            'message': f"{transaction['insider_name']} ({transaction['title']}) "
                      f"{transaction['transaction_type'].lower()}d "
                      f"{transaction['shares']:,.0f} shares @ ${transaction['price']:.2f}",
            'value': transaction['value'],
            'significance': transaction['significance_score'],
            'timestamp': transaction['transaction_date'],
            'priority': 'HIGH' if transaction['significance_score'] >= 80 else 'MEDIUM'
        }


# Singleton
insider_tracker = InsiderTracker()
