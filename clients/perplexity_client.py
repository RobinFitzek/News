"""
Enhanced Perplexity Client for Investment Analysis
Uses Perplexity's internet access for real-time market intelligence.
Budget-aware: adapts daily request limits from monthly EUR budget.
"""
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from core.config import PERPLEXITY_API_KEY
from datetime import datetime
from typing import Dict, Optional
import json
import logging
import time


class EnhancedPerplexityClient:
    """Perplexity client optimized for financial intelligence with budget tracking"""

    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai"
        self.logger = logging.getLogger(__name__)
        self._budget_tracker = None

        # Configure session with retry logic
        self.session = self._create_session()

    @property
    def budget_tracker(self):
        if self._budget_tracker is None:
            from core.budget_tracker import budget_tracker
            self._budget_tracker = budget_tracker
        return self._budget_tracker

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic and connection pooling"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key and len(self.api_key) > 10)

    def get_usage(self) -> dict:
        """Get current usage stats from budget tracker"""
        daily_limit = self.budget_tracker.get_daily_request_limit('perplexity')
        today_count = self.budget_tracker.get_today_request_count('perplexity')
        status = self.budget_tracker.get_budget_status().get('perplexity', {})

        return {
            "used_today": today_count,
            "daily_limit": daily_limit,
            "remaining": max(0, daily_limit - today_count),
            "is_configured": self.is_configured(),
            "monthly_budget_eur": status.get('monthly_budget_eur', 5.0),
            "spent_eur": status.get('spent_eur', 0),
            "remaining_eur": status.get('remaining_eur', 5.0),
        }

    def _call_api(self, system_prompt: str, user_query: str,
                  domains: list = None, recency: str = "day") -> Optional[str]:
        """Base API call with budget-aware limiting"""
        # Validation checks
        if not self.is_configured():
            self.logger.warning("Perplexity API not configured")
            print("  Perplexity API not configured")
            return None

        if not system_prompt or not user_query:
            self.logger.error("Empty prompt or query provided")
            return None

        if not self.budget_tracker.can_afford_request('perplexity'):
            daily_limit = self.budget_tracker.get_daily_request_limit('perplexity')
            self.logger.warning(f"Daily budget exhausted (limit: {daily_limit})")
            print(f"  Perplexity daily budget exhausted (limit: {daily_limit} req/day)")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Default financial domains
        if domains is None:
            domains = [
                "finance.yahoo.com",
                "bloomberg.com",
                "reuters.com",
                "seekingalpha.com",
                "marketwatch.com"
            ]

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "search_domain_filter": domains,
            "search_recency_filter": recency
        }

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    self.logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                response_data = response.json()

                if 'choices' not in response_data or not response_data['choices']:
                    self.logger.error(f"Invalid response structure: {response_data}")
                    return None

                content = response_data['choices'][0]['message']['content']

                if not content:
                    self.logger.warning("Empty response content")
                    return None

                # Extract token usage and log cost
                usage = response_data.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 800)
                output_tokens = usage.get('completion_tokens', 400)

                cost = self.budget_tracker.log_cost(
                    'perplexity', 'sonar', input_tokens, output_tokens
                )

                today_count = self.budget_tracker.get_today_request_count('perplexity')
                daily_limit = self.budget_tracker.get_daily_request_limit('perplexity')
                self.logger.info(f"API call successful. Usage: {today_count}/{daily_limit} (${cost:.4f})")
                print(f"  Perplexity: {today_count}/{daily_limit} today (${cost:.4f})")

                # Clear any auth alert on success
                try:
                    from core.database import db
                    db.clear_system_alert('perplexity_auth')
                except Exception:
                    pass

                return content

            except requests.exceptions.Timeout:
                self.logger.error(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None

            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None
                self.logger.error(f"HTTP error {status_code}: {e}")

                if status_code == 401:
                    print("  Perplexity API key invalid")
                    try:
                        from core.database import db
                        db.raise_system_alert(
                            'perplexity_auth',
                            'Perplexity API Key Invalid',
                            'The Perplexity API key is rejected (401 Unauthorized). News analysis is disabled until you update the key.',
                            severity='error', service='perplexity',
                            action_url='/settings', action_label='Update API Key')
                    except Exception:
                        pass
                elif status_code == 403:
                    print("  Perplexity API access forbidden")
                    try:
                        from core.database import db
                        db.raise_system_alert(
                            'perplexity_auth',
                            'Perplexity API Access Forbidden',
                            'The Perplexity API returned 403 Forbidden. Check your plan or API key permissions.',
                            severity='error', service='perplexity',
                            action_url='/settings', action_label='Check API Key')
                    except Exception:
                        pass
                elif status_code and status_code >= 500:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                return None

            except json.JSONDecodeError:
                self.logger.error("JSON decode error")
                return None

            except Exception as e:
                self.logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None

        self.logger.error(f"All {max_retries} attempts failed")
        return None

    # ========== STRUCTURED INTELLIGENCE QUERIES ==========

    def get_breaking_news(self, ticker: str) -> Dict:
        """Get breaking news with structured sentiment analysis"""
        system_prompt = """Du bist ein Echtzeit-Finanzanalyst mit Internet-Zugang.
Analysiere die aktuellsten Nachrichten und gib strukturierte Daten zurück.
Antworte NUR im folgenden JSON-kompatiblen Format:

SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
NEWS_SCORE: [1-10] (1=sehr negativ, 10=sehr positiv)
BREAKING: [Ja/Nein]
HEADLINES:
- [Headline 1]
- [Headline 2]
- [Headline 3]
KEY_EVENTS:
- [Event 1 mit Datum]
- [Event 2 mit Datum]
MARKET_REACTION: [Kurze Beschreibung der Marktreaktion]
CATALYST_ALERT: [Bevorstehende wichtige Events wie Earnings, FDA-Entscheidungen etc.]
"""
        query = f"""Aktuelle Nachrichten für {ticker} Aktie in den letzten 24-48 Stunden.
Fokus auf: Kursbewegungen, Analystenratings, Unternehmensnews, Sektortrends.
Gib nur verifizierte, faktische Informationen."""

        result = self._call_api(system_prompt, query, recency="day")
        return {"ticker": ticker, "raw": result, "type": "breaking_news"}

    def get_market_sentiment(self, ticker: str) -> Dict:
        """Get current market sentiment and analyst opinions"""
        system_prompt = """Du bist ein Sentiment-Analyst mit Echtzeit-Internet-Zugang.
Analysiere die aktuelle Marktstimmung und gib strukturierte Daten zurück.

Format EXAKT so:
OVERALL_SENTIMENT: [SEHR_BULLISH/BULLISH/NEUTRAL/BEARISH/SEHR_BEARISH]
SENTIMENT_SCORE: [1-100] (1=extrem negativ, 100=extrem positiv)
ANALYST_CONSENSUS: [Strong Buy/Buy/Hold/Sell/Strong Sell]
PRICE_TARGETS:
- Hoch: $[Preis]
- Durchschnitt: $[Preis]
- Tief: $[Preis]
RECENT_RATINGS:
- [Analyst/Bank]: [Upgrade/Downgrade/Reiterate] zu [Rating] (Datum)
SOCIAL_BUZZ: [Hoch/Mittel/Niedrig]
INSTITUTIONAL_ACTIVITY: [Beschreibung aktueller Insiderkäufe/-verkäufe]
"""
        query = f"""Aktuelle Marktstimmung und Analystenratings für {ticker}.
Inkludiere: Kursziele, aktuelle Upgrades/Downgrades, institutionelle Aktivität."""

        result = self._call_api(system_prompt, query)
        return {"ticker": ticker, "raw": result, "type": "sentiment"}

    def get_sector_intelligence(self, sector: str) -> Dict:
        """Get sector-wide market intelligence"""
        system_prompt = """Du bist ein Sektor-Analyst mit Internet-Zugang.
Gib einen strukturierten Sektor-Report:

SECTOR_TREND: [STARK_AUFWÄRTS/AUFWÄRTS/SEITWÄRTS/ABWÄRTS/STARK_ABWÄRTS]
SECTOR_SCORE: [1-10]
TOP_PERFORMERS: [Ticker1, Ticker2, Ticker3]
LAGGARDS: [Ticker1, Ticker2, Ticker3]
KEY_DRIVERS:
- [Treiber 1]
- [Treiber 2]
RISKS:
- [Risiko 1]
- [Risiko 2]
UPCOMING_CATALYSTS:
- [Catalyst mit Datum]
RECOMMENDATION: [Übergewichten/Neutral/Untergewichten]
"""
        query = f"""Aktueller Status des {sector} Sektors.
Trend, Top-Performer, Risiken, und bevorstehende Ereignisse."""

        result = self._call_api(system_prompt, query, recency="week")
        return {"sector": sector, "raw": result, "type": "sector_intel"}

    def get_risk_scan(self, ticker: str) -> Dict:
        """Scan for potential risks and red flags"""
        system_prompt = """Du bist ein Risiko-Analyst mit Internet-Zugang.
Scanne nach aktuellen Risiken und Warnzeichen:

RISK_LEVEL: [NIEDRIG/MITTEL/HOCH/KRITISCH]
RISK_SCORE: [1-10] (10=höchstes Risiko)
RED_FLAGS:
- [Flag 1 mit Quelle]
- [Flag 2 mit Quelle]
REGULATORY_RISKS: [Beschreibung oder "Keine bekannt"]
LITIGATION_RISKS: [Beschreibung oder "Keine bekannt"]
COMPETITIVE_THREATS: [Beschreibung]
FINANCIAL_CONCERNS: [Beschreibung oder "Keine bekannt"]
SHORT_INTEREST: [Hoch/Mittel/Niedrig] ([Prozent]%)
INSIDER_ACTIVITY: [Käufe/Verkäufe/Neutral]
OVERALL_ASSESSMENT: [Kurzfassung]
"""
        query = f"""Risiko-Scan für {ticker}:
Aktuelle Risiken, rote Flaggen, regulatorische Probleme, Rechtsstreitigkeiten,
Wettbewerbsdruck, finanzielle Bedenken, Short-Interest, Insider-Aktivität."""

        result = self._call_api(system_prompt, query)
        return {"ticker": ticker, "raw": result, "type": "risk_scan"}

    def get_earnings_preview(self, ticker: str) -> Dict:
        """Get earnings preview and expectations"""
        system_prompt = """Du bist ein Earnings-Analyst mit Internet-Zugang.
Gib einen strukturierten Earnings-Report:

NEXT_EARNINGS_DATE: [Datum oder "TBD"]
EARNINGS_STATUS: [Bevorstehend/Gerade veröffentlicht/Keine Info]
EPS_ESTIMATE: $[Betrag]
REVENUE_ESTIMATE: $[Betrag]
WHISPER_NUMBER: [Falls bekannt]
HISTORY:
- Letzte 4 Quartale: [Beat/Miss/Inline] mit [+/-X%]
EXPECTATIONS: [Hoch/Niedrig]
KEY_METRICS_TO_WATCH:
- [Metrik 1]
- [Metrik 2]
POTENTIAL_MOVERS: [Was könnte den Kurs bewegen]
"""
        query = f"""Earnings-Vorschau für {ticker}:
Nächstes Earnings-Datum, Erwartungen, historische Performance,
wichtige Metriken, potenzielle Kursbeweger."""

        result = self._call_api(system_prompt, query)
        return {"ticker": ticker, "raw": result, "type": "earnings"}

    def get_competitive_landscape(self, ticker: str) -> Dict:
        """Analyze competitive position"""
        system_prompt = """Du bist ein Wettbewerbsanalyst mit Internet-Zugang.
Analysiere die Wettbewerbsposition:

MARKET_POSITION: [Leader/Challenger/Follower/Niche]
MARKET_SHARE: [X%] in [Markt]
MAIN_COMPETITORS:
- [Competitor 1]: [Stärken/Schwächen]
- [Competitor 2]: [Stärken/Schwächen]
COMPETITIVE_ADVANTAGE: [Beschreibung]
THREATS:
- [Bedrohung 1]
- [Bedrohung 2]
MOAT_STRENGTH: [Stark/Mittel/Schwach]
RECENT_COMPETITIVE_NEWS: [Aktuelle Entwicklungen]
"""
        query = f"""Wettbewerbsanalyse für {ticker}:
Marktposition, Hauptkonkurrenten, Wettbewerbsvorteile, Bedrohungen."""

        result = self._call_api(system_prompt, query, recency="week")
        return {"ticker": ticker, "raw": result, "type": "competitive"}

    # ========== INTEGRATED ANALYSIS METHODS ==========

    def full_intelligence_scan(self, ticker: str) -> Dict:
        """Complete intelligence scan combining multiple queries (uses 3 API calls)"""
        print(f"  Perplexity Full Intelligence Scan for {ticker}...")

        news = self.get_breaking_news(ticker)
        sentiment = self.get_market_sentiment(ticker)
        risks = self.get_risk_scan(ticker)

        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "news": news,
            "sentiment": sentiment,
            "risks": risks,
            "api_calls_used": 3
        }

    def quick_scan(self, ticker: str) -> Dict:
        """Quick single-call intelligence scan optimized for daily cycles"""
        system_prompt = """Du bist ein Elite-Finanzanalyst mit Echtzeit-Internet-Zugang.
Führe einen schnellen aber umfassenden Scan durch.

Format EXAKT so:
NEWS_SUMMARY: [2-3 Sätze zu aktuellen Entwicklungen]
SENTIMENT: [BULLISH/NEUTRAL/BEARISH]
RISK_FLAG: [Ja/Nein - kurze Begründung]
ANALYST_TREND: [Upgrades/Downgrades/Stabil]
UPCOMING: [Nächstes wichtiges Event mit Datum]
QUICK_TAKE: [1 Satz Investment-Einschätzung]
"""
        query = f"""Schneller Intelligenz-Scan für {ticker}:
Aktuelle News, Marktstimmung, Risiken, Analystentrend, kommende Events."""

        result = self._call_api(system_prompt, query)
        return {"ticker": ticker, "raw": result, "type": "quick_scan"}

    def discover_trending_stocks(self, sector: str = None, focus: str = "balanced", limit: int = 5) -> Dict:
        """Discover new interesting stocks using Perplexity's real-time internet access"""
        focus_guidance = {
            "growth": "hochgradig wachstumsstarke Unternehmen mit disruptivem Potenzial, starkem Umsatzwachstum und Marktführerschaft",
            "value": "unterbewertete Quality-Aktien mit soliden Fundamentaldaten, niedrigem KGV/KBV und Aufholpotenzial",
            "dividend": "dividendenstarke Unternehmen mit stabilen Ausschüttungen, solider Dividendenhistorie und nachhaltigem Cashflow",
            "balanced": "ausgewogene Mischung aus Wachstum, Bewertung und Qualität"
        }

        focus_desc = focus_guidance.get(focus, focus_guidance["balanced"])
        sector_filter = f" im {sector} Sektor" if sector else " über alle Sektoren"

        system_prompt = f"""Du bist ein Elite Stock-Scout mit Echtzeit-Internet-Zugang zu Finanzmärkten.
Deine Aufgabe: Finde die interessantesten Aktien basierend auf aktuellen Markttrends, News und Momentum.

FOKUS: {focus_desc}
SEKTOR: {sector_filter}

Analysiere:
- Aktuelle Markttrends und Momentum
- Breaking News und Katalysatoren
- Analystenmeinungen und Upgrades
- Technische Ausbrüche und Volumenspitzen
- Sektorrotation und makroökonomische Treiber

WICHTIG: Antworte mit einer detaillierten Analyse UND am Ende mit dieser EXAKTEN Struktur:

RECOMMENDED_STOCKS:
- TICKER: [Kurzer Grund] | Score: [1-100] | Catalyst: [Nächster wichtiger Event]
- TICKER: [Kurzer Grund] | Score: [1-100] | Catalyst: [Nächster wichtiger Event]
...

Gib {min(limit, 10)} konkrete Empfehlungen."""

        query = f"""Welche Aktien sind JETZT besonders interessant?
Zeitraum: Aktuelle Marktlage (heute/diese Woche)
Fokus: {focus}
Sektor: {sector if sector else 'Alle Sektoren'}

Berücksichtige:
1. Aktuelle News und Katalysatoren (letzte 24-48h)
2. Momentum und Volumen
3. Analystenmeinungen und Upgrades
4. Technische Setups (Ausbrüche, Unterstützungen)
5. Makro-Trends und Sektorrotation

Nenne {min(limit, 10)} Top-Picks mit klarer Begründung."""

        result = self._call_api(system_prompt, query, recency="day")

        if not result:
            return {
                "success": False,
                "error": "API call failed",
                "stocks": []
            }

        parsed_stocks = self._parse_stock_recommendations(result, limit)

        return {
            "success": True,
            "sector": sector,
            "focus": focus,
            "raw_analysis": result,
            "stocks": parsed_stocks,
            "timestamp": datetime.now().isoformat(),
            "type": "stock_discovery"
        }

    def _parse_stock_recommendations(self, raw_text: str, limit: int) -> list:
        """Parse structured stock recommendations from Perplexity response"""
        import re

        stocks = []

        if "RECOMMENDED_STOCKS:" not in raw_text:
            ticker_pattern = r'\b([A-Z]{1,5})\b'
            potential_tickers = re.findall(ticker_pattern, raw_text)
            common_words = {'THE', 'A', 'AN', 'AND', 'OR', 'FOR', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY', 'WITH'}
            valid_tickers = [t for t in potential_tickers if t not in common_words and len(t) <= 5]

            seen = set()
            for ticker in valid_tickers:
                if ticker not in seen and len(stocks) < limit:
                    stocks.append({
                        'ticker': ticker,
                        'reason': 'Mentioned in analysis',
                        'score': 50,
                        'catalyst': 'See full analysis',
                        'confidence': 'medium'
                    })
                    seen.add(ticker)

            return stocks

        rec_section = raw_text.split("RECOMMENDED_STOCKS:")[1]

        pattern = r'-\s*([A-Z]{1,5})\s*:\s*\[?([^\]|]+)\]?\s*\|\s*Score:\s*(\d+)\s*\|\s*Catalyst:\s*(.+?)(?=\n-|\n\n|$)'
        matches = re.findall(pattern, rec_section, re.MULTILINE | re.DOTALL)

        for match in matches[:limit]:
            ticker, reason, score, catalyst = match
            score = int(score)

            if score >= 80:
                confidence = 'high'
            elif score >= 60:
                confidence = 'medium'
            else:
                confidence = 'low'

            stocks.append({
                'ticker': ticker.strip(),
                'reason': reason.strip(),
                'score': max(0, min(100, score)),
                'catalyst': catalyst.strip(),
                'confidence': confidence
            })

        return stocks


# Singleton
pplx_client = EnhancedPerplexityClient()
