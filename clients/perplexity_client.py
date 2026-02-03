"""
Enhanced Perplexity Client for Investment Analysis
Uses Perplexity's internet access for real-time market intelligence.
"""
import requests
from core.config import PERPLEXITY_API_KEY
from datetime import datetime
from typing import Dict, Optional
import json


class EnhancedPerplexityClient:
    """Perplexity client optimized for financial intelligence with structured outputs"""
    
    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai"
        self.requests_used_today = 0
        self.daily_limit = 33  # ~$5/month budget
        self.last_reset_date = datetime.now().date()
    
    def _check_daily_reset(self):
        """Reset counter if new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.requests_used_today = 0
            self.last_reset_date = today
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key and len(self.api_key) > 10)
    
    def get_usage(self) -> dict:
        """Get current usage stats"""
        self._check_daily_reset()
        return {
            "used_today": self.requests_used_today,
            "daily_limit": self.daily_limit,
            "remaining": self.daily_limit - self.requests_used_today,
            "is_configured": self.is_configured()
        }
    
    def _call_api(self, system_prompt: str, user_query: str, 
                  domains: list = None, recency: str = "day") -> Optional[str]:
        """Base API call with structured prompts"""
        self._check_daily_reset()
        
        if not self.is_configured():
            print("⚠️ Perplexity API not configured")
            return None
            
        if self.requests_used_today >= self.daily_limit:
            print(f"⚠️ Perplexity daily limit reached ({self.daily_limit})")
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
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            self.requests_used_today += 1
            print(f"✅ Perplexity: {self.requests_used_today}/{self.daily_limit} today")
            
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"❌ Perplexity API Error: {e}")
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
        print(f"🌐 Perplexity Full Intelligence Scan for {ticker}...")
        
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
📰 NEWS_SUMMARY: [2-3 Sätze zu aktuellen Entwicklungen]
📊 SENTIMENT: [BULLISH/NEUTRAL/BEARISH]
⚠️ RISK_FLAG: [Ja/Nein - kurze Begründung]
🎯 ANALYST_TREND: [Upgrades/Downgrades/Stabil]
📅 UPCOMING: [Nächstes wichtiges Event mit Datum]
💡 QUICK_TAKE: [1 Satz Investment-Einschätzung]
"""
        query = f"""Schneller Intelligenz-Scan für {ticker}:
Aktuelle News, Marktstimmung, Risiken, Analystentrend, kommende Events."""
        
        result = self._call_api(system_prompt, query)
        return {"ticker": ticker, "raw": result, "type": "quick_scan"}


# Singleton
pplx_client = EnhancedPerplexityClient()
