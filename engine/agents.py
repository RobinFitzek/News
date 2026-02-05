"""
Investment Analysis Agents
Multi-agent system for stock analysis using Perplexity and Gemini APIs.
"""
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
import yfinance as yf
from core.database import db
import logging
from typing import Dict, Optional

class InvestmentSwarm:
    """Multi-Agent System for automated investment analysis"""

    def __init__(self):
        self.pplx = pplx_client
        self.gemini = gemini_client
        self.logger = logging.getLogger(__name__)
        self._reload_api_keys()
    
    def _reload_api_keys(self):
        """Reload API keys from database"""
        pplx_key = db.get_api_key("perplexity")
        gemini_key = db.get_api_key("gemini")
        
        if pplx_key:
            self.pplx.api_key = pplx_key
        if gemini_key:
            self.gemini.reload_api_key(gemini_key)

    def analyze_single_stock(self, ticker: str, strategy: str = "balanced") -> dict:
        """Manual full analysis for a single stock (Stage 2 + Stage 3)"""
        print(f"🔧 Manual Analysis for {ticker}...")
        stock_data = self._get_stock_data(ticker)
        
        # Fake Stage 1 result to pass to Stage 2
        candidate = {
            'ticker': ticker,
            'score': 100, # Manual override
            'initial_reason': "Manually requested",
            'data': stock_data
        }
        
        # Run Stage 2
        stage2_results = self.stage2_analyze(candidate, strategy)
        candidate.update(stage2_results)
        
        # Run Stage 3
        final_result = self.stage3_synthesize(candidate, strategy)
        
        return final_result
    
    def stage1_scan(self, tickers: list, variant: str = "balanced") -> list:
        """Stage 1: Flash-8b Quick Scan with robust error handling"""
        if not tickers:
            self.logger.warning("Empty ticker list provided to stage1_scan")
            return []

        print(f"\n🚀 STAGE 1: Scanning {len(tickers)} tickers with Flash-8b (Variant: {variant})")
        candidates = []
        errors = []

        for ticker in tickers:
            try:
                # Validate ticker
                if not ticker or not isinstance(ticker, str):
                    self.logger.warning(f"Invalid ticker: {ticker}")
                    continue

                ticker = ticker.upper().strip()

                # Get stock data with error handling
                stock_data = self._get_stock_data(ticker)

                # Skip if data fetch failed
                if 'error' in stock_data:
                    self.logger.warning(f"Skipping {ticker} due to data error: {stock_data['error']}")
                    errors.append({'ticker': ticker, 'error': stock_data['error']})
                    continue

                # Variant-specific nuances
                variant_prompt = ""
                if variant == "conservative":
                    variant_prompt = "Fokus: Maximale Sicherheit, Blue-Chips, stabile Dividenden. Sei extrem kritisch."
                elif variant == "aggressive":
                    variant_prompt = "Fokus: High-Reward, Small-Caps, Turnaround-Chancen. Akzeptiere hohes Risiko für explosives Wachstum."
                else:  # balanced
                    variant_prompt = "Fokus: Ausgewogenes Verhältnis. Qualität zu fairem Preis."

                prompt = f"""
                Quick Scan für {ticker}:
                Daten: {stock_data}
                {variant_prompt}

                Gib einen 'Interest Score' von 0-100 basierend auf der Strategie.
                0 = Uninteressant / Hohes Risiko
                100 = Top Kandidat / Passt perfekt

                Format genau so: "Score: [0-100] | Grund: [Kurz]"
                """

                # Call Gemini API with error handling
                response = self.gemini.generate(prompt, tier='flash-8b')

                # Check if response is an error message
                if response.startswith("⚠️") or response.startswith("❌"):
                    self.logger.warning(f"API error for {ticker}: {response}")
                    errors.append({'ticker': ticker, 'error': response})
                    continue

                # Parse score
                import re
                score_match = re.search(r"Score:\s*(\d+)", response)
                score = int(score_match.group(1)) if score_match else 0

                # Validate score range
                score = max(0, min(100, score))

                print(f"  👉 {ticker}: {score} - {response.split('|')[-1].strip()[:50] if '|' in response else 'N/A'}...")

                candidates.append({
                    'ticker': ticker,
                    'score': score,
                    'initial_reason': response,
                    'data': stock_data
                })

            except ValueError as e:
                self.logger.error(f"Value error scanning {ticker}: {e}")
                errors.append({'ticker': ticker, 'error': str(e)})

            except Exception as e:
                self.logger.error(f"Unexpected error scanning {ticker}: {e}", exc_info=True)
                errors.append({'ticker': ticker, 'error': str(e)})
                print(f"⚠️ Scan error {ticker}: {e}")

        # Log summary
        self.logger.info(f"Stage 1 complete: {len(candidates)} successful, {len(errors)} errors")
        if errors:
            self.logger.warning(f"Errors encountered: {errors}")

        # Sort by score descending
        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def stage2_analyze(self, candidate: dict, variant: str = "balanced") -> dict:
        """Stage 2: Flash Deep Dive (Fundamental + Technical)"""
        ticker = candidate['ticker']
        print(f"\n🔬 STAGE 2: Deep Dive Analysis for {ticker} (Flash)")
        
        results = {
            'ticker': ticker, 
            'stage1_score': candidate['score'],
            'stage1_reason': candidate['initial_reason']
        }
        stock_data = candidate['data']
        
        # Fundamental Analysis
        fund_focus = "finanzielle Stabilität und faire Bewertung"
        if variant == "aggressive": fund_focus = "Umsatzwachstum, Cash-Burn und Marktpotential"
        if variant == "conservative": fund_focus = "Schuldendeckung, Free Cashflow und Dividendenkontinuität"
        
        fund_prompt = f"""
        Analysiere {ticker} fundamental. Fokus: {fund_focus}.
        Daten: {stock_data}
        
        Bewerte: KGV/KBV, Wachstum, Schulden.
        Kurz und prägnant auf Deutsch.
        """
        results['fundamental'] = self.gemini.generate(fund_prompt, tier='flash')
        
        # Technical Analysis
        tech_focus = "Trendbestätigung und gleitende Durchschnitte"
        if variant == "aggressive": tech_focus = "Volatilität, RSI-Extrema und Ausbruchsmuster"
        if variant == "conservative": tech_focus = "Langfristige Unterstützungen und Trendstabilität"

        tech_prompt = f"""
        Analysiere {ticker} technisch. Fokus: {tech_focus}.
        Kursverlauf: {stock_data.get('price_history', 'N/A')}

        Erkenne Trends und Support/Resistance.
        Kurz und prägnant auf Deutsch.
        """
        results['technical'] = self.gemini.generate(tech_prompt, tier='flash')

        # Market Insights & News (Perplexity) - PRIMARY SOURCE
        if self.pplx.is_configured():
            print(f"  📰 Fetching real-time market insights for {ticker}...")
            news_scan = self.pplx.quick_scan(ticker)
            results['news'] = news_scan['raw'] if news_scan and news_scan.get('raw') else "No recent news or market insights available"
        else:
            results['news'] = "⚠️ Perplexity API not configured - market insights unavailable"

        return results

    def stage3_synthesize(self, analysis_result: dict, variant: str = "balanced") -> dict:
        """Stage 3: Pro Final Verdict (Detailed Risk Assessment)"""
        ticker = analysis_result['ticker']
        print(f"\n⚖️ STAGE 3: Final Recommendation for {ticker} (Pro)")
        
        # Variant Context
        var_desc = "Ausgewogen"
        if variant == "conservative": var_desc = "Konservativ (Sicherheit & Werterhalt)"
        if variant == "aggressive": var_desc = "Aggressiv (Wachstum & Spekulation)"
        
        prompt = f"""
        Handle als Senior-Portfoliomanager. Analysevariante: {var_desc}.
        Synthetisiere die Analysen für {ticker}:
        
        Fundamental: {analysis_result['fundamental']}
        Technisch: {analysis_result['technical']}
        Quick Scan: {analysis_result['stage1_reason']}
        
        Aufgabe:
        1. Gib eine glasklare Handels-Empfehlung (Strong Buy, Buy, Hold, Sell, Strong Sell).
        2. Führe eine detaillierte Risiko-Bewertung durch.
        3. Bestimme den Risk Score (1-10), wobei 1 = Sicher, 10 = Totalverlustrisiko.
        4. Nenne Kursziele (Base/Bull/Bear Case) falls möglich.
        
        Format ZWINGEND:
        Signal: [SIGNAL]
        Risk Score: [1-10]
        Risk Level: [Low/Medium/High/Extreme]
        Confidence: [0-100]%
        Kursziel (12m): [Preis oder N/A]
        Begründung: [Detaillierte Analyse auf Deutsch]
        Risiko-Faktoren: [Konkrete Gefahren auf Deutsch]
        """
        
        response = self.gemini.generate(prompt, tier='pro')
        analysis_result['recommendation'] = response
        
        # Parse Risk Score & Signal for DB
        import re
        risk_match = re.search(r"Risk Score:\s*(\d+)", response)
        analysis_result['risk_score'] = int(risk_match.group(1)) if risk_match else 5
        
        print(f"  🏁 {ticker}: {response.split(chr(10))[0]} (Risk: {analysis_result['risk_score']})")
        
        return analysis_result

    def _get_stock_data(self, ticker: str) -> dict:
        """Get stock data from yfinance with comprehensive error handling"""
        if not ticker or not isinstance(ticker, str):
            self.logger.error("Invalid ticker provided")
            return {'error': 'Invalid ticker', 'ticker': ticker}

        ticker = ticker.upper().strip()

        try:
            # Set timeout for yfinance requests
            stock = yf.Ticker(ticker)

            # Get stock info with error handling
            try:
                info = stock.info
                if not info or len(info) < 3:  # yfinance returns minimal dict on error
                    self.logger.warning(f"Minimal or no data returned for {ticker}")
                    return {
                        'ticker': ticker,
                        'error': 'No data available',
                        'name': ticker
                    }
            except Exception as e:
                self.logger.error(f"Failed to get info for {ticker}: {e}")
                info = {}

            # Get historical data with error handling
            price_summary = "N/A"
            try:
                hist = stock.history(period="1mo", timeout=10)

                if not hist.empty and len(hist) > 0:
                    try:
                        latest = float(hist['Close'].iloc[-1])
                        oldest = float(hist['Close'].iloc[0])

                        if oldest > 0:
                            change = ((latest - oldest) / oldest) * 100
                            price_summary = f"Aktuell: ${latest:.2f}, 30-Tage Änderung: {change:+.1f}%"
                        else:
                            price_summary = f"Aktuell: ${latest:.2f}"
                    except (IndexError, ValueError, ZeroDivisionError) as e:
                        self.logger.warning(f"Error calculating price change for {ticker}: {e}")
                        price_summary = "Preisverlauf nicht verfügbar"
            except Exception as e:
                self.logger.warning(f"Failed to get history for {ticker}: {e}")

            # Safely extract data with defaults
            result = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'pe_ratio': info.get('trailingPE'),
                'market_cap': info.get('marketCap'),
                'revenue_growth': info.get('revenueGrowth'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'target_price': info.get('targetMeanPrice'),
                'price_summary': price_summary,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }

            self.logger.info(f"Successfully fetched data for {ticker}")
            return result

        except ValueError as e:
            self.logger.error(f"Invalid ticker format {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': f'Invalid ticker format: {str(e)}',
                'name': ticker
            }

        except ConnectionError as e:
            self.logger.error(f"Network error fetching {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': 'Network error - check internet connection',
                'name': ticker
            }

        except Exception as e:
            self.logger.error(f"Unexpected error fetching data for {ticker}: {e}", exc_info=True)
            return {
                'ticker': ticker,
                'error': str(e),
                'name': ticker
            }

# Singleton
swarm = InvestmentSwarm()
