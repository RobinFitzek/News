"""
Investment Analysis Agents
Multi-agent system: Quant screening (math) + Perplexity (news) + Gemini (research notes).
No buy/sell signals — facts and context only.
"""
from clients.perplexity_client import pplx_client
from clients.gemini_client import gemini_client
from clients.custom_provider_client import custom_provider_client
from engine.quant_screener import quant_screener
import yfinance as yf
from core.database import db
import logging
import re
from typing import Dict, Optional

class InvestmentSwarm:
    """Multi-Agent System for automated investment analysis"""

    def __init__(self):
        self.pplx = pplx_client
        self.gemini = gemini_client
        self.screener = quant_screener
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
        """Manual full analysis for a single stock"""
        print(f"Manual Analysis for {ticker}...")

        # Stage 1: Quant screen (free)
        quant_result = self.screener.screen_ticker(ticker, variant=strategy)
        if not quant_result or 'error' in quant_result:
            stock_data = self._get_stock_data(ticker)
            quant_result = {
                'ticker': ticker,
                'score': 50,
                'composite_score': 50,
                'signal': 'Neutral',
                'initial_reason': 'Quant data unavailable, manual override',
                'data': stock_data,
                'valuation': {}, 'technicals': {}, 'momentum': {}, 'quality': {},
                'anomalies': [],
            }

        candidate = quant_result
        candidate['data'] = candidate.get('data', self._get_stock_data(ticker))

        # Stage 2: News only (Perplexity)
        stage2_results = self.stage2_analyze(candidate, strategy)
        candidate.update(stage2_results)

        # Stage 3: Research note (Gemini)
        final_result = self.stage3_synthesize(candidate, strategy)

        return final_result

    def stage1_scan(self, tickers: list, variant: str = "balanced") -> list:
        """Stage 1: Quantitative screening (zero API cost, pure math)"""
        if not tickers:
            self.logger.warning("Empty ticker list provided to stage1_scan")
            return []

        # Use quant screener instead of Gemini
        results = self.screener.screen_batch(tickers, variant)

        self.logger.info(f"Stage 1 (quant screen) complete: {len(results)} candidates")
        return results

    def stage2_analyze(self, candidate: dict, variant: str = "balanced") -> dict:
        """Stage 2: News summarization only (Perplexity). No AI fundamental/technical analysis."""
        ticker = candidate['ticker']
        print(f"\n  STAGE 2: News & Market Intelligence for {ticker}")

        results = {
            'ticker': ticker,
            'stage1_score': candidate.get('score', candidate.get('composite_score', 0)),
            'stage1_reason': candidate.get('initial_reason', ''),
        }

        # Quant data already computed in Stage 1 — pass it through
        results['quant_metrics'] = {
            'valuation': candidate.get('valuation', {}),
            'technicals': candidate.get('technicals', {}),
            'momentum': candidate.get('momentum', {}),
            'quality': candidate.get('quality', {}),
            'anomalies': candidate.get('anomalies', []),
            'composite_score': candidate.get('composite_score', candidate.get('score', 0)),
            'signal': candidate.get('signal', 'Neutral'),
        }

        # Fundamental + Technical analysis is now computed by quant screener (math)
        # Generate text summaries from quant data for DB compatibility
        qm = results['quant_metrics']
        val = qm.get('valuation', {})
        tech = qm.get('technicals', {})
        mom = qm.get('momentum', {})
        qual = qm.get('quality', {})

        results['fundamental'] = (
            f"P/E: {val.get('pe_ratio', 'N/A')} (vs Sektor: {val.get('pe_vs_sector', 'N/A')}x) | "
            f"PEG: {val.get('peg_ratio', 'N/A')} | P/B vs Sektor: {val.get('pb_vs_sector', 'N/A')}x | "
            f"D/E: {qual.get('debt_to_equity', 'N/A')} | ROE: {qual.get('roe', 'N/A')}% | "
            f"FCF Yield: {qual.get('fcf_yield', 'N/A')}% | Current Ratio: {qual.get('current_ratio', 'N/A')}"
        )
        results['technical'] = (
            f"RSI(14): {tech.get('rsi_14', 'N/A')} | SMA Signal: {tech.get('sma_cross_signal', 'N/A')} | "
            f"52W Position: {tech.get('price_vs_52w_range', 'N/A')} | Bollinger: {tech.get('bollinger_position', 'N/A')} | "
            f"1M Return: {mom.get('return_1m', 'N/A')}% | 3M Return: {mom.get('return_3m', 'N/A')}% | "
            f"6M Return: {mom.get('return_6m', 'N/A')}% | "
            f"vs SPY 1M: {mom.get('excess_1m', 'N/A')}% | 3M: {mom.get('excess_3m', 'N/A')}% | 6M: {mom.get('excess_6m', 'N/A')}%"
        )

        # Market Insights & News
        stage2_provider = custom_provider_client.get_provider_for_role('stage2_news')
        if stage2_provider:
            print(f"    Fetching stage2 news via {stage2_provider.get('name')}...")
            provider_news = custom_provider_client.generate(
                stage2_provider,
                system_prompt=(
                    "You are a financial news analyst. Return a concise, factual news brief "
                    "with catalysts, risks, and cited sources when available."
                ),
                user_prompt=(
                    f"Ticker: {ticker}. Provide latest relevant market/news context in 6-10 bullets. "
                    "Include sentiment (bullish/neutral/bearish), key events, and source names/URLs if present."
                ),
                temperature=0.2,
                max_tokens=700,
            )
            if provider_news:
                results['news'] = provider_news
                results['stage2_provider'] = stage2_provider.get('name')
            else:
                if self.pplx.is_configured():
                    news_scan = self.pplx.quick_scan(ticker)
                    results['news'] = news_scan['raw'] if news_scan and news_scan.get('raw') else "No recent news available"
                    results['stage2_provider'] = 'Perplexity (fallback)'
                else:
                    results['news'] = "Custom stage2 provider failed - fallback unavailable"
                    results['stage2_provider'] = stage2_provider.get('name')
        elif self.pplx.is_configured():
            print(f"    Fetching real-time news for {ticker}...")
            news_scan = self.pplx.quick_scan(ticker)
            results['news'] = news_scan['raw'] if news_scan and news_scan.get('raw') else "No recent news available"
            results['stage2_provider'] = 'Perplexity'
        else:
            results['news'] = "No stage2 provider configured"
            results['stage2_provider'] = 'none'

        return results

    def stage3_synthesize(self, analysis_result: dict, variant: str = "balanced") -> dict:
        """Stage 3: Research note with risks and catalysts. NO buy/sell recommendations."""
        ticker = analysis_result['ticker']
        print(f"\n  STAGE 3: Research Note for {ticker}")

        qm = analysis_result.get('quant_metrics', {})
        anomaly_text = ""
        for a in qm.get('anomalies', []):
            anomaly_text += f"  - {a['description']}\n"
        if not anomaly_text:
            anomaly_text = "  Keine Anomalien erkannt."

        insider = analysis_result.get('insider_activity', {})
        insider_text = ""
        if insider and insider.get('has_activity'):
            insider_text = f"\nInsider-Aktivitaet (Folge dem Geld):\n{insider.get('fact_summary', 'Insider-Aktivitaet vorhanden.')}\n"

        prompt = f"""Du bist ein Research-Analyst. Schreibe eine kurze Research-Notiz fuer {ticker}.

Quantitative Daten:
- Bewertung: P/E vs Sektor: {qm.get('valuation', {}).get('pe_vs_sector', 'N/A')}x, PEG: {qm.get('valuation', {}).get('peg_ratio', 'N/A')}, P/B vs Sektor: {qm.get('valuation', {}).get('pb_vs_sector', 'N/A')}x
- Technisch: RSI(14): {qm.get('technicals', {}).get('rsi_14', 'N/A')}, SMA-Signal: {qm.get('technicals', {}).get('sma_cross_signal', 'N/A')}, 52W-Position: {qm.get('technicals', {}).get('price_vs_52w_range', 'N/A')}
- Momentum: 1M: {qm.get('momentum', {}).get('return_1m', 'N/A')}%, 3M: {qm.get('momentum', {}).get('return_3m', 'N/A')}%, 6M: {qm.get('momentum', {}).get('return_6m', 'N/A')}% (vs SPY: {qm.get('momentum', {}).get('excess_1m', 'N/A')}%, {qm.get('momentum', {}).get('excess_3m', 'N/A')}%, {qm.get('momentum', {}).get('excess_6m', 'N/A')}%)
- Qualitaet: D/E: {qm.get('quality', {}).get('debt_to_equity', 'N/A')}, ROE: {qm.get('quality', {}).get('roe', 'N/A')}%, FCF Yield: {qm.get('quality', {}).get('fcf_yield', 'N/A')}%
- Anomalien:
{anomaly_text}
- Quant Signal: {qm.get('signal', 'Neutral')} (Score: {qm.get('composite_score', 'N/A')}/100){insider_text}

Aktuelle Nachrichten:
{analysis_result.get('news', 'Keine Nachrichten verfuegbar')}

Aufgabe:
Erstelle eine strukturierte Analyse mit klaren Bull/Bear Argumenten, einer Risikobewertung und überprüfbaren Quellen aus den Nachrichten.
Falls der Nachrichtentext URLs oder explizite Quellen enthält, MÜSSEN diese im Abschnitt "Quellen" aufgelistet werden.

KEINE Kauf/Verkauf-Empfehlung. Nur Fakten und Kontext.

Zwingendes Format:
Risk Score: [1-10]
Bull Case: [Argumente für die Aktie - 2-3 Sätze]
Bear Case: [Die größten Risiken und Schwächen - 2-3 Sätze]
Quellen: [Liste der URLs oder Publikationen]
Zusammenfassung: [Kurzer Gesamteindruck]
"""

        stage3_provider = custom_provider_client.get_provider_for_role('stage3_synthesis')
        if stage3_provider:
            print(f"    Synthesizing via {stage3_provider.get('name')}...")
            response = custom_provider_client.generate(
                stage3_provider,
                system_prompt="You are a precise financial research writer. Follow the required format exactly.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=1200,
            )
            if response:
                analysis_result['stage3_provider'] = stage3_provider.get('name')
            else:
                response = self.gemini.generate(prompt, tier='flash')
                analysis_result['stage3_provider'] = 'Gemini (fallback)'
        else:
            response = self.gemini.generate(prompt, tier='flash')
            analysis_result['stage3_provider'] = 'Gemini'

        if not response:
            response = "Risk Score: 5\nBull Case: Not available\nBear Case: Not available\nQuellen: N/A\nZusammenfassung: Analysis provider unavailable."
        analysis_result['recommendation'] = response

        # Parse sections for DB
        risk_match = re.search(r"Risk Score:\s*(\d+)", response, re.IGNORECASE)
        analysis_result['risk_score'] = int(risk_match.group(1)) if risk_match else 5

        def extract_section(text: str, header: str) -> str:
            pattern = rf"{header}:\s*(.*?)(?=\n(?:Risk Score|Bull Case|Bear Case|Quellen|Zusammenfassung):|$)"
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        analysis_result['bull_case'] = extract_section(response, "Bull Case")
        analysis_result['bear_case'] = extract_section(response, "Bear Case")
        analysis_result['sources'] = extract_section(response, "Quellen")

        # Fallbacks in case prompt format wasn't strictly followed
        if not analysis_result['bull_case'] and not analysis_result['bear_case']:
             analysis_result['bull_case'] = "Konnte nicht aus der Antwort extrahiert werden."
             analysis_result['bear_case'] = "Konnte nicht aus der Antwort extrahiert werden."

        # Signal comes from quant screener, not AI
        analysis_result['signal'] = qm.get('signal', 'Neutral')

        print(f"    {ticker}: {analysis_result['signal']} | Risk: {analysis_result['risk_score']}/10 | Score: {qm.get('composite_score', 'N/A')}/100")

        return analysis_result

    def _get_stock_data(self, ticker: str) -> dict:
        """Get stock data from yfinance with comprehensive error handling"""
        if not ticker or not isinstance(ticker, str):
            self.logger.error("Invalid ticker provided")
            return {'error': 'Invalid ticker', 'ticker': ticker}

        ticker = ticker.upper().strip()

        try:
            stock = yf.Ticker(ticker)

            try:
                info = stock.info
                if not info or len(info) < 3:
                    self.logger.warning(f"Minimal or no data returned for {ticker}")
                    return {
                        'ticker': ticker,
                        'error': 'No data available',
                        'name': ticker
                    }
            except Exception as e:
                self.logger.error(f"Failed to get info for {ticker}: {e}")
                info = {}

            price_summary = "N/A"
            try:
                hist = stock.history(period="1mo", timeout=10)

                if not hist.empty and len(hist) > 0:
                    try:
                        latest = float(hist['Close'].iloc[-1])
                        oldest = float(hist['Close'].iloc[0])

                        if oldest > 0:
                            change = ((latest - oldest) / oldest) * 100
                            price_summary = f"Aktuell: ${latest:.2f}, 30-Tage Aenderung: {change:+.1f}%"
                        else:
                            price_summary = f"Aktuell: ${latest:.2f}"
                    except (IndexError, ValueError, ZeroDivisionError) as e:
                        self.logger.warning(f"Error calculating price change for {ticker}: {e}")
                        price_summary = "Preisverlauf nicht verfuegbar"
            except Exception as e:
                self.logger.warning(f"Failed to get history for {ticker}: {e}")

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
