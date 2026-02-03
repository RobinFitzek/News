"""
Strategy Engine for Investment Algorithm
Manages investment strategies, risk classification, and dynamic prompt generation.
"""
import json
from typing import Dict, List, Optional, Any
from core.config import ASSET_CATEGORIES, TIME_HORIZONS, STRATEGY_PRESETS
from core.database import db
import yfinance as yf


class StrategyManager:
    """Manages investment strategies and their configurations"""
    
    def __init__(self):
        self.presets = STRATEGY_PRESETS
        self._load_active_strategy()
    
    def _load_active_strategy(self):
        """Load the currently active strategy from settings"""
        variant = db.get_setting('analysis_variant') or 'balanced'
        self.active_strategy = self.get_strategy(variant)
    
    def get_strategy(self, name: str) -> Dict:
        """Get strategy by name, from DB or presets"""
        # Try database first
        db_strategy = db.get_strategy(name)
        if db_strategy:
            return {
                'name': db_strategy['name'],
                'description': db_strategy['description'],
                'risk_tolerance': db_strategy['risk_tolerance'],
                'time_horizon': db_strategy['time_horizon'],
                'asset_mix': json.loads(db_strategy['asset_mix']) if db_strategy['asset_mix'] else {},
                'scan_frequency': db_strategy['scan_frequency']
            }
        # Fall back to presets
        if name in self.presets:
            preset = self.presets[name]
            return {
                'name': name,
                'description': f'{name.capitalize()} Strategy',
                **preset
            }
        return self.presets['balanced']
    
    def get_all_strategies(self) -> List[Dict]:
        """Get all available strategies"""
        return db.get_strategies(active_only=True)
    
    def set_active_strategy(self, name: str):
        """Set the active strategy"""
        db.set_setting('analysis_variant', name)
        self._load_active_strategy()
    
    def get_asset_mix_for_category(self, category: str) -> int:
        """Get the target percentage for an asset category"""
        asset_mix = self.active_strategy.get('asset_mix', {})
        return asset_mix.get(category, 0)


class RiskClassifier:
    """Classifies tickers by risk category"""
    
    def __init__(self):
        self.categories = ASSET_CATEGORIES
    
    def classify_ticker(self, ticker: str) -> Dict:
        """Classify a ticker based on its characteristics"""
        # Check if already classified in DB
        existing = db.get_category(ticker)
        if existing:
            return existing
        
        # Auto-classify based on yfinance data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            category = self._determine_category(info)
            risk_level = self._calculate_risk_level(info, category)
            time_horizon = self._suggest_time_horizon(category)
            
            # Save to DB
            db.set_category(ticker, category, risk_level, time_horizon)
            
            return {
                'ticker': ticker,
                'category': category,
                'risk_level': risk_level,
                'time_horizon': time_horizon,
                'auto_classified': True
            }
        except Exception as e:
            print(f"⚠️ Could not classify {ticker}: {e}")
            return {
                'ticker': ticker,
                'category': 'growth',  # Default
                'risk_level': 5,
                'time_horizon': 'medium',
                'auto_classified': False
            }
    
    def _determine_category(self, info: Dict) -> str:
        """Determine asset category from stock info"""
        quote_type = info.get('quoteType', '').upper()
        market_cap = info.get('marketCap', 0) or 0
        
        # ETF Detection
        if quote_type == 'ETF' or 'ETF' in info.get('longName', '').upper():
            return 'etf'
        
        # Market cap based classification
        if market_cap > 200_000_000_000:  # >200B = Mega Cap
            return 'blue_chip'
        elif market_cap > 10_000_000_000:  # >10B = Large Cap
            return 'blue_chip'
        elif market_cap > 2_000_000_000:   # >2B = Mid Cap
            return 'growth'
        elif market_cap > 300_000_000:     # >300M = Small Cap
            return 'startup'
        else:                               # <300M = Micro Cap
            return 'speculative'
    
    def _calculate_risk_level(self, info: Dict, category: str) -> int:
        """Calculate risk level 1-10"""
        base_range = self.categories[category]['risk_range']
        base_risk = (base_range[0] + base_range[1]) // 2
        
        # Adjustments
        adjustments = 0
        
        # Debt adjustment
        debt_ratio = info.get('debtToEquity', 0) or 0
        if debt_ratio > 200:
            adjustments += 2
        elif debt_ratio > 100:
            adjustments += 1
        
        # Profitability adjustment
        if info.get('profitMargins', 0) and info['profitMargins'] < 0:
            adjustments += 1
        
        # Volatility adjustment (beta)
        beta = info.get('beta', 1) or 1
        if beta > 1.5:
            adjustments += 1
        elif beta < 0.8:
            adjustments -= 1
        
        return max(1, min(10, base_risk + adjustments))
    
    def _suggest_time_horizon(self, category: str) -> str:
        """Suggest time horizon based on category"""
        category_to_horizon = {
            'etf': 'long',
            'blue_chip': 'long',
            'growth': 'medium',
            'startup': 'short',
            'speculative': 'short'
        }
        return category_to_horizon.get(category, 'medium')


class DynamicPromptBuilder:
    """Builds dynamic prompts based on strategy, category, and context"""
    
    def __init__(self):
        self.categories = ASSET_CATEGORIES
        self.horizons = TIME_HORIZONS
    
    def build_scan_prompt(self, ticker: str, strategy: str, category: str = None) -> str:
        """Build Stage 1 scan prompt"""
        strategy_focus = self._get_strategy_focus(strategy)
        category_focus = self._get_category_focus(category) if category else ""
        
        return f"""
Quick Scan für {ticker}:
Analyse-Strategie: {strategy_focus}
{category_focus}

Aufgabe: Bewerte das Investment nach der Strategie.
Gib einen 'Interest Score' von 0-100.
0 = Uninteressant / Passt nicht zur Strategie
100 = Top Kandidat / Perfekte Übereinstimmung

Format: "Score: [0-100] | Grund: [Kurze Begründung]"
"""
    
    def build_analysis_prompt(self, ticker: str, strategy: str, 
                               category: str, stock_data: Dict) -> str:
        """Build Stage 2 analysis prompt"""
        strategy_focus = self._get_strategy_focus(strategy)
        analysis_points = self._get_analysis_points(category)
        horizon = self._get_time_horizon_focus(strategy)
        
        return f"""
Tiefenanalyse für {ticker}:
Strategie: {strategy_focus}
Analysefokus: {analysis_points}
Zeithorizont: {horizon}
Daten: {json.dumps(stock_data, indent=2, default=str)}

Führe eine detaillierte Analyse durch:
1. Fundamentale Bewertung (KGV, KBV, Wachstum)
2. Technische Signale (Trend, Momentum)
3. Risikofaktoren

Antworte auf Deutsch, kurz und prägnant.
"""
    
    def build_synthesis_prompt(self, ticker: str, strategy: str,
                                category: str, analysis_results: Dict) -> str:
        """Build Stage 3 synthesis prompt"""
        strategy_desc = self._get_strategy_description(strategy)
        risk_focus = self._get_risk_focus(category)
        
        return f"""
Handle als Senior-Portfoliomanager.
Analysevariante: {strategy_desc}
Risikoprofil: {risk_focus}

Synthetisiere die Analysen für {ticker}:
Fundamental: {analysis_results.get('fundamental', 'N/A')}
Technisch: {analysis_results.get('technical', 'N/A')}
Quick Scan: {analysis_results.get('stage1_reason', 'N/A')}

Aufgabe:
1. Glasklare Handels-Empfehlung (Strong Buy, Buy, Hold, Sell, Strong Sell)
2. Detaillierte Risiko-Bewertung
3. Risk Score (1-10), 1=Sicher, 10=Totalverlustrisiko
4. Kursziele (Base/Bull/Bear) falls möglich

Format ZWINGEND:
Signal: [SIGNAL]
Risk Score: [1-10]
Risk Level: [Low/Medium/High/Extreme]
Confidence: [0-100]%
Kursziel (12m): [Preis oder N/A]
Begründung: [Detaillierte Analyse auf Deutsch]
Risiko-Faktoren: [Konkrete Gefahren]
"""
    
    def _get_strategy_focus(self, strategy: str) -> str:
        """Get strategy-specific focus text"""
        focuses = {
            'conservative': 'Konservativ - Maximale Sicherheit, Blue-Chips, stabile Dividenden. Extrem kritisch bewerten.',
            'balanced': 'Ausgewogen - Qualität zu fairem Preis. Balance zwischen Wachstum und Sicherheit.',
            'aggressive': 'Aggressiv - High-Reward, Growth-Chancen. Hohes Risiko für explosives Wachstum akzeptabel.'
        }
        return focuses.get(strategy, focuses['balanced'])
    
    def _get_category_focus(self, category: str) -> str:
        """Get category-specific analysis focus"""
        if not category or category not in self.categories:
            return ""
        
        cat_info = self.categories[category]
        focus_points = ', '.join(cat_info.get('analysis_focus', []))
        return f"Asset-Kategorie: {category.upper()} ({cat_info['description']})\nAnalysefokus: {focus_points}"
    
    def _get_analysis_points(self, category: str) -> str:
        """Get specific analysis points for category"""
        if not category or category not in self.categories:
            return "Allgemeine Fundamentalanalyse"
        
        return ', '.join(self.categories[category].get('analysis_focus', []))
    
    def _get_time_horizon_focus(self, strategy: str) -> str:
        """Get time horizon focus for strategy"""
        preset = STRATEGY_PRESETS.get(strategy, STRATEGY_PRESETS['balanced'])
        horizon = preset.get('time_horizon', 'medium_term')
        horizon_info = self.horizons.get(horizon, self.horizons['medium_term'])
        return f"{horizon_info['description']} (Fokus: {horizon_info['focus']})"
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get full strategy description"""
        descriptions = {
            'conservative': 'Konservativ (Sicherheit & Werterhalt)',
            'balanced': 'Ausgewogen (Qualität zu fairem Preis)',
            'aggressive': 'Aggressiv (Wachstum & Spekulation)'
        }
        return descriptions.get(strategy, descriptions['balanced'])
    
    def _get_risk_focus(self, category: str) -> str:
        """Get risk focus for category"""
        if not category or category not in self.categories:
            return "Standard Risikobewertung"
        
        cat_info = self.categories[category]
        risk_min, risk_max = cat_info['risk_range']
        return f"{category.upper()} (Risiko {risk_min}-{risk_max}, Volatilität: {cat_info['volatility']})"


# Singletons
strategy_manager = StrategyManager()
risk_classifier = RiskClassifier()
prompt_builder = DynamicPromptBuilder()
