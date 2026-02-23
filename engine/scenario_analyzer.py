"""
Scenario Analyzer
Maps portfolio holdings to factor sensitivities (beta, rate sensitivity,
sector exposure) and runs predefined or custom stress scenarios to estimate
portfolio impact. Pure math-based — uses yfinance for beta and sector data.

PRESET SCENARIOS:
- rate_hike: Interest rates +100bps
- rate_cut: Interest rates -100bps
- market_crash: Market drops 20%
- market_rally: Market rallies 15%
- tech_rotation: Tech sells off, defensives rally
- defensive_shift: Cyclicals drop, utilities/staples gain
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging

logger = logging.getLogger(__name__)

# Sector sensitivity profiles (empirical approximations)
SECTOR_RATE_SENSITIVITY = {
    'Technology': -0.8,           # Growth stocks hurt by higher rates
    'Real Estate': -1.2,          # REITs very rate-sensitive
    'Utilities': -0.9,            # Bond proxies
    'Consumer Discretionary': -0.5,
    'Financial Services': 0.6,    # Banks benefit from higher rates
    'Financials': 0.6,
    'Healthcare': -0.2,           # Relatively insensitive
    'Consumer Defensive': -0.1,
    'Consumer Staples': -0.1,
    'Energy': 0.1,                # Commodity-driven
    'Industrials': -0.3,
    'Communication Services': -0.4,
    'Basic Materials': -0.2,
    'Materials': -0.2,
}

# Sector behavior in rotation scenarios
SECTOR_ROTATION_PROFILE = {
    'tech_rotation': {
        'Technology': -0.15,
        'Communication Services': -0.10,
        'Consumer Discretionary': -0.08,
        'Utilities': 0.06,
        'Consumer Defensive': 0.05,
        'Consumer Staples': 0.05,
        'Healthcare': 0.04,
        'Financials': 0.03,
        'Financial Services': 0.03,
        'Energy': 0.02,
        'Industrials': 0.02,
        'Real Estate': 0.01,
        'Basic Materials': 0.01,
        'Materials': 0.01,
    },
    'defensive_shift': {
        'Consumer Discretionary': -0.12,
        'Technology': -0.10,
        'Industrials': -0.08,
        'Energy': -0.06,
        'Communication Services': -0.05,
        'Financials': -0.04,
        'Financial Services': -0.04,
        'Basic Materials': -0.05,
        'Materials': -0.05,
        'Utilities': 0.08,
        'Consumer Defensive': 0.07,
        'Consumer Staples': 0.07,
        'Healthcare': 0.05,
        'Real Estate': 0.03,
    },
}


class ScenarioAnalyzer:
    """Run stress scenarios against current portfolio holdings."""

    PRESET_SCENARIOS = {
        'rate_hike': {
            'name': 'Interest Rates +100bps',
            'description': 'Federal Reserve raises rates by 100 basis points',
            'type': 'rate',
            'magnitude': 1.0,  # +1% rate change
        },
        'rate_cut': {
            'name': 'Interest Rates -100bps',
            'description': 'Federal Reserve cuts rates by 100 basis points',
            'type': 'rate',
            'magnitude': -1.0,
        },
        'market_crash': {
            'name': 'Market Crash -20%',
            'description': 'Broad market decline of 20% (bear market territory)',
            'type': 'market',
            'magnitude': -0.20,
        },
        'market_rally': {
            'name': 'Market Rally +15%',
            'description': 'Broad market rally of 15%',
            'type': 'market',
            'magnitude': 0.15,
        },
        'tech_rotation': {
            'name': 'Tech Rotation',
            'description': 'Money flows from growth/tech into value/defensive sectors',
            'type': 'rotation',
            'rotation_profile': 'tech_rotation',
        },
        'defensive_shift': {
            'name': 'Defensive Shift',
            'description': 'Risk-off: cyclicals sell off, defensive sectors outperform',
            'type': 'rotation',
            'rotation_profile': 'defensive_shift',
        },
    }

    def __init__(self):
        self._sensitivity_cache = {}
        self._cache_duration = timedelta(minutes=30)

    def get_preset_scenarios(self) -> List[Dict]:
        """Return list of available preset scenarios."""
        return [
            {'key': key, **{k: v for k, v in scenario.items() if k != 'rotation_profile'}}
            for key, scenario in self.PRESET_SCENARIOS.items()
        ]

    def get_portfolio_sensitivities(self) -> Dict:
        """
        Calculate factor sensitivities for current portfolio.
        Returns beta, rate sensitivity, and sector exposure breakdown.
        """
        holdings = db.get_portfolio_holdings()
        active = [h for h in holdings if h.get('shares', 0) > 0]

        if not active:
            return {'message': 'No active holdings', 'holdings': [], 'portfolio_beta': None}

        total_value = 0
        enriched = []

        for h in active:
            ticker = h.get('ticker', '')
            shares = h.get('shares', 0)
            price = h.get('current_price', h.get('avg_cost', 0))
            value = shares * price
            total_value += value

            sensitivity = self._get_ticker_sensitivity(ticker)

            enriched.append({
                'ticker': ticker,
                'value': value,
                'beta': sensitivity.get('beta', 1.0),
                'sector': sensitivity.get('sector', 'Unknown'),
                'rate_sensitivity': sensitivity.get('rate_sensitivity', 0),
            })

        if total_value == 0:
            return {'message': 'Portfolio value is zero', 'holdings': enriched, 'portfolio_beta': None}

        # Portfolio-weighted metrics
        portfolio_beta = sum(h['value'] / total_value * h['beta'] for h in enriched)
        portfolio_rate_sens = sum(h['value'] / total_value * h['rate_sensitivity'] for h in enriched)

        # Sector exposure
        sector_exposure = {}
        for h in enriched:
            sector = h['sector']
            weight = h['value'] / total_value * 100
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

        return {
            'total_value': total_value,
            'portfolio_beta': round(portfolio_beta, 2),
            'portfolio_rate_sensitivity': round(portfolio_rate_sens, 2),
            'sector_exposure': {k: round(v, 1) for k, v in sorted(sector_exposure.items(), key=lambda x: -x[1])},
            'holdings': enriched,
        }

    def run_scenario(self, scenario_name: str, custom_params: Optional[Dict] = None) -> Dict:
        """
        Run a scenario against current portfolio.
        Returns estimated impact per holding and total portfolio impact.
        """
        if scenario_name not in self.PRESET_SCENARIOS and not custom_params:
            return {'error': f"Unknown scenario '{scenario_name}'. Use get_preset_scenarios() for options."}

        scenario = custom_params if custom_params else self.PRESET_SCENARIOS[scenario_name]
        sensitivities = self.get_portfolio_sensitivities()

        if not sensitivities.get('holdings'):
            return {'error': 'No active holdings to analyze', 'scenario': scenario.get('name', scenario_name)}

        holdings = sensitivities['holdings']
        total_value = sensitivities.get('total_value', 0)

        impacts = self._calculate_impact(holdings, scenario)

        total_impact_pct = sum(i['impact_pct'] * i['weight'] for i in impacts) if impacts else 0
        total_impact_dollar = total_impact_pct / 100 * total_value

        # Worst and best positions
        impacts_sorted = sorted(impacts, key=lambda x: x['impact_pct'])
        worst = impacts_sorted[0] if impacts_sorted else None
        best = impacts_sorted[-1] if impacts_sorted else None

        return {
            'scenario': scenario.get('name', scenario_name),
            'description': scenario.get('description', ''),
            'total_value': total_value,
            'estimated_impact_pct': round(total_impact_pct, 2),
            'estimated_impact_dollar': round(total_impact_dollar, 0),
            'worst_position': {
                'ticker': worst['ticker'],
                'impact_pct': worst['impact_pct'],
            } if worst else None,
            'best_position': {
                'ticker': best['ticker'],
                'impact_pct': best['impact_pct'],
            } if best else None,
            'position_impacts': impacts,
        }

    def _calculate_impact(self, holdings: List[Dict], scenario: Dict) -> List[Dict]:
        """Calculate estimated impact for each holding under a given scenario."""
        total_value = sum(h.get('value', 0) for h in holdings)
        if total_value == 0:
            return []

        scenario_type = scenario.get('type', 'market')
        impacts = []

        for h in holdings:
            ticker = h['ticker']
            beta = h.get('beta', 1.0)
            sector = h.get('sector', 'Unknown')
            rate_sens = h.get('rate_sensitivity', 0)
            weight = h.get('value', 0) / total_value

            if scenario_type == 'market':
                # Impact = beta * market move
                magnitude = scenario.get('magnitude', 0)
                impact_pct = beta * magnitude * 100

            elif scenario_type == 'rate':
                # Impact = rate_sensitivity * rate change magnitude
                magnitude = scenario.get('magnitude', 0)
                impact_pct = rate_sens * magnitude * 5  # Scale factor for rates

            elif scenario_type == 'rotation':
                # Impact from sector rotation profile
                profile_key = scenario.get('rotation_profile', '')
                profile = SECTOR_ROTATION_PROFILE.get(profile_key, {})
                sector_impact = profile.get(sector, 0)
                # Also factor in beta for amplification
                impact_pct = sector_impact * 100 * (0.5 + 0.5 * beta)

            else:
                impact_pct = 0

            impacts.append({
                'ticker': ticker,
                'sector': sector,
                'beta': round(beta, 2),
                'weight': round(weight * 100, 1),
                'impact_pct': round(impact_pct, 2),
                'impact_dollar': round(h.get('value', 0) * impact_pct / 100, 0),
            })

        return impacts

    def _get_ticker_sensitivity(self, ticker: str) -> Dict:
        """Get beta, sector, and rate sensitivity for a single ticker. Cached."""
        cache_key = ticker.upper()
        if cache_key in self._sensitivity_cache:
            entry = self._sensitivity_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            beta = info.get('beta', 1.0)
            if beta is None:
                beta = 1.0

            sector = info.get('sector', 'Unknown')
            rate_sensitivity = SECTOR_RATE_SENSITIVITY.get(sector, 0)

            result = {
                'beta': float(beta),
                'sector': sector,
                'rate_sensitivity': rate_sensitivity,
            }

            self._sensitivity_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now(),
            }

            return result

        except Exception as e:
            logger.debug(f"Sensitivity fetch failed for {ticker}: {e}")
            return {'beta': 1.0, 'sector': 'Unknown', 'rate_sensitivity': 0}


# Singleton
scenario_analyzer = ScenarioAnalyzer()
