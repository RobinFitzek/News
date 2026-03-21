"""
Geopolitical Scenario Stress Tester (#39)

Defines 6 named geopolitical scenarios with estimated sector impact vectors.
When the geo scanner detects a matching event (severity >= 8), the matching
scenario is automatically cross-referenced with the current portfolio to
estimate P&L impact.

Usage:
    from engine.geo_scenario import geo_scenarios
    result = geo_scenarios.run_scenario("taiwan_blockade")
    match  = geo_scenarios.find_matching_scenario(geo_scan_text)
"""
from typing import Dict, List, Optional, Tuple
from core.database import db
import logging

logger = logging.getLogger(__name__)

# ── Scenario definitions ───────────────────────────────────────────────────
# Each scenario has:
#   name            — human-readable title
#   description     — one-sentence summary
#   keywords        — trigger words in geo scan text (lower-case)
#   sector_impacts  — {sector_key: pct_change_estimate} (negative = loss)
#   historical_analog — real event this is modelled after
#   severity_threshold — minimum geo severity before auto-triggering

SCENARIOS: Dict[str, Dict] = {
    "taiwan_blockade": {
        "name": "Taiwan Strait Blockade",
        "description": "China blockades Taiwan, halting TSMC production and disrupting global semiconductor supply chains.",
        "keywords": ["taiwan", "tsmc", "strait", "blockade", "pla", "china taiwan", "invasion"],
        "sector_impacts": {
            "technology": -0.18,
            "semiconductors": -0.35,
            "consumer_electronics": -0.12,
            "energy": +0.08,
            "defense": +0.15,
            "financials": -0.06,
            "utilities": -0.02,
            "healthcare": -0.01,
        },
        "historical_analog": "1973 Arab Oil Embargo (supply shock + geopolitical panic)",
        "severity_threshold": 8,
    },
    "opec_production_cut": {
        "name": "OPEC+ Major Production Cut (−20%)",
        "description": "OPEC+ announces a surprise 20% production cut, driving oil above $120/barrel.",
        "keywords": ["opec", "oil cut", "production cut", "crude", "oil price spike", "brent surge"],
        "sector_impacts": {
            "energy": +0.22,
            "oil_gas": +0.28,
            "transportation": -0.14,
            "airlines": -0.20,
            "consumer_discretionary": -0.08,
            "industrials": -0.07,
            "utilities": -0.04,
            "technology": -0.03,
        },
        "historical_analog": "2022 OPEC+ October surprise cut",
        "severity_threshold": 7,
    },
    "russia_escalation": {
        "name": "Russia Military Escalation (Europe)",
        "description": "Russia expands conflict into NATO-adjacent territory, triggering Article 4 consultations and energy price spikes.",
        "keywords": ["russia", "ukraine", "nato", "escalation", "war expand", "missile strike", "belarus"],
        "sector_impacts": {
            "energy": +0.12,
            "defense": +0.20,
            "european_equities": -0.15,
            "financials": -0.08,
            "technology": -0.05,
            "materials": +0.06,
            "agriculture": +0.10,
            "consumer_discretionary": -0.09,
        },
        "historical_analog": "February 2022 Russia invasion of Ukraine",
        "severity_threshold": 8,
    },
    "us_iran_conflict": {
        "name": "US–Iran Military Confrontation",
        "description": "US strikes Iranian nuclear facilities; Iran retaliates by closing the Strait of Hormuz.",
        "keywords": ["iran", "hormuz", "strait of hormuz", "persian gulf", "nuclear", "iran strike"],
        "sector_impacts": {
            "energy": +0.25,
            "oil_gas": +0.30,
            "defense": +0.18,
            "airlines": -0.16,
            "consumer_discretionary": -0.10,
            "financials": -0.07,
            "technology": -0.05,
            "materials": +0.04,
        },
        "historical_analog": "1991 Gulf War oil shock",
        "severity_threshold": 8,
    },
    "eu_energy_crisis": {
        "name": "European Energy Supply Crisis",
        "description": "Major disruption to European gas supply causes energy rationing, industrial shutdowns, and recession fears.",
        "keywords": ["gas shortage", "energy crisis", "europe gas", "lng", "nord stream", "rationing", "energy supply"],
        "sector_impacts": {
            "european_equities": -0.18,
            "energy": +0.15,
            "industrials": -0.12,
            "chemicals": -0.14,
            "consumer_discretionary": -0.11,
            "utilities": -0.06,
            "financials": -0.07,
            "healthcare": -0.02,
        },
        "historical_analog": "2022 European gas crisis post-Nord Stream",
        "severity_threshold": 7,
    },
    "china_us_tariffs": {
        "name": "US–China Tariff Escalation (All-Out Trade War)",
        "description": "US raises tariffs on all Chinese goods to 50%+; China retaliates with equivalent measures and rare-earth export bans.",
        "keywords": ["tariff", "trade war", "china tariff", "rare earth", "decoupling", "trade escalation", "sanctions china"],
        "sector_impacts": {
            "technology": -0.15,
            "semiconductors": -0.20,
            "consumer_electronics": -0.12,
            "materials": -0.08,
            "industrials": -0.06,
            "agriculture": -0.05,
            "energy": +0.02,
            "defense": +0.08,
        },
        "historical_analog": "2018–2019 US-China Trade War",
        "severity_threshold": 7,
    },
}

# Sector key normalisers — map yfinance sector strings to our scenario keys
_SECTOR_MAP = {
    "technology": "technology",
    "information technology": "technology",
    "semiconductors": "semiconductors",
    "consumer electronics": "consumer_electronics",
    "energy": "energy",
    "oil & gas": "oil_gas",
    "utilities": "utilities",
    "financials": "financials",
    "health care": "healthcare",
    "healthcare": "healthcare",
    "industrials": "industrials",
    "materials": "materials",
    "consumer discretionary": "consumer_discretionary",
    "consumer staples": "consumer_discretionary",
    "defense": "defense",
    "aerospace & defense": "defense",
    "transportation": "transportation",
    "airlines": "airlines",
    "agriculture": "agriculture",
    "chemicals": "chemicals",
}


class GeoScenarioRunner:
    """Cross-reference geo scenarios with current portfolio to estimate impact."""

    def get_all_scenarios(self) -> Dict[str, Dict]:
        return SCENARIOS

    def run_scenario(self, scenario_key: str) -> Dict:
        """
        Run a named scenario against the current portfolio.

        Returns:
            {
              scenario: <scenario dict>,
              portfolio_impact_pct: <float>,   # estimated total portfolio P&L %
              holdings_impact: [
                {ticker, sector, weight_pct, sector_impact_pct, estimated_pnl_pct},
                ...
              ],
              ran_at: <iso datetime>,
            }
        """
        scenario = SCENARIOS.get(scenario_key)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_key}. "
                             f"Available: {list(SCENARIOS.keys())}")

        holdings = self._get_portfolio_holdings()
        total_value = sum(h.get('market_value', 0) or 0 for h in holdings)

        holdings_impact = []
        total_impact_pct = 0.0

        for h in holdings:
            ticker = h.get('ticker', '')
            sector_raw = (h.get('sector') or '').lower()
            sector_key = _SECTOR_MAP.get(sector_raw, sector_raw.replace(' ', '_'))
            impact_pct = scenario['sector_impacts'].get(sector_key, 0.0)

            mv = h.get('market_value') or 0
            weight = (mv / total_value * 100) if total_value > 0 else 0
            holding_impact = weight * impact_pct / 100

            holdings_impact.append({
                'ticker': ticker,
                'sector': sector_raw or 'unknown',
                'weight_pct': round(weight, 2),
                'sector_impact_pct': round(impact_pct * 100, 1),
                'estimated_pnl_pct': round(holding_impact, 3),
            })
            total_impact_pct += holding_impact

        holdings_impact.sort(key=lambda x: x['estimated_pnl_pct'])

        return {
            'scenario_key': scenario_key,
            'scenario': scenario,
            'portfolio_impact_pct': round(total_impact_pct, 2),
            'holdings_impact': holdings_impact,
            'total_holdings': len(holdings_impact),
            'ran_at': __import__('datetime').datetime.now().isoformat(),
        }

    def find_matching_scenario(self, geo_scan_text: str) -> Optional[Tuple[str, Dict]]:
        """
        Given a geo scan summary string, find the best-matching scenario
        by keyword overlap. Returns (scenario_key, scenario_dict) or None.
        """
        if not geo_scan_text:
            return None
        text_lower = geo_scan_text.lower()
        best_key = None
        best_score = 0
        for key, scenario in SCENARIOS.items():
            score = sum(1 for kw in scenario['keywords'] if kw in text_lower)
            if score > best_score:
                best_score = score
                best_key = key
        if best_score >= 2:  # require at least 2 keyword hits
            return best_key, SCENARIOS[best_key]
        return None

    def _get_portfolio_holdings(self) -> List[Dict]:
        """Fetch open portfolio positions with sector info."""
        try:
            from engine.portfolio_manager import portfolio_manager
            holdings = portfolio_manager.get_portfolio_holdings()
            # Enrich with sector from yfinance if missing
            enriched = []
            for h in holdings:
                if not h.get('sector'):
                    try:
                        import yfinance as yf
                        info = yf.Ticker(h['ticker']).info
                        h['sector'] = info.get('sector', '')
                        # estimate market value if missing
                        if not h.get('market_value') and info.get('currentPrice'):
                            shares = h.get('shares', h.get('quantity', 0)) or 0
                            h['market_value'] = float(info['currentPrice']) * float(shares)
                    except Exception:
                        pass
                enriched.append(h)
            return enriched
        except Exception as e:
            logger.warning(f"GeoScenarioRunner._get_portfolio_holdings: {e}")
            return []

    def auto_trigger_on_geo_scan(self, geo_scan_text: str,
                                  severity: float) -> Optional[Dict]:
        """
        Called after a geopolitical scan. If severity >= threshold and a matching
        scenario is found, runs it automatically. Returns result dict or None.
        """
        match = self.find_matching_scenario(geo_scan_text)
        if not match:
            return None
        scenario_key, scenario = match
        if severity < scenario.get('severity_threshold', 8):
            return None
        logger.info(f"Auto-triggering scenario '{scenario_key}' "
                    f"(severity={severity:.1f} >= {scenario['severity_threshold']})")
        return self.run_scenario(scenario_key)


geo_scenarios = GeoScenarioRunner()
