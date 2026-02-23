"""
AI Analysis Cross-Check System
Validates AI-generated financial claims against actual market data from yfinance.
Detects hallucinated metrics, inaccurate numbers, and assigns a trust score.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

import yfinance as yf

logger = logging.getLogger(__name__)


class AICrossCheck:
    """Cross-checks AI-generated analysis against actual financial data."""

    # Regex patterns for extracting financial metric claims from analysis text.
    # Each entry: (metric_name, yfinance_key, list of (pattern, group_index, multiplier))
    METRIC_PATTERNS = {
        'P/E Ratio': {
            'yf_keys': ['trailingPE', 'forwardPE'],
            'patterns': [
                # "P/E of 25.3", "P/E: 25.3", "P/E ratio of 25.3", "PE ratio: 25"
                (r'(?:P/?E|PE)\s*(?:ratio)?\s*(?:of|:|is|at)?\s*(\d+\.?\d*)', 1, 1.0),
                # "trading at 25x earnings", "25.3x earnings"
                (r'(\d+\.?\d*)\s*x\s*earnings', 1, 1.0),
                # "P/E 25.3"
                (r'(?:P/?E|PE)\s+(\d+\.?\d*)', 1, 1.0),
            ],
        },
        'Market Cap': {
            'yf_keys': ['marketCap'],
            'patterns': [
                # "$2.5T market cap", "$2.5 trillion market cap"
                (r'\$\s*(\d+\.?\d*)\s*(?:T|trillion)\s*(?:market\s*cap(?:italization)?)?', 1, 1e12),
                # "$250B market cap", "$250 billion market cap"
                (r'\$\s*(\d+\.?\d*)\s*(?:B|billion)\s*(?:market\s*cap(?:italization)?)?', 1, 1e9),
                # "$250M market cap", "$250 million market cap"
                (r'\$\s*(\d+\.?\d*)\s*(?:M|million)\s*(?:market\s*cap(?:italization)?)?', 1, 1e6),
                # "market capitalization of $2.5 trillion"
                (r'market\s*cap(?:italization)?\s*(?:of|:|\s+is)\s*\$\s*(\d+\.?\d*)\s*(?:T|trillion)', 1, 1e12),
                (r'market\s*cap(?:italization)?\s*(?:of|:|\s+is)\s*\$\s*(\d+\.?\d*)\s*(?:B|billion)', 1, 1e9),
                (r'market\s*cap(?:italization)?\s*(?:of|:|\s+is)\s*\$\s*(\d+\.?\d*)\s*(?:M|million)', 1, 1e6),
            ],
        },
        'Revenue Growth': {
            'yf_keys': ['revenueGrowth'],
            'patterns': [
                # "revenue grew 15%", "revenue growth of 15%", "revenue increased 15%"
                (r'revenue\s*(?:grew|growth|increased|rising|up)\s*(?:of|by|:)?\s*(\d+\.?\d*)\s*%', 1, 0.01),
                # "15% revenue growth"
                (r'(\d+\.?\d*)\s*%\s*revenue\s*growth', 1, 0.01),
            ],
        },
        'EPS': {
            'yf_keys': ['trailingEps'],
            'patterns': [
                # "EPS of $6.50", "EPS: $6.50", "EPS is $6.50"
                (r'EPS\s*(?:of|:|is|at)\s*\$?\s*(\d+\.?\d*)', 1, 1.0),
                # "earnings per share: $6.50", "earnings per share of $6.50"
                (r'earnings\s*per\s*share\s*(?:of|:|is|at)?\s*\$?\s*(\d+\.?\d*)', 1, 1.0),
            ],
        },
        'Dividend Yield': {
            'yf_keys': ['dividendYield'],
            'patterns': [
                # "dividend yield of 2.5%", "dividend yield: 2.5%"
                (r'dividend\s*yield\s*(?:of|:|is|at)?\s*(\d+\.?\d*)\s*%', 1, 0.01),
                # "yields 2.5%"
                (r'yields?\s+(\d+\.?\d*)\s*%', 1, 0.01),
            ],
        },
        'Price': {
            'yf_keys': ['currentPrice'],
            'patterns': [
                # "trading at $150", "trades at $150.25"
                (r'trad(?:ing|es)\s*at\s*\$\s*(\d+\.?\d*)', 1, 1.0),
                # "current price of $150", "current price: $150"
                (r'current\s*price\s*(?:of|:|is|at)\s*\$\s*(\d+\.?\d*)', 1, 1.0),
                # "priced at $150", "share price of $150"
                (r'(?:share\s*)?price(?:d)?\s*(?:of|at|:)\s*\$\s*(\d+\.?\d*)', 1, 1.0),
            ],
        },
        '52-Week High': {
            'yf_keys': ['fiftyTwoWeekHigh'],
            'patterns': [
                # "52-week high of $200", "52 week high: $200"
                (r'52[\s-]*week\s*high\s*(?:of|:|is|at)?\s*\$\s*(\d+\.?\d*)', 1, 1.0),
            ],
        },
        '52-Week Low': {
            'yf_keys': ['fiftyTwoWeekLow'],
            'patterns': [
                # "52-week low of $100", "52 week low: $100"
                (r'52[\s-]*week\s*low\s*(?:of|:|is|at)?\s*\$\s*(\d+\.?\d*)', 1, 1.0),
            ],
        },
    }

    def check_analysis(self, ticker: str, analysis_text: str) -> Dict:
        """
        Cross-check AI-generated analysis against actual yfinance data.

        Returns:
        {
            'ticker': str,
            'claims_found': int,
            'claims_verified': int,
            'accuracy': float (0-1),
            'details': [
                {'metric': 'P/E Ratio', 'ai_value': 25.3, 'actual_value': 24.8,
                 'score': 1.0, 'status': 'accurate'},
                ...
            ],
            'unverifiable_claims': [...],
            'trust_score': float,
            'warning': str or None
        }
        """
        if not ticker or not analysis_text:
            logger.warning("Empty ticker or analysis text provided to check_analysis")
            return {
                'ticker': ticker or '',
                'claims_found': 0,
                'claims_verified': 0,
                'accuracy': 0.0,
                'details': [],
                'unverifiable_claims': [],
                'trust_score': 0.0,
                'warning': 'No analysis text provided',
            }

        ticker = ticker.upper().strip()
        claims = self._extract_claims(analysis_text)
        actual_data = self._get_actual_data(ticker)

        if not claims:
            logger.info(f"No verifiable financial claims found in analysis for {ticker}")
            return {
                'ticker': ticker,
                'claims_found': 0,
                'claims_verified': 0,
                'accuracy': 1.0,
                'details': [],
                'unverifiable_claims': [],
                'trust_score': 1.0,
                'warning': None,
            }

        details = []
        unverifiable = []
        scores = []

        for claim in claims:
            metric_name = claim['metric']
            ai_value = claim['value']
            yf_keys = self.METRIC_PATTERNS[metric_name]['yf_keys']

            # Find the actual value from yfinance data
            actual_value = None
            for key in yf_keys:
                val = actual_data.get(key)
                if val is not None:
                    actual_value = val
                    break

            if actual_value is None:
                unverifiable.append({
                    'metric': metric_name,
                    'ai_value': ai_value,
                    'reason': 'No actual data available from yfinance',
                })
                logger.debug(f"{ticker}: Cannot verify {metric_name} = {ai_value} (no yfinance data)")
                continue

            # Compare claimed vs actual
            score, status = self._compare(ai_value, actual_value)
            scores.append(score)

            detail = {
                'metric': metric_name,
                'ai_value': ai_value,
                'actual_value': actual_value,
                'score': score,
                'status': status,
            }
            details.append(detail)

            log_msg = (
                f"{ticker} {metric_name}: AI={ai_value}, "
                f"Actual={actual_value}, Score={score}, Status={status}"
            )
            if score < 0.5:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

        # Calculate overall accuracy and trust score
        claims_verified = len(details)
        accuracy = sum(scores) / len(scores) if scores else 0.0
        trust_score = self._calculate_trust_score(
            accuracy, claims_verified, len(unverifiable)
        )

        warning = None
        if accuracy < 0.6 and claims_verified > 0:
            warning = (
                f"Low accuracy ({accuracy:.0%}): AI analysis for {ticker} contains "
                f"significant discrepancies from actual market data. "
                f"Verified {claims_verified} claims, "
                f"{sum(1 for d in details if d['score'] == 0.0)} were inaccurate."
            )
            logger.warning(f"Cross-check warning for {ticker}: {warning}")

        result = {
            'ticker': ticker,
            'claims_found': len(claims),
            'claims_verified': claims_verified,
            'accuracy': round(accuracy, 4),
            'details': details,
            'unverifiable_claims': unverifiable,
            'trust_score': round(trust_score, 4),
            'warning': warning,
        }

        logger.info(
            f"Cross-check complete for {ticker}: "
            f"{claims_verified}/{len(claims)} verified, "
            f"accuracy={accuracy:.2%}, trust={trust_score:.2%}"
        )

        return result

    def _extract_claims(self, text: str) -> List[Dict]:
        """Extract financial metric claims from analysis text using regex."""
        claims = []
        seen_metrics = set()

        for metric_name, config in self.METRIC_PATTERNS.items():
            for pattern, group_idx, multiplier in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if metric_name in seen_metrics:
                        break
                    try:
                        raw_value = float(match.group(group_idx))
                        scaled_value = raw_value * multiplier
                        claims.append({
                            'metric': metric_name,
                            'value': scaled_value,
                            'raw_text': match.group(0),
                        })
                        seen_metrics.add(metric_name)
                        logger.debug(
                            f"Extracted claim: {metric_name} = {scaled_value} "
                            f"(raw: {raw_value}, multiplier: {multiplier}) "
                            f"from '{match.group(0)}'"
                        )
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse match for {metric_name}: {e}")
                        continue
                if metric_name in seen_metrics:
                    break

        logger.info(f"Extracted {len(claims)} financial claims from analysis text")
        return claims

    def _get_actual_data(self, ticker: str) -> Dict:
        """Get actual financial data from yfinance for cross-checking."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or len(info) < 3:
                logger.warning(f"Minimal or no yfinance data for {ticker}")
                return {}

            # Extract the specific keys we need for cross-checking
            relevant_keys = [
                'trailingPE', 'forwardPE', 'marketCap', 'revenueGrowth',
                'trailingEps', 'dividendYield', 'currentPrice',
                'regularMarketPrice', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
            ]

            data = {}
            for key in relevant_keys:
                val = info.get(key)
                if val is not None:
                    try:
                        data[key] = float(val)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert {key}={val} to float for {ticker}")

            # Fallback: use regularMarketPrice if currentPrice is missing
            if 'currentPrice' not in data and 'regularMarketPrice' in data:
                data['currentPrice'] = data['regularMarketPrice']

            logger.info(
                f"Fetched {len(data)} actual metrics for {ticker} from yfinance"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to fetch yfinance data for {ticker}: {e}")
            return {}

    def _compare(
        self, claimed: float, actual: float, tolerance: float = 0.10
    ) -> Tuple[float, str]:
        """
        Compare claimed vs actual value.

        Scoring:
        - 1.0 (accurate): within 10% tolerance
        - 0.5 (approximate): within 25% tolerance
        - 0.0 (inaccurate): beyond 25% tolerance

        Returns (score, status).
        """
        if actual == 0:
            # Avoid division by zero; check if claimed is also ~0
            if abs(claimed) < 1e-6:
                return (1.0, 'accurate')
            return (0.0, 'inaccurate')

        deviation = abs(claimed - actual) / abs(actual)

        if deviation <= tolerance:
            return (1.0, 'accurate')
        elif deviation <= 0.25:
            return (0.5, 'approximate')
        else:
            return (0.0, 'inaccurate')

    def _calculate_trust_score(
        self, accuracy: float, verified_count: int, unverifiable_count: int
    ) -> float:
        """
        Calculate an overall trust score considering accuracy, coverage, and
        unverifiable claims.

        - Base: the raw accuracy of verified claims
        - Penalty: reduce trust if many claims could not be verified
        - Bonus: higher trust if more claims were successfully verified
        """
        if verified_count == 0 and unverifiable_count == 0:
            # No claims at all -- nothing to distrust
            return 1.0

        if verified_count == 0:
            # Claims were found but none could be verified
            return 0.5

        total_claims = verified_count + unverifiable_count
        coverage = verified_count / total_claims

        # Trust = accuracy weighted by coverage
        # If we could only verify a small fraction, we are less confident
        trust = accuracy * (0.7 + 0.3 * coverage)

        # Small bonus for having more verified data points (caps at 5+)
        if verified_count >= 5:
            trust = min(1.0, trust + 0.05)
        elif verified_count >= 3:
            trust = min(1.0, trust + 0.02)

        return max(0.0, min(1.0, trust))


# Singleton
ai_crosscheck = AICrossCheck()
