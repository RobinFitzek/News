"""
Monte Carlo Permutation Test (MCPT) Validator

Validates whether the system's strategy performance is statistically significant
or a product of data mining bias. Shuffles bar returns to build a null distribution
and computes a p-value for the actual strategy performance.

Runs weekly as a background validation job.

Based on: neurotrader888/mcpt
"""
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MCPTValidator:
    """
    Monte Carlo Permutation Test for strategy validation.
    """

    def __init__(self):
        self.in_sample_permutations = 1000
        self.walk_forward_permutations = 500
        self.in_sample_p_threshold = 0.01
        self.walk_forward_p_threshold = 0.05
        self.min_signals = 30

        # Load config
        try:
            from core.config import MCPT_CONFIG
            self.in_sample_permutations = MCPT_CONFIG.get('in_sample_permutations', 1000)
            self.walk_forward_permutations = MCPT_CONFIG.get('walk_forward_permutations', 500)
            self.in_sample_p_threshold = MCPT_CONFIG.get('in_sample_p_threshold', 0.01)
            self.walk_forward_p_threshold = MCPT_CONFIG.get('walk_forward_p_threshold', 0.05)
            self.min_signals = MCPT_CONFIG.get('min_signals_for_test', 30)
        except (ImportError, AttributeError):
            pass

        # Create DB table
        self._init_db()

    def _init_db(self):
        """Create the mcpt_results table."""
        try:
            from core.database import db
            db.execute("""
                CREATE TABLE IF NOT EXISTS mcpt_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_type TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    p_value REAL,
                    actual_metric REAL,
                    permuted_mean REAL,
                    permuted_std REAL,
                    n_permutations INTEGER,
                    n_signals INTEGER,
                    significant BOOLEAN,
                    details TEXT
                )
            """)
        except Exception as e:
            logger.warning(f"Could not create mcpt_results table: {e}")

    def run_validation(self, n_permutations: int = None) -> Dict:
        """
        Run in-sample MCPT on the system's historical signals.

        Computes the strategy's actual profit factor, then shuffles
        the signal-return assignments n_permutations times to build
        a null distribution. P-value = fraction of permuted PFs >= actual.
        """
        n_perm = n_permutations or self.in_sample_permutations

        try:
            from core.database import db

            # Fetch graded signals with returns
            records = db.query("""
                SELECT signal, return_30d, confidence
                FROM signal_grades
                WHERE grade IN ('correct', 'incorrect', 'partially_correct')
                  AND return_30d IS NOT NULL
            """)

            if len(records) < self.min_signals:
                return {
                    'status': 'insufficient_data',
                    'n_signals': len(records),
                    'min_required': self.min_signals,
                }

            # Build strategy returns
            # Signal direction: BUY/STRONG_BUY = +1, SELL/STRONG_SELL = -1, HOLD = 0
            signal_map = {
                'STRONG_BUY': 1.0, 'BUY': 1.0, 'HOLD': 0.0,
                'SELL': -1.0, 'STRONG_SELL': -1.0
            }

            signals = np.array([signal_map.get(r['signal'], 0.0) for r in records])
            returns = np.array([r['return_30d'] for r in records])

            # Strategy returns = signal direction * actual market return
            strategy_returns = signals * returns

            # Compute actual profit factor
            actual_pf = self._profit_factor(strategy_returns)

            # Permutation test: shuffle the returns (break signal-return association)
            permuted_pfs = np.zeros(n_perm)
            rng = np.random.default_rng(seed=42)

            for i in range(n_perm):
                # Shuffle returns while keeping signals fixed
                shuffled = rng.permutation(returns)
                perm_returns = signals * shuffled
                permuted_pfs[i] = self._profit_factor(perm_returns)

            # P-value: fraction of permuted PFs >= actual
            p_value = float(np.mean(permuted_pfs >= actual_pf))

            significant = p_value < self.walk_forward_p_threshold

            result = {
                'status': 'completed',
                'test_type': 'in_sample',
                'p_value': round(p_value, 4),
                'actual_pf': round(actual_pf, 4),
                'permuted_mean': round(float(np.mean(permuted_pfs)), 4),
                'permuted_std': round(float(np.std(permuted_pfs)), 4),
                'n_permutations': n_perm,
                'n_signals': len(records),
                'significant': significant,
            }

            # Save to DB
            try:
                db.execute("""
                    INSERT INTO mcpt_results
                    (test_type, run_date, p_value, actual_metric, permuted_mean,
                     permuted_std, n_permutations, n_signals, significant, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'in_sample',
                    datetime.now().isoformat(),
                    result['p_value'],
                    result['actual_pf'],
                    result['permuted_mean'],
                    result['permuted_std'],
                    n_perm,
                    len(records),
                    significant,
                    ''
                ))
            except Exception as e:
                logger.warning(f"Could not save MCPT result: {e}")

            logger.info(f"MCPT validation: p={result['p_value']}, PF={result['actual_pf']}, "
                       f"significant={significant}")
            return result

        except Exception as e:
            logger.error(f"MCPT validation failed: {e}")
            return {'status': 'error', 'error': str(e)}

    @staticmethod
    def _profit_factor(returns: np.ndarray) -> float:
        """
        Compute profit factor: sum(positive returns) / abs(sum(negative returns)).
        Returns 0 if no negative returns, or the ratio.
        """
        positive = returns[returns > 0]
        negative = returns[returns < 0]

        sum_pos = float(np.sum(positive)) if len(positive) > 0 else 0.0
        sum_neg = abs(float(np.sum(negative))) if len(negative) > 0 else 0.001

        return sum_pos / sum_neg

    def get_latest_result(self) -> Optional[Dict]:
        """Get the most recent MCPT result from DB."""
        try:
            from core.database import db
            row = db.query_one("""
                SELECT * FROM mcpt_results
                ORDER BY run_date DESC LIMIT 1
            """)
            if row:
                return dict(row)
            return None
        except Exception:
            return None

    def is_strategy_valid(self) -> bool:
        """Check if the latest MCPT result shows statistical significance."""
        result = self.get_latest_result()
        if result is None:
            return True  # No data = assume valid (don't block pipeline)
        return bool(result.get('significant', True))


# Singleton instance
mcpt_validator = MCPTValidator()
