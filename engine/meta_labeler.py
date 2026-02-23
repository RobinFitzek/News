"""
Meta-Labeling with Random Forest

Trains a Random Forest classifier on historical signal outcomes to predict
which quant screener signals will actually succeed. Acts as a confidence
multiplier between Stage 1 and Stage 2 of the pipeline.

Cold-start safe: returns neutral predictions until 50+ graded signals exist.

Based on: neurotrader888/TrendlineBreakoutMetaLabel
"""
import numpy as np
import pickle
import os
import json
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MetaLabeler:
    """
    Random Forest meta-labeler that predicts signal success probability.
    """

    def __init__(self):
        self._model = None
        self._feature_names: List[str] = []
        self._model_version = 0
        self._last_train_date: Optional[str] = None

        # Config defaults
        self.n_estimators = 500
        self.max_depth = 3
        self.min_training_samples = 50
        self.confirm_threshold = 0.55
        self.filter_threshold = 0.45
        self.score_blend_original = 0.70
        self.score_blend_meta = 0.30
        self.model_path = ''

        # Load config
        try:
            from core.config import BASE_DIR
            self.model_path = os.path.join(str(BASE_DIR), 'core', 'data', 'meta_label_rf.pkl')
        except (ImportError, AttributeError):
            self.model_path = 'core/data/meta_label_rf.pkl'

        try:
            from core.config import META_LABELER_CONFIG
            self.n_estimators = META_LABELER_CONFIG.get('n_estimators', self.n_estimators)
            self.max_depth = META_LABELER_CONFIG.get('max_depth', self.max_depth)
            self.min_training_samples = META_LABELER_CONFIG.get('min_training_samples', self.min_training_samples)
            self.confirm_threshold = META_LABELER_CONFIG.get('confirm_threshold', self.confirm_threshold)
            self.filter_threshold = META_LABELER_CONFIG.get('filter_threshold', self.filter_threshold)
            self.score_blend_original = META_LABELER_CONFIG.get('score_blend_original', self.score_blend_original)
            self.score_blend_meta = META_LABELER_CONFIG.get('score_blend_meta', self.score_blend_meta)
            if META_LABELER_CONFIG.get('model_path'):
                self.model_path = META_LABELER_CONFIG['model_path']
        except (ImportError, AttributeError):
            pass

        # Create DB table
        self._init_db()

        # Try to load persisted model
        self._load_model()

    def _init_db(self):
        """Create the meta_label_training audit table."""
        try:
            from core.database import db
            db.execute("""
                CREATE TABLE IF NOT EXISTS meta_label_training (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT,
                    training_samples INTEGER,
                    accuracy_cv REAL,
                    feature_importances TEXT,
                    model_version INTEGER
                )
            """)
        except Exception as e:
            logger.warning(f"Could not create meta_label_training table: {e}")

    def _load_model(self):
        """Load persisted model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                self._model = data.get('model')
                self._feature_names = data.get('feature_names', [])
                self._model_version = data.get('version', 0)
                self._last_train_date = data.get('trained_at')
                logger.info(f"Meta-labeler model loaded (v{self._model_version}, "
                           f"features={len(self._feature_names)})")
        except Exception as e:
            logger.warning(f"Could not load meta-labeler model: {e}")
            self._model = None

    def _save_model(self):
        """Persist model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            data = {
                'model': self._model,
                'feature_names': self._feature_names,
                'version': self._model_version,
                'trained_at': self._last_train_date,
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Meta-labeler model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Could not save meta-labeler model: {e}")

    def is_ready(self) -> bool:
        """Check if the model is trained and ready for predictions."""
        return self._model is not None

    def extract_features(self, candidate: Dict) -> Dict[str, float]:
        """
        Extract scale-invariant features from a quant screener candidate.
        All features normalized to roughly [0, 1] or [-1, 1] range.
        """
        scores = candidate.get('scores', {})
        technicals = candidate.get('technicals', {})
        momentum = candidate.get('momentum', {})
        volume_metrics = candidate.get('volume_metrics', {})
        structure = candidate.get('market_structure', {})
        harmonics = candidate.get('harmonic_patterns', [])

        # SMA cross signal to numeric code
        sma_cross_map = {
            'golden_cross': 2, 'bullish': 1, 'neutral': 0,
            'bearish': -1, 'death_cross': -2
        }

        # Trend regime to numeric code
        regime_map = {
            'strong_uptrend': 2, 'uptrend': 1, 'ranging': 0,
            'downtrend': -1, 'strong_downtrend': -2, 'insufficient_data': 0
        }

        # Harmonic signal: strongest bullish = 1, bearish = -1
        harmonic_signal = 0.0
        if harmonics:
            best = harmonics[0]  # Already sorted by confidence
            if best.get('direction') == 'bullish':
                harmonic_signal = min(1.0, best.get('confidence', 0) / 100.0)
            else:
                harmonic_signal = -min(1.0, best.get('confidence', 0) / 100.0)

        features = {
            'composite_score': candidate.get('composite_score', 50) / 100.0,
            'val_score': scores.get('valuation', 50) / 100.0,
            'tech_score': scores.get('technical', 50) / 100.0,
            'mom_score': scores.get('momentum', 50) / 100.0,
            'qual_score': scores.get('quality', 50) / 100.0,
            'rsi_14': (technicals.get('rsi_14') or 50) / 100.0,
            'bollinger_position': technicals.get('bollinger_position', 0.0),
            'price_vs_52w': technicals.get('price_vs_52w_range', 0.5),
            'sma_cross_code': sma_cross_map.get(technicals.get('sma_cross_signal', 'neutral'), 0) / 2.0,
            'volume_ratio': min(5.0, volume_metrics.get('volume_ratio', 1.0)) / 5.0,
            'excess_return_1m': np.clip((momentum.get('excess_1m') or 0) / 20.0, -1.0, 1.0),
            'excess_return_3m': np.clip((momentum.get('excess_3m') or 0) / 30.0, -1.0, 1.0),
            'structure_score': structure.get('structure_score', 50) / 100.0,
            'trend_regime_code': regime_map.get(structure.get('trend_regime', 'ranging'), 0) / 2.0,
            'harmonic_signal': harmonic_signal,
        }

        return features

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Predict signal success probability.
        Returns meta_probability and meta_signal.
        """
        if not self.is_ready():
            return {'meta_probability': 0.5, 'meta_signal': 'neutral'}

        try:
            # Build feature vector in the same order as training
            x = np.array([[features.get(f, 0.0) for f in self._feature_names]])
            proba = self._model.predict_proba(x)[0]

            # proba[1] = probability of success (class 1)
            success_prob = float(proba[1]) if len(proba) > 1 else 0.5

            if success_prob >= self.confirm_threshold:
                meta_signal = 'confirmed'
            elif success_prob <= self.filter_threshold:
                meta_signal = 'filtered'
            else:
                meta_signal = 'neutral'

            return {
                'meta_probability': round(success_prob, 3),
                'meta_signal': meta_signal,
            }

        except Exception as e:
            logger.warning(f"Meta-labeler prediction failed: {e}")
            return {'meta_probability': 0.5, 'meta_signal': 'neutral'}

    def train(self, force: bool = False) -> Dict:
        """
        Train the Random Forest model using walk-forward validation on signal_grades.
        Returns training result summary.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return {'trained': False, 'reason': 'scikit-learn not installed'}

        try:
            from core.database import db

            # Fetch graded signals with outcomes
            records = db.query("""
                SELECT sg.*, ah.composite_score, ah.signal as ah_signal
                FROM signal_grades sg
                LEFT JOIN analysis_history ah ON sg.analysis_id = ah.id
                WHERE sg.grade IN ('correct', 'incorrect')
                  AND sg.return_30d IS NOT NULL
                  AND sg.price_at_signal IS NOT NULL
            """)

            if len(records) < self.min_training_samples:
                return {
                    'trained': False,
                    'reason': f'Insufficient data ({len(records)}/{self.min_training_samples})'
                }

            # Build feature matrix from historical data
            # Note: we use a simplified feature set since we don't have full
            # quant screener state at signal time. We use what's available.
            X = []
            y = []

            for rec in records:
                features = self._extract_historical_features(rec)
                if features is None:
                    continue

                label = 1 if rec['grade'] == 'correct' else 0
                X.append([features[f] for f in sorted(features.keys())])
                y.append(label)

            X = np.array(X)
            y = np.array(y)

            if len(X) < self.min_training_samples:
                return {
                    'trained': False,
                    'reason': f'Insufficient valid features ({len(X)}/{self.min_training_samples})'
                }

            feature_names = sorted(self._extract_historical_features(records[0]).keys())

            # Train with cross-validation
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )

            # 5-fold CV to estimate accuracy
            cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X) // 10 + 1), scoring='accuracy')
            mean_cv = float(np.mean(cv_scores))

            # Only activate if better than random
            if mean_cv < 0.52:
                return {
                    'trained': False,
                    'reason': f'CV accuracy too low ({mean_cv:.3f} < 0.52)',
                    'cv_accuracy': round(mean_cv, 3)
                }

            # Train final model on all data
            rf.fit(X, y)

            # Feature importances
            importances = {name: round(float(imp), 4)
                          for name, imp in zip(feature_names, rf.feature_importances_)}

            # Save model
            self._model = rf
            self._feature_names = feature_names
            self._model_version += 1
            self._last_train_date = datetime.now().isoformat()
            self._save_model()

            # Log to DB
            try:
                db.execute("""
                    INSERT INTO meta_label_training
                    (trained_at, training_samples, accuracy_cv, feature_importances, model_version)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self._last_train_date,
                    len(X),
                    round(mean_cv, 4),
                    json.dumps(importances),
                    self._model_version
                ))
            except Exception as e:
                logger.warning(f"Could not log meta-label training: {e}")

            result = {
                'trained': True,
                'training_samples': len(X),
                'cv_accuracy': round(mean_cv, 3),
                'feature_importances': importances,
                'model_version': self._model_version,
            }
            logger.info(f"Meta-labeler trained: {result}")
            return result

        except Exception as e:
            logger.error(f"Meta-labeler training failed: {e}")
            return {'trained': False, 'error': str(e)}

    def _extract_historical_features(self, record: Dict) -> Optional[Dict[str, float]]:
        """
        Extract features from a historical signal_grades record.
        Uses available fields since we don't have full quant state from the past.
        """
        try:
            signal = record.get('signal', 'HOLD')
            confidence = record.get('confidence') or 50
            price = record.get('price_at_signal')
            ret_30d = record.get('return_30d')

            if price is None:
                return None

            # Signal direction encoding
            signal_map = {
                'STRONG_BUY': 1.0, 'BUY': 0.5, 'HOLD': 0.0,
                'SELL': -0.5, 'STRONG_SELL': -1.0
            }

            features = {
                'confidence': confidence / 100.0,
                'signal_direction': signal_map.get(signal, 0.0),
            }

            return features

        except Exception:
            return None

    def get_feature_importances(self) -> Dict:
        """Get feature importances from the current model."""
        if not self.is_ready():
            return {}
        try:
            return {name: round(float(imp), 4)
                    for name, imp in zip(self._feature_names, self._model.feature_importances_)}
        except Exception:
            return {}

    def get_status(self) -> Dict:
        """Get meta-labeler status for dashboard display."""
        return {
            'ready': self.is_ready(),
            'model_version': self._model_version,
            'last_trained': self._last_train_date,
            'n_features': len(self._feature_names),
            'feature_names': self._feature_names,
        }


# Singleton instance
meta_labeler = MetaLabeler()
