"""
ML Signal Predictor
Tích hợp với comprehensive_aggregator.py để kết hợp Rule-based + ML predictions

Usage trong comprehensive_aggregator.py:
    from ai_server.models.xgboost import MLSignalPredictor
    
    predictor = MLSignalPredictor()
    ml_signal, ml_conf, proba = predictor.predict(indicators, patterns, sr_data, news)
    
    # Combine với rule-based
    final_signal, final_conf = predictor.combine_signals(
        rule_signal, rule_conf,
        ml_signal, ml_conf
    )
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))


class MLSignalPredictor:
    """
    ML Signal Predictor - Wrapper để dễ dàng integrate với hệ thống hiện tại
    
    Features:
    - Lazy loading (load model khi cần)
    - Fallback to rule-based nếu ML fail
    - Configurable ML weight
    - Combine rule-based + ML signals
    """
    
    def __init__(self, model_path: str = None, ml_weight: float = 0.4):
        """
        Args:
            model_path: Path to trained model. If None, will load latest.
            ml_weight: Weight of ML signal (0-1). Rule weight = 1 - ml_weight
        """
        self.model_path = model_path
        self.ml_weight = ml_weight
        self.model = None
        self.extractor = None
        self.is_loaded = False
        self.load_error = None
        
        # Config
        self.config_path = Path("ai_models/ml_config.json")
        self.load_config()
    
    def load_config(self):
        """Load ML config"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.ml_weight = config.get('ml_weight', 0.4)
                self.model_path = config.get('model_path', None)
                self.enabled = config.get('enabled', True)
            else:
                self.enabled = True
        except Exception as e:
            logger.warning(f"Could not load ML config: {e}")
            self.enabled = True
    
    def save_config(self):
        """Save ML config"""
        try:
            config = {
                'ml_weight': self.ml_weight,
                'model_path': self.model_path,
                'enabled': self.enabled
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save ML config: {e}")
    
    def _lazy_load(self):
        """Lazy load model khi cần"""
        if self.is_loaded:
            return
        
        try:
            from .deep_learning_model import DeepLearningTrader
            from .feature_extractor import FeatureExtractor
            
            self.model = DeepLearningTrader()
            self.extractor = FeatureExtractor()
            
            # Try to load saved model
            try:
                self.model.load(self.model_path)
                self.is_loaded = True
                logger.info("✅ ML model loaded successfully")
            except FileNotFoundError:
                logger.warning("⚠️ No trained ML model found. Will use rule-based only.")
                self.is_loaded = False
                
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"❌ Error loading ML model: {e}")
            self.is_loaded = False
    
    def predict(self, data: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict signal từ data
        
        Args:
            data: Dict containing:
                - indicators: Dict with M15, H1 timeframe data
                - patterns: Dict with candle_patterns, price_patterns
                - trendline_sr: Dict with supports, resistances, trend
                - news: List of news items
                
        Returns:
            Tuple[signal, confidence, probabilities]
            - signal: 'BUY', 'SELL', or 'HOLD'
            - confidence: 0-100
            - probabilities: {'BUY': x, 'SELL': y, 'HOLD': z}
        """
        if not self.enabled:
            return 'HOLD', 0.0, {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
        
        self._lazy_load()
        
        if not self.is_loaded or self.model is None:
            return 'HOLD', 0.0, {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
        
        try:
            return self.model.predict_from_data(data)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 'HOLD', 0.0, {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
    
    def predict_from_aggregator(self, indicators: Dict, patterns: Dict, 
                                 trendline_sr: Dict, news: list = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict từ các components riêng lẻ (format của comprehensive_aggregator)
        
        Args:
            indicators: Dict với keys M15, H1 chứa indicator values
            patterns: Dict với candle_patterns, price_patterns
            trendline_sr: Dict với supports, resistances, trend
            news: List of news items
            
        Returns:
            Tuple[signal, confidence, probabilities]
        """
        data = {
            'indicators': indicators,
            'patterns': patterns,
            'trendline_sr': trendline_sr,
            'news': news or []
        }
        return self.predict(data)
    
    def combine_signals(self, 
                        rule_signal: str, rule_confidence: float,
                        ml_signal: str = None, ml_confidence: float = None) -> Tuple[str, float]:
        """
        Combine rule-based và ML signals
        
        Args:
            rule_signal: Signal từ rule-based ('BUY', 'SELL', 'HOLD')
            rule_confidence: Confidence từ rule-based (0-100)
            ml_signal: Signal từ ML (optional, will predict if None)
            ml_confidence: Confidence từ ML (optional)
            
        Returns:
            Tuple[final_signal, final_confidence]
        """
        # If ML not provided or disabled, return rule-based
        if ml_signal is None or not self.enabled or not self.is_loaded:
            return rule_signal, rule_confidence
        
        # Convert signals to scores (-1 to 1)
        signal_score = {'SELL': -1, 'HOLD': 0, 'BUY': 1}
        
        rule_score = signal_score.get(rule_signal, 0) * (rule_confidence / 100)
        ml_score = signal_score.get(ml_signal, 0) * (ml_confidence / 100)
        
        # Weighted combination
        rule_weight = 1 - self.ml_weight
        combined_score = rule_weight * rule_score + self.ml_weight * ml_score
        
        # Combined confidence
        combined_confidence = rule_weight * rule_confidence + self.ml_weight * ml_confidence
        
        # Determine final signal
        if combined_score > 0.2:
            final_signal = 'BUY'
        elif combined_score < -0.2:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        return final_signal, combined_confidence
    
    def get_status(self) -> Dict:
        """Get ML predictor status"""
        return {
            'enabled': self.enabled,
            'is_loaded': self.is_loaded,
            'ml_weight': self.ml_weight,
            'model_path': self.model_path,
            'load_error': self.load_error,
            'training_history': self.model.training_history if self.model else {}
        }
    
    def set_weight(self, weight: float):
        """Set ML weight (0-1)"""
        self.ml_weight = max(0, min(1, weight))
        self.save_config()
    
    def enable(self):
        """Enable ML predictions"""
        self.enabled = True
        self.save_config()
    
    def disable(self):
        """Disable ML predictions (use rule-based only)"""
        self.enabled = False
        self.save_config()


# ============================================================================
# Global instance for easy access
# ============================================================================

_global_predictor: Optional[MLSignalPredictor] = None

def get_ml_predictor() -> MLSignalPredictor:
    """Get global ML predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = MLSignalPredictor()
    return _global_predictor


def predict_signal(data: Dict) -> Tuple[str, float, Dict[str, float]]:
    """Quick function to predict signal"""
    predictor = get_ml_predictor()
    return predictor.predict(data)


def combine_with_rules(rule_signal: str, rule_confidence: float,
                       data: Dict = None) -> Tuple[str, float]:
    """
    Combine rule-based signal với ML prediction
    
    Args:
        rule_signal: Signal từ rule-based
        rule_confidence: Confidence từ rule-based
        data: Data dict để ML predict (optional)
        
    Returns:
        Tuple[final_signal, final_confidence]
    """
    predictor = get_ml_predictor()
    
    if data:
        ml_signal, ml_confidence, _ = predictor.predict(data)
    else:
        ml_signal, ml_confidence = None, None
    
    return predictor.combine_signals(rule_signal, rule_confidence, ml_signal, ml_confidence)


if __name__ == "__main__":
    # Test
    predictor = MLSignalPredictor()
    print(f"Status: {predictor.get_status()}")
