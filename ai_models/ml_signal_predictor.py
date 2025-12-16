"""
🤖 ML Signal Predictor for comprehensive_aggregator.py
======================================================
Combines Rule-based signals with Deep Learning predictions.
Self-learning model that improves over time.

Usage:
    from ai_models import get_ml_predictor
    
    predictor = get_ml_predictor()
    
    # Option 1: Get ML signal directly
    signal, confidence, probs = predictor.predict(data)
    
    # Option 2: Combine with rule-based
    final_signal, final_conf = predictor.combine_signals(
        rule_signal='BUY', rule_confidence=70,
        ml_signal='BUY', ml_confidence=85
    )
    
    # Option 3: Learn from trade result
    predictor.learn_from_trade(data, 'BUY', 'WIN', profit_pips=25)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Setup dedicated shadow learning logger
shadow_logger = logging.getLogger('shadow_learning')
shadow_logger.setLevel(logging.INFO)

# Create logs directory if not exists
os.makedirs('logs', exist_ok=True)

# File handler for shadow learning log
_shadow_handler = logging.FileHandler('logs/shadow_learning.log', encoding='utf-8')
_shadow_handler.setLevel(logging.INFO)
_shadow_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
shadow_logger.addHandler(_shadow_handler)

# Prevent propagation to root logger (avoid duplicate console output)
shadow_logger.propagate = False


class MLSignalPredictor:
    """
    ML Signal Predictor - Wrapper combining Rule-based + Deep Learning
    
    Features:
    - Deep Learning model (LSTM + Attention)
    - Configurable ML weight (default 40% ML, 60% rules)
    - Online learning from trade results
    - GPU acceleration when available
    - Automatic fallback to rule-based if ML fails
    """
    
    def __init__(self, ml_weight: float = 0.0):
        """
        Args:
            ml_weight: Weight of ML signal (0-1). Rule weight = 1 - ml_weight
                       Set to 0 for Shadow Learning mode (AI learns but doesn't affect signals)
        """
        self.ml_weight = ml_weight
        self.enabled = True
        self.shadow_learning = True  # AI still predicts/learns even when weight=0
        self.dl_trader = None
        self.is_loaded = False
        self.load_error = None
        
        # Track shadow predictions for learning
        self._last_shadow_prediction = None
        self._shadow_predictions_count = 0
        
        # Config file
        self.config_path = Path("ai_models/ml_config.json")
        self._load_config()
        
        # Lazy load DL model
        self._init_dl_model()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.ml_weight = config.get('ml_weight', 0.0)
                self.enabled = config.get('enabled', True)
                self.shadow_learning = config.get('shadow_learning', True)
                
                mode = "Shadow Learning" if self.ml_weight == 0 and self.shadow_learning else f"Active ({self.ml_weight:.0%})"
                logger.info(f"📁 ML config loaded: weight={self.ml_weight}, mode={mode}")
        except Exception as e:
            logger.warning(f"Could not load ML config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "Shadow Learning (AI learns but doesn't affect signals)" if self.ml_weight == 0 and self.shadow_learning else f"Active ({self.ml_weight:.0%} influence)"
            config = {
                'ml_weight': self.ml_weight,
                'enabled': self.enabled,
                'shadow_learning': self.shadow_learning,
                'last_updated': datetime.now().isoformat(),
                'notes': f"Current mode: {mode}"
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save ML config: {e}")
    
    def _init_dl_model(self):
        """Initialize Deep Learning model"""
        try:
            from .deep_learning_model import get_dl_trader
            self.dl_trader = get_dl_trader()
            self.is_loaded = self.dl_trader.is_ready
            
            if self.is_loaded:
                logger.info(f"✅ ML Predictor ready (DL model on {self.dl_trader.device})")
            else:
                logger.warning("⚠️ DL model not ready, will use rule-based only")
                
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            logger.error(f"❌ Could not init DL model: {e}")
    
    def predict(self, data: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict signal using Deep Learning model
        
        Args:
            data: Dict containing:
                - indicators: Dict with M15, H1 data
                - patterns: Dict with candle_patterns, price_patterns
                - trendline_sr: Dict with supports, resistances, trend
                - news: List of news items
                
        Returns:
            (signal, confidence, probabilities)
            - signal: 'BUY', 'SELL', or 'HOLD'
            - confidence: 0-100
            - probabilities: {'BUY': x, 'SELL': y, 'HOLD': z, 'market_regime': str}
        """
        if not self.enabled:
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
        
        if not self.is_loaded or self.dl_trader is None:
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
        
        try:
            return self.dl_trader.predict(data)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
    
    def predict_from_components(
        self,
        indicators: Dict,
        patterns: Dict,
        trendline_sr: Dict,
        news: list = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict from separate components (for comprehensive_aggregator compatibility)
        """
        data = {
            'indicators': indicators,
            'patterns': patterns,
            'trendline_sr': trendline_sr,
            'news': news or []
        }
        return self.predict(data)
    
    def combine_signals(
        self,
        rule_signal: str,
        rule_confidence: float,
        ml_signal: str = None,
        ml_confidence: float = None,
        data: Dict = None
    ) -> Tuple[str, float]:
        """
        Combine rule-based and ML signals
        
        Args:
            rule_signal: Signal from rule-based ('BUY', 'SELL', 'HOLD')
            rule_confidence: Confidence from rule-based (0-100)
            ml_signal: Signal from ML (optional)
            ml_confidence: Confidence from ML (optional)
            data: Input data for shadow learning (optional)
            
        Returns:
            (final_signal, final_confidence)
        """
        # Shadow Learning Mode: AI predicts for learning but doesn't affect signal
        if self.ml_weight == 0 and self.shadow_learning and self.is_loaded:
            # Get ML prediction for learning purposes (if not already provided)
            if ml_signal is None and data is not None:
                ml_signal, ml_confidence, _ = self.predict(data)
            
            # Store shadow prediction for later learning
            if ml_signal is not None:
                # Get symbol if available
                symbol = "Unknown"
                if data and isinstance(data, dict):
                    if 'symbol' in data:
                        symbol = data['symbol']
                    elif 'indicators' in data:
                        # Try to extract from indicators
                        for tf_data in data.get('indicators', {}).values():
                            if isinstance(tf_data, dict) and 'symbol' in tf_data:
                                symbol = tf_data['symbol']
                                break
                
                self._last_shadow_prediction = {
                    'symbol': symbol,
                    'ml_signal': ml_signal,
                    'ml_confidence': ml_confidence,
                    'rule_signal': rule_signal,
                    'rule_confidence': rule_confidence,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                self._shadow_predictions_count += 1
                
                # Check if AI and Rule agree
                agree = "✅ AGREE" if ml_signal == rule_signal else "⚠️ DISAGREE"
                
                # Log EVERY prediction to file
                shadow_logger.info(
                    f"[{symbol}] AI: {ml_signal} ({ml_confidence:.1f}%) | "
                    f"Rule: {rule_signal} ({rule_confidence:.1f}%) | {agree}"
                )
                
                # Summary to console every 10 predictions
                if self._shadow_predictions_count % 10 == 0:
                    logger.info(f"🔮 Shadow Learning: {self._shadow_predictions_count} predictions logged to logs/shadow_learning.log")
            
            # Return pure rule-based signal (AI doesn't influence)
            return rule_signal, rule_confidence
        
        # If ML not available, return rule-based
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
        if combined_score > 0.15:
            final_signal = 'BUY'
        elif combined_score < -0.15:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Boost confidence if both agree
        if rule_signal == ml_signal and rule_signal != 'HOLD':
            combined_confidence = min(100, combined_confidence * 1.1)
        
        # Reduce confidence if they disagree
        elif rule_signal != ml_signal and rule_signal != 'HOLD' and ml_signal != 'HOLD':
            combined_confidence *= 0.7
        
        return final_signal, round(combined_confidence, 1)
    
    def learn_from_trade(
        self,
        data: Dict,
        predicted_signal: str,
        result: str,  # 'WIN', 'LOSS', 'BREAKEVEN'
        profit_pips: float = 0
    ):
        """
        Learn from trade result (online learning)
        
        Args:
            data: Original input data
            predicted_signal: What was predicted
            result: Trade outcome ('WIN', 'LOSS', 'BREAKEVEN')
            profit_pips: Profit/loss in pips
        """
        if not self.is_loaded or self.dl_trader is None:
            return
        
        try:
            self.dl_trader.learn_from_result(data, predicted_signal, result, profit_pips)
            
            # Save model periodically
            if self.dl_trader.total_predictions % 50 == 0:
                self.dl_trader.save()
                
        except Exception as e:
            logger.error(f"Learning error: {e}")
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        mode = "Shadow Learning" if self.ml_weight == 0 and self.shadow_learning else f"Active ({self.ml_weight:.0%})"
        stats = {
            'enabled': self.enabled,
            'is_loaded': self.is_loaded,
            'ml_weight': self.ml_weight,
            'shadow_learning': self.shadow_learning,
            'mode': mode,
            'shadow_predictions_count': self._shadow_predictions_count,
            'load_error': self.load_error
        }
        
        if self.dl_trader:
            stats.update(self.dl_trader.get_stats())
        
        return stats
    
    def get_last_shadow_prediction(self) -> Optional[Dict]:
        """Get the last shadow prediction (for debugging/monitoring)"""
        return self._last_shadow_prediction
    
    def enable_shadow_learning(self):
        """Enable shadow learning mode (AI learns but weight=0)"""
        self.shadow_learning = True
        self.ml_weight = 0.0
        self.save_config()
        logger.info("🔮 Shadow Learning enabled: AI will predict and learn but not affect signals")
    
    def disable_shadow_learning(self):
        """Disable shadow learning (when weight=0, AI won't predict at all)"""
        self.shadow_learning = False
        self.save_config()
        logger.info("⛔ Shadow Learning disabled")
    
    def set_weight(self, weight: float):
        """Set ML weight (0-1)"""
        self.ml_weight = max(0, min(1, weight))
        self.save_config()
        logger.info(f"📊 ML weight set to {self.ml_weight:.0%}")
    
    def enable(self):
        """Enable ML predictions"""
        self.enabled = True
        self.save_config()
        logger.info("✅ ML predictions enabled")
    
    def disable(self):
        """Disable ML predictions"""
        self.enabled = False
        self.save_config()
        logger.info("⛔ ML predictions disabled")


# ============================================================================
# Singleton
# ============================================================================
_predictor_instance = None


def get_ml_predictor() -> MLSignalPredictor:
    """Get singleton instance of MLSignalPredictor"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLSignalPredictor()
    return _predictor_instance


# ============================================================================
# Test
# ============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Testing ML Signal Predictor...")
    
    predictor = get_ml_predictor()
    print(f"\n📊 Stats: {predictor.get_stats()}")
    
    # Test data
    test_data = {
        'indicators': {
            'M15': {'RSI14': 65, 'MACD_12_26_9': 0.8, 'ADX14': 28, 'close': 1.1050},
            'H1': {'RSI14': 60, 'MACD_12_26_9': 0.5, 'ADX14': 32, 'close': 1.1050}
        },
        'patterns': {'candle_patterns': [], 'price_patterns': []},
        'trendline_sr': {'trend_direction': 'up', 'distance_to_support': 0.8},
        'news': []
    }
    
    # Test prediction
    signal, conf, probs = predictor.predict(test_data)
    print(f"\n🎯 ML Prediction: {signal} ({conf:.1f}%)")
    print(f"   Probabilities: {probs}")
    
    # Test combine
    combined_sig, combined_conf = predictor.combine_signals(
        rule_signal='BUY', rule_confidence=75,
        ml_signal=signal, ml_confidence=conf
    )
    print(f"\n🔗 Combined: {combined_sig} ({combined_conf:.1f}%)")
    print(f"   Rule: BUY (75%) | ML: {signal} ({conf:.1f}%) | Weight: {predictor.ml_weight:.0%}")
    
    print("\n✅ Test completed!")
