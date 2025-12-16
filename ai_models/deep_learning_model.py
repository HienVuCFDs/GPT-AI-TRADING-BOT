"""
🧠 Deep Learning Trading Model for comprehensive_aggregator.py
==============================================================
Standalone DL model that learns from trading data over time.
Self-contained, GPU-optimized, auto-learning from results.

Features:
- LSTM + Attention architecture
- Online learning (learns from each trade result)
- GPU acceleration when available
- Auto-saves best model
- Feature extraction from indicators, patterns, S/R, news

Author: Trading Bot AI
Version: 1.0
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from collections import deque

import numpy as np

# PyTorch imports with GPU detection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# GPU Setup
# ============================================================================
def get_device():
    """Get optimal device (GPU if available)"""
    if not TORCH_AVAILABLE:
        return None
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 GPU detected: {gpu_name}")
        # Enable cuDNN optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        return device
    else:
        logger.info("⚠️ No GPU, using CPU")
        return torch.device('cpu')


# ============================================================================
# Neural Network Architecture
# ============================================================================
class AttentionLayer(nn.Module):
    """Self-attention layer for focusing on important features"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        weights = self.attention(x)  # (batch, seq_len, 1)
        attended = torch.sum(x * weights, dim=1)  # (batch, hidden_size)
        return attended, weights


class TradingLSTM(nn.Module):
    """
    LSTM + Attention model for trading signal prediction
    
    Architecture:
    - Input embedding layer
    - Bidirectional LSTM
    - Self-attention
    - Multi-head output (signal, confidence, market regime)
    """
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3  # BUY, SELL, HOLD
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Output heads
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Market regime head (trending/ranging/volatile)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # up_trend, down_trend, ranging, volatile
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size) or (batch, input_size)
        Returns:
            signal_logits, confidence, regime_logits, attention_weights
        """
        # Handle 2D input (single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Attention
        attended, attn_weights = self.attention(lstm_out)  # (batch, hidden_size * 2)
        
        # Output heads
        signal_logits = self.signal_head(attended)  # (batch, 3)
        confidence = self.confidence_head(attended)  # (batch, 1)
        regime_logits = self.regime_head(attended)  # (batch, 4)
        
        return signal_logits, confidence.squeeze(-1), regime_logits, attn_weights


# ============================================================================
# Feature Extractor
# ============================================================================

# Import data normalizer for proper scaling
try:
    from .data_normalizer import get_normalizer
    NORMALIZER_AVAILABLE = True
except ImportError:
    try:
        from ai_models.data_normalizer import get_normalizer
        NORMALIZER_AVAILABLE = True
    except ImportError:
        NORMALIZER_AVAILABLE = False
        logger.warning("⚠️ Data normalizer not available, using default normalization")


class FeatureExtractor:
    """Extract features from trading data for DL model"""
    
    # Feature dimensions
    INDICATOR_FEATURES = 24  # RSI, MACD, ADX, etc.
    PATTERN_FEATURES = 16    # Candle patterns, price patterns
    SR_FEATURES = 8          # Support/Resistance
    NEWS_FEATURES = 4        # News sentiment
    TOTAL_FEATURES = INDICATOR_FEATURES + PATTERN_FEATURES + SR_FEATURES + NEWS_FEATURES
    
    def __init__(self, use_normalizer: bool = True):
        self.feature_stats = {}  # For normalization
        self.use_normalizer = use_normalizer and NORMALIZER_AVAILABLE
        self._normalizer = None
        
        if self.use_normalizer:
            try:
                self._normalizer = get_normalizer()
                logger.info("✅ Using fitted normalizer for feature extraction")
            except Exception as e:
                logger.warning(f"Could not load normalizer: {e}")
                self.use_normalizer = False
    
    def extract(self, data: Dict, symbol: str = None) -> np.ndarray:
        """
        Extract features from comprehensive_aggregator data format
        
        Args:
            data: Dict containing indicators, patterns, trendline_sr, news
            symbol: Optional symbol for per-symbol normalization
            
        Returns:
            numpy array of shape (TOTAL_FEATURES,)
        """
        # Try to get symbol from data if not provided
        if symbol is None:
            symbol = data.get('symbol')
        
        features = []
        
        # 1. Indicator features (24)
        features.extend(self._extract_indicators(data.get('indicators', {}), symbol))
        
        # 2. Pattern features (16)
        features.extend(self._extract_patterns(data.get('patterns', {})))
        
        # 3. Support/Resistance features (8)
        features.extend(self._extract_sr(data.get('trendline_sr', {})))
        
        # 4. News features (4)
        features.extend(self._extract_news(data.get('news', [])))
        
        # Pad or truncate to TOTAL_FEATURES
        features = features[:self.TOTAL_FEATURES]
        while len(features) < self.TOTAL_FEATURES:
            features.append(0.0)
        
        # Final validation: replace NaN/Inf with 0
        features = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in features]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_indicators(self, indicators: Dict, symbol: str = None) -> List[float]:
        """Extract indicator features with proper normalization"""
        features = []
        
        # Get M15 and H1 data
        m15 = indicators.get('M15', indicators)
        h1 = indicators.get('H1', {})
        
        def safe_get(d, *keys, default=0.0):
            for key in keys:
                if isinstance(d, dict) and key in d:
                    val = d[key]
                    if val is not None:
                        try:
                            return float(val)
                        except:
                            pass
            return default
        
        def normalize(value: float, feature_key: str, fallback_fn=None) -> float:
            """Normalize using fitted normalizer or fallback function"""
            if self.use_normalizer and self._normalizer:
                return self._normalizer.normalize_value(value, feature_key, symbol)
            # Use fallback function or return raw value
            if fallback_fn:
                return fallback_fn(value)
            return value
        
        # M15 indicators (12) - Use normalizer when available
        features.append(normalize(safe_get(m15, 'RSI14', 'rsi14', 'rsi'), 'M15_RSI14', self._normalize_rsi))
        features.append(normalize(safe_get(m15, 'StochK_14_3', 'stochk', 'stoch_k'), 'M15_StochK', lambda x: x / 100))
        features.append(normalize(safe_get(m15, 'StochD_14_3', 'stochd', 'stoch_d'), 'M15_StochD', lambda x: x / 100))
        features.append(normalize(safe_get(m15, 'MACD_12_26_9', 'macd'), 'M15_MACD', self._normalize_macd))
        features.append(normalize(safe_get(m15, 'MACDh_12_26_9', 'macd_hist'), 'M15_MACDh', self._normalize_macd))
        features.append(normalize(safe_get(m15, 'ADX14', 'adx14', 'adx'), 'M15_ADX14', lambda x: x / 100))
        features.append(self._calc_bb_position(m15))  # Already normalized [0,1]
        features.append(self._calc_price_vs_ema(m15, 'EMA20', 'ema20'))  # Already normalized
        features.append(self._calc_price_vs_ema(m15, 'EMA50', 'ema50'))  # Already normalized
        features.append(normalize(safe_get(m15, 'ATR14', 'atr14', 'atr') / max(safe_get(m15, 'close', default=1), 1), 'M15_ATR_pct', None))
        features.append(normalize(safe_get(m15, 'CCI20', 'cci20', 'cci'), 'M15_CCI20', self._normalize_cci))
        features.append(normalize(safe_get(m15, 'WILLR14', 'willr', 'williams_r', default=-50), 'M15_WILLR', lambda x: x / 100))
        
        # H1 indicators (12) - Use normalizer when available
        features.append(normalize(safe_get(h1, 'RSI14', 'rsi14', 'rsi'), 'H1_RSI14', self._normalize_rsi))
        features.append(normalize(safe_get(h1, 'StochK_14_3', 'stochk', 'stoch_k'), 'H1_StochK', lambda x: x / 100))
        features.append(normalize(safe_get(h1, 'StochD_14_3', 'stochd', 'stoch_d'), 'H1_StochD', lambda x: x / 100))
        features.append(normalize(safe_get(h1, 'MACD_12_26_9', 'macd'), 'H1_MACD', self._normalize_macd))
        features.append(normalize(safe_get(h1, 'MACDh_12_26_9', 'macd_hist'), 'H1_MACDh', self._normalize_macd))
        features.append(normalize(safe_get(h1, 'ADX14', 'adx14', 'adx'), 'H1_ADX14', lambda x: x / 100))
        features.append(self._calc_bb_position(h1))  # Already normalized [0,1]
        features.append(self._calc_price_vs_ema(h1, 'EMA20', 'ema20'))  # Already normalized
        features.append(self._calc_price_vs_ema(h1, 'EMA50', 'ema50'))  # Already normalized
        features.append(normalize(safe_get(h1, 'ATR14', 'atr14', 'atr') / max(safe_get(h1, 'close', default=1), 1), 'H1_ATR_pct', None))
        features.append(normalize(safe_get(h1, 'CCI20', 'cci20', 'cci'), 'H1_CCI20', self._normalize_cci))
        features.append(normalize(safe_get(h1, 'WILLR14', 'willr', 'williams_r', default=-50), 'H1_WILLR', lambda x: x / 100))
        
        return features[:24]
    
    def _extract_patterns(self, patterns: Dict) -> List[float]:
        """Extract pattern features"""
        features = []
        
        # Candle patterns (8)
        candle = patterns.get('candle_patterns', patterns.get('candle', {}))
        if isinstance(candle, list):
            # Handle list of pattern names (strings) or list of dicts
            bullish = sum(1 for p in candle if isinstance(p, str) and 'bull' in p.lower() or 
                         (isinstance(p, dict) and 'bull' in str(p.get('type', '')).lower()))
            bearish = sum(1 for p in candle if isinstance(p, str) and 'bear' in p.lower() or
                         (isinstance(p, dict) and 'bear' in str(p.get('type', '')).lower()))
            features.extend([bullish / 5, bearish / 5, (bullish - bearish) / 5])
            features.extend([0.0] * 5)  # Padding
        elif isinstance(candle, dict):
            features.append(candle.get('bullish_count', 0) / 5)
            features.append(candle.get('bearish_count', 0) / 5)
            features.append(candle.get('score', 0) / 10)
            features.append(candle.get('doji', 0))
            features.append(candle.get('hammer', 0))
            features.append(candle.get('engulfing', 0))
            features.append(candle.get('morning_star', 0))
            features.append(candle.get('evening_star', 0))
        else:
            features.extend([0.0] * 8)  # Default padding for unexpected types
        
        # Price patterns (8)
        price = patterns.get('price_patterns', patterns.get('price', {}))
        if isinstance(price, list):
            # Handle list of pattern names (strings) or list of dicts
            bullish = sum(1 for p in price if isinstance(p, str) and 'bull' in p.lower() or
                         (isinstance(p, dict) and p.get('direction') == 'bullish'))
            bearish = sum(1 for p in price if isinstance(p, str) and 'bear' in p.lower() or
                         (isinstance(p, dict) and p.get('direction') == 'bearish'))
            features.extend([bullish / 3, bearish / 3, (bullish - bearish) / 3])
            features.extend([0.0] * 5)
        elif isinstance(price, dict):
            features.append(price.get('bullish_count', 0) / 3)
            features.append(price.get('bearish_count', 0) / 3)
            features.append(price.get('score', 0) / 10)
            features.append(float(price.get('double_bottom', False)))
            features.append(float(price.get('double_top', False)))
            features.append(float(price.get('head_shoulders', False)))
            features.append(float(price.get('triangle', False)))
            features.append(float(price.get('wedge', False)))
        else:
            features.extend([0.0] * 8)  # Default padding for unexpected types
        
        return features[:16]
    
    def _extract_sr(self, sr_data: Dict) -> List[float]:
        """Extract Support/Resistance features"""
        features = []
        
        # Distance to S/R
        features.append(sr_data.get('distance_to_support', 0) / 100)
        features.append(sr_data.get('distance_to_resistance', 0) / 100)
        
        # S/R ratio
        supports = sr_data.get('support_levels', sr_data.get('supports', []))
        resistances = sr_data.get('resistance_levels', sr_data.get('resistances', []))
        total = len(supports) + len(resistances)
        features.append(len(supports) / max(total, 1))
        
        # Trend
        trend = sr_data.get('trend_direction', sr_data.get('trend', 'sideway')).lower()
        features.append(1.0 if trend == 'up' else (-1.0 if trend == 'down' else 0.0))
        
        # Trend strength
        features.append(sr_data.get('trend_strength', 0) / 100)
        
        # Position in range
        features.append(sr_data.get('position_in_range', 0.5))
        
        # Is sideway
        features.append(1.0 if 'sideway' in trend or 'range' in trend else 0.0)
        
        # Breakout potential
        features.append(sr_data.get('breakout_probability', 0.5))
        
        return features[:8]
    
    def _extract_news(self, news: List) -> List[float]:
        """Extract news sentiment features"""
        features = []
        
        def parse_impact(impact_val) -> float:
            """Convert impact to numeric value"""
            if isinstance(impact_val, (int, float)):
                return float(impact_val)
            if isinstance(impact_val, str):
                impact_lower = impact_val.lower()
                if 'high' in impact_lower:
                    return 1.0
                elif 'medium' in impact_lower or 'mid' in impact_lower:
                    return 0.5
                elif 'low' in impact_lower:
                    return 0.25
            return 0.0
        
        if isinstance(news, dict):
            features.append(float(news.get('sentiment', 0) or 0))
            features.append(parse_impact(news.get('impact', 0)))
            features.append(float(news.get('news_count', 0) or 0) / 10)
            features.append(float(news.get('high_impact_count', 0) or 0) / 5)
        elif isinstance(news, list):
            if news:
                sentiments = []
                impacts = []
                high_impact = 0
                for n in news:
                    if isinstance(n, dict):
                        sent = n.get('sentiment', 0)
                        if isinstance(sent, (int, float)):
                            sentiments.append(float(sent))
                        imp = parse_impact(n.get('impact', 0))
                        impacts.append(imp)
                        if imp >= 0.8:  # High impact threshold
                            high_impact += 1
                            
                features.append(np.mean(sentiments) if sentiments else 0.0)
                features.append(np.mean(impacts) if impacts else 0.0)
                features.append(len(news) / 10)
                features.append(high_impact / 5)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features[:4]
    
    # Normalization helpers
    def _normalize_rsi(self, rsi: float) -> float:
        return (rsi - 50) / 50 if rsi else 0.0
    
    def _normalize_macd(self, macd: float) -> float:
        return np.tanh(macd / 10) if macd else 0.0
    
    def _normalize_cci(self, cci: float) -> float:
        return np.tanh(cci / 200) if cci else 0.0
    
    def _calc_bb_position(self, data: Dict) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        upper = data.get('BBU_20_2.0', data.get('bb_upper', 0))
        lower = data.get('BBL_20_2.0', data.get('bb_lower', 0))
        close = data.get('close', 0)
        
        if upper and lower and close and upper > lower:
            return (close - lower) / (upper - lower)
        return 0.5
    
    def _calc_price_vs_ema(self, data: Dict, *keys) -> float:
        """Calculate price position vs EMA"""
        close = data.get('close', 0)
        ema = None
        for key in keys:
            ema = data.get(key)
            if ema:
                break
        
        if close and ema:
            return np.tanh((close - ema) / ema * 100)
        return 0.0


# ============================================================================
# Online Learning Dataset
# ============================================================================
class TradingDataset(Dataset):
    """Dataset for online learning from trade results"""
    
    def __init__(self, max_samples: int = 10000):
        self.features = deque(maxlen=max_samples)
        self.labels = deque(maxlen=max_samples)
        self.weights = deque(maxlen=max_samples)  # Sample weights based on recency
    
    def add_sample(self, features: np.ndarray, label: int, weight: float = 1.0):
        """Add a training sample"""
        self.features.append(features)
        self.labels.append(label)
        self.weights.append(weight)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.weights[idx], dtype=torch.float32)
        )


# ============================================================================
# Main Deep Learning Trader
# ============================================================================
class DeepLearningTrader:
    """
    Deep Learning Trading Model with Online Learning
    
    Features:
    - LSTM + Attention architecture
    - GPU acceleration
    - Online learning from trade results
    - Auto-save best model
    - Confidence calibration
    """
    
    SIGNAL_MAP = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    SIGNAL_TO_IDX = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    REGIME_MAP = {0: 'UP_TREND', 1: 'DOWN_TREND', 2: 'RANGING', 3: 'VOLATILE'}
    
    def __init__(
        self,
        model_dir: str = "ai_models/saved",
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        min_confidence: float = 0.6
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        
        # Device
        self.device = get_device()
        
        # Components
        self.model = None
        self.optimizer = None
        self.feature_extractor = FeatureExtractor()
        self.dataset = TradingDataset()
        
        # Stats
        self.total_predictions = 0
        self.correct_predictions = 0
        self.trade_history = deque(maxlen=1000)
        
        # Initialize
        self._init_model()
        self._load_if_exists()
    
    def _init_model(self):
        """Initialize the neural network"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available. DL model disabled.")
            return
        
        self.model = TradingLSTM(
            input_size=FeatureExtractor.TOTAL_FEATURES,
            hidden_size=self.hidden_size
        )
        
        if self.device:
            self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"🧠 DL Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _load_if_exists(self):
        """Load saved model if exists"""
        model_path = self.model_dir / "dl_trader_best.pt"
        
        if model_path.exists() and self.model:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.total_predictions = checkpoint.get('total_predictions', 0)
                self.correct_predictions = checkpoint.get('correct_predictions', 0)
                logger.info(f"✅ Loaded DL model from {model_path}")
                logger.info(f"   Accuracy: {self.accuracy:.1%} ({self.total_predictions} predictions)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model: {e}")
    
    def save(self):
        """Save model checkpoint"""
        if not self.model:
            return
        
        model_path = self.model_dir / "dl_trader_best.pt"
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_predictions': self.total_predictions,
                'correct_predictions': self.correct_predictions,
                'timestamp': datetime.now().isoformat()
            }, model_path)
            logger.info(f"💾 Model saved to {model_path}")
        except Exception as e:
            logger.error(f"❌ Could not save model: {e}")
    
    @property
    def accuracy(self) -> float:
        """Get prediction accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return TORCH_AVAILABLE and self.model is not None
    
    @torch.inference_mode()
    def predict(self, data: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict trading signal from data
        
        Args:
            data: Dict with indicators, patterns, trendline_sr, news
            
        Returns:
            (signal, confidence, probabilities)
        """
        if not self.is_ready:
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
        
        try:
            # Extract features
            features = self.feature_extractor.extract(data)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32)
            if self.device:
                x = x.to(self.device)
            x = x.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            self.model.eval()
            signal_logits, confidence, regime_logits, _ = self.model(x)
            
            # Get probabilities
            probs = F.softmax(signal_logits, dim=-1).squeeze().cpu().numpy()
            confidence = confidence.item()
            
            # Get signal
            signal_idx = np.argmax(probs)
            signal = self.SIGNAL_MAP[signal_idx]
            
            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                signal = 'HOLD'
            
            # Get market regime
            regime_probs = F.softmax(regime_logits, dim=-1).squeeze().cpu().numpy()
            regime_idx = np.argmax(regime_probs)
            regime = self.REGIME_MAP[regime_idx]
            
            return signal, confidence * 100, {
                'SELL': float(probs[0]) * 100,
                'HOLD': float(probs[1]) * 100,
                'BUY': float(probs[2]) * 100,
                'market_regime': regime
            }
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
    
    def learn_from_result(
        self,
        data: Dict,
        predicted_signal: str,
        actual_result: str,  # 'WIN', 'LOSS', 'BREAKEVEN'
        profit_pips: float = 0
    ):
        """
        Online learning from trade result
        
        Args:
            data: Original input data
            predicted_signal: What the model predicted
            actual_result: Trade outcome
            profit_pips: Profit/loss in pips
        """
        if not self.is_ready:
            return
        
        # Determine correct label based on result
        if actual_result == 'WIN':
            # Prediction was correct
            correct_label = self.SIGNAL_TO_IDX.get(predicted_signal, 1)
            self.correct_predictions += 1
        elif actual_result == 'LOSS':
            # Prediction was wrong - opposite would have been better
            if predicted_signal == 'BUY':
                correct_label = self.SIGNAL_TO_IDX['SELL']
            elif predicted_signal == 'SELL':
                correct_label = self.SIGNAL_TO_IDX['BUY']
            else:
                correct_label = self.SIGNAL_TO_IDX['HOLD']
        else:
            # Breakeven - HOLD would have been better
            correct_label = self.SIGNAL_TO_IDX['HOLD']
        
        self.total_predictions += 1
        
        # Extract features and add to dataset
        features = self.feature_extractor.extract(data)
        weight = 1.0 + abs(profit_pips) / 100  # Higher weight for larger moves
        self.dataset.add_sample(features, correct_label, weight)
        
        # Save trade to history
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'signal': predicted_signal,
            'result': actual_result,
            'profit_pips': profit_pips
        })
        
        # Train on batch if enough samples
        if len(self.dataset) >= 32 and len(self.dataset) % 10 == 0:
            self._train_batch()
    
    def _train_batch(self, batch_size: int = 32, epochs: int = 1):
        """Train on recent samples"""
        if not self.is_ready or len(self.dataset) < batch_size:
            return
        
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        for epoch in range(epochs):
            for features, labels, weights in dataloader:
                if self.device:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    weights = weights.to(self.device)
                
                # Forward
                signal_logits, _, _, _ = self.model(features)
                
                # Weighted cross entropy loss
                loss = F.cross_entropy(signal_logits, labels, reduction='none')
                loss = (loss * weights).mean()
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Auto-save periodically
        if self.total_predictions % 100 == 0:
            self.save()
        
        logger.debug(f"📚 Trained on {len(self.dataset)} samples, loss: {total_loss:.4f}")
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        return {
            'is_ready': self.is_ready,
            'device': str(self.device) if self.device else 'N/A',
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': f"{self.accuracy:.1%}",
            'dataset_size': len(self.dataset),
            'trade_history_size': len(self.trade_history),
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def train_supervised(
        self,
        data: Dict,
        label: str,  # 'BUY', 'SELL', 'HOLD'
        weight: float = 1.0
    ):
        """
        Supervised training with known label
        
        Args:
            data: Input data dict
            label: Correct signal label
            weight: Sample weight
        """
        if not self.is_ready:
            return
        
        # Extract features and add to dataset
        features = self.feature_extractor.extract(data)
        label_idx = self.SIGNAL_TO_IDX.get(label, 1)  # Default to HOLD
        self.dataset.add_sample(features, label_idx, weight)
    
    def run_training_epoch(self, batch_size: int = 32) -> float:
        """
        Run one training epoch on the dataset
        
        Returns:
            Average loss for the epoch
        """
        if not self.is_ready or len(self.dataset) < batch_size:
            return 0.0
        
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        n_batches = 0
        
        for features, labels, weights in dataloader:
            if self.device:
                features = features.to(self.device)
                labels = labels.to(self.device)
                weights = weights.to(self.device)
            
            # Forward
            signal_logits, _, _, _ = self.model(features)
            
            # Weighted cross entropy loss
            loss = F.cross_entropy(signal_logits, labels, reduction='none')
            loss = (loss * weights).mean()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def clear_dataset(self):
        """Clear training dataset for fresh training"""
        self.dataset = TradingDataset(max_samples=10000)


# ============================================================================
# Singleton & Helper
# ============================================================================
_dl_trader_instance = None


def get_dl_trader() -> DeepLearningTrader:
    """Get singleton instance of DeepLearningTrader"""
    global _dl_trader_instance
    if _dl_trader_instance is None:
        _dl_trader_instance = DeepLearningTrader()
    return _dl_trader_instance


# ============================================================================
# Test
# ============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Testing Deep Learning Trading Model...")
    
    # Create instance
    trader = get_dl_trader()
    print(f"\n📊 Stats: {trader.get_stats()}")
    
    # Test prediction
    test_data = {
        'indicators': {
            'M15': {'RSI14': 45, 'MACD_12_26_9': 0.5, 'ADX14': 25, 'close': 1.1000},
            'H1': {'RSI14': 55, 'MACD_12_26_9': 0.3, 'ADX14': 30, 'close': 1.1000}
        },
        'patterns': {
            'candle_patterns': [{'type': 'bullish_engulfing'}],
            'price_patterns': []
        },
        'trendline_sr': {
            'trend_direction': 'up',
            'distance_to_support': 0.5,
            'distance_to_resistance': 1.2
        },
        'news': []
    }
    
    signal, confidence, probs = trader.predict(test_data)
    print(f"\n🎯 Prediction:")
    print(f"   Signal: {signal}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Probabilities: {probs}")
    
    # Test learning
    print("\n📚 Testing online learning...")
    trader.learn_from_result(test_data, signal, 'WIN', 15.0)
    trader.learn_from_result(test_data, signal, 'LOSS', -10.0)
    
    print(f"\n📊 Stats after learning: {trader.get_stats()}")
    print("\n✅ Test completed!")
