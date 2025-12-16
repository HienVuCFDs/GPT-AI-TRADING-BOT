"""
Feature Processor cho Trading AI Models
Chuyển đổi raw data từ indicators, patterns, SR thành features cho models
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingFeatureProcessor:
    """
    Process trading data thành features cho AI models
    
    Features:
    - Indicators: RSI, MACD, ADX, ATR, Stoch, EMA, BB (20 features)
    - Patterns: Candlestick + Price patterns (29 features)
    - SR: Support/Resistance levels (10 features)
    """
    
    # Indicator names to extract
    INDICATOR_NAMES = [
        'RSI14', 'RSI', 'rsi',
        'MACD_12_26_9', 'MACD', 'macd',
        'MACDs_12_26_9', 'MACD_signal', 'macd_signal',
        'MACDh_12_26_9', 'MACD_hist', 'macd_hist',
        'ADX14', 'ADX', 'adx',
        'ATR14', 'ATR', 'atr',
        'StochK_14_3', 'STOCHk', 'stoch_k',
        'StochD_14_3', 'STOCHd', 'stoch_d',
        'EMA20', 'ema20',
        'EMA50', 'ema50',
        'EMA200', 'ema200',
        'BB_upper', 'bb_upper', 'BBU',
        'BB_middle', 'bb_middle', 'BBM',
        'BB_lower', 'bb_lower', 'BBL',
        'close', 'Close',
        'high', 'High',
        'low', 'Low',
        'open', 'Open',
        'volume', 'Volume',
    ]
    
    # Pattern names
    CANDLE_PATTERNS = [
        'bullish_engulfing', 'bearish_engulfing',
        'hammer', 'shooting_star',
        'doji', 'morning_star', 'evening_star',
        'three_white_soldiers', 'three_black_crows',
        'bullish_harami', 'bearish_harami',
        'tweezer_top', 'tweezer_bottom',
        'piercing_line', 'dark_cloud_cover',
    ]
    
    PRICE_PATTERNS = [
        'double_top', 'double_bottom',
        'head_shoulders', 'inverse_head_shoulders',
        'ascending_triangle', 'descending_triangle',
        'symmetric_triangle', 'wedge_up', 'wedge_down',
        'channel_up', 'channel_down',
        'flag_bull', 'flag_bear',
        'cup_handle',
    ]
    
    def __init__(self):
        self.n_indicators = 20
        self.n_patterns = len(self.CANDLE_PATTERNS) + len(self.PRICE_PATTERNS)
        self.n_sr = 10
        
    def process_training_sample(self, data: Dict) -> Dict[str, np.ndarray]:
        """
        Process 1 training sample
        
        Returns:
            Dict with keys: indicators, patterns, sr_features
        """
        indicators = self._extract_indicators(data)
        patterns = self._extract_patterns(data)
        sr_features = self._extract_sr_features(data)
        
        return {
            'indicators': indicators,
            'patterns': patterns,
            'sr_features': sr_features,
        }
    
    def _extract_indicators(self, data: Dict) -> np.ndarray:
        """Extract và normalize indicators"""
        indicators = data.get('indicators', {})
        
        # Get M15 or first available timeframe
        if isinstance(indicators, dict):
            tf_data = indicators.get('M15', indicators.get('H1', indicators))
            if isinstance(tf_data, list) and tf_data:
                tf_data = tf_data[-1]  # Get latest
        else:
            tf_data = {}
        
        features = np.zeros(self.n_indicators, dtype=np.float32)
        
        # Extract key indicators
        feature_map = [
            ('RSI14', 'RSI', 'rsi'),
            ('MACD_12_26_9', 'MACD', 'macd'),
            ('MACDs_12_26_9', 'MACD_signal', 'macd_signal'),
            ('MACDh_12_26_9', 'MACD_hist', 'macd_hist'),
            ('ADX14', 'ADX', 'adx'),
            ('ATR14', 'ATR', 'atr'),
            ('StochK_14_3', 'STOCHk', 'stoch_k'),
            ('StochD_14_3', 'STOCHd', 'stoch_d'),
            ('EMA20', 'ema20'),
            ('EMA50', 'ema50'),
            ('EMA200', 'ema200'),
            ('BB_upper', 'BBU', 'bb_upper'),
            ('BB_middle', 'BBM', 'bb_middle'),
            ('BB_lower', 'BBL', 'bb_lower'),
            ('close', 'Close'),
            ('high', 'High'),
            ('low', 'Low'),
            ('open', 'Open'),
            ('volume', 'Volume'),
            ('OBV', 'obv'),
        ]
        
        for i, names in enumerate(feature_map[:self.n_indicators]):
            value = None
            for name in names if isinstance(names, tuple) else [names]:
                value = tf_data.get(name)
                if value is not None:
                    break
            features[i] = float(value) if value is not None else 0.0
        
        # Normalize
        features = self._normalize_indicators(features)
        
        return features
    
    def _normalize_indicators(self, features: np.ndarray) -> np.ndarray:
        """Normalize indicators to reasonable ranges"""
        normalized = features.copy()
        
        # RSI: 0-100 -> 0-1
        if abs(normalized[0]) > 0:
            normalized[0] = normalized[0] / 100.0
        
        # Stoch K, D: 0-100 -> 0-1
        if abs(normalized[6]) > 0:
            normalized[6] = normalized[6] / 100.0
        if abs(normalized[7]) > 0:
            normalized[7] = normalized[7] / 100.0
        
        # ADX: 0-100 -> 0-1
        if abs(normalized[4]) > 0:
            normalized[4] = normalized[4] / 100.0
        
        return normalized
    
    def _extract_patterns(self, data: Dict) -> np.ndarray:
        """Extract pattern features"""
        patterns_data = data.get('patterns', {})
        candle = patterns_data.get('candle_patterns', patterns_data.get('candlestick', {}))
        price = patterns_data.get('price_patterns', patterns_data.get('chart', {}))
        
        features = np.zeros(self.n_patterns, dtype=np.float32)
        
        # Candle patterns
        for i, pattern in enumerate(self.CANDLE_PATTERNS):
            value = candle.get(pattern, 0)
            if isinstance(value, dict):
                value = value.get('detected', 0) or value.get('strength', 0)
            features[i] = float(value) if value else 0.0
        
        # Price patterns
        offset = len(self.CANDLE_PATTERNS)
        for i, pattern in enumerate(self.PRICE_PATTERNS):
            value = price.get(pattern, 0)
            if isinstance(value, dict):
                value = value.get('detected', 0) or value.get('confidence', 0)
            features[offset + i] = float(value) if value else 0.0
        
        return features
    
    def _extract_sr_features(self, data: Dict) -> np.ndarray:
        """Extract support/resistance features"""
        sr_data = data.get('trendline_sr', data.get('sr_levels', {}))
        
        features = np.zeros(self.n_sr, dtype=np.float32)
        
        # Get current price
        indicators = data.get('indicators', {})
        tf_data = indicators.get('M15', indicators)
        if isinstance(tf_data, list) and tf_data:
            tf_data = tf_data[-1]
        current_price = tf_data.get('close', tf_data.get('Close', 100.0)) if isinstance(tf_data, dict) else 100.0
        
        # Supports
        supports = sr_data.get('supports', sr_data.get('support', []))
        if isinstance(supports, list) and supports:
            nearest_support = min(supports, key=lambda x: abs(current_price - (x if isinstance(x, (int, float)) else x.get('price', 0))))
            if isinstance(nearest_support, dict):
                nearest_support = nearest_support.get('price', 0)
            features[0] = (current_price - nearest_support) / current_price if current_price > 0 else 0
            features[1] = float(nearest_support)
        
        # Resistances
        resistances = sr_data.get('resistances', sr_data.get('resistance', []))
        if isinstance(resistances, list) and resistances:
            nearest_resistance = min(resistances, key=lambda x: abs(current_price - (x if isinstance(x, (int, float)) else x.get('price', 0))))
            if isinstance(nearest_resistance, dict):
                nearest_resistance = nearest_resistance.get('price', 0)
            features[2] = (nearest_resistance - current_price) / current_price if current_price > 0 else 0
            features[3] = float(nearest_resistance)
        
        # Trend
        trend = sr_data.get('trend', sr_data.get('trend_direction', 'sideways'))
        if trend in ['up', 'bullish', 'uptrend']:
            features[4] = 1.0
        elif trend in ['down', 'bearish', 'downtrend']:
            features[4] = -1.0
        else:
            features[4] = 0.0
        
        # Trend strength
        features[5] = float(sr_data.get('trend_strength', sr_data.get('strength', 0.5)))
        
        # SR ratio
        if features[1] > 0 and features[3] > 0:
            features[6] = (current_price - features[1]) / (features[3] - features[1]) if features[3] != features[1] else 0.5
        
        # Trendline touches
        features[7] = float(sr_data.get('support_touches', 0))
        features[8] = float(sr_data.get('resistance_touches', 0))
        
        # Breakout status
        features[9] = float(sr_data.get('breakout', 0))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        names = []
        
        # Indicator names
        indicator_names = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'atr',
            'stoch_k', 'stoch_d', 'ema20', 'ema50', 'ema200',
            'bb_upper', 'bb_middle', 'bb_lower',
            'close', 'high', 'low', 'open', 'volume', 'obv'
        ]
        names.extend(indicator_names[:self.n_indicators])
        
        # Pattern names
        names.extend(self.CANDLE_PATTERNS)
        names.extend(self.PRICE_PATTERNS)
        
        # SR names
        sr_names = [
            'dist_to_support', 'support_level', 'dist_to_resistance', 'resistance_level',
            'trend_direction', 'trend_strength', 'sr_ratio',
            'support_touches', 'resistance_touches', 'breakout'
        ]
        names.extend(sr_names)
        
        return names
