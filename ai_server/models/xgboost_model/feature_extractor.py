"""
Feature Extractor for ML Trading Model
Chuyển đổi raw data từ indicators, patterns, SR, news thành features cho ML model
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract features từ trading data để train ML model
    
    Features được chia thành các nhóm:
    1. Technical Indicators (20 features)
    2. Pattern Features (15 features)
    3. Support/Resistance Features (10 features)
    4. News/Sentiment Features (5 features)
    5. Derived Features (10 features)
    
    Total: ~60 features
    """
    
    # Feature names for reference
    FEATURE_NAMES = [
        # Technical Indicators - M15
        'rsi_m15', 'stoch_k_m15', 'macd_m15', 'macd_signal_m15', 'macd_hist_m15',
        'adx_m15', 'atr_m15', 'bb_position_m15',
        # Technical Indicators - H1
        'rsi_h1', 'stoch_k_h1', 'macd_h1', 'macd_signal_h1', 'macd_hist_h1',
        'adx_h1', 'atr_h1', 'bb_position_h1',
        # Price vs EMA
        'price_vs_ema20_m15', 'price_vs_ema50_m15',
        'price_vs_ema20_h1', 'price_vs_ema50_h1',
        # Pattern Features
        'candle_pattern_bullish', 'candle_pattern_bearish', 'candle_pattern_score',
        'price_pattern_bullish', 'price_pattern_bearish', 'price_pattern_confidence',
        'pattern_overall_bias',  # -1 bearish, 0 neutral, 1 bullish
        # Support/Resistance
        'distance_to_support', 'distance_to_resistance', 'sr_ratio',
        'trend_strength', 'trend_direction',  # -1 down, 0 sideways, 1 up
        # Sideway trading features
        'is_sideway', 'position_in_range', 'sideway_signal',
        # Derived Features
        'rsi_divergence', 'macd_crossover', 'momentum_score',
        'volatility_ratio', 'trend_consistency',
        # Signal context
        'buy_count_m15', 'sell_count_m15', 'buy_count_h1', 'sell_count_h1',
        # News sentiment (if available)
        'news_sentiment', 'news_impact',
    ]
    
    def __init__(self):
        self.feature_count = len(self.FEATURE_NAMES)
        logger.info(f"FeatureExtractor initialized with {self.feature_count} features")
    
    def extract_from_pending_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """
        Extract features từ 1 pending file
        
        Returns:
            Tuple[features, signal_type, confidence]
            - features: numpy array of features
            - signal_type: BUY, SELL, HOLD
            - confidence: 0-100
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.extract_features(data)
            
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return None, None, None
    
    def extract_features(self, data: Dict) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """
        Extract features từ data dict
        
        Args:
            data: Dict chứa indicators, patterns, trendline_sr, news
            
        Returns:
            Tuple[features, signal_type, confidence]
        """
        try:
            features = []
            
            # 1. Technical Indicators
            indicators = data.get('indicators', {})
            
            # M15 timeframe
            m15_raw = indicators.get('M15', {})
            m15 = m15_raw[-1] if isinstance(m15_raw, list) and m15_raw else m15_raw if isinstance(m15_raw, dict) else {}
            features.extend(self._extract_indicator_features(m15))
            
            # H1 timeframe
            h1_raw = indicators.get('H1', {})
            h1 = h1_raw[-1] if isinstance(h1_raw, list) and h1_raw else h1_raw if isinstance(h1_raw, dict) else {}
            features.extend(self._extract_indicator_features(h1))
            
            # Price vs EMA
            close_m15 = m15.get('close', m15.get('Close', 0)) if isinstance(m15, dict) else 0
            close_h1 = h1.get('close', h1.get('Close', 0)) if isinstance(h1, dict) else 0
            
            ema20_m15 = m15.get('ema_20', m15.get('EMA_20', m15.get('EMA20', close_m15))) if isinstance(m15, dict) else close_m15
            ema50_m15 = m15.get('ema_50', m15.get('EMA_50', m15.get('EMA50', close_m15))) if isinstance(m15, dict) else close_m15
            ema20_h1 = h1.get('ema_20', h1.get('EMA_20', h1.get('EMA20', close_h1))) if isinstance(h1, dict) else close_h1
            ema50_h1 = h1.get('ema_50', h1.get('EMA_50', h1.get('EMA50', close_h1))) if isinstance(h1, dict) else close_h1
            
            features.append(self._safe_ratio(close_m15 - ema20_m15, close_m15))
            features.append(self._safe_ratio(close_m15 - ema50_m15, close_m15))
            features.append(self._safe_ratio(close_h1 - ema20_h1, close_h1))
            features.append(self._safe_ratio(close_h1 - ema50_h1, close_h1))
            
            # 2. Pattern Features
            patterns = data.get('patterns', {})
            features.extend(self._extract_pattern_features(patterns))
            
            # 3. Support/Resistance Features
            sr_data = data.get('trendline_sr', {})
            features.extend(self._extract_sr_features(sr_data, close_h1))
            
            # 4. Derived Features
            features.extend(self._extract_derived_features(m15, h1))
            
            # 5. Signal context
            features.append(m15.get('buy_count', 0))
            features.append(m15.get('sell_count', 0))
            features.append(h1.get('buy_count', 0))
            features.append(h1.get('sell_count', 0))
            
            # 6. News sentiment
            news = data.get('news', [])
            features.extend(self._extract_news_features(news))
            
            # Get target (signal)
            signal_type = data.get('signal_type', 'HOLD')
            confidence = data.get('confidence', 50)
            
            # Pad features to expected length
            while len(features) < self.feature_count:
                features.append(0.0)
            
            # Truncate if too many
            features = features[:self.feature_count]
            
            return np.array(features, dtype=np.float32), signal_type, confidence
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None, None, None
    
    def _extract_indicator_features(self, tf_data: Any) -> List[float]:
        """Extract features từ 1 timeframe
        
        Args:
            tf_data: Can be Dict or List of dicts (take last element)
        """
        features = []
        
        # Handle list format (take last element)
        if isinstance(tf_data, list):
            tf_data = tf_data[-1] if tf_data else {}
        if not isinstance(tf_data, dict):
            tf_data = {}
        
        # RSI (0-100, normalize to 0-1)
        rsi = tf_data.get('rsi', tf_data.get('RSI', 50))
        features.append(rsi / 100.0)
        
        # Stochastic K (0-100, normalize to 0-1)
        stoch_k = tf_data.get('stoch_k', 50)
        features.append(stoch_k / 100.0)
        
        # MACD (normalize by ATR)
        atr = tf_data.get('atr', 1)
        macd = tf_data.get('macd', 0)
        macd_signal = tf_data.get('macd_signal', 0)
        macd_hist = tf_data.get('macd_hist', 0)
        
        features.append(self._safe_ratio(macd, atr))
        features.append(self._safe_ratio(macd_signal, atr))
        features.append(self._safe_ratio(macd_hist, atr))
        
        # ADX (0-100, normalize to 0-1)
        adx = tf_data.get('adx', 25)
        features.append(adx / 100.0)
        
        # ATR (as percentage of close)
        close = tf_data.get('close', 1)
        features.append(self._safe_ratio(atr, close) * 100)  # ATR%
        
        # Bollinger Band position (-1 to 1)
        bb_upper = tf_data.get('bb_upper', close)
        bb_lower = tf_data.get('bb_lower', close)
        bb_mid = (bb_upper + bb_lower) / 2
        bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1
        bb_position = (close - bb_mid) / (bb_range / 2) if bb_range > 0 else 0
        features.append(np.clip(bb_position, -1, 1))
        
        return features
    
    def _extract_pattern_features(self, patterns: Dict) -> List[float]:
        """Extract pattern features"""
        features = []
        
        # Candle patterns
        candle_patterns = patterns.get('candle_patterns', [])
        bullish_count = 0
        bearish_count = 0
        total_score = 0
        
        for p in candle_patterns:
            signal = p.get('signal', 'Neutral')
            score = p.get('score', 0)
            if signal == 'Bullish':
                bullish_count += 1
            elif signal == 'Bearish':
                bearish_count += 1
            total_score += score
        
        features.append(bullish_count)
        features.append(bearish_count)
        features.append(total_score)
        
        # Price patterns
        price_patterns = patterns.get('price_patterns', [])
        price_bullish = 0
        price_bearish = 0
        price_confidence = 0
        
        for p in price_patterns:
            signal = p.get('signal', 'Neutral')
            conf = p.get('confidence', 0)
            if signal == 'Bullish':
                price_bullish += 1
                price_confidence += conf
            elif signal == 'Bearish':
                price_bearish += 1
                price_confidence -= conf
        
        features.append(price_bullish)
        features.append(price_bearish)
        features.append(price_confidence)
        
        # Overall bias
        bias = patterns.get('overall_bias', 'NEUTRAL')
        bias_score = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}.get(bias, 0)
        features.append(bias_score)
        
        return features
    
    def _extract_sr_features(self, sr_data: Dict, current_price: float) -> List[float]:
        """Extract Support/Resistance features with SIDEWAY TRADING logic"""
        features = []
        
        # Parse support levels
        supports = sr_data.get('supports', [])
        resistances = sr_data.get('resistances', [])
        
        # Convert to floats
        support_levels = []
        for s in supports:
            try:
                support_levels.append(float(s))
            except:
                pass
        
        resistance_levels = []
        for r in resistances:
            try:
                resistance_levels.append(float(r))
            except:
                pass
        
        # Distance to nearest support (as %)
        if support_levels and current_price > 0:
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            distance_support = (current_price - nearest_support) / current_price * 100
        else:
            distance_support = 5.0  # Default 5%
        
        # Distance to nearest resistance (as %)
        if resistance_levels and current_price > 0:
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            distance_resistance = (nearest_resistance - current_price) / current_price * 100
        else:
            distance_resistance = 5.0  # Default 5%
        
        features.append(np.clip(distance_support, 0, 20))
        features.append(np.clip(distance_resistance, 0, 20))
        
        # SR ratio (positive = closer to support = potential buy)
        sr_ratio = distance_resistance - distance_support
        features.append(np.clip(sr_ratio / 10, -1, 1))
        
        # Trend
        trend = sr_data.get('trend', sr_data.get('trend_direction', 'Sideways'))
        trend_map = {
            'Strong Uptrend': (1.0, 1.0),
            'Uptrend': (0.5, 0.7),
            'Weak Uptrend': (0.2, 0.3),
            'Sideways': (0.0, 0.0),
            'Sideway': (0.0, 0.0),
            'Weak Downtrend': (-0.2, 0.3),
            'Downtrend': (-0.5, 0.7),
            'Strong Downtrend': (-1.0, 1.0),
        }
        trend_direction, trend_strength = trend_map.get(trend, (0.0, 0.0))
        features.append(trend_strength)
        features.append(trend_direction)
        
        # ========== NEW: SIDEWAY TRADING FEATURES ==========
        # Is Sideway market?
        is_sideway = 1.0 if trend in ['Sideways', 'Sideway'] else 0.0
        features.append(is_sideway)
        
        # Sideway range info
        sideway_range = sr_data.get('sideway_range', {})
        if sideway_range and is_sideway:
            range_low = sideway_range.get('range_low', current_price * 0.97)
            range_high = sideway_range.get('range_high', current_price * 1.03)
            range_width = range_high - range_low if range_high > range_low else 1
            
            # Position in range (0 = at low/support, 1 = at high/resistance)
            position_in_range = (current_price - range_low) / range_width if range_width > 0 else 0.5
            position_in_range = np.clip(position_in_range, 0, 1)
        else:
            position_in_range = 0.5
        
        features.append(position_in_range)
        
        # Sideway signal: BUY if near support (position < 0.3), SELL if near resistance (position > 0.7)
        # This feature encodes the sideway trading rule
        if is_sideway:
            if position_in_range < 0.30:
                sideway_signal = 1.0   # BUY zone (near support)
            elif position_in_range > 0.70:
                sideway_signal = -1.0  # SELL zone (near resistance)
            else:
                sideway_signal = 0.0   # NEUTRAL zone (middle)
        else:
            sideway_signal = 0.0
        
        features.append(sideway_signal)
        
        return features
    
    def _extract_derived_features(self, m15: Dict, h1: Dict) -> List[float]:
        """Extract derived/computed features"""
        features = []
        
        # RSI divergence (H1 vs M15)
        rsi_m15 = m15.get('rsi', 50)
        rsi_h1 = h1.get('rsi', 50)
        rsi_div = (rsi_h1 - rsi_m15) / 100
        features.append(rsi_div)
        
        # MACD crossover signal
        macd_h1 = h1.get('macd', 0)
        macd_signal_h1 = h1.get('macd_signal', 0)
        macd_cross = 1 if macd_h1 > macd_signal_h1 else (-1 if macd_h1 < macd_signal_h1 else 0)
        features.append(macd_cross)
        
        # Momentum score (combination of indicators)
        momentum = 0
        if rsi_h1 > 50: momentum += 0.25
        if rsi_h1 < 50: momentum -= 0.25
        if h1.get('stoch_k', 50) > 50: momentum += 0.25
        if h1.get('stoch_k', 50) < 50: momentum -= 0.25
        if macd_h1 > 0: momentum += 0.25
        if macd_h1 < 0: momentum -= 0.25
        if h1.get('macd_hist', 0) > 0: momentum += 0.25
        if h1.get('macd_hist', 0) < 0: momentum -= 0.25
        features.append(momentum)
        
        # Volatility ratio (M15 ATR / H1 ATR)
        atr_m15 = m15.get('atr', 1)
        atr_h1 = h1.get('atr', 1)
        vol_ratio = atr_m15 / atr_h1 if atr_h1 > 0 else 1
        features.append(np.clip(vol_ratio, 0, 3))
        
        # Trend consistency (do M15 and H1 agree?)
        m15_signal = m15.get('overall_signal', 'Hold')
        h1_signal = h1.get('overall_signal', 'Hold')
        if m15_signal == h1_signal:
            consistency = 1.0
        elif 'Hold' in [m15_signal, h1_signal]:
            consistency = 0.5
        else:
            consistency = 0.0  # Conflicting signals
        features.append(consistency)
        
        return features
    
    def _extract_news_features(self, news: List) -> List[float]:
        """Extract news sentiment features"""
        if not news:
            return [0.0, 0.0]  # No news
        
        sentiment_score = 0
        impact_score = 0
        
        for n in news:
            sentiment = n.get('sentiment', 'neutral')
            impact = n.get('impact', 'low')
            
            if sentiment == 'positive':
                sentiment_score += 1
            elif sentiment == 'negative':
                sentiment_score -= 1
            
            impact_map = {'low': 1, 'medium': 2, 'high': 3}
            impact_score += impact_map.get(impact, 1)
        
        # Normalize
        sentiment_score = np.clip(sentiment_score / max(len(news), 1), -1, 1)
        impact_score = np.clip(impact_score / (len(news) * 3), 0, 1)
        
        return [sentiment_score, impact_score]
    
    def _safe_ratio(self, a: float, b: float) -> float:
        """Safe division"""
        if b == 0:
            return 0.0
        return a / b
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.FEATURE_NAMES.copy()


if __name__ == "__main__":
    # Test
    extractor = FeatureExtractor()
    print(f"Feature count: {extractor.feature_count}")
    print(f"Feature names: {extractor.get_feature_names()[:10]}...")
