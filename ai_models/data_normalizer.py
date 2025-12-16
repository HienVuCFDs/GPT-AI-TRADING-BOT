"""
🔧 Data Normalizer for AI Training
==================================
Chuẩn hóa dữ liệu để tránh Overfitting/Underfitting

Features:
- StandardScaler với fit trên training data
- Outlier detection và xử lý
- Per-symbol normalization
- Validation split
- Data quality checks

Author: Trading Bot AI
Version: 1.0
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class TradingDataNormalizer:
    """
    Chuẩn hóa dữ liệu trading cho AI training
    
    Giải quyết các vấn đề:
    1. Hardcoded ranges → Adaptive scaling từ data thực
    2. Outliers → Clip và IQR filtering  
    3. Symbol differences → Per-symbol stats
    4. Data imbalance → Class weighting
    """
    
    # Feature groups với expected ranges
    FEATURE_GROUPS = {
        'rsi': {'min': 0, 'max': 100, 'default': 50},
        'stoch': {'min': 0, 'max': 100, 'default': 50},
        'macd': {'min': -1, 'max': 1, 'default': 0},  # After normalization
        'adx': {'min': 0, 'max': 100, 'default': 25},
        'cci': {'min': -200, 'max': 200, 'default': 0},
        'atr': {'min': 0, 'max': 1, 'default': 0.01},  # Relative to price
        'bb_position': {'min': 0, 'max': 1, 'default': 0.5},
        'ema_position': {'min': -1, 'max': 1, 'default': 0},
        'williams_r': {'min': -100, 'max': 0, 'default': -50},
    }
    
    def __init__(self, save_dir: str = "ai_models/saved"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics per feature
        self.feature_stats: Dict[str, Dict] = {}
        
        # Per-symbol statistics
        self.symbol_stats: Dict[str, Dict] = {}
        
        # Global statistics
        self.global_stats = {
            'n_samples': 0,
            'n_symbols': 0,
            'class_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'last_updated': None
        }
        
        # Outlier thresholds (IQR multiplier)
        self.outlier_threshold = 1.5
        
        # Load existing stats if available
        self._load_stats()
    
    def _load_stats(self):
        """Load saved normalization statistics"""
        stats_file = self.save_dir / "normalizer_stats.pkl"
        if stats_file.exists():
            try:
                with open(stats_file, 'rb') as f:
                    saved = pickle.load(f)
                self.feature_stats = saved.get('feature_stats', {})
                self.symbol_stats = saved.get('symbol_stats', {})
                self.global_stats = saved.get('global_stats', self.global_stats)
                logger.info(f"📁 Loaded normalizer stats: {self.global_stats['n_samples']} samples")
            except Exception as e:
                logger.warning(f"Could not load normalizer stats: {e}")
    
    def save_stats(self):
        """Save normalization statistics"""
        stats_file = self.save_dir / "normalizer_stats.pkl"
        try:
            self.global_stats['last_updated'] = datetime.now().isoformat()
            with open(stats_file, 'wb') as f:
                pickle.dump({
                    'feature_stats': self.feature_stats,
                    'symbol_stats': self.symbol_stats,
                    'global_stats': self.global_stats
                }, f)
            logger.info(f"💾 Saved normalizer stats")
        except Exception as e:
            logger.error(f"Could not save normalizer stats: {e}")
    
    # =========================================================================
    # FIT: Learn statistics from training data
    # =========================================================================
    
    def fit(self, training_data: List[Dict], verbose: bool = True):
        """
        Fit normalizer trên training data
        
        Args:
            training_data: List of dicts, each containing:
                - indicators: Dict with M15, H1 data
                - patterns: Dict
                - symbol: str
                - signal_type: 'BUY', 'SELL', 'HOLD'
        """
        if verbose:
            print(f"\n📊 Fitting normalizer on {len(training_data)} samples...")
        
        # Collect all feature values
        feature_values = defaultdict(list)
        symbol_values = defaultdict(lambda: defaultdict(list))
        
        for sample in training_data:
            symbol = sample.get('symbol', 'UNKNOWN')
            signal_type = sample.get('signal_type', 'HOLD')
            
            # Update class distribution
            if signal_type in self.global_stats['class_distribution']:
                self.global_stats['class_distribution'][signal_type] += 1
            
            # Extract indicator values
            indicators = sample.get('indicators', {})
            m15 = indicators.get('M15', indicators)
            h1 = indicators.get('H1', {})
            
            # Collect values per feature
            for tf_name, tf_data in [('M15', m15), ('H1', h1)]:
                for key, value in tf_data.items() if isinstance(tf_data, dict) else []:
                    if value is not None and isinstance(value, (int, float)):
                        feature_key = f"{tf_name}_{key}"
                        feature_values[feature_key].append(float(value))
                        symbol_values[symbol][feature_key].append(float(value))
        
        # Calculate statistics for each feature
        for feature_key, values in feature_values.items():
            if len(values) < 10:  # Need minimum samples
                continue
            
            values_arr = np.array(values)
            
            # Remove outliers using IQR
            q1, q3 = np.percentile(values_arr, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            clean_values = values_arr[(values_arr >= lower_bound) & (values_arr <= upper_bound)]
            
            if len(clean_values) < 5:
                clean_values = values_arr  # Fallback to all values
            
            self.feature_stats[feature_key] = {
                'mean': float(np.mean(clean_values)),
                'std': float(np.std(clean_values)) + 1e-8,  # Avoid division by zero
                'min': float(np.min(clean_values)),
                'max': float(np.max(clean_values)),
                'q1': float(q1),
                'q3': float(q3),
                'n_samples': len(values),
                'n_outliers': len(values) - len(clean_values)
            }
        
        # Calculate per-symbol statistics
        for symbol, sym_features in symbol_values.items():
            self.symbol_stats[symbol] = {}
            for feature_key, values in sym_features.items():
                if len(values) >= 5:
                    values_arr = np.array(values)
                    self.symbol_stats[symbol][feature_key] = {
                        'mean': float(np.mean(values_arr)),
                        'std': float(np.std(values_arr)) + 1e-8,
                        'n_samples': len(values)
                    }
        
        # Update global stats
        self.global_stats['n_samples'] = len(training_data)
        self.global_stats['n_symbols'] = len(symbol_values)
        
        if verbose:
            print(f"✅ Fitted on {len(training_data)} samples, {len(self.feature_stats)} features")
            print(f"   Symbols: {list(symbol_values.keys())[:5]}...")
            print(f"   Class distribution: {self.global_stats['class_distribution']}")
        
        # Save stats
        self.save_stats()
        
        return self
    
    # =========================================================================
    # TRANSFORM: Normalize data using learned statistics
    # =========================================================================
    
    def normalize_value(
        self, 
        value: float, 
        feature_key: str, 
        symbol: str = None,
        method: str = 'zscore'
    ) -> float:
        """
        Normalize a single value
        
        Args:
            value: Raw value
            feature_key: Feature name (e.g., 'M15_RSI14')
            symbol: Optional symbol for per-symbol normalization
            method: 'zscore', 'minmax', or 'robust'
            
        Returns:
            Normalized value in range [-1, 1] or [0, 1]
        """
        if value is None:
            return 0.0
        
        # Get statistics (prefer symbol-specific, fallback to global)
        stats = None
        if symbol and symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol].get(feature_key)
        
        if stats is None:
            stats = self.feature_stats.get(feature_key)
        
        if stats is None:
            # Fallback to default normalization based on feature type
            return self._default_normalize(value, feature_key)
        
        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            normalized = (value - stats['mean']) / stats['std']
            # Clip to [-3, 3] to handle outliers
            return float(np.clip(normalized, -3, 3) / 3)
        
        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            range_val = stats['max'] - stats['min']
            if range_val < 1e-8:
                return 0.5
            normalized = (value - stats['min']) / range_val
            return float(np.clip(normalized, 0, 1))
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = (stats['q1'] + stats['q3']) / 2
            iqr = stats['q3'] - stats['q1']
            if iqr < 1e-8:
                return 0.0
            normalized = (value - median) / iqr
            return float(np.clip(normalized, -2, 2) / 2)
        
        return self._default_normalize(value, feature_key)
    
    def _default_normalize(self, value: float, feature_key: str) -> float:
        """Default normalization based on feature type"""
        key_lower = feature_key.lower()
        
        # RSI: [0, 100] → [-1, 1]
        if 'rsi' in key_lower:
            return (value - 50) / 50
        
        # Stochastic: [0, 100] → [0, 1]
        if 'stoch' in key_lower:
            return value / 100
        
        # ADX: [0, 100] → [0, 1]
        if 'adx' in key_lower:
            return value / 100
        
        # CCI: typically [-200, 200] → [-1, 1]
        if 'cci' in key_lower:
            return np.tanh(value / 200)
        
        # MACD: variable → [-1, 1]
        if 'macd' in key_lower:
            return np.tanh(value / 0.01)  # Assume small MACD values
        
        # Williams %R: [-100, 0] → [-1, 0]
        if 'willr' in key_lower or 'williams' in key_lower:
            return value / 100
        
        # Default: tanh for unknown features
        return float(np.tanh(value))
    
    def normalize_indicators(
        self, 
        indicators: Dict, 
        symbol: str = None
    ) -> Dict:
        """
        Normalize all indicators in a dictionary
        
        Args:
            indicators: Dict with raw indicator values
            symbol: Optional symbol for per-symbol normalization
            
        Returns:
            Dict with normalized values
        """
        normalized = {}
        
        for tf_name in ['M15', 'H1', 'M30', 'H4', 'D1']:
            tf_data = indicators.get(tf_name, {})
            if not isinstance(tf_data, dict):
                continue
            
            normalized[tf_name] = {}
            for key, value in tf_data.items():
                if value is not None and isinstance(value, (int, float)):
                    feature_key = f"{tf_name}_{key}"
                    normalized[tf_name][key] = self.normalize_value(
                        value, feature_key, symbol, method='zscore'
                    )
                else:
                    normalized[tf_name][key] = value
        
        # Also normalize flat indicators (no timeframe prefix)
        for key, value in indicators.items():
            if key not in ['M15', 'H1', 'M30', 'H4', 'D1'] and isinstance(value, (int, float)):
                normalized[key] = self.normalize_value(
                    value, key, symbol, method='zscore'
                )
        
        return normalized
    
    # =========================================================================
    # DATA QUALITY CHECKS
    # =========================================================================
    
    def check_data_quality(self, training_data: List[Dict]) -> Dict:
        """
        Check data quality và trả về báo cáo
        
        Returns:
            Dict with quality metrics and issues
        """
        report = {
            'total_samples': len(training_data),
            'issues': [],
            'warnings': [],
            'class_balance': {},
            'missing_features': defaultdict(int),
            'outlier_count': 0,
            'duplicate_count': 0,
            'quality_score': 100  # Start with perfect score
        }
        
        # Check class balance
        class_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        seen_hashes = set()
        
        for sample in training_data:
            signal_type = sample.get('signal_type', 'HOLD')
            class_counts[signal_type] = class_counts.get(signal_type, 0) + 1
            
            # Check for duplicates (simple hash)
            sample_hash = hash(json.dumps(sample.get('indicators', {}), sort_keys=True, default=str))
            if sample_hash in seen_hashes:
                report['duplicate_count'] += 1
            seen_hashes.add(sample_hash)
            
            # Check missing features
            indicators = sample.get('indicators', {})
            for tf in ['M15', 'H1']:
                tf_data = indicators.get(tf, {})
                for required in ['RSI14', 'MACD_12_26_9', 'ADX14']:
                    if required not in tf_data:
                        report['missing_features'][f"{tf}_{required}"] += 1
        
        report['class_balance'] = class_counts
        
        # Check class imbalance
        total = sum(class_counts.values())
        if total > 0:
            min_ratio = min(class_counts.values()) / total
            max_ratio = max(class_counts.values()) / total
            
            if max_ratio > 0.7:
                report['issues'].append(f"⚠️ Severe class imbalance: {class_counts}")
                report['quality_score'] -= 30
            elif max_ratio > 0.5:
                report['warnings'].append(f"Class slightly imbalanced: {class_counts}")
                report['quality_score'] -= 10
        
        # Check sample size
        if total < 100:
            report['issues'].append(f"⚠️ Very few samples ({total}). Need 500+ for reliable training.")
            report['quality_score'] -= 40
        elif total < 500:
            report['warnings'].append(f"Limited samples ({total}). Recommend 500+ for best results.")
            report['quality_score'] -= 15
        
        # Check duplicates
        if report['duplicate_count'] > total * 0.1:
            report['issues'].append(f"⚠️ {report['duplicate_count']} duplicate samples ({report['duplicate_count']/total*100:.1f}%)")
            report['quality_score'] -= 20
        
        # Check missing features
        for feature, count in report['missing_features'].items():
            if count > total * 0.2:
                report['warnings'].append(f"Feature {feature} missing in {count} samples")
                report['quality_score'] -= 5
        
        report['quality_score'] = max(0, report['quality_score'])
        
        return report
    
    def print_quality_report(self, report: Dict):
        """Print formatted quality report"""
        print("\n" + "="*60)
        print("📊 DATA QUALITY REPORT")
        print("="*60)
        
        print(f"\n📈 Total samples: {report['total_samples']}")
        print(f"📊 Class balance: {report['class_balance']}")
        print(f"🔄 Duplicates: {report['duplicate_count']}")
        
        if report['issues']:
            print(f"\n❌ ISSUES ({len(report['issues'])}):")
            for issue in report['issues']:
                print(f"   {issue}")
        
        if report['warnings']:
            print(f"\n⚠️ WARNINGS ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"   {warning}")
        
        score = report['quality_score']
        if score >= 80:
            emoji = "✅"
            status = "GOOD"
        elif score >= 50:
            emoji = "⚠️"
            status = "ACCEPTABLE"
        else:
            emoji = "❌"
            status = "POOR"
        
        print(f"\n{emoji} Quality Score: {score}/100 ({status})")
        print("="*60)
    
    # =========================================================================
    # TRAIN/VALIDATION SPLIT
    # =========================================================================
    
    def train_val_split(
        self, 
        data: List[Dict], 
        val_ratio: float = 0.2,
        stratify: bool = True,
        shuffle: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into training and validation sets
        
        Args:
            data: List of samples
            val_ratio: Fraction for validation (default 20%)
            stratify: Keep class balance in both sets
            shuffle: Shuffle before splitting
            
        Returns:
            (train_data, val_data)
        """
        if shuffle:
            import random
            data = data.copy()
            random.shuffle(data)
        
        if not stratify:
            split_idx = int(len(data) * (1 - val_ratio))
            return data[:split_idx], data[split_idx:]
        
        # Stratified split
        by_class = defaultdict(list)
        for sample in data:
            signal_type = sample.get('signal_type', 'HOLD')
            by_class[signal_type].append(sample)
        
        train_data = []
        val_data = []
        
        for signal_type, samples in by_class.items():
            split_idx = int(len(samples) * (1 - val_ratio))
            train_data.extend(samples[:split_idx])
            val_data.extend(samples[split_idx:])
        
        # Shuffle again
        if shuffle:
            import random
            random.shuffle(train_data)
            random.shuffle(val_data)
        
        logger.info(f"📊 Split: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    # =========================================================================
    # CLASS WEIGHTS for imbalanced data
    # =========================================================================
    
    def compute_class_weights(self, training_data: List[Dict] = None) -> Dict[str, float]:
        """
        Compute class weights for imbalanced data
        
        Returns:
            Dict with weights for each class
        """
        if training_data:
            class_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            for sample in training_data:
                signal_type = sample.get('signal_type', 'HOLD')
                class_counts[signal_type] = class_counts.get(signal_type, 0) + 1
        else:
            class_counts = self.global_stats['class_distribution']
        
        total = sum(class_counts.values())
        if total == 0:
            return {'BUY': 1.0, 'SELL': 1.0, 'HOLD': 1.0}
        
        # Inverse frequency weighting
        n_classes = len([c for c in class_counts.values() if c > 0])
        weights = {}
        for cls, count in class_counts.items():
            if count > 0:
                weights[cls] = total / (n_classes * count)
            else:
                weights[cls] = 1.0
        
        # Normalize to max = 1
        max_weight = max(weights.values())
        weights = {k: v / max_weight for k, v in weights.items()}
        
        logger.info(f"📊 Class weights: {weights}")
        return weights
    
    def get_summary(self) -> Dict:
        """Get summary of normalizer state"""
        return {
            'n_features': len(self.feature_stats),
            'n_symbols': len(self.symbol_stats),
            'n_samples_fitted': self.global_stats['n_samples'],
            'class_distribution': self.global_stats['class_distribution'],
            'last_updated': self.global_stats['last_updated'],
            'symbols': list(self.symbol_stats.keys())[:10]
        }


# ============================================================================
# Singleton instance
# ============================================================================
_normalizer_instance = None


def get_normalizer() -> TradingDataNormalizer:
    """Get singleton instance of TradingDataNormalizer"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TradingDataNormalizer()
    return _normalizer_instance


# ============================================================================
# Test
# ============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Testing Data Normalizer...")
    
    normalizer = get_normalizer()
    
    # Create test data
    test_data = []
    for i in range(100):
        test_data.append({
            'symbol': 'XAUUSD' if i % 3 == 0 else 'EURUSD',
            'signal_type': ['BUY', 'SELL', 'HOLD'][i % 3],
            'indicators': {
                'M15': {
                    'RSI14': 30 + i % 40,
                    'MACD_12_26_9': 0.001 * (i - 50),
                    'ADX14': 20 + i % 30,
                    'StochK_14_3': i % 100,
                    'close': 2000 + i
                },
                'H1': {
                    'RSI14': 35 + i % 30,
                    'MACD_12_26_9': 0.002 * (i - 50),
                    'ADX14': 25 + i % 25,
                }
            }
        })
    
    # Fit
    normalizer.fit(test_data)
    
    # Check quality
    report = normalizer.check_data_quality(test_data)
    normalizer.print_quality_report(report)
    
    # Split
    train, val = normalizer.train_val_split(test_data)
    print(f"\n📊 Split: {len(train)} train, {len(val)} val")
    
    # Compute weights
    weights = normalizer.compute_class_weights(test_data)
    print(f"📊 Class weights: {weights}")
    
    # Test normalization
    sample = test_data[0]
    normalized = normalizer.normalize_indicators(sample['indicators'], sample['symbol'])
    print(f"\n🔧 Original M15_RSI14: {sample['indicators']['M15']['RSI14']}")
    print(f"🔧 Normalized M15_RSI14: {normalized['M15']['RSI14']:.4f}")
    
    print("\n✅ Test completed!")
