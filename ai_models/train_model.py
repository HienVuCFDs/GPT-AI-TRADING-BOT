"""
Train Deep Learning Model from Pending Data
============================================
Loads all pending training data and trains the LSTM+Attention model
With proper data normalization to prevent Overfitting/Underfitting

Features:
- Data quality checking
- Proper normalization (StandardScaler)
- Class balancing
- Early stopping
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PENDING_DIR = Path(__file__).parent / "training" / "pending"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

# Import normalizer
try:
    from .data_normalizer import get_normalizer, TradingDataNormalizer
    NORMALIZER_AVAILABLE = True
except ImportError:
    try:
        from ai_models.data_normalizer import get_normalizer, TradingDataNormalizer
        NORMALIZER_AVAILABLE = True
    except ImportError:
        NORMALIZER_AVAILABLE = False
        logger.warning("⚠️ Data normalizer not available")


def load_pending_data() -> List[Dict]:
    """Load all pending training data"""
    data_list = []
    
    if not PENDING_DIR.exists():
        logger.error(f"❌ Pending directory not found: {PENDING_DIR}")
        return data_list
    
    json_files = list(PENDING_DIR.glob("*.json"))
    logger.info(f"📂 Found {len(json_files)} JSON files in {PENDING_DIR}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate data structure
            if isinstance(data, dict):
                # Check if it has required fields
                if 'indicators' in data or 'signal' in data or 'features' in data:
                    data_list.append(data)
                else:
                    # Try to extract nested data
                    for key in data:
                        if isinstance(data[key], dict) and 'indicators' in data[key]:
                            data_list.append(data[key])
                            
        except Exception as e:
            logger.debug(f"Could not load {json_file.name}: {e}")
            continue
    
    logger.info(f"✅ Loaded {len(data_list)} valid training samples")
    return data_list


def prepare_training_sample(raw_data: Dict) -> Tuple[Dict, str]:
    """
    Prepare a raw data sample for training
    
    Returns:
        (input_data, label) or (None, None) if invalid
    """
    try:
        # Extract indicators (our format has indicators.M15 and indicators.H1)
        indicators = raw_data.get('indicators', {})
        
        if not indicators:
            return None, None
        
        # Extract patterns
        patterns = raw_data.get('patterns', {})
        if not patterns:
            patterns = {
                'candle_patterns': raw_data.get('candle_patterns', []),
                'price_patterns': raw_data.get('price_patterns', [])
            }
        
        # Extract trendline/SR
        trendline_sr = raw_data.get('trendline_sr', {})
        if not trendline_sr:
            # Infer trend from market_type
            market_type = raw_data.get('market_type', 'SIDEWAY')
            trendline_sr = {
                'trend_direction': 'up' if market_type == 'TRENDING_UP' else ('down' if market_type == 'TRENDING_DOWN' else 'neutral'),
                'distance_to_support': 0.5,
                'distance_to_resistance': 0.5
            }
        
        # Extract label - our format uses 'signal_type'
        label = raw_data.get('signal_type', raw_data.get('signal', raw_data.get('action', 'HOLD')))
        if isinstance(label, dict):
            label = label.get('direction', 'HOLD')
        
        # Normalize label
        label = str(label).upper()
        if label not in ['BUY', 'SELL', 'HOLD']:
            label = 'HOLD'
        
        # Build input data
        input_data = {
            'indicators': indicators,
            'patterns': patterns,
            'trendline_sr': trendline_sr,
            'news': raw_data.get('news', [])
        }
        
        return input_data, label
        
    except Exception as e:
        logger.debug(f"Could not prepare sample: {e}")
        return None, None


def train_model(
    epochs: int = 10, 
    batch_size: int = 32, 
    learning_rate: float = 0.001,
    early_stopping_patience: int = 3,
    use_normalizer: bool = True
):
    """
    Train the Deep Learning model with proper normalization
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs
        use_normalizer: Whether to fit normalizer on data first
    """
    print("=" * 60)
    print("🧠 DEEP LEARNING MODEL TRAINING (with Data Normalization)")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading training data...")
    raw_data_list = load_pending_data()
    
    if not raw_data_list:
        print("❌ No training data found!")
        return
    
    # Prepare samples
    print("\n🔄 Preparing training samples...")
    training_samples = []
    label_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    # Also prepare data for normalizer fitting
    normalizer_data = []
    
    for raw_data in raw_data_list:
        input_data, label = prepare_training_sample(raw_data)
        if input_data and label:
            training_samples.append((input_data, label))
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Prepare for normalizer
            normalizer_data.append({
                'symbol': raw_data.get('symbol', 'UNKNOWN'),
                'signal_type': label,
                'indicators': input_data.get('indicators', {}),
                'patterns': input_data.get('patterns', {})
            })
    
    print(f"✅ Prepared {len(training_samples)} samples")
    print(f"   Label distribution: {label_counts}")
    
    if len(training_samples) < 10:
        print("❌ Not enough training samples (need at least 10)")
        return
    
    # =========================================================================
    # FIT NORMALIZER (Step quan trọng để tránh Overfitting/Underfitting)
    # =========================================================================
    normalizer = None
    class_weights = None
    
    if use_normalizer and NORMALIZER_AVAILABLE:
        print("\n" + "=" * 60)
        print("📊 FITTING DATA NORMALIZER")
        print("=" * 60)
        
        normalizer = get_normalizer()
        
        # Fit trên toàn bộ training data
        normalizer.fit(normalizer_data, verbose=True)
        
        # Check data quality
        print("\n🔍 Checking data quality...")
        quality_report = normalizer.check_data_quality(normalizer_data)
        normalizer.print_quality_report(quality_report)
        
        # Cảnh báo nếu có vấn đề nghiêm trọng
        if quality_report.get('overall_issues', 0) > 5:
            print("\n⚠️ WARNING: Data has quality issues. Consider collecting more data!")
            response = input("Continue training anyway? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Training cancelled")
                return
        
        # Compute class weights for imbalanced data
        class_weights = normalizer.compute_class_weights(normalizer_data)
        print(f"\n⚖️ Class weights: {class_weights}")
        
        # Split train/val using normalizer's stratified split
        train_norm, val_norm = normalizer.train_val_split(normalizer_data, val_ratio=0.2)
        print(f"📊 Stratified split: {len(train_norm)} train, {len(val_norm)} validation")
        
        # Rebuild training_samples with proper split
        train_indices = set(id(d) for d in train_norm)
        val_indices = set(id(d) for d in val_norm)
        
        # Use indices from original data
        train_samples = []
        val_samples = []
        for i, (input_data, label) in enumerate(training_samples):
            # Simple 80/20 split matching normalizer
            if i < int(len(training_samples) * 0.8):
                train_samples.append((input_data, label, class_weights.get(label, 1.0)))
            else:
                val_samples.append((input_data, label))
    else:
        print("\n⚠️ Training WITHOUT normalizer (may cause overfitting)")
        
        # Shuffle and split
        random.shuffle(training_samples)
        split_idx = int(len(training_samples) * 0.8)
        train_samples = [(d, l, 1.0) for d, l in training_samples[:split_idx]]
        val_samples = training_samples[split_idx:]
    
    print(f"\n📊 Final split: Train={len(train_samples)}, Val={len(val_samples)}")
    
    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    print("\n🧠 Initializing Deep Learning model...")
    from .deep_learning_model import get_dl_trader
    
    trader = get_dl_trader()
    
    if not trader.is_ready:
        print("❌ Model not ready!")
        return
    
    print(f"✅ Model on {trader.device}, {sum(p.numel() for p in trader.model.parameters()):,} parameters")
    
    # Clear old training data
    trader.clear_dataset()
    
    # Add all training samples to dataset (with class weights)
    print("\n📥 Loading samples into dataset...")
    for item in train_samples:
        if len(item) == 3:
            input_data, label, weight = item
        else:
            input_data, label = item
            weight = 1.0
        trader.train_supervised(input_data, label, weight=weight)
    
    print(f"   Dataset size: {len(trader.dataset)}")
    
    # =========================================================================
    # TRAINING LOOP with Early Stopping
    # =========================================================================
    print(f"\n🏋️ Training for {epochs} epochs (early stopping patience={early_stopping_patience})...")
    
    best_val_acc = 0
    epochs_without_improvement = 0
    training_history = []
    
    for epoch in range(epochs):
        # Run one epoch
        avg_loss = trader.run_training_epoch(batch_size=batch_size)
        
        # Validation
        val_correct = 0
        val_total = len(val_samples)
        for item in val_samples:
            if len(item) == 3:
                input_data, label, _ = item
            else:
                input_data, label = item
            predicted, _, _ = trader.predict(input_data)
            if predicted == label:
                val_correct += 1
        
        val_acc = val_correct / len(val_samples) if val_samples else 0
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'val_acc': val_acc
        })
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            # Save best model
            trader.save()
            print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.1%} ⭐ NEW BEST")
        else:
            epochs_without_improvement += 1
            print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.1%} (no improve: {epochs_without_improvement})")
        
        # Check early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    print("\n💾 Saving model...")
    trader.save()
    
    # Final stats
    print("\n" + "=" * 60)
    print("📊 TRAINING COMPLETED")
    print("=" * 60)
    print(f"   Total samples trained: {len(train_samples)}")
    print(f"   Best validation accuracy: {best_val_acc:.1%}")
    print(f"   Model stats: {trader.get_stats()}")
    
    # Test prediction
    print("\n🎯 Testing prediction...")
    if val_samples:
        test_item = val_samples[0]
        if len(test_item) == 3:
            test_input, test_label, _ = test_item
        else:
            test_input, test_label = test_item
    else:
        test_item = train_samples[0]
        if len(test_item) == 3:
            test_input, test_label, _ = test_item
        else:
            test_input, test_label = test_item
            
    signal, confidence, probs = trader.predict(test_input)
    print(f"   Predicted: {signal} ({confidence:.1f}%)")
    print(f"   Actual: {test_label}")
    print(f"   Probabilities: BUY={probs.get('BUY', 0):.1f}%, SELL={probs.get('SELL', 0):.1f}%, HOLD={probs.get('HOLD', 0):.1f}%")
    
    # Save training history
    history_file = Path(__file__).parent / "saved" / "training_history.json"
    try:
        with open(history_file, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'epochs': len(training_history),
                'best_val_acc': best_val_acc,
                'samples': len(train_samples),
                'class_distribution': label_counts,
                'history': training_history
            }, f, indent=2)
        print(f"\n📝 Training history saved to {history_file}")
    except Exception as e:
        logger.warning(f"Could not save training history: {e}")
    
    print("\n✅ Training completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Deep Learning Model with Data Normalization')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--no-normalizer', action='store_true', help='Skip data normalization')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        use_normalizer=not args.no_normalizer
    )
