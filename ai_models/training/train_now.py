"""Quick training script - fixed"""
import sys
import os

# Setup path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=" * 60)
print("🧠 DEEP LEARNING TRADING MODEL TRAINER")
print("=" * 60)

try:
    from ai_models.deep_learning_model import DeepLearningTrader
    
    print("\n📦 Creating trainer...")
    trader = DeepLearningTrader()
    
    print("\n📊 Loading and preparing data...")
    X, y = trader.prepare_data('ai_models/training/pending')
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    print("\n🏋️ Training model (50 epochs)...")
    history = trader.train(X, y, epochs=50, batch_size=32)
    
    print("\n💾 Saving model...")
    save_path = trader.save('trading_model')
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"   Final Train Accuracy: {history['train_acc'][-1]*100:.2f}%")
    print(f"   Final Val Accuracy: {history['val_acc'][-1]*100:.2f}%")
    print(f"   Model saved: {save_path}")
    print("=" * 60)
    
except Exception as e:
    import traceback
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
