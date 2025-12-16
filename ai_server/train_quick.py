"""
Quick Train AI Server Models
============================
Train CNN-LSTM và Transformer từ local pending data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PENDING_DIR = Path("ai_models/training/pending")
SAVE_DIR = Path("ai_server/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> List[Dict]:
    """Load all pending JSON files"""
    data_list = []
    json_files = list(PENDING_DIR.glob("*.json"))
    logger.info(f"📂 Found {len(json_files)} JSON files")
    
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if 'indicators' in data and 'signal_type' in data:
                    data_list.append(data)
        except:
            continue
    
    logger.info(f"✅ Loaded {len(data_list)} valid samples")
    return data_list


def extract_features(data: Dict) -> Tuple[np.ndarray, int]:
    """Extract features from one sample"""
    features = []
    
    # Indicators M15
    m15 = data.get('indicators', {}).get('M15', {})
    features.extend([
        m15.get('RSI14', 50) / 100,
        m15.get('ADX14', 25) / 100,
        np.tanh(m15.get('MACD_12_26_9', 0) / 10),
        m15.get('StochK_14_3', 50) / 100,
        m15.get('ATR14', 1) / 10,
    ])
    
    # Indicators H1
    h1 = data.get('indicators', {}).get('H1', {})
    features.extend([
        h1.get('RSI14', 50) / 100,
        h1.get('ADX14', 25) / 100,
        np.tanh(h1.get('MACD_12_26_9', 0) / 10),
        h1.get('StochK_14_3', 50) / 100,
        h1.get('ATR14', 1) / 10,
    ])
    
    # Market type
    market = data.get('market_type', 'SIDEWAY')
    features.append(1.0 if 'UP' in market else (0.0 if 'DOWN' in market else 0.5))
    
    # Confidence
    features.append(data.get('confidence', 50) / 100)
    
    # Pad to fixed size
    while len(features) < 20:
        features.append(0.0)
    
    # Label
    signal = data.get('signal_type', 'HOLD')
    label_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    label = label_map.get(signal, 1)
    
    return np.array(features[:20], dtype=np.float32), label


def train_xgboost(X: np.ndarray, y: np.ndarray, epochs: int = 100):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("🌳 TRAINING XGBOOST MODEL")
    print("="*60)
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train
        model = xgb.XGBClassifier(
            n_estimators=epochs,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='hist',  # Fast histogram-based training
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        print(f"\n✅ XGBoost Validation Accuracy: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
        
        # Save
        save_path = SAVE_DIR / "xgboost_model.json"
        model.save_model(str(save_path))
        print(f"💾 Saved to {save_path}")
        
        return model, acc
        
    except ImportError:
        print("❌ XGBoost not installed. Run: pip install xgboost")
        return None, 0


def train_simple_nn(X: np.ndarray, y: np.ndarray, epochs: int = 50):
    """Train simple Neural Network with PyTorch"""
    print("\n" + "="*60)
    print("🧠 TRAINING NEURAL NETWORK")
    print("="*60)
    
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Simple MLP model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 3)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t.to(device))
                val_pred = val_out.argmax(dim=1).cpu()
                val_acc = (val_pred == y_val_t).float().mean()
            print(f"   Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc*100:.1f}%")
    
    # Final eval
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t.to(device))
        val_pred = val_out.argmax(dim=1).cpu()
        val_acc = (val_pred == y_val_t).float().mean()
    
    print(f"\n✅ Neural Network Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save
    save_path = SAVE_DIR / "simple_nn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved to {save_path}")
    
    return model, val_acc.item()


def main():
    print("="*60)
    print("🚀 AI SERVER MODELS TRAINING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading training data...")
    data_list = load_data()
    
    if len(data_list) < 100:
        print(f"❌ Not enough data ({len(data_list)} samples). Need at least 100.")
        return
    
    # Extract features
    print("\n🔄 Extracting features...")
    X_list = []
    y_list = []
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for data in data_list:
        try:
            x, y = extract_features(data)
            X_list.append(x)
            y_list.append(y)
            label_counts[y] += 1
        except:
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Extracted {len(X)} samples")
    print(f"   Labels: SELL={label_counts[0]}, HOLD={label_counts[1]}, BUY={label_counts[2]}")
    
    # Train XGBoost
    xgb_model, xgb_acc = train_xgboost(X, y, epochs=100)
    
    # Train Neural Network
    nn_model, nn_acc = train_simple_nn(X, y, epochs=50)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"   XGBoost Accuracy: {xgb_acc*100:.2f}%")
    print(f"   Neural Net Accuracy: {nn_acc*100:.2f}%")
    print(f"   Models saved in: {SAVE_DIR}")
    print("\n✅ Training completed!")


if __name__ == '__main__':
    main()
