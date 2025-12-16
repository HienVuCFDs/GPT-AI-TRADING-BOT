"""
🤖 Ensemble Trading Model
========================
Kết hợp nhiều mô hình để tạo signal chính xác hơn:
- PyTorch Neural Network (Dense layers)
- XGBoost (Gradient Boosting)
- LSTM (Sequence model)

Weighted voting để ra quyết định cuối cùng.
"""

import os
import sys
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import base components
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - install with: pip install xgboost")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .feature_extractor import FeatureExtractor


# ============================================================================
# Individual Models
# ============================================================================

class PyTorchNN(nn.Module):
    """Dense Neural Network"""
    
    def __init__(self, n_features: int = 46, n_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class LSTMModel(nn.Module):
    """LSTM for sequence patterns"""
    
    def __init__(self, n_features: int = 46, hidden_size: int = 64, 
                 num_layers: int = 2, n_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Reshape input for LSTM: (batch, seq_len=1, features)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        out = lstm_out[:, -1, :]
        return self.fc(out)


# ============================================================================
# Ensemble Model
# ============================================================================

class EnsembleTrader:
    """
    Ensemble Trading Model
    Kết hợp PyTorch NN, XGBoost, LSTM với weighted voting
    """
    
    SIGNAL_MAP = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    SIGNAL_MAP_REVERSE = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    def __init__(self, model_dir: str = "ai_models/trained/ensemble"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = FeatureExtractor()
        self.n_features = len(self.extractor.FEATURE_NAMES)
        
        # Models
        self.pytorch_nn = None
        self.xgboost_model = None
        self.lstm_model = None
        self.scaler = None
        
        # Weights cho ensemble (có thể tự động điều chỉnh)
        self.weights = {
            'pytorch_nn': 0.40,
            'xgboost': 0.35,
            'lstm': 0.25
        }
        
        # Performance tracking
        self.model_performance = {
            'pytorch_nn': {'accuracy': 0, 'predictions': 0},
            'xgboost': {'accuracy': 0, 'predictions': 0},
            'lstm': {'accuracy': 0, 'predictions': 0}
        }
        
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, validation_split: float = 0.2) -> Dict:
        """
        Train tất cả models trong ensemble
        
        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,)
            epochs: Training epochs cho neural networks
            validation_split: Ratio for validation
            
        Returns:
            Training history và metrics
        """
        logger.info(f"🚀 Training Ensemble Model with {len(X)} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        history = {}
        
        # 1. Train PyTorch NN
        if TORCH_AVAILABLE:
            logger.info("📊 Training PyTorch Neural Network...")
            history['pytorch_nn'] = self._train_pytorch(
                X_train_scaled, y_train, X_val_scaled, y_val, epochs
            )
        
        # 2. Train XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("📊 Training XGBoost...")
            history['xgboost'] = self._train_xgboost(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
        
        # 3. Train LSTM
        if TORCH_AVAILABLE:
            logger.info("📊 Training LSTM...")
            history['lstm'] = self._train_lstm(
                X_train_scaled, y_train, X_val_scaled, y_val, epochs
            )
        
        # Update weights based on validation accuracy
        self._update_weights(history)
        
        self.is_trained = True
        
        # Save models
        self.save()
        
        return history
    
    def _train_pytorch(self, X_train, y_train, X_val, y_val, epochs) -> Dict:
        """Train PyTorch NN"""
        self.pytorch_nn = PyTorchNN(self.n_features).to(self.device)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        weights = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.pytorch_nn.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # Data loaders
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_val_acc = 0
        best_state = None
        patience = 0
        
        for epoch in range(epochs):
            self.pytorch_nn.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.pytorch_nn(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.pytorch_nn.eval()
            with torch.no_grad():
                val_outputs = self.pytorch_nn(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item() * 100
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.pytorch_nn.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break
        
        if best_state:
            self.pytorch_nn.load_state_dict(best_state)
        
        logger.info(f"   PyTorch NN: {best_val_acc:.2f}% validation accuracy")
        return {'val_accuracy': best_val_acc, 'epochs': epoch + 1}
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train XGBoost"""
        self.xgboost_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.xgboost_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = self.xgboost_model.predict(X_val)
        val_acc = (val_pred == y_val).mean() * 100
        
        logger.info(f"   XGBoost: {val_acc:.2f}% validation accuracy")
        return {'val_accuracy': val_acc}
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, epochs) -> Dict:
        """Train LSTM"""
        self.lstm_model = LSTMModel(self.n_features).to(self.device)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        weights = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_val_acc = 0
        best_state = None
        
        for epoch in range(epochs):
            self.lstm_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(X_val_t)
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item() * 100
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.lstm_model.state_dict().copy()
        
        if best_state:
            self.lstm_model.load_state_dict(best_state)
        
        logger.info(f"   LSTM: {best_val_acc:.2f}% validation accuracy")
        return {'val_accuracy': best_val_acc}
    
    def _update_weights(self, history: Dict):
        """Cập nhật weights dựa trên validation accuracy"""
        total_acc = 0
        
        for model_name, result in history.items():
            acc = result.get('val_accuracy', 0)
            self.model_performance[model_name]['accuracy'] = acc
            total_acc += acc
        
        if total_acc > 0:
            for model_name in self.weights:
                if model_name in history:
                    self.weights[model_name] = history[model_name]['val_accuracy'] / total_acc
        
        logger.info(f"📊 Updated weights: {self.weights}")
    
    def predict(self, data: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict signal using ensemble voting
        
        Args:
            data: Dict với indicators, patterns, trendline_sr, news
            
        Returns:
            Tuple[signal, confidence, probabilities]
        """
        if not self.is_trained:
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
        
        # Extract features
        features, _, _ = self.extractor.extract_features(data)
        if features is None:
            return 'HOLD', 0.0, {'SELL': 33.3, 'HOLD': 33.4, 'BUY': 33.3}
        
        # Scale
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        # PyTorch NN
        if self.pytorch_nn is not None:
            self.pytorch_nn.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.pytorch_nn(x)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predictions['pytorch_nn'] = proba
        
        # XGBoost
        if self.xgboost_model is not None:
            proba = self.xgboost_model.predict_proba(features_scaled)[0]
            predictions['xgboost'] = proba
        
        # LSTM
        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.lstm_model(x)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predictions['lstm'] = proba
        
        # Weighted average
        final_proba = np.zeros(3)
        total_weight = 0
        
        for model_name, proba in predictions.items():
            weight = self.weights.get(model_name, 0)
            final_proba += proba * weight
            total_weight += weight
        
        if total_weight > 0:
            final_proba /= total_weight
        
        # Get final signal
        pred_class = np.argmax(final_proba)
        signal = self.SIGNAL_MAP_REVERSE[pred_class]
        confidence = final_proba[pred_class] * 100
        
        probabilities = {
            'SELL': final_proba[0] * 100,
            'HOLD': final_proba[1] * 100,
            'BUY': final_proba[2] * 100
        }
        
        return signal, confidence, probabilities
    
    def predict_with_details(self, data: Dict) -> Dict:
        """Predict với chi tiết từng model"""
        signal, confidence, probabilities = self.predict(data)
        
        # Get individual predictions
        features, _, _ = self.extractor.extract_features(data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        individual = {}
        
        if self.pytorch_nn is not None:
            self.pytorch_nn.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.pytorch_nn(x)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(proba)
                individual['pytorch_nn'] = {
                    'signal': self.SIGNAL_MAP_REVERSE[pred],
                    'confidence': proba[pred] * 100,
                    'weight': self.weights['pytorch_nn']
                }
        
        if self.xgboost_model is not None:
            proba = self.xgboost_model.predict_proba(features_scaled)[0]
            pred = np.argmax(proba)
            individual['xgboost'] = {
                'signal': self.SIGNAL_MAP_REVERSE[pred],
                'confidence': proba[pred] * 100,
                'weight': self.weights['xgboost']
            }
        
        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.lstm_model(x)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(proba)
                individual['lstm'] = {
                    'signal': self.SIGNAL_MAP_REVERSE[pred],
                    'confidence': proba[pred] * 100,
                    'weight': self.weights['lstm']
                }
        
        return {
            'final_signal': signal,
            'final_confidence': confidence,
            'probabilities': probabilities,
            'individual_predictions': individual,
            'weights': self.weights
        }
    
    def save(self, path: str = None):
        """Save ensemble model"""
        save_dir = Path(path) if path else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'n_features': self.n_features,
            'weights': self.weights,
            'model_performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        
        # Save PyTorch NN
        if self.pytorch_nn is not None:
            torch.save(self.pytorch_nn.state_dict(), save_dir / 'pytorch_nn.pth')
        
        # Save XGBoost
        if self.xgboost_model is not None:
            joblib.dump(self.xgboost_model, save_dir / 'xgboost.pkl')
        
        # Save LSTM
        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), save_dir / 'lstm.pth')
        
        logger.info(f"✅ Ensemble model saved to {save_dir}")
    
    def load(self, path: str = None):
        """Load ensemble model"""
        load_dir = Path(path) if path else self.model_dir
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")
        
        # Load config
        with open(load_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.n_features = config['n_features']
        self.weights = config['weights']
        self.model_performance = config.get('model_performance', {})
        
        # Load scaler
        scaler_path = load_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load PyTorch NN
        nn_path = load_dir / 'pytorch_nn.pth'
        if nn_path.exists():
            self.pytorch_nn = PyTorchNN(self.n_features).to(self.device)
            self.pytorch_nn.load_state_dict(torch.load(nn_path, map_location=self.device))
            self.pytorch_nn.eval()
        
        # Load XGBoost
        xgb_path = load_dir / 'xgboost.pkl'
        if xgb_path.exists():
            self.xgboost_model = joblib.load(xgb_path)
        
        # Load LSTM
        lstm_path = load_dir / 'lstm.pth'
        if lstm_path.exists():
            self.lstm_model = LSTMModel(self.n_features).to(self.device)
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            self.lstm_model.eval()
        
        self.is_trained = True
        logger.info(f"✅ Ensemble model loaded from {load_dir}")


# ============================================================================
# Continuous Learning
# ============================================================================

class ContinuousLearner:
    """
    Tự động cập nhật model khi có data mới
    """
    
    def __init__(self, ensemble: EnsembleTrader, 
                 pending_dir: str = "ai_models/training/pending",
                 min_new_samples: int = 50):
        self.ensemble = ensemble
        self.pending_dir = Path(pending_dir)
        self.min_new_samples = min_new_samples
        self.processed_files = set()
        self.load_processed_list()
    
    def load_processed_list(self):
        """Load danh sách files đã xử lý"""
        processed_file = self.ensemble.model_dir / 'processed_files.json'
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                self.processed_files = set(json.load(f))
    
    def save_processed_list(self):
        """Save danh sách files đã xử lý"""
        processed_file = self.ensemble.model_dir / 'processed_files.json'
        with open(processed_file, 'w') as f:
            json.dump(list(self.processed_files), f)
    
    def check_new_data(self) -> int:
        """Kiểm tra số lượng data mới"""
        if not self.pending_dir.exists():
            return 0
        
        all_files = set(str(f) for f in self.pending_dir.glob("*.json"))
        new_files = all_files - self.processed_files
        return len(new_files)
    
    def retrain_if_needed(self) -> bool:
        """Retrain nếu có đủ data mới"""
        new_count = self.check_new_data()
        
        if new_count < self.min_new_samples:
            logger.info(f"📊 {new_count} new samples (need {self.min_new_samples})")
            return False
        
        logger.info(f"🔄 Found {new_count} new samples, starting incremental training...")
        
        # Collect all data
        X_list, y_list = [], []
        extractor = FeatureExtractor()
        
        for file in self.pending_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features, signal, _ = extractor.extract_features(data)
                if features is not None and signal in self.ensemble.SIGNAL_MAP:
                    X_list.append(features)
                    y_list.append(self.ensemble.SIGNAL_MAP[signal])
                
                self.processed_files.add(str(file))
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if len(X_list) < self.min_new_samples:
            return False
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Retrain
        history = self.ensemble.train(X, y, epochs=50)
        
        # Save processed list
        self.save_processed_list()
        
        logger.info(f"✅ Incremental training complete!")
        return True


# ============================================================================
# Quick functions
# ============================================================================

def train_ensemble(pending_dir: str = "ai_models/training/pending", 
                   epochs: int = 100) -> EnsembleTrader:
    """Quick function to train ensemble model"""
    
    from pathlib import Path
    import json
    
    ensemble = EnsembleTrader()
    extractor = FeatureExtractor()
    
    # Load data
    pending_path = Path(pending_dir)
    files = list(pending_path.glob("*.json"))
    
    print(f"📁 Found {len(files)} training files")
    
    X_list, y_list = [], []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features, signal, _ = extractor.extract_features(data)
            if features is not None and signal in ensemble.SIGNAL_MAP:
                X_list.append(features)
                y_list.append(ensemble.SIGNAL_MAP[signal])
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"📊 Loaded {len(X)} samples")
    
    # Train
    history = ensemble.train(X, y, epochs=epochs)
    
    return ensemble


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 ENSEMBLE TRADING MODEL")
    print("="*60 + "\n")
    
    # Train ensemble
    ensemble = train_ensemble()
    
    print("\n✅ Ensemble model ready!")
    print(f"   Weights: {ensemble.weights}")
