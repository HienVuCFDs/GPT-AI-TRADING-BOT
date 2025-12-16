"""
Deep Learning Trading Model
Multi-input Neural Network để predict trading signals

Architecture:
- Input: 60+ features (indicators, patterns, SR, news)
- Hidden layers: Dense với BatchNorm và Dropout
- Output: 3 classes (BUY, HOLD, SELL)
"""

import os
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Using sklearn fallback.")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not installed.")


# ============================================================================
# PyTorch Deep Learning Model
# ============================================================================

if TORCH_AVAILABLE:
    
    class TradingDataset(Dataset):
        """Custom Dataset for trading data"""
        
        def __init__(self, features: np.ndarray, labels: np.ndarray):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    
    class TradingNeuralNetwork(nn.Module):
        """
        Deep Neural Network for Trading Signal Prediction
        
        Architecture:
        - Input layer: n_features
        - Hidden layer 1: 128 units + BatchNorm + ReLU + Dropout(0.3)
        - Hidden layer 2: 64 units + BatchNorm + ReLU + Dropout(0.3)
        - Hidden layer 3: 32 units + BatchNorm + ReLU + Dropout(0.2)
        - Output layer: 3 units (BUY, HOLD, SELL)
        """
        
        def __init__(self, n_features: int, n_classes: int = 3):
            super(TradingNeuralNetwork, self).__init__()
            
            self.network = nn.Sequential(
                # Layer 1
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Layer 2
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Layer 3
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                # Output layer
                nn.Linear(32, n_classes)
            )
        
        def forward(self, x):
            return self.network(x)
        
        def predict_proba(self, x):
            """Get probability distribution"""
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                proba = torch.softmax(logits, dim=1)
            return proba


class DeepLearningTrader:
    """
    Main class for Deep Learning Trading Model
    
    Supports:
    - PyTorch Neural Network (primary)
    - Sklearn GradientBoosting (fallback)
    
    Features:
    - Train từ pending files
    - Incremental learning (thêm data mới)
    - Save/Load models
    - Predict signals với confidence
    """
    
    SIGNAL_MAP = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    SIGNAL_MAP_REVERSE = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    def __init__(self, model_dir: str = "ai_models/trained"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        self.n_features = 0
        self.is_trained = False
        self.use_torch = TORCH_AVAILABLE
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': 0
        }
        
        logger.info(f"DeepLearningTrader initialized. Device: {self.device}")
    
    def prepare_data(self, pending_dir: str = "ai_models/training/pending") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load và prepare data từ pending files
        
        Returns:
            Tuple[X, y] - features và labels
        """
        from .feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        self.feature_names = extractor.get_feature_names()
        self.n_features = len(self.feature_names)
        
        X_list = []
        y_list = []
        
        pending_path = Path(pending_dir)
        files = list(pending_path.glob("*.json"))
        
        logger.info(f"Loading {len(files)} files from {pending_dir}...")
        
        processed = 0
        skipped = 0
        
        for file in files:
            try:
                features, signal_type, confidence = extractor.extract_from_pending_file(str(file))
                
                if features is not None and signal_type is not None:
                    # Only use samples with reasonable confidence
                    if confidence >= 40:  # Min confidence threshold
                        X_list.append(features)
                        y_list.append(self.SIGNAL_MAP.get(signal_type, 1))  # Default to HOLD
                        processed += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                skipped += 1
        
        logger.info(f"Processed: {processed}, Skipped: {skipped}")
        
        if not X_list:
            raise ValueError("No valid training data found!")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001, val_split: float = 0.2) -> Dict:
        """
        Train model
        
        Args:
            X: Features array
            y: Labels array
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            
        Returns:
            Dict with training history
        """
        
        if self.use_torch:
            return self._train_torch(X, y, epochs, batch_size, learning_rate, val_split)
        else:
            return self._train_sklearn(X, y, val_split)
    
    def _train_torch(self, X: np.ndarray, y: np.ndarray,
                     epochs: int, batch_size: int, 
                     learning_rate: float, val_split: float) -> Dict:
        """Train PyTorch model"""
        
        logger.info("🧠 Training PyTorch Neural Network...")
        
        # Create dataset
        dataset = TradingDataset(X, y)
        
        # Split train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.n_features = X.shape[1]
        self.model = TradingNeuralNetwork(self.n_features, n_classes=3).to(self.device)
        
        # Loss and optimizer
        # Use class weights to handle imbalanced data
        class_counts = np.bincount(y, minlength=3)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * 3
        weights = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.training_history['epochs'] = epoch + 1
        
        # Load best model
        self._load_checkpoint('best_model.pt')
        
        self.is_trained = True
        
        logger.info(f"✅ Training completed! Best Val Loss: {best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_sklearn(self, X: np.ndarray, y: np.ndarray, val_split: float) -> Dict:
        """Train sklearn fallback model"""
        
        logger.info("🌲 Training Sklearn GradientBoosting...")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
        
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['epochs'] = 1
        
        self.is_trained = True
        
        logger.info(f"✅ Training completed! Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Classification report
        y_pred = self.model.predict(X_val)
        logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred, target_names=['SELL', 'HOLD', 'BUY'])}")
        
        return self.training_history
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict signal từ features
        
        Args:
            features: Feature array (1D or 2D)
            
        Returns:
            Tuple[signal, confidence, probabilities]
        """
        if not self.is_trained:
            logger.warning("Model not trained!")
            return 'HOLD', 0.0, {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Handle NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.use_torch and isinstance(self.model, nn.Module):
            # PyTorch prediction
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features).to(self.device)
                proba = self.model.predict_proba(x).cpu().numpy()[0]
        else:
            # Sklearn prediction
            proba = self.model.predict_proba(features)[0]
        
        # Get prediction
        pred_class = np.argmax(proba)
        confidence = proba[pred_class] * 100
        
        signal = self.SIGNAL_MAP_REVERSE[pred_class]
        
        probabilities = {
            'SELL': float(proba[0]) * 100,
            'HOLD': float(proba[1]) * 100,
            'BUY': float(proba[2]) * 100
        }
        
        return signal, confidence, probabilities
    
    def predict_from_data(self, data: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict từ raw data dict (như từ comprehensive_aggregator)
        
        Args:
            data: Dict chứa indicators, patterns, trendline_sr, news
            
        Returns:
            Tuple[signal, confidence, probabilities]
        """
        from .feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        features, _, _ = extractor.extract_features(data)
        
        if features is None:
            return 'HOLD', 0.0, {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
        
        return self.predict(features)
    
    def save(self, name: str = "trading_model"):
        """Save model to disk"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.model_dir / f"{name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_torch and isinstance(self.model, nn.Module):
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_features': self.n_features,
                'training_history': self.training_history,
            }, save_dir / 'model.pt')
        else:
            # Save sklearn model
            joblib.dump(self.model, save_dir / 'model.joblib')
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, save_dir / 'scaler.joblib')
        
        # Save config
        config = {
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'use_torch': self.use_torch,
            'training_history': self.training_history,
            'created_at': timestamp,
            'signal_map': self.SIGNAL_MAP,
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update latest symlink
        latest_path = self.model_dir / 'latest'
        if latest_path.exists():
            latest_path.unlink()
        
        # Write latest path
        with open(self.model_dir / 'latest.txt', 'w') as f:
            f.write(str(save_dir))
        
        logger.info(f"✅ Model saved to {save_dir}")
        
        return str(save_dir)
    
    def load(self, path: str = None):
        """Load model from disk"""
        
        if path is None:
            # Load latest
            latest_file = self.model_dir / 'latest.txt'
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    path = f.read().strip()
            else:
                raise FileNotFoundError("No saved model found!")
        
        load_dir = Path(path)
        
        # Load config
        with open(load_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.n_features = config['n_features']
        self.feature_names = config.get('feature_names', [])
        self.use_torch = config.get('use_torch', True)
        self.training_history = config.get('training_history', {})
        
        # Load scaler - try both naming conventions
        scaler_path = load_dir / 'scaler.joblib'
        if not scaler_path.exists():
            scaler_path = load_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load model - try both naming conventions
        if self.use_torch and TORCH_AVAILABLE:
            model_path = load_dir / 'model.pt'
            if not model_path.exists():
                model_path = load_dir / 'model.pth'
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = TradingNeuralNetwork(self.n_features, n_classes=3).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            self.model = joblib.load(load_dir / 'model.joblib')
        
        self.is_trained = True
        
        logger.info(f"✅ Model loaded from {load_dir}")
    
    def _save_checkpoint(self, filename: str):
        """Save checkpoint during training"""
        if self.use_torch and isinstance(self.model, nn.Module):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_features': self.n_features,
            }, self.model_dir / filename)
    
    def _load_checkpoint(self, filename: str):
        """Load checkpoint"""
        if self.use_torch:
            checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================================
# Quick training function
# ============================================================================

def train_model(pending_dir: str = "ai_models/training/pending",
                epochs: int = 100,
                save_name: str = "trading_model") -> DeepLearningTrader:
    """
    Quick function để train model
    
    Usage:
        model = train_model()
        signal, conf, proba = model.predict_from_data(data)
    """
    
    trader = DeepLearningTrader()
    
    # Prepare data
    X, y = trader.prepare_data(pending_dir)
    
    # Train
    history = trader.train(X, y, epochs=epochs)
    
    # Save
    save_path = trader.save(save_name)
    
    print(f"\n{'='*50}")
    print(f"✅ Training Complete!")
    print(f"{'='*50}")
    print(f"📊 Epochs: {history.get('epochs', 0)}")
    print(f"📈 Final Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"📈 Final Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"💾 Model saved: {save_path}")
    print(f"{'='*50}")
    
    return trader


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Train model
    model = train_model(epochs=100)
