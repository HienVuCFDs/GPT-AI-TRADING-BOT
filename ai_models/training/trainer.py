"""
🎯 Model Trainer for ai_models/
================================
Train và fine-tune Deep Learning model từ collected data.

Usage:
    from ai_models.training import train_model
    
    # Train with collected data
    train_model(epochs=50, batch_size=32)
    
    # Or use directly
    from ai_models.training.trainer import ModelTrainer
    trainer = ModelTrainer()
    trainer.train(epochs=50)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Setup
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
AI_MODELS_DIR = BASE_DIR / "ai_models"
DATA_DIR = Path(__file__).parent / "data"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


class TrainingDataset(Dataset):
    """Dataset for training examples"""
    
    def __init__(self, examples: List[Dict]):
        self.examples = [e for e in examples if e.get('has_outcome')]
        self.feature_extractor = None
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """Initialize feature extractor from ai_models"""
        try:
            sys.path.insert(0, str(AI_MODELS_DIR))
            from deep_learning_model import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
        except ImportError as e:
            logger.warning(f"Could not import FeatureExtractor: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Extract features
        if self.feature_extractor:
            data = {
                'indicators': ex.get('indicators', {}),
                'patterns': ex.get('patterns', {}),
                'trendline_sr': ex.get('trendline_sr', {}),
                'news': []
            }
            features = self.feature_extractor.extract(data)
        else:
            features = [0.0] * 48  # Default
        
        # Label: 0=SELL, 1=HOLD, 2=BUY
        signal_type = ex.get('signal_type', 'HOLD')
        outcome = ex.get('outcome', 'BREAKEVEN')
        
        # Use actual outcome as label for supervised learning
        if outcome == 'WIN':
            if signal_type == 'BUY':
                label = 2  # BUY was correct
            else:
                label = 0  # SELL was correct
        elif outcome == 'LOSS':
            if signal_type == 'BUY':
                label = 0  # Should have been SELL
            else:
                label = 2  # Should have been BUY
        else:
            label = 1  # HOLD/BREAKEVEN
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class ModelTrainer:
    """
    Trainer for ai_models/deep_learning_model.py
    """
    
    def __init__(self):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load model from ai_models/"""
        try:
            sys.path.insert(0, str(AI_MODELS_DIR))
            from deep_learning_model import get_dl_trader
            
            trader = get_dl_trader()
            self.model = trader.model
            self.model.to(self.device)
            
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
            
            logger.info(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Could not load model: {e}")
            raise
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training data"""
        data_file = DATA_DIR / "training_examples.json"
        
        if not data_file.exists():
            logger.warning("No training data found. Run data_collector first.")
            return None, None
        
        with open(data_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # Filter examples with outcomes
        valid_examples = [e for e in examples if e.get('has_outcome')]
        
        if len(valid_examples) < 10:
            logger.warning(f"Not enough training data ({len(valid_examples)} examples)")
            return None, None
        
        # Create dataset
        dataset = TrainingDataset(valid_examples)
        
        # Split 80/20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"📊 Data loaded: {train_size} train, {val_size} val examples")
        
        return train_loader, val_loader
    
    def train(self, epochs: int = 50, early_stop_patience: int = 10) -> Dict:
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            early_stop_patience: Stop if no improvement after N epochs
            
        Returns:
            Training history dict
        """
        train_loader, val_loader = self.load_data()
        
        if train_loader is None:
            return {'error': 'No training data'}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🏋️ Training for {epochs} epochs on {self.device}")
        print("=" * 50)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        print("=" * 50)
        print(f"✅ Training completed!")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Final Val Acc: {val_acc:.1f}%")
        
        # Save final model
        self._save_to_ai_models()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        path = CHECKPOINTS_DIR / "best_model.pt"
        torch.save(checkpoint, path)
        logger.debug(f"💾 Checkpoint saved: epoch {epoch}, loss {val_loss:.4f}")
    
    def _save_to_ai_models(self):
        """Save trained model to ai_models/saved/"""
        try:
            save_dir = AI_MODELS_DIR / "saved"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model weights
            model_path = save_dir / f"model_{timestamp}.pt"
            torch.save(self.model.state_dict(), model_path)
            
            # Save as latest
            latest_path = save_dir / "model_latest.pt"
            torch.save(self.model.state_dict(), latest_path)
            
            # Save training history
            history_path = save_dir / f"history_{timestamp}.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"💾 Model saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Could not save to ai_models/: {e}")
    
    def load_checkpoint(self, path: Path = None):
        """Load from checkpoint"""
        if path is None:
            path = CHECKPOINTS_DIR / "best_model.pt"
        
        if not path.exists():
            logger.warning(f"No checkpoint found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return True


def train_model(epochs: int = 50, batch_size: int = 32) -> Dict:
    """
    Convenience function to train model
    
    Usage:
        from ai_models.training import train_model
        history = train_model(epochs=50)
    """
    trainer = ModelTrainer()
    return trainer.train(epochs=epochs)


# ========================================
# CLI
# ========================================
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Train AI Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--collect', action='store_true', help='Collect data first')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🤖 AI MODEL TRAINER")
    print("=" * 60)
    
    # Optionally collect data first
    if args.collect:
        from data_collector import get_collector
        collector = get_collector()
        collector.collect_from_analysis_results(days_back=30)
        collector.collect_from_mt5(days_back=30)
        collector.export_for_training()
    
    # Train
    trainer = ModelTrainer()
    history = trainer.train(epochs=args.epochs)
    
    print("\n✅ Training completed!")
