# AI Models for comprehensive_aggregator.py
# Standalone ML/DL models that learn from trading data over time
"""
AI Models package for Trading Bot
==================================
Standalone AI/ML models for comprehensive_aggregator.py

Structure:
    ai_models/
    ├── __init__.py              # Package exports
    ├── deep_learning_model.py   # LSTM + Attention model
    ├── ml_signal_predictor.py   # Wrapper/interface
    ├── feature_extractor.py     # Feature extraction
    ├── ml_config.json           # Configuration
    ├── saved/                   # Saved models
    └── training/                # Training system
        ├── pending/             # Pending training files (1996+ files)
        ├── data/                # Processed training data
        ├── checkpoints/         # Training checkpoints
        ├── data_collector.py    # Data collection
        ├── trainer.py           # Model trainer
        └── utils.py             # Utilities

Usage:
    # For predictions
    from ai_models import get_ml_predictor
    predictor = get_ml_predictor()
    signal, confidence, probs = predictor.predict(data)
    
    # For training
    from ai_models.training import train_model
    history = train_model(epochs=50)
    
    # Or use trainer directly
    from ai_models.training.trainer import ModelTrainer
    trainer = ModelTrainer()
    trainer.train(epochs=50)
"""

__version__ = "1.0.0"

from .ml_signal_predictor import MLSignalPredictor, get_ml_predictor
from .deep_learning_model import DeepLearningTrader, get_dl_trader

# Training exports
from .training import (
    TrainingDataCollector,
    get_collector,
    ModelTrainer,
    train_model,
    sync_pending_data,
    cleanup_old_files
)

__all__ = [
    # Models
    'MLSignalPredictor',
    'get_ml_predictor',
    'DeepLearningTrader', 
    'get_dl_trader',
    # Training
    'TrainingDataCollector',
    'get_collector',
    'ModelTrainer',
    'train_model',
    'sync_pending_data',
    'cleanup_old_files'
]
