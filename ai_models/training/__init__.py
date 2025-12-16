# AI Training System for Trading Bot
# ==================================
# Centralized training data management for ai_models/
# Local only - NO server communication

__version__ = "1.0.0"

from .data_collector import TrainingDataCollector, get_collector
from .trainer import ModelTrainer, train_model
from .utils import sync_pending_data, cleanup_old_files
from .data_collector_local import LocalTrainingDataCollector, get_local_collector

__all__ = [
    'TrainingDataCollector',
    'get_collector', 
    'ModelTrainer',
    'train_model',
    'sync_pending_data',
    'cleanup_old_files',
    'LocalTrainingDataCollector',
    'get_local_collector'
]
