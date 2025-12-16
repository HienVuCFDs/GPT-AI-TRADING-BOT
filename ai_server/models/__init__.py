"""
AI Server Models Package
========================

Organized structure:
- models/xgboost/     - XGBoost & Ensemble models (Port 5001)
- models/cnn_lstm/    - CNN-LSTM Pro model (Port 5002)
- models/transformer/ - Transformer Pro model (Port 5003)
- models/common/      - Shared utilities
"""

# XGBoost / Ensemble
from .xgboost_model import FeatureExtractor, EnsembleTrader, MLSignalPredictor

# CNN-LSTM Pro
from .cnn_lstm import CNNLSTMProModel
from .cnn_lstm.cnn_lstm_pro import create_cnn_lstm_pro_model

# Transformer Pro
from .transformer import TransformerProModel
from .transformer.transformer_pro import create_transformer_pro_model

# Common utilities
from .common import TradingFeatureProcessor

__all__ = [
    # XGBoost / Ensemble
    'FeatureExtractor',
    'EnsembleTrader',
    'MLSignalPredictor',
    # CNN-LSTM Pro
    'CNNLSTMProModel',
    'create_cnn_lstm_pro_model',
    # Transformer Pro
    'TransformerProModel',
    'create_transformer_pro_model',
    # Common
    'TradingFeatureProcessor',
]
