"""
🤖 AI Server Package - Flask APIs for Trading AI
=================================================
Includes:
- XGBoost model (port 5001)
- CNN-LSTM model (port 5002)
- Transformer model (port 5003)

Structure:
    ai_server/
    ├── trading_ai_server.py      # Main server with 3 models
    ├── auto_collect_training_data.py # Auto collect data for server
    ├── training/                 # Server's training data
    └── models/                   # Saved models

Note: Training data collection is now handled locally by:
    ai_models/training/data_collector_local.py

Usage:
    # Start server
    python ai_server/trading_ai_server.py
"""

__version__ = "1.0.0"
__author__ = "VU HIEN CFDs"

__all__ = []
