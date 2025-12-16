# AI Training System for Trading Bot
# ===================================
# Training system for ai_models/ (comprehensive_aggregator.py)

## Structure

```
ai_models/
 __init__.py              # Package exports
 deep_learning_model.py   # LSTM + Attention model (766,025 params)
 ml_signal_predictor.py   # Wrapper/interface
 feature_extractor.py     # Feature extraction
 ml_config.json           # Configuration
 saved/                   # Saved models
 training/                # <-- You are here
     __init__.py          # Training package exports
     data_collector.py    # Collect training data
     trainer.py           # Train model
     utils.py             # Utilities
     data/                # Training data storage
     pending/             # Pending data (1996+ files)
     checkpoints/         # Training checkpoints
```

## Quick Start

### Collect Training Data
`python
from ai_models.training import get_collector
collector = get_collector()
collector.collect_from_analysis_results()
collector.collect_from_mt5()
`

### Train Model
`python
from ai_models.training import train_model
history = train_model(epochs=50)
`

### CLI
`ash
python train_ai_model.py --status
python train_ai_model.py --collect --train 50
`

## Model Info
- Architecture: LSTM + Attention (766,025 params)
- GPU: CUDA supported
