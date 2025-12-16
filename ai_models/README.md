# 🤖 AI Models for Trading Bot

Standalone Deep Learning models that learn and improve over time.

## Features

- **LSTM + Attention** architecture for sequential pattern recognition
- **GPU acceleration** (CUDA) for fast inference
- **Online learning** from trade results
- **Self-improving** - accuracy increases with more data

## Usage

```python
from ai_models import get_ml_predictor

# Get predictor instance
predictor = get_ml_predictor()

# Prepare data
data = {
    'indicators': {
        'M15': {'RSI14': 65, 'MACD_12_26_9': 0.8, 'ADX14': 28, 'close': 2620.5},
        'H1': {'RSI14': 60, 'MACD_12_26_9': 0.5, 'ADX14': 32, 'close': 2620.5}
    },
    'patterns': {'candle_patterns': ['bullish_engulfing'], 'price_patterns': []},
    'trendline_sr': {'trend_direction': 'up', 'supports': [2600], 'resistances': [2650]},
    'news': []
}

# Get prediction
signal, confidence, probabilities = predictor.predict(data)
print(f"Signal: {signal}, Confidence: {confidence:.1f}%")

# Combine with rule-based signal
combined_signal, combined_conf = predictor.combine_signals(
    rule_signal='BUY', rule_confidence=75,
    ml_signal=signal, ml_confidence=confidence
)
```

## Learning from Trades

```python
# After trade closes with profit
predictor.learn_from_trade(data, 'BUY', 'WIN', profit_pips=25)

# After trade closes with loss
predictor.learn_from_trade(data, 'SELL', 'LOSS', profit_pips=-15)
```

## Configuration

Edit `ml_config.json`:

```json
{
  "ml_weight": 0.4,      // 0-1, weight of ML signal in combination
  "enabled": true        // enable/disable ML predictions
}
```

## Model Architecture

```
Input (48 features)
    ├── Indicators (24): RSI, MACD, Stochastic, ADX, BB, EMA, ATR, CCI, WillR
    ├── Patterns (16): Candle patterns, Price patterns  
    └── S/R & Trend (8): Support/Resistance, Trend strength/direction

LSTM (48 → 64 → 32) + Attention
    ↓
Output (3 classes): BUY, SELL, HOLD
```

## Stats

```python
stats = predictor.get_stats()
# {
#   'device': 'cuda',
#   'total_predictions': 150,
#   'correct_predictions': 95,
#   'accuracy': '63.3%',
#   'dataset_size': 100,
#   'parameters': 766025
# }
```

## Files

- `__init__.py` - Package exports
- `deep_learning_model.py` - LSTM + Attention model
- `ml_signal_predictor.py` - High-level predictor wrapper
- `ml_config.json` - Configuration
- `models/` - Saved model weights (created after training)
