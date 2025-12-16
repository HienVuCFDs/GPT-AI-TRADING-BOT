"""
CNN-LSTM Pro Model for Trading
==============================

Architecture (~6.8M parameters):
- Multi-Scale Temporal Convolution Network (TCN)
- Cross-Modal Attention
- Hierarchical LSTM with Highway connections
- Causal Masking for temporal consistency

Port: 5002
"""

from .cnn_lstm_pro import CNNLSTMProModel

__all__ = ['CNNLSTMProModel']
