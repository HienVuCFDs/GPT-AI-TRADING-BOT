"""
Transformer Pro Model for Trading
=================================

Architecture (~9.3M parameters):
- Hierarchical Transformer Encoder
- Mixture of Experts (MoE)
- Rotary Position Embeddings (RoPE)
- Multi-Scale Feature Fusion
- Causal Masking for temporal consistency

Port: 5003
"""

from .transformer_pro import TransformerProModel

__all__ = ['TransformerProModel']
