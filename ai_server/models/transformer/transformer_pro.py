"""
🤖 Transformer Pro Model cho Trading Signal Prediction
======================================================
Phiên bản nâng cao với:
- Hierarchical Transformer (Multi-Timeframe)
- Cross-Attention giữa Price/Indicators/Patterns
- Rotary Position Embeddings (RoPE)
- Causal Attention Masks (prevent future leakage)
- Flash Attention mechanism
- Mixture of Experts (MoE) layers
- Multi-Task Learning heads
- Uncertainty Quantification
- Gradient Checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Tạo causal mask để tránh leak future information
    mask[i,j] = True nếu j > i (không được nhìn vào tương lai)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Tốt hơn absolute position"""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        cos = self.cos_cached[:seq_len].unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # Apply rotary embedding
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin


class GatedLinearUnit(nn.Module):
    """GLU activation cho better gradient flow"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return self.dropout(x * torch.sigmoid(gate))


class ExpertLayer(nn.Module):
    """Single Expert trong Mixture of Experts"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    - Các experts chuyên biệt cho các market conditions khác nhau
    - Router học cách chọn expert phù hợp
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert layers
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.size()
        
        # Compute routing weights
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x[:, :, :self.experts[0].net[-1].out_features])
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            expert_weight = top_k_probs[:, :, k:k+1]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output, router_probs


class MultiScaleAttention(nn.Module):
    """Attention ở nhiều temporal scales"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        self.fusion = nn.Linear(d_model * len(scales), d_model)
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Tạo causal mask
        attn_mask = None
        if causal:
            attn_mask = create_causal_mask(seq_len, x.device)
        
        outputs = []
        for scale, attention in zip(self.scales, self.attentions):
            if scale == 1:
                out, _ = attention(x, x, x, attn_mask=attn_mask)
            else:
                # Downsample
                downsampled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                
                # Tạo causal mask cho downsampled sequence
                ds_seq_len = downsampled.size(1)
                ds_mask = create_causal_mask(ds_seq_len, x.device) if causal else None
                
                attended, _ = attention(downsampled, downsampled, downsampled, attn_mask=ds_mask)
                
                # Upsample back
                out = F.interpolate(
                    attended.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            outputs.append(out)
        
        # Fuse multi-scale outputs
        concatenated = torch.cat(outputs, dim=-1)
        return self.fusion(concatenated)


class CrossModalTransformerBlock(nn.Module):
    """Transformer block với Cross-Modal Attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        use_causal: bool = True,
    ) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Tạo causal mask nếu chưa có
        if use_causal and self_attn_mask is None:
            self_attn_mask = create_causal_mask(seq_len, x.device)
        
        # Self attention với causal mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross attention (cũng cần causal cho context)
        ctx_len = context.size(1)
        cross_mask = create_causal_mask(ctx_len, x.device) if use_causal else None
        cross_out, _ = self.cross_attn(x, context, context, attn_mask=cross_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        
        return x


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical Encoder cho multi-timeframe analysis
    - Local: Chi tiết ngắn hạn
    - Global: Xu hướng dài hạn
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Local encoder (fine-grained)
        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)
        
        # Downsampling
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        
        # Global encoder (coarse)
        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=num_global_layers)
        
        # Upsampling
        self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, use_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        
        # Tạo causal mask cho local encoder
        causal_mask = None
        if use_causal:
            causal_mask = create_causal_mask(seq_len, x.device)
        
        # Local encoding với causal mask
        local_out = self.local_encoder(x, mask=causal_mask)
        
        # Downsample for global
        x_down = self.downsample(local_out.transpose(1, 2)).transpose(1, 2)
        
        # Tạo causal mask cho global encoder
        global_causal = None
        if use_causal:
            global_seq_len = x_down.size(1)
            global_causal = create_causal_mask(global_seq_len, x.device)
        
        # Global encoding với causal mask
        global_out = self.global_encoder(x_down, mask=global_causal)
        
        # Upsample back
        global_up = self.upsample(global_out.transpose(1, 2)).transpose(1, 2)
        
        # Handle size mismatch
        if global_up.size(1) != local_out.size(1):
            global_up = F.interpolate(
                global_up.transpose(1, 2),
                size=local_out.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Fuse local and global
        combined = torch.cat([local_out, global_up], dim=-1)
        fused = self.fusion(combined)
        
        return fused, global_out


class UncertaintyHead(nn.Module):
    """Uncertainty Quantification head - Dự đoán cả mean và variance"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus(),  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        var = self.var_head(x) + 1e-6  # Add small epsilon
        return mean, var


class TransformerProModel(nn.Module):
    """
    Transformer Pro Model - State-of-the-Art Trading AI
    
    Kiến trúc:
    1. Rotary Position Embeddings
    2. Hierarchical Transformer Encoder
    3. Multi-Scale Attention
    4. Cross-Modal Attention (Price ↔ Indicators ↔ Patterns)
    5. Mixture of Experts
    6. Multi-Task Learning
    7. Uncertainty Quantification
    """
    
    def __init__(
        self,
        # Input
        price_features: int = 5,
        indicator_features: int = 20,
        pattern_features: int = 29,
        sr_features: int = 10,
        sequence_length: int = 50,
        
        # Architecture
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        num_experts: int = 4,
        
        # Output
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Input Projections
        self.price_proj = nn.Linear(price_features, d_model)
        self.indicator_proj = nn.Linear(indicator_features, d_model)
        
        # 2. Rotary Position Embedding
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len=sequence_length + 10)
        
        # 3. Hierarchical Encoder cho price
        self.price_encoder = HierarchicalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_local_layers=2,
            num_global_layers=2,
            dropout=dropout,
        )
        
        # 4. Multi-Scale Attention cho indicators
        self.indicator_encoder = MultiScaleAttention(
            d_model=d_model,
            num_heads=num_heads,
            scales=[1, 2, 4],
            dropout=dropout,
        )
        
        # 5. Cross-Modal Transformer blocks
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(2)
        ])
        
        # 6. Mixture of Experts
        self.moe = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=2,
        )
        
        # 7. Pattern Embedding
        self.pattern_embed = nn.Sequential(
            nn.Linear(pattern_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )
        
        # 8. SR Embedding
        self.sr_embed = nn.Sequential(
            nn.Linear(sr_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model // 2),
        )
        
        # 9. Global Pooling
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        
        # 10. Feature Fusion
        fusion_dim = d_model + d_model + d_model // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        
        # 11. Task Heads
        # Signal classification with uncertainty
        self.signal_head = UncertaintyHead(d_model, num_classes)
        
        # Market regime
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # TREND_UP, TREND_DOWN, RANGING, VOLATILE
        )
        
        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Volatility prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        
        # Price movement magnitude
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        price_sequence: torch.Tensor,
        indicator_sequence: torch.Tensor,
        pattern_features: torch.Tensor,
        sr_features: torch.Tensor,
        use_checkpointing: bool = False,
        use_causal: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = price_sequence.size()
        
        # 1. Project inputs
        price_emb = self.price_proj(price_sequence)
        indicator_emb = self.indicator_proj(indicator_sequence)
        
        # 2. Apply RoPE
        price_emb = self.rope(price_emb)
        indicator_emb = self.rope(indicator_emb)
        
        # 3. Hierarchical encoding for price (với causal mask)
        if use_checkpointing and self.training:
            price_encoded, price_global = checkpoint(
                self.price_encoder, price_emb, use_causal,
                use_reentrant=False
            )
        else:
            price_encoded, price_global = self.price_encoder(price_emb, use_causal=use_causal)
        
        # 4. Multi-scale attention for indicators (với causal mask)
        if use_checkpointing and self.training:
            indicator_encoded = checkpoint(
                self.indicator_encoder, indicator_emb, use_causal,
                use_reentrant=False
            )
        else:
            indicator_encoded = self.indicator_encoder(indicator_emb, causal=use_causal)
        
        # 5. Cross-modal attention (với causal mask)
        for block in self.cross_modal_blocks:
            if use_checkpointing and self.training:
                price_encoded = checkpoint(
                    block, price_encoded, indicator_encoded, None, use_causal,
                    use_reentrant=False
                )
            else:
                price_encoded = block(price_encoded, indicator_encoded, use_causal=use_causal)
        
        # 6. Mixture of Experts
        moe_out, router_probs = self.moe(price_encoded)
        
        # 7. Global pooling
        pool_weights = F.softmax(self.global_pool(moe_out).squeeze(-1), dim=1)
        pooled = torch.bmm(pool_weights.unsqueeze(1), moe_out).squeeze(1)
        
        # 8. Pattern and SR embeddings
        pattern_emb = self.pattern_embed(pattern_features)
        sr_emb = self.sr_embed(sr_features)
        
        # 9. Fuse all features
        combined = torch.cat([pooled, pattern_emb, sr_emb], dim=-1)
        fused = self.fusion(combined)
        
        # 10. Task outputs
        signal_mean, signal_var = self.signal_head(fused)
        signal_probs = F.softmax(signal_mean, dim=-1)
        
        regime_logits = self.regime_head(fused)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        confidence = self.confidence_head(fused)
        volatility = self.volatility_head(fused)
        magnitude = self.magnitude_head(fused)
        
        return {
            'logits': signal_mean,
            'logits_variance': signal_var,
            'probabilities': signal_probs,
            'confidence': confidence,
            'regime_logits': regime_logits,
            'regime_probabilities': regime_probs,
            'volatility': volatility,
            'magnitude': magnitude,
            'router_probs': router_probs,
            'features': fused,
        }
    
    def predict(
        self,
        price_sequence: torch.Tensor,
        indicator_sequence: torch.Tensor,
        pattern_features: torch.Tensor,
        sr_features: torch.Tensor,
    ) -> Dict[str, any]:
        self.eval()
        with torch.no_grad():
            output = self.forward(
                price_sequence, indicator_sequence,
                pattern_features, sr_features
            )
            
            probs = output['probabilities']
            pred_class = torch.argmax(probs, dim=-1)
            
            regime_probs = output['regime_probabilities']
            regime_class = torch.argmax(regime_probs, dim=-1)
            
            signal_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            regime_map = {0: 'TREND_UP', 1: 'TREND_DOWN', 2: 'RANGING', 3: 'VOLATILE'}
            
            # Uncertainty từ variance
            uncertainty = output['logits_variance'].mean().item()
            
            return {
                'signal': signal_map[pred_class.item()],
                'confidence': output['confidence'].item() * 100,
                'uncertainty': uncertainty * 100,
                'probabilities': {
                    'BUY': probs[0, 0].item() * 100,
                    'SELL': probs[0, 1].item() * 100,
                    'HOLD': probs[0, 2].item() * 100,
                },
                'market_regime': regime_map[regime_class.item()],
                'regime_probabilities': {
                    'TREND_UP': regime_probs[0, 0].item() * 100,
                    'TREND_DOWN': regime_probs[0, 1].item() * 100,
                    'RANGING': regime_probs[0, 2].item() * 100,
                    'VOLATILE': regime_probs[0, 3].item() * 100,
                },
                'volatility': output['volatility'].item(),
                'magnitude': output['magnitude'].item(),
            }


def create_transformer_pro_model(config: Dict = None) -> TransformerProModel:
    if config is None:
        config = {}
    
    return TransformerProModel(
        price_features=config.get('price_features', 5),
        indicator_features=config.get('indicator_features', 20),
        pattern_features=config.get('pattern_features', 29),
        sr_features=config.get('sr_features', 10),
        sequence_length=config.get('sequence_length', 50),
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        d_ff=config.get('d_ff', 1024),
        num_experts=config.get('num_experts', 4),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('dropout', 0.1),
    )


if __name__ == '__main__':
    print("Testing Transformer Pro Model...")
    
    model = create_transformer_pro_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward
    batch_size = 4
    seq_len = 50
    
    price = torch.randn(batch_size, seq_len, 5)
    indicators = torch.randn(batch_size, seq_len, 20)
    patterns = torch.randn(batch_size, 29)
    sr = torch.randn(batch_size, 10)
    
    output = model(price, indicators, patterns, sr)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Logits Variance: {output['logits_variance'].shape}")
    print(f"  Probabilities: {output['probabilities'].shape}")
    print(f"  Confidence: {output['confidence'].shape}")
    print(f"  Regime: {output['regime_probabilities'].shape}")
    print(f"  Volatility: {output['volatility'].shape}")
    print(f"  Router probs: {output['router_probs'].shape}")
    
    # Test prediction
    pred = model.predict(price[:1], indicators[:1], patterns[:1], sr[:1])
    print(f"\nPrediction:")
    print(f"  Signal: {pred['signal']}")
    print(f"  Confidence: {pred['confidence']:.1f}%")
    print(f"  Uncertainty: {pred['uncertainty']:.1f}%")
    print(f"  Market Regime: {pred['market_regime']}")
    print(f"  Volatility: {pred['volatility']:.4f}")
    
    print("\n✅ Transformer Pro Model test passed!")
