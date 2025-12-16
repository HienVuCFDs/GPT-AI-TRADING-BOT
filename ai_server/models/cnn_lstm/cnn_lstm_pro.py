"""
🧠 CNN+LSTM Pro Model cho Trading Signal Prediction
===================================================
Phiên bản nâng cao với:
- Multi-Scale Temporal CNN (capture patterns ở nhiều scale)
- Dilated Causal Convolutions (WaveNet-style)
- Causal Attention Masks (prevent future leakage)
- Bidirectional LSTM với Attention
- Residual & Skip Connections
- Feature Pyramid Network cho multi-resolution
- Squeeze-and-Excitation blocks
- Temporal Attention với learnable queries
- Gradient Checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from torch.utils.checkpoint import checkpoint


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Tạo causal mask để tránh leak future information
    mask[i,j] = True nếu j > i (không được nhìn vào tương lai)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block để học channel importance"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        b, c, _ = x.size()
        squeeze = self.squeeze(x).view(b, c)
        excitation = self.excitation(squeeze).view(b, c, 1)
        return x * excitation


class DilatedCausalConv(nn.Module):
    """Dilated Causal Convolution (WaveNet-style) cho temporal patterns"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal: chỉ dùng past và current, không dùng future
        out = self.conv(x)
        # Remove future values
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        return F.gelu(self.bn(out))


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block với residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.conv1 = DilatedCausalConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = DilatedCausalConv(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        
        return F.gelu(out + residual)


class MultiScaleTCN(nn.Module):
    """
    Multi-Scale Temporal Convolutional Network
    - Capture patterns ở nhiều temporal scales
    - Exponentially increasing dilation
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        num_levels = len(hidden_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_ch = input_channels if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_channels = hidden_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # -> (batch, seq_len, channels)
        return out.transpose(1, 2)


class MultiHeadTemporalAttention(nn.Module):
    """Multi-Head Attention với Temporal Bias và Causal Masking"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 500,
        causal: bool = True,  # Enable causal masking by default
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative positional bias
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, max_len, max_len))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.size()
        
        # Project
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores với relative positional bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores + self.relative_bias[:, :seq_len, :seq_len].unsqueeze(0)
        
        # Apply causal mask - CRITICAL để tránh leak future
        if self.causal:
            causal_mask = create_causal_mask(seq_len, query.device)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context), attn_weights.mean(dim=1)  # Average over heads


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention giữa Price và Indicators"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.price_to_indicator = MultiHeadTemporalAttention(d_model, num_heads, dropout)
        self.indicator_to_price = MultiHeadTemporalAttention(d_model, num_heads, dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        price_features: torch.Tensor,
        indicator_features: torch.Tensor,
    ) -> torch.Tensor:
        # Cross attention
        price_attended, _ = self.price_to_indicator(price_features, indicator_features, indicator_features)
        indicator_attended, _ = self.indicator_to_price(indicator_features, price_features, price_features)
        
        # Fusion
        fused = torch.cat([price_attended, indicator_attended], dim=-1)
        return self.fusion(fused)


class HierarchicalLSTM(nn.Module):
    """Hierarchical LSTM cho multi-scale temporal learning"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Fine-grained LSTM
        self.lstm_fine = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Coarse LSTM (downsampled)
        self.lstm_coarse = nn.LSTM(
            hidden_size * 2, hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.output_size = hidden_size * 4  # Both LSTMs bidirectional
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        # Fine-grained processing
        fine_out, _ = self.lstm_fine(x)  # (batch, seq_len, hidden*2)
        
        # Downsample for coarse processing (every 5th step)
        coarse_input = fine_out[:, ::5, :]  # (batch, seq_len//5, hidden*2)
        coarse_out, _ = self.lstm_coarse(coarse_input)
        
        # Upsample coarse back
        coarse_upsampled = F.interpolate(
            coarse_out.transpose(1, 2),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        # Combine
        combined = torch.cat([fine_out, coarse_upsampled], dim=-1)
        
        return combined, fine_out


class TemporalQueryAttention(nn.Module):
    """Learnable Query-based Temporal Attention"""
    
    def __init__(self, d_model: int, num_queries: int = 4):
        super().__init__()
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        nn.init.xavier_uniform_(self.queries)
        
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(d_model * num_queries, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross attention: queries attend to sequence
        attended, _ = self.attention(queries, x, x)
        
        # Flatten và project
        attended = attended.reshape(batch_size, -1)
        return self.output_proj(attended)


class PatternEmbedding(nn.Module):
    """Advanced Pattern Embedding với learnable pattern prototypes"""
    
    def __init__(
        self,
        num_candle_patterns: int = 15,
        num_price_patterns: int = 14,
        embedding_dim: int = 64,
    ):
        super().__init__()
        
        total_patterns = num_candle_patterns + num_price_patterns
        
        # Learnable pattern embeddings
        self.pattern_embeddings = nn.Embedding(total_patterns, embedding_dim)
        
        # Pattern interaction network
        self.interaction = nn.Sequential(
            nn.Linear(total_patterns, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Softmax(dim=-1),
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(embedding_dim + 32, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )
    
    def forward(self, pattern_features: torch.Tensor) -> torch.Tensor:
        batch_size = pattern_features.size(0)
        num_patterns = pattern_features.size(1)
        
        # Get embeddings for active patterns
        indices = torch.arange(num_patterns, device=pattern_features.device)
        all_embeddings = self.pattern_embeddings(indices)  # (num_patterns, embed_dim)
        
        # Weight by pattern presence
        weighted = pattern_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
        aggregated = weighted.sum(dim=1)  # (batch, embed_dim)
        
        # Pattern interaction
        interaction = self.interaction(pattern_features)
        
        # Combine
        combined = torch.cat([aggregated, interaction], dim=-1)
        return self.output(combined)


class MarketRegimeDetector(nn.Module):
    """Detect market regime: Trending, Ranging, Volatile"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4),  # TREND_UP, TREND_DOWN, RANGING, VOLATILE
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.detector(x)


class CNNLSTMProModel(nn.Module):
    """
    CNN+LSTM Pro Model - Production-Ready Trading AI
    
    Kiến trúc nâng cao:
    1. Multi-Scale TCN với Dilated Convolutions
    2. Cross-Modal Attention (Price ↔ Indicators)
    3. Hierarchical BiLSTM
    4. Temporal Query Attention
    5. Pattern Embedding Network
    6. Multi-Task Learning (Signal + Regime + Confidence)
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
        tcn_channels: List[int] = [64, 128, 256],
        lstm_hidden: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        
        # Output
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Multi-Scale TCN cho price
        self.price_tcn = MultiScaleTCN(
            input_channels=price_features,
            hidden_channels=tcn_channels,
            dropout=dropout,
        )
        
        # 2. Indicator encoder
        self.indicator_encoder = nn.Sequential(
            nn.Linear(indicator_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, tcn_channels[-1]),
        )
        
        # 3. Cross-Modal Attention
        self.cross_attention = CrossModalAttention(
            d_model=tcn_channels[-1],
            num_heads=num_heads // 2,
            dropout=dropout,
        )
        
        # 4. Hierarchical LSTM
        self.hierarchical_lstm = HierarchicalLSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
        )
        
        # 5. Temporal Query Attention
        self.temporal_query = TemporalQueryAttention(
            d_model=self.hierarchical_lstm.output_size,
            num_queries=4,
        )
        
        # 6. Pattern Embedding
        self.pattern_embedding = PatternEmbedding(
            num_candle_patterns=15,
            num_price_patterns=14,
            embedding_dim=64,
        )
        
        # 7. SR Embedding
        self.sr_embedding = nn.Sequential(
            nn.Linear(sr_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )
        
        # 8. Feature Fusion
        fusion_input = self.hierarchical_lstm.output_size + 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # 9. Market Regime Detector
        self.regime_detector = MarketRegimeDetector(d_model)
        
        # 10. Signal Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 4, 128),  # +4 for regime
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )
        
        # 11. Confidence Estimator
        self.confidence = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # 12. Risk Estimator (bonus)
        self.risk_estimator = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        price_sequence: torch.Tensor,
        indicator_sequence: torch.Tensor,
        pattern_features: torch.Tensor,
        sr_features: torch.Tensor,
        use_checkpointing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = price_sequence.size(0)
        
        # 1. TCN cho price (với optional gradient checkpointing)
        if use_checkpointing and self.training:
            price_encoded = checkpoint(self.price_tcn, price_sequence, use_reentrant=False)
        else:
            price_encoded = self.price_tcn(price_sequence)
        
        # 2. Encode indicators
        indicator_encoded = self.indicator_encoder(indicator_sequence)
        
        # 3. Cross-Modal Attention (với causal mask được apply tự động)
        if use_checkpointing and self.training:
            fused_sequence = checkpoint(
                self.cross_attention, price_encoded, indicator_encoded,
                use_reentrant=False
            )
        else:
            fused_sequence = self.cross_attention(price_encoded, indicator_encoded)
        
        # 4. Hierarchical LSTM
        if use_checkpointing and self.training:
            lstm_out, fine_out = checkpoint(
                self.hierarchical_lstm, fused_sequence,
                use_reentrant=False
            )
        else:
            lstm_out, fine_out = self.hierarchical_lstm(fused_sequence)
        
        # 5. Temporal Query Attention
        temporal_features = self.temporal_query(lstm_out)
        
        # 6. Pattern embedding
        pattern_emb = self.pattern_embedding(pattern_features)
        
        # 7. SR embedding
        sr_emb = self.sr_embedding(sr_features)
        
        # 8. Fusion
        combined = torch.cat([temporal_features, pattern_emb, sr_emb], dim=-1)
        fused = self.fusion(combined)
        
        # 9. Market regime
        regime_logits = self.regime_detector(fused)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # 10. Classification (regime-aware)
        classifier_input = torch.cat([fused, regime_probs], dim=-1)
        logits = self.classifier(classifier_input)
        probs = F.softmax(logits, dim=-1)
        
        # 11. Confidence
        confidence = self.confidence(fused)
        
        # 12. Risk
        risk = self.risk_estimator(fused)
        
        return {
            'logits': logits,
            'probabilities': probs,
            'confidence': confidence,
            'risk': risk,
            'regime_logits': regime_logits,
            'regime_probabilities': regime_probs,
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
            
            return {
                'signal': signal_map[pred_class.item()],
                'confidence': output['confidence'].item() * 100,
                'risk': output['risk'].item() * 100,
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
            }


def create_cnn_lstm_pro_model(config: Dict = None) -> CNNLSTMProModel:
    if config is None:
        config = {}
    
    return CNNLSTMProModel(
        price_features=config.get('price_features', 5),
        indicator_features=config.get('indicator_features', 20),
        pattern_features=config.get('pattern_features', 29),
        sr_features=config.get('sr_features', 10),
        sequence_length=config.get('sequence_length', 50),
        tcn_channels=config.get('tcn_channels', [64, 128, 256]),
        lstm_hidden=config.get('lstm_hidden', 128),
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('dropout', 0.2),
    )


if __name__ == '__main__':
    print("Testing CNN+LSTM Pro Model...")
    
    model = create_cnn_lstm_pro_model()
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
    print(f"  Probabilities: {output['probabilities'].shape}")
    print(f"  Confidence: {output['confidence'].shape}")
    print(f"  Risk: {output['risk'].shape}")
    print(f"  Regime: {output['regime_probabilities'].shape}")
    
    # Test prediction
    pred = model.predict(price[:1], indicators[:1], patterns[:1], sr[:1])
    print(f"\nPrediction:")
    print(f"  Signal: {pred['signal']}")
    print(f"  Confidence: {pred['confidence']:.1f}%")
    print(f"  Risk: {pred['risk']:.1f}%")
    print(f"  Market Regime: {pred['market_regime']}")
    
    print("\n✅ CNN+LSTM Pro Model test passed!")
