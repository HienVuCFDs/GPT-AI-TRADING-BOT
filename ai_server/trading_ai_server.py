"""
🚀 Trading AI Multi-Model Server
================================
Server chạy 3 models trên 3 ports độc lập:
- Port 5001: XGBoost (tabular indicators)
- Port 5002: CNN+LSTM (patterns + trends)
- Port 5003: Transformer (self-attention)

Mỗi model có endpoint:
- POST /predict - Dự đoán signal
- GET /health - Health check
- GET /info - Model info
"""

import os
import sys
import json
import logging
import threading
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import pickle

import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# ============================================================================
# GPU Configuration & Utilities
# ============================================================================
def setup_gpu():
    """Setup GPU and return device info"""
    gpu_info = {
        'available': False,
        'device': 'cpu',
        'name': 'CPU',
        'memory_total': 0,
        'memory_free': 0,
        'cuda_version': None,
        'cudnn_version': None
    }
    
    if torch.cuda.is_available():
        gpu_info['available'] = True
        gpu_info['device'] = 'cuda'
        gpu_info['name'] = torch.cuda.get_device_name(0)
        gpu_info['cuda_version'] = torch.version.cuda
        gpu_info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        
        # Get memory info
        gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_info['memory_free'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
        
        # Set default tensor type to CUDA
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Can cause issues, skip
        
        print(f"🎮 GPU Detected: {gpu_info['name']}")
        print(f"   CUDA Version: {gpu_info['cuda_version']}")
        print(f"   cuDNN Version: {gpu_info['cudnn_version']}")
        print(f"   Memory: {gpu_info['memory_total']:.1f} GB total, {gpu_info['memory_free']:.1f} GB free")
        print(f"   cuDNN Benchmark: Enabled")
    else:
        print("⚠️ GPU not available, using CPU")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['device'] = 'mps'  # Apple Silicon
            gpu_info['name'] = 'Apple MPS'
            gpu_info['available'] = True
            print(f"🍎 Apple MPS Detected")
    
    return gpu_info


def get_optimal_device(prefer_gpu: bool = True) -> torch.device:
    """Get optimal device for inference"""
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


# Initialize GPU on module load
GPU_INFO = setup_gpu()


# ============================================================================
# Request Tracking System
# ============================================================================
@dataclass
class UserRequest:
    """Track individual user request"""
    id: str
    timestamp: datetime
    user_info: Dict  # email, name, phone, etc.
    model: str
    symbol: str
    request_data: Dict
    response: Dict
    processing_time: float  # milliseconds
    ip_address: str
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_info': self.user_info,
            'model': self.model,
            'symbol': self.symbol,
            'signal': self.response.get('signal', 'N/A'),
            'confidence': self.response.get('confidence', 0),
            'processing_time': self.processing_time,
            'ip_address': self.ip_address
        }


class RequestTracker:
    """Track all requests to AI server"""
    
    def __init__(self, max_history: int = 1000):
        self.requests: deque = deque(maxlen=max_history)
        self.stats = {
            'total_requests': 0,
            'requests_by_model': {'xgboost': 0, 'cnn_lstm': 0, 'transformer': 0},
            'requests_by_user': {},
            'requests_today': 0,
            'last_reset': datetime.now().date()
        }
        self.lock = threading.Lock()
    
    def add_request(self, user_request: UserRequest):
        """Add a request to tracking"""
        with self.lock:
            self.requests.append(user_request)
            self.stats['total_requests'] += 1
            
            # Track by model
            if user_request.model in self.stats['requests_by_model']:
                self.stats['requests_by_model'][user_request.model] += 1
            
            # Track by user
            user_email = user_request.user_info.get('email', 'anonymous')
            if user_email not in self.stats['requests_by_user']:
                self.stats['requests_by_user'][user_email] = {
                    'count': 0,
                    'last_request': None,
                    'user_info': user_request.user_info
                }
            self.stats['requests_by_user'][user_email]['count'] += 1
            self.stats['requests_by_user'][user_email]['last_request'] = user_request.timestamp.isoformat()
            
            # Reset daily counter
            if datetime.now().date() != self.stats['last_reset']:
                self.stats['requests_today'] = 0
                self.stats['last_reset'] = datetime.now().date()
            self.stats['requests_today'] += 1
    
    def get_recent_requests(self, limit: int = 50) -> List[Dict]:
        """Get recent requests"""
        with self.lock:
            recent = list(self.requests)[-limit:]
            return [r.to_dict() for r in reversed(recent)]
    
    def get_active_users(self) -> List[Dict]:
        """Get list of active users"""
        with self.lock:
            users = []
            for email, data in self.stats['requests_by_user'].items():
                users.append({
                    'email': email,
                    'name': data['user_info'].get('name', 'Unknown'),
                    'phone': data['user_info'].get('phone', 'N/A'),
                    'request_count': data['count'],
                    'last_request': data['last_request']
                })
            # Sort by request count
            users.sort(key=lambda x: x['request_count'], reverse=True)
            return users
    
    def get_stats(self) -> Dict:
        """Get overall stats"""
        with self.lock:
            return {
                'total_requests': self.stats['total_requests'],
                'requests_today': self.stats['requests_today'],
                'requests_by_model': self.stats['requests_by_model'].copy(),
                'active_users': len(self.stats['requests_by_user']),
                'uptime': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds()
            }


# Global request tracker
request_tracker = RequestTracker()

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Pro models only (Basic models removed)
from models.cnn_lstm import CNNLSTMProModel
from models.cnn_lstm.cnn_lstm_pro import create_cnn_lstm_pro_model
from models.transformer import TransformerProModel
from models.transformer.transformer_pro import create_transformer_pro_model
from models.common import TradingFeatureProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class ServerConfig:
    """Configuration cho server"""
    # Ports
    xgboost_port: int = 5001
    cnn_lstm_port: int = 5002
    transformer_port: int = 5003
    
    # Model paths
    models_dir: str = "ai_server/saved_models"
    
    # Device
    use_gpu: bool = True
    
    # Debug
    debug: bool = False


class BaseModelServer:
    """Base class cho model servers"""
    
    def __init__(self, name: str, port: int, config: ServerConfig):
        self.name = name
        self.port = port
        self.config = config
        self.logger = logging.getLogger(f"Server.{name}")
        
        # Flask app
        self.app = Flask(f"{name}_server")
        CORS(self.app)
        
        # Model
        self.model = None
        
        # 🎮 GPU Setup - Use optimal device
        self.device = get_optimal_device(config.use_gpu)
        self.gpu_info = GPU_INFO
        self.loaded = False
        
        # Log device info
        self.logger.info(f"🎮 {name} using device: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"   GPU: {GPU_INFO['name']}")
            self.logger.info(f"   Memory: {GPU_INFO['memory_total']:.1f} GB")
        
        # Stats
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            # Update GPU memory info
            gpu_memory = {}
            if self.device.type == 'cuda':
                gpu_memory = {
                    'allocated': torch.cuda.memory_allocated(0) / 1024**3,
                    'cached': torch.cuda.memory_reserved(0) / 1024**3,
                    'total': GPU_INFO['memory_total']
                }
            
            return jsonify({
                'status': 'healthy' if self.loaded else 'loading',
                'model': self.name,
                'port': self.port,
                'device': str(self.device),
                'gpu_name': GPU_INFO['name'] if GPU_INFO['available'] else 'N/A',
                'gpu_memory': gpu_memory,
                'requests_served': self.request_count,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            })
        
        @self.app.route('/info', methods=['GET'])
        def info():
            return jsonify(self.get_model_info())
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            if not self.loaded:
                return jsonify({'error': 'Model not loaded'}), 503
            
            try:
                start_time = time.time()
                data = request.json
                
                # Extract user info from request
                user_info = data.get('user_info', {})
                symbol = data.get('symbol', 'UNKNOWN')
                
                result = self.predict(data)
                self.request_count += 1
                
                # Track request
                processing_time = (time.time() - start_time) * 1000
                import uuid
                user_request = UserRequest(
                    id=str(uuid.uuid4())[:8],
                    timestamp=datetime.now(),
                    user_info=user_info,
                    model=self.name,
                    symbol=symbol,
                    request_data={'indicators_count': len(data.get('indicators', {}))},
                    response=result,
                    processing_time=processing_time,
                    ip_address=request.remote_addr or 'unknown'
                )
                request_tracker.add_request(user_request)
                
                return jsonify(result)
            except Exception as e:
                self.logger.error(f"Prediction error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
    
    def load_model(self):
        """Load model - override in subclass"""
        raise NotImplementedError
    
    def predict(self, data: Dict) -> Dict:
        """Make prediction - override in subclass"""
        raise NotImplementedError
    
    def get_model_info(self) -> Dict:
        """Get model info"""
        return {
            'name': self.name,
            'port': self.port,
            'loaded': self.loaded,
            'device': str(self.device),
        }
    
    def run(self, threaded: bool = True):
        """Run the server"""
        self.load_model()
        self.app.run(
            host='0.0.0.0',
            port=self.port,
            debug=self.config.debug,
            threaded=threaded,
            use_reloader=False,
        )


class XGBoostServer(BaseModelServer):
    """XGBoost model server - uses trained ensemble model"""
    
    def __init__(self, config: ServerConfig):
        super().__init__("xgboost", config.xgboost_port, config)
        self.feature_names = None
        self.scaler = None
    
    def load_model(self):
        """Load XGBoost model from trained ensemble"""
        try:
            import joblib
            # Import xgboost before loading pickle file to ensure class is registered
            import xgboost
            
            # Try ensemble model first (already trained)
            ensemble_path = Path("ai_server/models/xgboost_model/trained/ensemble")
            xgb_path = ensemble_path / "xgboost.pkl"
            scaler_path = ensemble_path / "scaler.pkl"
            
            if xgb_path.exists():
                self.model = joblib.load(xgb_path)
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                self.loaded = True
                self.logger.info(f"✅ XGBoost model loaded from {xgb_path}")
            else:
                # Fallback: random prediction for testing
                self.model = None
                self.loaded = True
                self.logger.warning(f"XGBoost model not found at {xgb_path}, using fallback")
        
        except Exception as e:
            self.logger.error(f"Error loading XGBoost: {e}")
            self.loaded = False
    
    def predict(self, data: Dict) -> Dict:
        """XGBoost prediction using trained sklearn model"""
        try:
            # Extract features
            features = self._extract_features(data)
            
            if self.model is not None:
                # Scale features if scaler available
                if self.scaler is not None:
                    features = self.scaler.transform([features])[0]
                
                # Predict with sklearn XGBClassifier
                probs = self.model.predict_proba([features])[0]
            else:
                # Fallback: random prediction for testing
                probs = np.random.dirichlet([1, 1, 1])
            
            # Map to signal
            signal_idx = np.argmax(probs)
            signals = ['BUY', 'SELL', 'HOLD']
            
            return {
                'model': 'xgboost',
                'signal': signals[signal_idx],
                'confidence': float(probs[signal_idx]) * 100,
                'probabilities': {
                    'BUY': float(probs[0]) * 100,
                    'SELL': float(probs[1]) * 100,
                    'HOLD': float(probs[2]) * 100,
                },
                'timestamp': datetime.now().isoformat(),
            }
        
        except Exception as e:
            self.logger.error(f"XGBoost prediction error: {e}")
            raise
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract 46 features matching the trained model"""
        indicators = data.get('indicators', {})
        m15_data = indicators.get('M15', indicators)
        h1_data = indicators.get('H1', {})
        patterns = data.get('patterns', {})
        support_resistance = data.get('support_resistance', {})
        sideway = data.get('sideway_analysis', {})
        news = data.get('news_analysis', {})
        
        def safe_get(d, *keys, default=0.0):
            for key in keys:
                val = d.get(key) if isinstance(d, dict) else None
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return default
        
        # 46 features matching training config
        features = [
            # M15 indicators (8)
            safe_get(m15_data, 'RSI14', 'rsi14', 'RSI'),
            safe_get(m15_data, 'StochK_14_3', 'stochk_14_3', 'STOCH_K'),
            safe_get(m15_data, 'MACD_12_26_9', 'macd_12_26_9', 'MACD'),
            safe_get(m15_data, 'MACDs_12_26_9', 'macds_12_26_9', 'MACD_SIGNAL'),
            safe_get(m15_data, 'MACDh_12_26_9', 'macdh_12_26_9', 'MACD_HIST'),
            safe_get(m15_data, 'ADX14', 'adx14', 'ADX'),
            safe_get(m15_data, 'ATR14', 'atr14', 'ATR'),
            self._calc_bb_position(m15_data),  # bb_position_m15
            
            # H1 indicators (8)
            safe_get(h1_data, 'RSI14', 'rsi14', 'RSI'),
            safe_get(h1_data, 'StochK_14_3', 'stochk_14_3', 'STOCH_K'),
            safe_get(h1_data, 'MACD_12_26_9', 'macd_12_26_9', 'MACD'),
            safe_get(h1_data, 'MACDs_12_26_9', 'macds_12_26_9', 'MACD_SIGNAL'),
            safe_get(h1_data, 'MACDh_12_26_9', 'macdh_12_26_9', 'MACD_HIST'),
            safe_get(h1_data, 'ADX14', 'adx14', 'ADX'),
            safe_get(h1_data, 'ATR14', 'atr14', 'ATR'),
            self._calc_bb_position(h1_data),  # bb_position_h1
            
            # Price vs EMA (4)
            self._calc_price_vs_ema(m15_data, 'EMA20'),
            self._calc_price_vs_ema(m15_data, 'EMA50'),
            self._calc_price_vs_ema(h1_data, 'EMA20'),
            self._calc_price_vs_ema(h1_data, 'EMA50'),
            
            # Candle patterns (3)
            safe_get(patterns, 'candle_bullish', 'bullish_candle', default=0),
            safe_get(patterns, 'candle_bearish', 'bearish_candle', default=0),
            safe_get(patterns, 'candle_score', 'candle_pattern_score', default=0),
            
            # Price patterns (4)
            safe_get(patterns, 'price_bullish', 'bullish_pattern', default=0),
            safe_get(patterns, 'price_bearish', 'bearish_pattern', default=0),
            safe_get(patterns, 'price_confidence', 'pattern_confidence', default=0.5),
            safe_get(patterns, 'overall_bias', 'bias', default=0),
            
            # Support/Resistance (4)
            safe_get(support_resistance, 'distance_to_support', default=0),
            safe_get(support_resistance, 'distance_to_resistance', default=0),
            safe_get(support_resistance, 'sr_ratio', default=0.5),
            safe_get(support_resistance, 'trend_strength', default=0),
            
            # Trend/Sideway (5)
            safe_get(sideway, 'trend_direction', default=0),
            1.0 if safe_get(sideway, 'is_sideway', default=0) else 0.0,
            safe_get(sideway, 'position_in_range', default=0.5),
            safe_get(sideway, 'sideway_signal', default=0),
            safe_get(support_resistance, 'trend_strength', default=0),
            
            # Technical signals (4)
            safe_get(patterns, 'rsi_divergence', default=0),
            safe_get(patterns, 'macd_crossover', default=0),
            safe_get(patterns, 'momentum_score', default=0),
            safe_get(m15_data, 'volatility_ratio', default=1.0),
            
            # Indicator signals (4)
            safe_get(m15_data, 'buy_count', 'signal_buy_count', default=0),
            safe_get(m15_data, 'sell_count', 'signal_sell_count', default=0),
            safe_get(h1_data, 'buy_count', 'signal_buy_count', default=0),
            safe_get(h1_data, 'sell_count', 'signal_sell_count', default=0),
            
            # News (2)
            safe_get(news, 'sentiment', 'news_sentiment', default=0),
            safe_get(news, 'impact', 'news_impact', default=0),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calc_bb_position(self, data: Dict) -> float:
        """Calculate price position within Bollinger Bands (0-1)"""
        close = data.get('close', data.get('Close', 0))
        bb_upper = data.get('BBU_20_2.0', data.get('bb_upper', data.get('BB_Upper', 0)))
        bb_lower = data.get('BBL_20_2.0', data.get('bb_lower', data.get('BB_Lower', 0)))
        
        try:
            close, bb_upper, bb_lower = float(close), float(bb_upper), float(bb_lower)
            if bb_upper > bb_lower:
                return (close - bb_lower) / (bb_upper - bb_lower)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        return 0.5
    
    def _calc_price_vs_ema(self, data: Dict, ema_key: str) -> float:
        """Calculate percentage distance from price to EMA"""
        close = data.get('close', data.get('Close', 0))
        ema = data.get(ema_key, data.get(ema_key.lower(), 0))
        
        try:
            close, ema = float(close), float(ema)
            if ema > 0:
                return (close - ema) / ema * 100
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        return 0.0
    
    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'type': 'XGBoost',
            'description': 'Gradient boosting for tabular indicator data',
            'best_for': 'Quick predictions based on current indicators',
        })
        return info


class CNNLSTMServer(BaseModelServer):
    """CNN+LSTM Pro model server"""
    
    def __init__(self, config: ServerConfig):
        super().__init__("cnn_lstm", config.cnn_lstm_port, config)
        self.normalizer = None
        self.processor = TradingFeatureProcessor()
    
    def load_model(self):
        """Load CNN+LSTM Pro model"""
        try:
            model_path = Path(self.config.models_dir) / "cnn_lstm_pro_best.pt"
            self.model = create_cnn_lstm_pro_model({
                'sequence_length': 50,
                'price_features': 5,
                'indicator_features': 20,
                'pattern_features': 29,
                'sr_features': 10,
            })
            self.logger.info("Using CNN+LSTM PRO model")
            
            normalizer_path = Path(self.config.models_dir) / "cnn_lstm_normalizer.pkl"
            
            if model_path.exists():
                # 🎮 Load to GPU if available
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"✅ Model loaded from {model_path}")
            else:
                self.logger.warning(f"Model not found at {model_path}, using random weights")
            
            # 🎮 Move model to GPU and optimize
            self.model.to(self.device)
            self.model.eval()
            
            # Enable inference optimizations
            if self.device.type == 'cuda':
                self.logger.info(f"🎮 CNN+LSTM running on GPU: {GPU_INFO['name']}")
                # Use half precision for faster inference on GPU (optional)
                # self.model.half()  # FP16 - can cause accuracy issues
            
            # Load normalizer
            if normalizer_path.exists():
                with open(normalizer_path, 'rb') as f:
                    self.normalizer = pickle.load(f)
            
            self.loaded = True
            self.logger.info(f"✅ CNN+LSTM loaded on {self.device}")
            
            # Log model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"   Parameters: {total_params:,}")
        
        except Exception as e:
            self.logger.error(f"Error loading CNN+LSTM: {e}", exc_info=True)
            self.loaded = False
    
    @torch.inference_mode()  # 🎮 Faster than torch.no_grad()
    def predict(self, data: Dict) -> Dict:
        """CNN+LSTM prediction with GPU optimization"""
        try:
            # Process input
            processed = self.processor.process_training_sample(data)
            
            # 🎮 Create tensors directly on GPU for better performance
            price_seq = torch.tensor(
                self._create_price_sequence(data), 
                dtype=torch.float32,
                device=self.device  # Create directly on GPU
            ).unsqueeze(0)
            
            indicator_seq = torch.tensor(
                np.tile(processed['indicators'], (50, 1)),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            pattern_feat = torch.tensor(
                processed['patterns'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            sr_feat = torch.tensor(
                processed['sr_features'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            # Predict
            result = self.model.predict(
                price_seq, indicator_seq, pattern_feat, sr_feat
            )
            
            response = {
                'model': 'cnn_lstm_pro',
                'signal': result['signal'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'market_regime': result.get('market_regime', 'UNKNOWN'),
                'regime_probabilities': result.get('regime_probabilities', {}),
                'risk': result.get('risk', 0.5),
                'timestamp': datetime.now().isoformat(),
            }
            
            return response
        
        except Exception as e:
            self.logger.error(f"CNN+LSTM prediction error: {e}")
            raise
    
    def _create_price_sequence(self, data: Dict) -> np.ndarray:
        """Create synthetic price sequence"""
        indicators = data.get('indicators', {})
        m15 = indicators.get('M15', indicators)
        
        close = m15.get('close', 100.0)
        atr = m15.get('ATR14', m15.get('atr', 1.0)) or 1.0
        
        seq_len = 50
        price_seq = np.zeros((seq_len, 5), dtype=np.float32)
        
        current = close
        for i in range(seq_len - 1, -1, -1):
            change = np.random.randn() * atr * 0.3
            
            if i < seq_len - 1:
                current = price_seq[i + 1, 3] - change
            
            high = current + abs(np.random.randn()) * atr * 0.5
            low = current - abs(np.random.randn()) * atr * 0.5
            open_p = low + (high - low) * np.random.random()
            close_p = low + (high - low) * np.random.random()
            volume = 1000 * (1 + np.random.random())
            
            price_seq[i] = [open_p, high, low, close_p, volume]
        
        price_seq[-1, 3] = close
        
        # Normalize
        mean = np.mean(price_seq[:, :4])
        std = np.std(price_seq[:, :4]) + 1e-8
        price_seq[:, :4] = (price_seq[:, :4] - mean) / std
        price_seq[:, 4] = (price_seq[:, 4] - np.mean(price_seq[:, 4])) / (np.std(price_seq[:, 4]) + 1e-8)
        
        return price_seq
    
    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'type': 'CNN+LSTM Pro',
            'description': 'Multi-Scale TCN + Cross-Modal Attention + Hierarchical LSTM',
            'best_for': 'Pattern-based signals with trend context and market regime detection',
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'features': ['Multi-Task Learning', 'Cross-Modal Attention', 'Causal Masking'],
        })
        return info


class TransformerServer(BaseModelServer):
    """Transformer Pro model server"""
    
    def __init__(self, config: ServerConfig):
        super().__init__("transformer", config.transformer_port, config)
        self.processor = TradingFeatureProcessor()
    
    def load_model(self):
        """Load Transformer Pro model"""
        try:
            model_path = Path(self.config.models_dir) / "transformer_pro_best.pt"
            self.model = create_transformer_pro_model({
                'sequence_length': 50,
                'price_features': 5,
                'indicator_features': 20,
                'pattern_features': 29,
                'sr_features': 10,
                'd_model': 256,
                'num_heads': 8,
                'num_experts': 4,
            })
            self.logger.info("Using Transformer PRO model")
            
            if model_path.exists():
                # 🎮 Load to GPU if available
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"✅ Model loaded from {model_path}")
            else:
                self.logger.warning(f"Model not found at {model_path}, using random weights")
            
            # 🎮 Move model to GPU and optimize
            self.model.to(self.device)
            self.model.eval()
            
            # Enable inference optimizations
            if self.device.type == 'cuda':
                self.logger.info(f"🎮 Transformer running on GPU: {GPU_INFO['name']}")
            
            self.loaded = True
            self.logger.info(f"✅ Transformer loaded on {self.device}")
            
            # Log model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"   Parameters: {total_params:,}")
        
        except Exception as e:
            self.logger.error(f"Error loading Transformer: {e}", exc_info=True)
            self.loaded = False
    
    @torch.inference_mode()  # 🎮 Faster than torch.no_grad()
    def predict(self, data: Dict) -> Dict:
        """Transformer prediction with GPU optimization"""
        try:
            # Process input
            processed = self.processor.process_training_sample(data)
            
            # Create tensors directly on GPU
            price_seq = torch.tensor(
                self._create_price_sequence(data),
                dtype=torch.float32,
                device=self.device  # 🎮 Create directly on GPU
            ).unsqueeze(0)
            
            indicator_seq = torch.tensor(
                np.tile(processed['indicators'], (50, 1)),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            pattern_feat = torch.tensor(
                processed['patterns'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            sr_feat = torch.tensor(
                processed['sr_features'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            # Predict
            result = self.model.predict(
                price_seq, indicator_seq, pattern_feat, sr_feat
            )
            
            response = {
                'model': 'transformer_pro',
                'signal': result['signal'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'market_regime': result.get('market_regime', 'UNKNOWN'),
                'regime_probabilities': result.get('regime_probabilities', {}),
                'uncertainty': result.get('uncertainty', 0),
                'volatility': result.get('volatility', 0),
                'magnitude': result.get('magnitude', 0),
                'timestamp': datetime.now().isoformat(),
            }
            
            return response
        
        except Exception as e:
            self.logger.error(f"Transformer prediction error: {e}")
            raise
    
    def _create_price_sequence(self, data: Dict) -> np.ndarray:
        """Create synthetic price sequence"""
        # Same as CNN+LSTM
        indicators = data.get('indicators', {})
        m15 = indicators.get('M15', indicators)
        
        close = m15.get('close', 100.0)
        atr = m15.get('ATR14', m15.get('atr', 1.0)) or 1.0
        
        seq_len = 50
        price_seq = np.zeros((seq_len, 5), dtype=np.float32)
        
        current = close
        for i in range(seq_len - 1, -1, -1):
            change = np.random.randn() * atr * 0.3
            
            if i < seq_len - 1:
                current = price_seq[i + 1, 3] - change
            
            high = current + abs(np.random.randn()) * atr * 0.5
            low = current - abs(np.random.randn()) * atr * 0.5
            open_p = low + (high - low) * np.random.random()
            close_p = low + (high - low) * np.random.random()
            volume = 1000 * (1 + np.random.random())
            
            price_seq[i] = [open_p, high, low, close_p, volume]
        
        price_seq[-1, 3] = close
        
        # Normalize
        mean = np.mean(price_seq[:, :4])
        std = np.std(price_seq[:, :4]) + 1e-8
        price_seq[:, :4] = (price_seq[:, :4] - mean) / std
        price_seq[:, 4] = (price_seq[:, 4] - np.mean(price_seq[:, 4])) / (np.std(price_seq[:, 4]) + 1e-8)
        
        return price_seq
    
    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'type': 'Transformer Pro',
            'description': 'Hierarchical Transformer + Mixture of Experts + RoPE + Multi-Scale Attention',
            'best_for': 'Complex patterns with long-range dependencies and uncertainty quantification',
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'features': ['Mixture of Experts', 'Uncertainty Quantification', 'Volatility Prediction', 'Causal Masking'],
        })
        return info


class MultiModelServer:
    """
    Main server quản lý cả 3 models
    
    Usage:
        python trading_ai_server.py --all  # Run all models
        python trading_ai_server.py --model xgboost  # Run specific model
    """
    
    def __init__(self, config: ServerConfig = None):
        self.config = config or ServerConfig()
        self.logger = logging.getLogger("MultiModelServer")
        
        # Create servers
        self.servers = {
            'xgboost': XGBoostServer(self.config),
            'cnn_lstm': CNNLSTMServer(self.config),
            'transformer': TransformerServer(self.config),
        }
        
        self.threads = {}
        
        # Dashboard server
        self.dashboard_port = 5000
        self.dashboard_app = self._create_dashboard_app()
    
    def _create_dashboard_app(self):
        """Create Flask app for dashboard"""
        template_dir = Path(__file__).parent / 'templates'
        app = Flask('dashboard', template_folder=str(template_dir))
        CORS(app)
        
        @app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @app.route('/api/status', methods=['GET'])
        def api_status():
            """Get status of all models"""
            status = {}
            for name, server in self.servers.items():
                try:
                    resp = requests.get(f'http://localhost:{server.port}/health', timeout=2)
                    status[name] = resp.json()
                except:
                    status[name] = {'status': 'offline', 'model': name}
            return jsonify(status)
        
        @app.route('/api/predict_all', methods=['POST'])
        def api_predict_all():
            """Predict using all models"""
            data = request.json
            results = {}
            
            for name, server in self.servers.items():
                try:
                    resp = requests.post(
                        f'http://localhost:{server.port}/predict',
                        json=data,
                        timeout=10
                    )
                    results[name] = resp.json()
                except Exception as e:
                    results[name] = {'error': str(e)}
            
            return jsonify(results)
        
        @app.route('/api/requests', methods=['GET'])
        def api_requests():
            """Get recent requests"""
            limit = request.args.get('limit', 50, type=int)
            return jsonify(request_tracker.get_recent_requests(limit))
        
        @app.route('/api/users', methods=['GET'])
        def api_users():
            """Get active users"""
            return jsonify(request_tracker.get_active_users())
        
        @app.route('/api/stats', methods=['GET'])
        def api_stats():
            """Get overall statistics"""
            return jsonify(request_tracker.get_stats())
        
        return app
    
    def start_all(self):
        """Start all model servers"""
        self.logger.info("="*60)
        self.logger.info("🚀 Starting Trading AI Multi-Model Server")
        self.logger.info("="*60)
        self.logger.info(f"📊 Dashboard   : http://localhost:{self.dashboard_port}")
        self.logger.info(f"XGBoost     : http://localhost:{self.config.xgboost_port}")
        self.logger.info(f"CNN+LSTM    : http://localhost:{self.config.cnn_lstm_port}")
        self.logger.info(f"Transformer : http://localhost:{self.config.transformer_port}")
        self.logger.info("="*60)
        
        # Start model servers in daemon threads
        for name, server in self.servers.items():
            thread = threading.Thread(target=server.run, daemon=True, name=f"{name}_thread")
            thread.start()
            self.threads[name] = thread
            time.sleep(1)  # Stagger startup
        
        self.logger.info("All servers started!")
        self.logger.info(f"🌐 Open Dashboard: http://localhost:{self.dashboard_port}")
        
        # Run dashboard on MAIN thread (this blocks and keeps process alive)
        try:
            self.dashboard_app.run(
                host='0.0.0.0',
                port=self.dashboard_port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
    
    def start_single(self, model_name: str):
        """Start a single model server"""
        if model_name not in self.servers:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.logger.info(f"Starting {model_name} server...")
        self.servers[model_name].run()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading AI Multi-Model Server (Pro Models)")
    parser.add_argument('--all', action='store_true', help='Start all model servers')
    parser.add_argument('--model', type=str, choices=['xgboost', 'cnn_lstm', 'transformer'],
                       help='Start specific model server')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Port customization
    parser.add_argument('--xgboost-port', type=int, default=5001)
    parser.add_argument('--cnn-lstm-port', type=int, default=5002)
    parser.add_argument('--transformer-port', type=int, default=5003)
    
    args = parser.parse_args()
    
    # Handle GPU/CPU flag
    use_gpu = args.gpu and not args.cpu
    
    config = ServerConfig(
        xgboost_port=args.xgboost_port,
        cnn_lstm_port=args.cnn_lstm_port,
        transformer_port=args.transformer_port,
        use_gpu=use_gpu,
        debug=args.debug,
    )
    
    print("="*60)
    print("🤖 Trading AI Server - Pro Models")
    print("="*60)
    print("   • XGBoost: Gradient Boosting (CPU)")
    print("   • CNN-LSTM Pro: Multi-Scale TCN + Attention (~6.8M params)")
    print("   • Transformer Pro: Hierarchical + MoE (~9.3M params)")
    print("-"*60)
    
    # 🎮 Display GPU info
    if GPU_INFO['available']:
        print(f"🎮 GPU Mode: ENABLED")
        print(f"   Device: {GPU_INFO['name']}")
        print(f"   CUDA: {GPU_INFO['cuda_version']}")
        print(f"   Memory: {GPU_INFO['memory_total']:.1f} GB")
    else:
        print(f"⚠️ GPU Mode: DISABLED (using CPU)")
        if not use_gpu:
            print(f"   Reason: --cpu flag specified")
        else:
            print(f"   Reason: CUDA not available")
    print("="*60)
    
    server = MultiModelServer(config)
    
    if args.all:
        server.start_all()
    elif args.model:
        server.start_single(args.model)
    else:
        # Default: start all
        server.start_all()


if __name__ == '__main__':
    main()
