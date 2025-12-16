"""
🤖 Mistral AI Client for Trading Bot GUI
Connects to local or remote Mistral AI Server

Usage:
    from ai_server.client import get_ai_client
    
    client = get_ai_client()
    
    # Check server
    if client.is_available():
        response = client.chat("XAUUSD đang uptrend, nên mua không?")
        print(response)
"""

import requests
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

@dataclass
class AIClientConfig:
    """AI Client configuration"""
    server_url: str = "http://localhost:8001"
    timeout: int = 120  # Longer timeout for AI responses
    retry_count: int = 2
    
    def __post_init__(self):
        self.server_url = self.server_url.rstrip("/")

# ============================================
# AI Client Class
# ============================================

class MistralAIClient:
    """Client to communicate with Mistral AI Server"""
    
    def __init__(self, config: AIClientConfig = None):
        self.config = config or AIClientConfig()
        self._last_health_check = None
        self._last_health_result = None
        
    # === Connection Methods ===
    
    def is_available(self) -> bool:
        """Check if AI server is available"""
        try:
            response = requests.get(
                f"{self.config.server_url}/health", 
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Server not available: {e}")
            return False
    
    def get_health(self) -> Dict[str, Any]:
        """Get server health status"""
        try:
            response = requests.get(
                f"{self.config.server_url}/health", 
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "code": response.status_code}
        except requests.exceptions.ConnectionError:
            return {"status": "offline", "message": "Cannot connect to server"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        try:
            response = requests.get(
                f"{self.config.server_url}/api/status", 
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def is_model_loaded(self) -> bool:
        """Check if AI model is loaded and ready"""
        health = self.get_health()
        return health.get("model_loaded", False)
    
    # === Model Control ===
    
    def load_model(self) -> Dict[str, Any]:
        """Request server to load model"""
        try:
            response = requests.post(
                f"{self.config.server_url}/api/model/load",
                timeout=300  # 5 minutes for model loading
            )
            return response.json()
        except requests.exceptions.Timeout:
            return {"success": False, "message": "Loading timeout - model may still be loading"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def unload_model(self) -> Dict[str, Any]:
        """Request server to unload model"""
        try:
            response = requests.post(
                f"{self.config.server_url}/api/model/unload",
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # === AI Analysis Methods ===
    
    def chat(self, message: str, context: str = None, 
             trading_data: Dict = None, max_tokens: int = 500,
             temperature: float = 0.3) -> str:
        """
        Chat with AI about trading
        
        Args:
            message: The question or message to send
            context: Additional context about the conversation
            trading_data: Current trading data (positions, signals, etc.)
            max_tokens: Maximum tokens in response
            temperature: Creativity (0.0-1.0, lower = more focused)
        
        Returns:
            AI response string
        """
        try:
            payload = {
                "message": message,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if context:
                payload["context"] = context
            if trading_data:
                payload["trading_data"] = trading_data
                
            response = requests.post(
                f"{self.config.server_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            result = response.json()
            
            if result.get("success"):
                return result["data"]["response"]
            else:
                return f"❌ Error: {result.get('error', 'Unknown error')}"
                
        except requests.exceptions.Timeout:
            return "⏱️ Request timeout - AI đang bận, vui lòng thử lại sau"
        except requests.exceptions.ConnectionError:
            return "❌ Không thể kết nối đến AI Server. Server có đang chạy không?"
        except Exception as e:
            return f"❌ Lỗi: {e}"
    
    def analyze_market(self, 
                       symbol: str,
                       price: float,
                       change_24h: float = None,
                       rsi: float = None,
                       macd_signal: str = None,
                       volume: str = None,
                       trend: str = None,
                       indicators: Dict = None,
                       patterns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market data with AI
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            price: Current price
            change_24h: 24h change percentage
            rsi: RSI value
            macd_signal: MACD signal (e.g., "Bullish crossover")
            volume: Volume description
            trend: Current trend
            indicators: Additional indicators dict
            patterns: List of detected patterns
        
        Returns:
            Analysis result dict with 'success', 'data', 'error' keys
        """
        try:
            payload = {
                "symbol": symbol,
                "price": price,
                "change_24h": change_24h,
                "rsi": rsi,
                "macd_signal": macd_signal,
                "volume": volume,
                "trend": trend,
                "indicators": indicators,
                "patterns": patterns
            }
            
            response = requests.post(
                f"{self.config.server_url}/api/analyze",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Cannot connect to AI server"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def evaluate_signal(self,
                        symbol: str,
                        action: str,
                        entry: float,
                        sl: float,
                        tp: float,
                        confidence: float,
                        indicators: Dict = None,
                        patterns: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trading signal quality
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            entry: Entry price
            sl: Stop loss price
            tp: Take profit price
            confidence: System confidence (0-100)
            indicators: Additional indicators
            patterns: Detected patterns
        
        Returns:
            Evaluation result dict
        """
        try:
            payload = {
                "symbol": symbol,
                "action": action,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
                "indicators": indicators,
                "patterns": patterns
            }
            
            response = requests.post(
                f"{self.config.server_url}/api/signal/evaluate",
                json=payload,
                timeout=self.config.timeout
            )
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Cannot connect to AI server"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # === Convenience Methods ===
    
    def quick_analyze(self, symbol: str, price: float, rsi: float = None) -> str:
        """Quick market analysis - returns just the analysis text"""
        result = self.analyze_market(symbol=symbol, price=price, rsi=rsi)
        if result.get("success"):
            return result["data"]["analysis"]
        return f"❌ {result.get('error', 'Analysis failed')}"
    
    def quick_evaluate(self, symbol: str, action: str, entry: float, sl: float, tp: float) -> str:
        """Quick signal evaluation - returns just the evaluation text"""
        result = self.evaluate_signal(
            symbol=symbol,
            action=action,
            entry=entry,
            sl=sl,
            tp=tp,
            confidence=50
        )
        if result.get("success"):
            return result["data"]["evaluation"]
        return f"❌ {result.get('error', 'Evaluation failed')}"

# ============================================
# Singleton Instance
# ============================================

_client_instance = None

def get_ai_client(server_url: str = "http://localhost:8000") -> MistralAIClient:
    """
    Get or create AI client singleton
    
    Args:
        server_url: AI server URL (default: localhost:8000)
    
    Returns:
        MistralAIClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = MistralAIClient(AIClientConfig(server_url=server_url))
    return _client_instance

def reset_client():
    """Reset the client singleton (useful for changing server URL)"""
    global _client_instance
    _client_instance = None

# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("🧪 Testing Mistral AI Client...")
    print("=" * 50)
    
    client = get_ai_client()
    
    # Check health
    print("\n📡 Checking server health...")
    health = client.get_health()
    print(f"   Health: {health}")
    
    if health.get("status") == "healthy":
        print("\n✅ Server is running!")
        
        # Check if model is loaded
        if health.get("model_loaded"):
            print("✅ Model is loaded and ready!")
            
            # Test chat
            print("\n💬 Testing chat...")
            response = client.chat("XAUUSD đang ở mức 2650, RSI 65, nên mua hay chờ?")
            print(f"   Response: {response[:500]}...")
            
            # Test analyze
            print("\n📊 Testing market analysis...")
            result = client.analyze_market(
                symbol="XAUUSD",
                price=2650.50,
                rsi=65,
                trend="Uptrend",
                macd_signal="Bullish"
            )
            if result.get("success"):
                print(f"   Analysis: {result['data']['analysis'][:500]}...")
            else:
                print(f"   Error: {result.get('error')}")
        else:
            print("⏳ Model is not loaded yet")
            print("   Loading model...")
            result = client.load_model()
            print(f"   Result: {result}")
    else:
        print(f"❌ Server not available: {health}")
        print("\n💡 Start the server with:")
        print("   python ai_server/server.py")
