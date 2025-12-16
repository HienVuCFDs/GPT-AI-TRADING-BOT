"""
🎯 VÍ DỤ: UI GỌI SERVER AI ĐỂ PHÂN TÍCH TRADING
===============================================

Flow:
1. UI tính indicators, patterns từ MT5 data
2. UI gửi data lên Server AI
3. Server AI phân tích và trả về signal
4. UI hiển thị signal cho user
5. User confirm → UI execute trade
"""

import requests
import json
from datetime import datetime

# Server AI URL
AI_SERVER_URL = "http://localhost:8001"


def send_analysis_request(
    symbol: str,
    current_price: float,
    indicators: dict,
    candle_patterns: list = None,
    price_patterns: list = None,
    support_levels: list = None,
    resistance_levels: list = None,
    candles: list = None,
    prompt: str = "Phân tích và đưa ra signal trading",
    user_id: str = "user_001",
    strategy_rules: str = None
):
    """
    Gửi request phân tích đến Server AI
    
    Args:
        symbol: Cặp tiền (XAUUSD, BTCUSD, ...)
        current_price: Giá hiện tại
        indicators: Dict các indicators đã tính {RSI: 45, MACD: 0.5, ...}
        candle_patterns: List mô hình nến ["Bullish Engulfing", ...]
        price_patterns: List mô hình giá ["Double Bottom", ...]
        support_levels: List các mức support [3200, 3180, ...]
        resistance_levels: List các mức resistance [3280, 3300, ...]
        candles: List các nến OHLC [{time, open, high, low, close}, ...]
        prompt: Câu hỏi/yêu cầu từ user
        user_id: ID của user
        strategy_rules: Quy tắc trading của user (optional)
    
    Returns:
        dict: Response từ Server AI
    """
    
    payload = {
        "user_id": user_id,
        "symbol": symbol,
        "timeframe": "H1",
        "current_price": current_price,
        "indicators": indicators,
        "candle_patterns": candle_patterns or [],
        "price_patterns": price_patterns or [],
        "support_levels": support_levels or [],
        "resistance_levels": resistance_levels or [],
        "candles": candles or [],
        "prompt": prompt,
        "strategy_rules": strategy_rules,
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{AI_SERVER_URL}/api/trading/analyze",
            json=payload,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================
# VÍ DỤ SỬ DỤNG
# ============================================

if __name__ == "__main__":
    
    # ============================================
    # VÍ DỤ 1: User A gửi data XAUUSD
    # ============================================
    print("=" * 60)
    print("📤 USER A: Gửi phân tích XAUUSD")
    print("=" * 60)
    
    result_a = send_analysis_request(
        user_id="user_A",
        symbol="XAUUSD",
        current_price=3245.50,
        
        # Indicators đã tính từ UI
        indicators={
            "RSI14": 28.5,           # Oversold
            "MACD": 0.85,
            "MACD_signal": 0.62,
            "EMA20": 3240.00,
            "EMA50": 3235.00,
            "ATR14": 15.5,
            "ADX": 32,
            "BB_upper": 3280,
            "BB_lower": 3210,
            "Stoch_K": 18,
            "Stoch_D": 22
        },
        
        # Mô hình nến phát hiện
        candle_patterns=["Bullish Engulfing", "Morning Doji Star"],
        
        # Mô hình giá phát hiện
        price_patterns=["Double Bottom"],
        
        # S/R levels
        support_levels=[3230, 3210, 3190],
        resistance_levels=[3260, 3280, 3300],
        
        # 5 nến gần nhất
        candles=[
            {"time": "2025-12-04 10:00", "open": 3248, "high": 3252, "low": 3240, "close": 3242},
            {"time": "2025-12-04 11:00", "open": 3242, "high": 3245, "low": 3235, "close": 3238},
            {"time": "2025-12-04 12:00", "open": 3238, "high": 3240, "low": 3230, "close": 3232},
            {"time": "2025-12-04 13:00", "open": 3232, "high": 3248, "low": 3230, "close": 3245},
            {"time": "2025-12-04 14:00", "open": 3245, "high": 3250, "low": 3243, "close": 3245.5},
        ],
        
        prompt="RSI đang oversold và có Bullish Engulfing tại support. Có nên BUY không?"
    )
    
    print(f"Response: {json.dumps(result_a, indent=2, ensure_ascii=False)}")
    
    
    # ============================================
    # VÍ DỤ 2: User B gửi data BTCUSD với strategy khác
    # ============================================
    print("\n" + "=" * 60)
    print("📤 USER B: Gửi phân tích BTCUSD (strategy khác)")
    print("=" * 60)
    
    result_b = send_analysis_request(
        user_id="user_B",
        symbol="BTCUSD",
        current_price=97500,
        
        indicators={
            "RSI14": 72.5,           # Overbought
            "MACD": 150,
            "MACD_signal": 120,
            "EMA20": 97000,
            "EMA50": 95500,
            "ATR14": 1200,
            "ADX": 45
        },
        
        candle_patterns=["Shooting Star"],
        price_patterns=["Rising Wedge"],
        
        support_levels=[96000, 94500, 93000],
        resistance_levels=[98000, 99500, 100000],
        
        # User B có strategy riêng
        strategy_rules="""
        - Chỉ SELL khi RSI > 75 VÀ có bearish pattern
        - Chỉ BUY khi RSI < 25 VÀ có bullish pattern
        - Risk/Reward tối thiểu 1:2
        - Không trade khi ADX < 20
        """,
        
        prompt="BTC đang overbought với Shooting Star. Có nên SELL không?"
    )
    
    print(f"Response: {json.dumps(result_b, indent=2, ensure_ascii=False)}")
    
    
    # ============================================
    # VÍ DỤ 3: Concurrent requests (nhiều user cùng lúc)
    # ============================================
    print("\n" + "=" * 60)
    print("📤 CONCURRENT: Nhiều user gửi cùng lúc")
    print("=" * 60)
    
    import concurrent.futures
    
    def user_request(user_id, symbol, price, rsi):
        return send_analysis_request(
            user_id=user_id,
            symbol=symbol,
            current_price=price,
            indicators={"RSI14": rsi, "MACD": 0.5, "EMA20": price * 0.99},
            prompt=f"RSI = {rsi}, nên trade không?"
        )
    
    # 3 users gửi cùng lúc
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(user_request, "user_C", "EURUSD", 1.0850, 35),
            executor.submit(user_request, "user_D", "GBPUSD", 1.2650, 68),
            executor.submit(user_request, "user_E", "USDJPY", 150.50, 55),
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("success"):
                data = result.get("data", {})
                print(f"✅ {data.get('user_id')}: {data.get('symbol')} → {data.get('action')} (confidence: {data.get('confidence')}%)")
            else:
                print(f"❌ Error: {result.get('error')}")


# ============================================
# OUTPUT EXAMPLE
# ============================================
"""
📤 USER A: Gửi phân tích XAUUSD
Response: {
  "success": true,
  "data": {
    "request_id": "abc12345",
    "user_id": "user_A",
    "symbol": "XAUUSD",
    "action": "BUY",
    "entry": 3245.50,
    "sl": 3214.50,
    "tp": 3292.00,
    "confidence": 78,
    "reason": "RSI oversold (28.5) + Bullish Engulfing at support 3230. Strong BUY signal."
  },
  "processing_time_ms": 2847
}
"""
