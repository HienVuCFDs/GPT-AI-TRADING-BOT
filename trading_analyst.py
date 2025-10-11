import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

class TradingAnalyst:
    def __init__(self, use_openai=False, api_key=None):
        self.use_openai = use_openai
        self.api_key = api_key

    def detect_candle_pattern(self, candle, prev_candle=None):
        """
        Phát hiện mẫu nến Price Action cơ bản.
        candle, prev_candle: dict chứa open, high, low, close
        Trả về: 'bullish_engulfing', 'bearish_engulfing', 'pin_bar', 'doji', 'hammer', 'shooting_star', None
        """
        open_ = candle.get("open")
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
        if open_ is None or high is None or low is None or close is None:
            return None

        body = abs(close - open_)
        range_ = high - low
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        # Doji: thân nhỏ hơn 10% range
        if body <= 0.1 * range_:
            return "doji"

        # Pin Bar: 1 bóng dài >= 2 thân, bóng kia nhỏ
        if lower_wick >= 2 * body and upper_wick <= 0.1 * body:
            return "hammer"
        if upper_wick >= 2 * body and lower_wick <= 0.1 * body:
            return "shooting_star"

        # Engulfing (cần prev_candle)
        if prev_candle:
            prev_open = prev_candle.get("open")
            prev_close = prev_candle.get("close")
            if prev_open is not None and prev_close is not None:
                # Bullish Engulfing: nến trước giảm, nến sau tăng, thân sau bao phủ thân trước
                if prev_close < prev_open and close > open_ and open_ < prev_close and close > prev_open:
                    return "bullish_engulfing"
                # Bearish Engulfing: ngược lại
                if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
                    return "bearish_engulfing"

        return None

    def analyze(self, combined_data):
        if not combined_data:
            return {
                "decision": "hold",
                "reason": "Không có dữ liệu để phân tích.",
                "summary": "Chưa cung cấp dữ liệu đầu vào hợp lệ."
            }

        candles = combined_data.get("candles", [])
        indicators = combined_data.get("indicators", [])
        news = combined_data.get("news", [])

        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        patterns_detected = {
            "bullish_engulfing": 0,
            "bearish_engulfing": 0,
            "pin_bar": 0,
            "doji": 0,
            "hammer": 0,
            "shooting_star": 0,
        }

        macd_diffs = []

        for i, candle in enumerate(candles):
            close = candle.get("close")
            open_ = candle.get("open")
            high = candle.get("high")
            low = candle.get("low")

            ind = indicators[i] if i < len(indicators) else {}

            EMA_20 = ind.get("EMA_20")
            EMA_50 = ind.get("EMA_50")
            MACD = ind.get("MACD")
            MACD_signal = ind.get("MACD_signal")
            ATR_14 = ind.get("ATR_14")
            BB_Upper = ind.get("BB_Upper")
            BB_Middle = ind.get("BB_Middle")
            BB_Lower = ind.get("BB_Lower")

            # EMA analysis
            if close is not None and EMA_20 is not None and EMA_50 is not None:
                if close > EMA_20 and close > EMA_50:
                    buy_signals += 1
                elif close < EMA_20 and close < EMA_50:
                    sell_signals += 1
                else:
                    hold_signals += 1

            # MACD
            if MACD is not None and MACD_signal is not None:
                macd_diffs.append(MACD - MACD_signal)

            # BB analysis
            if close is not None and BB_Upper is not None and BB_Lower is not None:
                if close <= BB_Lower:
                    buy_signals += 1
                elif close >= BB_Upper:
                    sell_signals += 1
                else:
                    hold_signals += 1

            # Price Action pattern
            prev_candle = candles[i - 1] if i > 0 else None
            pattern = self.detect_candle_pattern(candle, prev_candle)
            if pattern:
                patterns_detected[pattern] = patterns_detected.get(pattern, 0) + 1
                # Cộng điểm tín hiệu dựa vào pattern
                if pattern in ["bullish_engulfing", "hammer", "pin_bar"]:
                    buy_signals += 2  # tín hiệu mạnh
                elif pattern in ["bearish_engulfing", "shooting_star"]:
                    sell_signals += 2
                elif pattern == "doji":
                    hold_signals += 1  # thể hiện sự do dự thị trường

        avg_macd_diff = sum(macd_diffs) / len(macd_diffs) if macd_diffs else 0

        # Quyết định cuối cùng
        decision = "hold"
        reason = ""
        if buy_signals > sell_signals:
            decision = "buy"
            reason = f"Có nhiều tín hiệu mua hơn bán ({buy_signals} vs {sell_signals})."
        elif sell_signals > buy_signals:
            decision = "sell"
            reason = f"Có nhiều tín hiệu bán hơn mua ({sell_signals} vs {buy_signals})."
        else:
            reason = f"Tín hiệu mua và bán cân bằng hoặc không rõ ràng."

        # Quản lý rủi ro: tính SL, TP dựa ATR và giá vào lệnh gần nhất
        if candles and indicators:
            last_close = candles[-1].get("close")
            last_ATR = indicators[-1].get("ATR_14")
            if last_close is not None and last_ATR is not None:
                sl_distance = 1.5 * last_ATR
                tp_distance = 2 * sl_distance
                if decision == "buy":
                    SL = last_close - sl_distance
                    TP = last_close + tp_distance
                elif decision == "sell":
                    SL = last_close + sl_distance
                    TP = last_close - tp_distance
                else:
                    SL = None
                    TP = None
            else:
                SL = None
                TP = None
        else:
            SL = None
            TP = None

        summary = (
            f"Phân tích {len(candles)} cây nến.\n"
            f"Số tín hiệu mua: {buy_signals}, bán: {sell_signals}, giữ: {hold_signals}\n"
            f"Mẫu nến phát hiện: {patterns_detected}\n"
            f"Trung bình MACD - MACD_signal: {avg_macd_diff:.4f}\n"
            f"SL đề xuất: {SL if SL is not None else 'Chưa xác định'} | TP đề xuất: {TP if TP is not None else 'Chưa xác định'}\n"
            f"Không có tin tức tác động cao."  # Bạn có thể thêm phân tích tin tức sau
        )

        return {
            "decision": decision,
            "reason": reason,
            "summary": summary,
            "SL": SL,
            "TP": TP
        }


if __name__ == "__main__":
    import json
    import logging
    logging.basicConfig(level=logging.INFO)

    # Ví dụ dùng thử
    # Bạn thay combined_data bằng dữ liệu thực tế từ file hoặc API của bạn
    combined_data = {
        "candles": [
            {"time": "2025-05-23 20:00:00", "open": 142.7, "high": 143.0, "low": 142.3, "close": 142.55},
            {"time": "2025-05-23 20:15:00", "open": 142.55, "high": 143.2, "low": 142.5, "close": 143.1},
            # ... thêm nhiều cây nến hơn
        ],
        "indicators": [
            {"EMA_20": 143.6, "EMA_50": 144.3, "MACD": -0.56, "MACD_signal": -0.50, "ATR_14": 0.54, "BB_Upper": 144.8, "BB_Middle": 143.6, "BB_Lower": 142.4},
            {"EMA_20": 143.7, "EMA_50": 144.2, "MACD": -0.50, "MACD_signal": -0.48, "ATR_14": 0.55, "BB_Upper": 144.9, "BB_Middle": 143.7, "BB_Lower": 142.5},
            # ... tương ứng với candles
        ],
        "news": []
    }

    analyst = TradingAnalyst()
    result = analyst.analyze(combined_data)
    logging.info(json.dumps(result, indent=2, ensure_ascii=False))

def main():
    """Main function for standalone execution"""
    print("📊 Trading Analyst - Standalone Mode")
    print("=" * 50)
    
    # 🧹 AUTO CLEANUP before analysis
    print("🧹 Trading Analyst: Auto cleanup before processing...")
    try:
        cleanup_result = cleanup_trading_analyst_data(max_age_hours=48, keep_latest=10)
        print(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
              f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Example usage
    print("🤖 Testing Trading Analyst...")
    
    try:
        combined_data = {
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "candles": [
                {"time": "2025-05-23 20:00:00", "open": 142.7, "high": 143.0, "low": 142.3, "close": 142.55},
                {"time": "2025-05-23 20:15:00", "open": 142.55, "high": 143.2, "low": 142.5, "close": 143.1},
            ],
            "indicators": [
                {"EMA_20": 143.6, "EMA_50": 144.3, "MACD": -0.56, "MACD_signal": -0.50, "ATR_14": 0.54},
                {"EMA_20": 143.7, "EMA_50": 144.2, "MACD": -0.50, "MACD_signal": -0.48, "ATR_14": 0.55},
            ],
            "news": []
        }

        analyst = TradingAnalyst()
        result = analyst.analyze(combined_data)
        
        print("📋 Analysis Results:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n✅ Trading Analyst test completed successfully")
        
    except Exception as e:
        print(f"❌ Trading Analyst test failed: {e}")

def cleanup_trading_analyst_data(max_age_hours: int = 48, keep_latest: int = 10) -> Dict[str, Any]:
    """
    🧹 TRADING ANALYST: Dọn dẹp dữ liệu của module này
    Dọn dẹp analysis results và logs
    
    Args:
        max_age_hours: Tuổi tối đa của file (giờ)
        keep_latest: Số file mới nhất cần giữ lại
    """
    cleanup_stats = {
        'module_name': 'trading_analyst',
        'directories_cleaned': [],
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Thư mục mà Trading Analyst quản lý
    target_directories = [
        'trading_analysis',   # Trading analysis results
        'analyst_logs',      # Analyst logs
        'ai_output'         # AI analysis output
    ]
    
    for directory in target_directories:
        if os.path.exists(directory):
            result = _clean_directory(directory, max_age_hours, keep_latest)
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'files_deleted': result['deleted'],
                'space_freed_mb': result['space_freed']
            })
            cleanup_stats['total_files_deleted'] += result['deleted']
            cleanup_stats['total_space_freed_mb'] += result['space_freed']
        else:
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'status': 'not_found'
            })
    
    print(f"🧹 TRADING ANALYST cleanup complete: "
          f"{cleanup_stats['total_files_deleted']} files deleted, "
          f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_directory(directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
    """Helper function để clean một directory"""
    import os
    from datetime import datetime, timedelta
    
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả analyst files
        all_files = []
        for file_name in os.listdir(directory):
            if file_name.endswith(('.json', '.txt', '.log')):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_size = os.path.getsize(file_path)
                    all_files.append({
                        'path': file_path,
                        'time': file_time,
                        'size': file_size
                    })
        
        # Sắp xếp theo thời gian (mới nhất trước)
        all_files.sort(key=lambda x: x['time'], reverse=True)
        
        # Giữ lại keep_latest files mới nhất
        files_to_keep = all_files[:keep_latest]
        files_to_check = all_files[keep_latest:]
        
        # Xóa files cũ hơn max_age_hours
        for file_info in files_to_check:
            if file_info['time'] < cutoff_time:
                try:
                    os.remove(file_info['path'])
                    deleted_count += 1
                    space_freed += file_info['size'] / (1024 * 1024)  # Convert to MB
                except Exception as e:
                    print(f"Warning: Could not delete {file_info['path']}: {e}")
        
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")
    
    return {'deleted': deleted_count, 'space_freed': space_freed}

if __name__ == "__main__":
    main()
