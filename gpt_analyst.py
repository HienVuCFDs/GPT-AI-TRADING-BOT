import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

# Enhanced import with error handling
try:
    from trading_analyst import TradingAnalyst
    TRADING_ANALYST_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ TradingAnalyst not available: {e}")
    TRADING_ANALYST_AVAILABLE = False
    # Mock TradingAnalyst
    class TradingAnalyst:
        def __init__(self, *args, **kwargs):
            pass
        def analyze(self, data):
            return {
                "symbol": data.get("symbol", "Unknown"),
                "timeframe": data.get("timeframe", "Unknown"),
                "analysis": "Analysis not available - TradingAnalyst module missing",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0,
                "signals": []
            }

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/gpt_analyst_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file with enhanced error handling"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"Successfully loaded: {filepath}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Could not read file: {filepath} — {e}")
        return None

def analyze_symbol(symbol: str, timeframe: str = "M15", use_openai: bool = False, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced symbol analysis with better error handling and fallbacks"""
    logger.info(f"🔍 Starting analysis for {symbol} - {timeframe}...")
    
    # Setup file paths
    file_paths = {
        "candles": f"data/{symbol}_{timeframe}.json",
        "candles_alt": f"data/data_{symbol}_{timeframe}.json",
        "indicators": f"indicator_output/{symbol}_{timeframe}_indicators.json",
        "news": f"forexfactory_news/{symbol}.json",
        "output": f"analysis_output/{symbol}_{timeframe}_analysis.json"
    }
    
    # Create output directory
    os.makedirs("analysis_output", exist_ok=True)
    
    # Load data with fallbacks
    candles = load_json_file(file_paths["candles"]) or load_json_file(file_paths["candles_alt"])
    indicators = load_json_file(file_paths["indicators"])
    news = load_json_file(file_paths["news"]) or []
    
    # Check data availability
    data_status = {
        "candles_available": candles is not None,
        "indicators_available": indicators is not None,
        "news_available": len(news) > 0,
        "candles_count": len(candles) if candles else 0,
        "indicators_count": len(indicators) if indicators else 0,
        "news_count": len(news)
    }
    
    logger.info(f"Data status: {data_status}")
    
    # Fallback if no essential data
    if not candles and not indicators:
        logger.error("❌ No essential data available (candles or indicators)")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "No essential data available",
            "data_status": data_status,
            "timestamp": datetime.now().isoformat(),
            "analysis": "Cannot perform analysis without candles or indicators",            "confidence": 0,
            "signals": []
        }
    
    try:
        # Prepare combined data
        combined_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles or [],
            "indicators": indicators or [],
            "news": news,
            "data_status": data_status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Perform analysis
        logger.info(f"🤖 Starting analysis with TradingAnalyst...")
        analyst = TradingAnalyst(use_openai=use_openai, api_key=api_key)
        result = analyst.analyze(combined_data)
        
        # Enhance result with metadata
        result.update({
            "data_status": data_status,
            "analysis_timestamp": datetime.now().isoformat(),
            "analyst_available": TRADING_ANALYST_AVAILABLE,
            "file_paths": file_paths
        })
        
        # Save results
        try:
            with open(file_paths["output"], "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Analysis saved to: {file_paths['output']}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
        
        logger.info(f"✅ Analysis completed for {symbol} ({timeframe})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for {symbol} {timeframe}: {e}")
        error_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": str(e),
            "data_status": data_status,
            "timestamp": datetime.now().isoformat(),
            "analysis": "Analysis failed due to error",
            "confidence": 0,
            "signals": []
        }
        
        # Try to save error result
        try:
            with open(file_paths["output"], "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
        except:
            pass
        
        return error_result

def batch_analyze_symbols(symbols: List[str], timeframes: List[str] = ["M15"], **kwargs) -> Dict[str, Dict]:
    """Analyze multiple symbols and timeframes"""
    results = {}
    total = len(symbols) * len(timeframes)
    current = 0
    
    logger.info(f"🔄 Starting batch analysis for {len(symbols)} symbols, {len(timeframes)} timeframes")
    
    for symbol in symbols:
        results[symbol] = {}
        for timeframe in timeframes:
            current += 1
            logger.info(f"📊 Progress: {current}/{total} - {symbol} {timeframe}")
            
            try:
                result = analyze_symbol(symbol, timeframe, **kwargs)
                results[symbol][timeframe] = result
            except Exception as e:
                logger.error(f"Batch analysis failed for {symbol} {timeframe}: {e}")
                results[symbol][timeframe] = {
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe
                }
    
    logger.info(f"✅ Batch analysis completed: {current}/{total} processed")
    return results

def main():
    """Main function for standalone execution"""
    print("🔍 GPT Analyst - Standalone Mode")
    print("=" * 50)
    
    # 🧹 AUTO CLEANUP before analysis
    print("🧹 GPT Analyst: Auto cleanup before processing...")
    try:
        cleanup_result = cleanup_gpt_analyst_data(max_age_hours=48, keep_latest=10)
        print(f"✅ Cleaned {cleanup_result['total_files_deleted']} files, "
              f"freed {cleanup_result['total_space_freed_mb']:.2f} MB")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Default configuration
    symbol = "XAUUSD"
    timeframe = "M15"
    use_openai = False  # Set to True if you have OpenAI API key
    api_key = None      # Set your API key here
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   OpenAI: {'Enabled' if use_openai else 'Disabled (Mock mode)'}")
    print(f"   TradingAnalyst: {'Available' if TRADING_ANALYST_AVAILABLE else 'Not Available'}")
    print()
    
    # Run analysis
    try:
        result = analyze_symbol(symbol, timeframe, use_openai=use_openai, api_key=api_key)
        
        print("📋 Analysis Results:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("error"):
            print(f"\n❌ Analysis completed with errors")
        else:
            print(f"\n✅ Analysis completed successfully")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Main execution failed: {e}")

def cleanup_gpt_analyst_data(max_age_hours: int = 48, keep_latest: int = 10) -> Dict[str, Any]:
    """
    🧹 GPT ANALYST: Dọn dẹp dữ liệu của module này
    Dọn dẹp analysis results và logs
    
    Args:
        max_age_hours: Tuổi tối đa của file (giờ)
        keep_latest: Số file mới nhất cần giữ lại
    """
    cleanup_stats = {
        'module_name': 'gpt_analyst',
        'directories_cleaned': [],
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Thư mục mà GPT Analyst quản lý
    target_directories = [
        'analysis_results',  # Analysis output
        'gpt_output',       # GPT analysis results (if any)
        'logs'             # Analyst logs
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
    
    print(f"🧹 GPT ANALYST cleanup complete: "
          f"{cleanup_stats['total_files_deleted']} files deleted, "
          f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_directory(directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
    """Helper function để clean một directory"""
    import os
    from datetime import timedelta
    
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả analysis files
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
