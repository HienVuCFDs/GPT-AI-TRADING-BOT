"""
📊 Local Training Data Collector
=================================
Thu thập và chuẩn hóa dữ liệu từ các nguồn local cho training AI.
KHÔNG gửi dữ liệu đến server - chỉ lưu local.

Dữ liệu thu thập:
- Candles (OHLCV) từ data/
- Indicators từ indicator_output/
- Trendline/Support/Resistance từ trendline_sr/
- Candle patterns từ pattern_signals/
- Price patterns từ pattern_price/
- News từ news_output/
- Signals từ analysis_results/
- Execution results

Usage:
    from ai_models.training.data_collector_local import get_local_collector
    
    collector = get_local_collector()
    collector.collect_and_save(symbol="XAUUSD.", signal_type="BUY", confidence=75)
"""

import os
import sys
import json
import glob
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent  # my_trading_bot/
INDICATOR_DIR = BASE_DIR / "indicator_output"
PATTERN_DIR = BASE_DIR / "pattern_signals"
PATTERN_PRICE_DIR = BASE_DIR / "pattern_price"
NEWS_DIR = BASE_DIR / "news_output"
ANALYSIS_DIR = BASE_DIR / "analysis_results"
REPORTS_DIR = BASE_DIR / "reports"
TRENDLINE_SR_DIR = BASE_DIR / "trendline_sr"
DATA_DIR = BASE_DIR / "data"

# Output directory for training data
TRAINING_PENDING_DIR = Path(__file__).parent / "pending"


class LocalTrainingDataCollector:
    """
    Thu thập dữ liệu local cho training AI
    Lưu vào ai_models/training/pending/
    """
    
    def __init__(self):
        # Ensure pending directory exists
        TRAINING_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        
        # Track processed data to avoid duplicates
        self._processed_hashes = set()
        self._load_processed_hashes()
    
    def _load_processed_hashes(self):
        """Load hashes of already processed data"""
        hash_file = TRAINING_PENDING_DIR / ".processed_hashes"
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    self._processed_hashes = set(f.read().splitlines())
            except:
                pass
    
    def _save_processed_hash(self, data_hash: str):
        """Save hash to avoid reprocessing"""
        self._processed_hashes.add(data_hash)
        hash_file = TRAINING_PENDING_DIR / ".processed_hashes"
        try:
            with open(hash_file, 'a') as f:
                f.write(data_hash + '\n')
        except:
            pass
    
    def _compute_hash(self, data: Dict) -> str:
        """Compute hash for deduplication"""
        # Use key fields for hash
        key_data = {
            'symbol': data.get('symbol', ''),
            'signal_type': data.get('signal_type', ''),
            'timestamp': data.get('timestamp', '')[:16] if data.get('timestamp') else '',  # Truncate to minute
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:12]
    
    # ========================================
    # DATA COLLECTION METHODS
    # ========================================
    
    def collect_candles(self, symbol: str, timeframe: str = "M15", count: int = 50) -> List[Dict]:
        """Thu thập candles từ data/"""
        try:
            candle_file = DATA_DIR / f"{symbol}_{timeframe}.json"
            if candle_file.exists():
                with open(candle_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-count:]
        except Exception as e:
            logger.debug(f"Could not load candles for {symbol} {timeframe}: {e}")
        return []
    
    def collect_indicators(self, symbol: str, timeframe: str = "M15") -> Dict:
        """Thu thập indicators từ indicator_output/"""
        try:
            possible_files = [
                INDICATOR_DIR / f"{symbol}_{timeframe}_indicators.json",
                INDICATOR_DIR / f"{symbol}_{timeframe}.json",
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            return data[-1]
                        return data
        except Exception as e:
            logger.debug(f"Could not load indicators for {symbol}: {e}")
        return {}
    
    def collect_trendline_sr(self, symbol: str) -> Dict:
        """Thu thập Trendline và Support/Resistance"""
        try:
            sr_files = list(TRENDLINE_SR_DIR.glob(f"{symbol}*.json"))
            if sr_files:
                latest_file = max(sr_files, key=os.path.getmtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load trendline/SR for {symbol}: {e}")
        return {}
    
    def collect_candle_patterns(self, symbol: str, timeframes: List[str] = None) -> List[Dict]:
        """
        Thu thập mô hình nến từ pattern_signals/
        Files format: {symbol}_{timeframe}_priority_patterns.json
        """
        if timeframes is None:
            timeframes = ['M15', 'M30', 'H1']
        
        all_patterns = []
        try:
            for tf in timeframes:
                pattern_file = PATTERN_DIR / f"{symbol}_{tf}_priority_patterns.json"
                if pattern_file.exists():
                    with open(pattern_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Add timeframe to each pattern for reference
                            for p in data:
                                if isinstance(p, dict):
                                    p['source_timeframe'] = tf
                            all_patterns.extend(data)
                        elif isinstance(data, dict):
                            data['source_timeframe'] = tf
                            all_patterns.append(data)
            
            # Sort by time descending, take recent patterns
            all_patterns.sort(key=lambda x: x.get('time', ''), reverse=True)
            return all_patterns[:10]  # Max 10 recent patterns
            
        except Exception as e:
            logger.debug(f"Could not load candle patterns for {symbol}: {e}")
        return []
    
    def collect_price_patterns(self, symbol: str) -> List[Dict]:
        """
        Thu thập mô hình giá từ pattern_price/
        (Optional - có thể không có nếu tính năng không bật)
        """
        all_patterns = []
        try:
            pattern_files = list(PATTERN_PRICE_DIR.glob(f"{symbol}*.json"))
            for pattern_file in pattern_files:
                try:
                    with open(pattern_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_patterns.extend(data)
                        elif isinstance(data, dict):
                            all_patterns.append(data)
                except:
                    continue
            
            # Sort by time descending
            all_patterns.sort(key=lambda x: x.get('time', x.get('end_time', '')), reverse=True)
            return all_patterns[:5]  # Max 5 price patterns
            
        except Exception as e:
            logger.debug(f"Could not load price patterns for {symbol}: {e}")
        return []
    
    def collect_news(self, currency_filter: str = None) -> Dict:
        """
        Thu thập news thực từ 2 nguồn:
        1. news_forexfactory_*.json - Lịch kinh tế (calendar)
        2. recent_news_with_actual_*.json - Tin tức đã công bố với kết quả thực
        
        Args:
            currency_filter: Filter by currency (e.g., 'USD', 'EUR'). None = all
        """
        result = {
            'calendar': [],      # Lịch kinh tế sắp tới
            'recent_news': [],   # Tin vừa công bố
            'high_impact': [],   # Tin quan trọng (High impact)
        }
        
        try:
            # 1. Thu thập lịch kinh tế (forexfactory)
            calendar_files = sorted(NEWS_DIR.glob("news_forexfactory_*.json"), reverse=True)
            if calendar_files:
                with open(calendar_files[0], 'r', encoding='utf-8') as f:
                    calendar_data = json.load(f)
                    if isinstance(calendar_data, list):
                        for news in calendar_data:
                            # Filter by currency if specified
                            if currency_filter and news.get('currency') != currency_filter:
                                continue
                            result['calendar'].append({
                                'event': news.get('event', ''),
                                'currency': news.get('currency', ''),
                                'impact': news.get('impact', 'Low'),
                                'time_mt5': news.get('datetime_mt5', news.get('time_mt5', '')),
                                'forecast': news.get('forecast', ''),
                                'previous': news.get('previous', ''),
                                'actual': news.get('actual', ''),
                            })
                            if news.get('impact', '').upper() == 'HIGH':
                                result['high_impact'].append(news.get('event', ''))
            
            # 2. Thu thập tin tức đã công bố với actual values
            recent_files = sorted(NEWS_DIR.glob("recent_news_with_actual_*.json"), reverse=True)
            if recent_files:
                with open(recent_files[0], 'r', encoding='utf-8') as f:
                    recent_data = json.load(f)
                    if isinstance(recent_data, list):
                        for news in recent_data:
                            if currency_filter and news.get('currency') != currency_filter:
                                continue
                            result['recent_news'].append({
                                'event': news.get('event', ''),
                                'currency': news.get('currency', ''),
                                'impact': news.get('impact', 'Low'),
                                'actual': news.get('actual', ''),
                                'forecast': news.get('forecast', ''),
                                'surprise': self._calc_news_surprise(news),
                            })
            
            # Giới hạn số lượng
            result['calendar'] = result['calendar'][:10]
            result['recent_news'] = result['recent_news'][:5]
            
        except Exception as e:
            logger.debug(f"Could not load news: {e}")
        
        return result
    
    def _calc_news_surprise(self, news: Dict) -> str:
        """Tính surprise: actual vs forecast"""
        try:
            actual = news.get('actual', '')
            forecast = news.get('forecast', '')
            if actual and forecast:
                # Remove % and K, M, B suffixes for comparison
                actual_val = float(actual.replace('%', '').replace('K', '').replace('M', '').replace('B', ''))
                forecast_val = float(forecast.replace('%', '').replace('K', '').replace('M', '').replace('B', ''))
                diff = actual_val - forecast_val
                if diff > 0:
                    return 'BETTER'
                elif diff < 0:
                    return 'WORSE'
                else:
                    return 'AS_EXPECTED'
        except:
            pass
        return 'UNKNOWN'
    
    def collect_signal(self, symbol: str) -> Dict:
        """Thu thập signal từ analysis_results/"""
        try:
            signal_files = list(ANALYSIS_DIR.glob(f"{symbol}*_signal_*.json"))
            if signal_files:
                latest_file = max(signal_files, key=os.path.getmtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load signal for {symbol}: {e}")
        return {}
    
    # ========================================
    # STANDARDIZED DATA COLLECTION
    # ========================================
    
    def _infer_market_type(self, trendline_sr: Dict) -> str:
        """
        Infer market_type từ trendline_sr data
        Returns: TRENDING_UP, TRENDING_DOWN, SIDEWAY, UNKNOWN
        """
        if not trendline_sr:
            return 'UNKNOWN'
        
        trend = trendline_sr.get('trend', trendline_sr.get('trend_direction', ''))
        trend_lower = str(trend).lower()
        
        # Check for sideway
        if 'sideway' in trend_lower or 'side' in trend_lower or 'rang' in trend_lower:
            return 'SIDEWAY'
        if trendline_sr.get('sideway_range'):
            return 'SIDEWAY'
        
        # Check for uptrend
        if 'up' in trend_lower or 'bull' in trend_lower:
            return 'TRENDING_UP'
        
        # Check for downtrend
        if 'down' in trend_lower or 'bear' in trend_lower:
            return 'TRENDING_DOWN'
        
        return 'UNKNOWN'
    
    def _get_currency_from_symbol(self, symbol: str) -> str:
        """Extract currency from symbol for news filtering"""
        symbol_upper = symbol.upper()
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 'USD'  # Gold trades against USD
        if 'EUR' in symbol_upper:
            return 'EUR'
        if 'GBP' in symbol_upper:
            return 'GBP'
        if 'JPY' in symbol_upper:
            return 'JPY'
        if 'USD' in symbol_upper:
            return 'USD'
        if 'AUD' in symbol_upper:
            return 'AUD'
        if 'NZD' in symbol_upper:
            return 'NZD'
        if 'CAD' in symbol_upper:
            return 'CAD'
        if 'CHF' in symbol_upper:
            return 'CHF'
        # Crypto - no specific currency
        if any(c in symbol_upper for c in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'LTC']):
            return None
        return None
    
    def collect_all_for_symbol(self, symbol: str) -> Dict:
        """
        Thu thập TẤT CẢ dữ liệu THỰC cho 1 symbol
        Chuẩn hóa format cho training
        """
        # Collect trendline_sr first for market_type inference
        trendline_sr = self.collect_trendline_sr(symbol)
        market_type = self._infer_market_type(trendline_sr)
        
        # Get currency for news filtering
        currency = self._get_currency_from_symbol(symbol)
        
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_type": market_type,  # NEW: TRENDING_UP, TRENDING_DOWN, SIDEWAY
            
            # Candles - last 20 bars for context
            "candles": {
                "M15": self.collect_candles(symbol, "M15", 20),
                "M30": self.collect_candles(symbol, "M30", 20),
                "H1": self.collect_candles(symbol, "H1", 20),
            },
            
            # Indicators by timeframe
            "indicators": {
                "M15": self.collect_indicators(symbol, "M15"),
                "M30": self.collect_indicators(symbol, "M30"),
                "H1": self.collect_indicators(symbol, "H1"),
            },
            
            # Support/Resistance and Trend
            "trendline_sr": trendline_sr,
            
            # Patterns - REAL data from pattern_signals/
            "patterns": {
                "candle_patterns": self.collect_candle_patterns(symbol),  # Always collected
                "price_patterns": self.collect_price_patterns(symbol),    # Optional if enabled
            },
            
            # News - REAL data from news_output/
            "news": self.collect_news(currency_filter=currency),
            
            # Current signal
            "signal": self.collect_signal(symbol),
            
            # Data source
            "training_source": "local_collector_v2",
        }
        
        return data
    
    def standardize_training_record(
        self,
        symbol: str,
        signal_type: str,  # BUY/SELL/HOLD
        confidence: float,
        entry_price: float = 0,
        outcome: str = None,  # WIN/LOSS/PENDING
        profit_pips: float = None,
        execution_details: Dict = None
    ) -> Dict:
        """
        Tạo record chuẩn hóa cho training
        Bao gồm: indicators, patterns, news, market_type
        """
        # Collect all market data
        market_data = self.collect_all_for_symbol(symbol)
        
        # Extract key features for training
        indicators_m15 = market_data.get('indicators', {}).get('M15', {})
        indicators_h1 = market_data.get('indicators', {}).get('H1', {})
        patterns = market_data.get('patterns', {})
        news = market_data.get('news', {})
        
        record = {
            # Metadata
            "id": f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_type": market_data.get('market_type', 'UNKNOWN'),
            
            # Signal info
            "signal_type": signal_type.upper(),
            "confidence": confidence,
            "entry_price": entry_price,
            
            # Outcome (if known)
            "outcome": outcome,
            "profit_pips": profit_pips,
            
            # Indicators by timeframe
            "indicators": {
                "M15": indicators_m15,
                "H1": indicators_h1,
            },
            
            # Patterns - from pattern_signals/ and pattern_price/
            "patterns": {
                "candle_patterns": patterns.get('candle_patterns', []),
                "price_patterns": patterns.get('price_patterns', []),
            },
            
            # Trendline and Support/Resistance
            "trendline_sr": market_data.get('trendline_sr', {}),
            
            # News - from news_output/
            "news": news,
            
            # Execution details
            "execution": execution_details or {},
            
            # Source
            "training_source": "local_collector_v2",
            "version": "2.0",
        }
        
        return record
    
    # ========================================
    # SAVE METHODS
    # ========================================
    
    def save_training_record(self, record: Dict) -> bool:
        """
        Lưu record vào pending folder
        """
        try:
            # Check for duplicates
            data_hash = self._compute_hash(record)
            if data_hash in self._processed_hashes:
                logger.debug(f"Duplicate record skipped: {data_hash}")
                return False
            
            # Generate filename
            symbol = record.get('symbol', 'UNKNOWN').replace('.', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            signal_type = record.get('signal_type', 'UNKNOWN')
            filename = f"training_{symbol}_{signal_type}_{timestamp}_{data_hash}.json"
            
            filepath = TRAINING_PENDING_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False, default=str)
            
            # Save hash
            self._save_processed_hash(data_hash)
            
            logger.info(f"Saved training record: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training record: {e}")
            return False
    
    def collect_and_save(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        entry_price: float = 0,
        outcome: str = None,
        profit_pips: float = None,
        execution_details: Dict = None
    ) -> bool:
        """
        Thu thập và lưu dữ liệu training
        
        Usage:
            collector.collect_and_save(
                symbol="XAUUSD.",
                signal_type="BUY",
                confidence=75,
                entry_price=2350.50
            )
        """
        record = self.standardize_training_record(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            outcome=outcome,
            profit_pips=profit_pips,
            execution_details=execution_details
        )
        
        return self.save_training_record(record)
    
    def log_execution(self, action_type: str, symbol: str, ticket: int, 
                      details: Dict, success: bool, result_message: str = "") -> bool:
        """
        Log execution action để train model
        """
        record = {
            "type": "execution",
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "symbol": symbol,
            "ticket": ticket,
            "success": success,
            "result_message": result_message,
            "details": details
        }
        
        # Save to executions subfolder
        exec_dir = TRAINING_PENDING_DIR / "executions"
        exec_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"exec_{symbol.replace('.', '')}_{action_type}_{timestamp}.json"
        
        try:
            with open(exec_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about collected data"""
        pending_files = list(TRAINING_PENDING_DIR.glob("training_*.json"))
        exec_files = list((TRAINING_PENDING_DIR / "executions").glob("exec_*.json")) if (TRAINING_PENDING_DIR / "executions").exists() else []
        
        return {
            "pending_training_files": len(pending_files),
            "execution_logs": len(exec_files),
            "total_processed_hashes": len(self._processed_hashes),
            "pending_dir": str(TRAINING_PENDING_DIR)
        }


# Singleton instance
_collector_instance = None

def get_local_collector() -> LocalTrainingDataCollector:
    """Get singleton collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = LocalTrainingDataCollector()
    return _collector_instance


# ========================================
# CLI
# ========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Training Data Collector')
    parser.add_argument('--stats', action='store_true', help='Show stats')
    parser.add_argument('--test', type=str, help='Test collect for symbol')
    args = parser.parse_args()
    
    collector = get_local_collector()
    
    if args.stats:
        stats = collector.get_stats()
        print("\n📊 Training Data Stats:")
        print(f"   Pending files: {stats['pending_training_files']}")
        print(f"   Execution logs: {stats['execution_logs']}")
        print(f"   Processed hashes: {stats['total_processed_hashes']}")
        print(f"   Directory: {stats['pending_dir']}")
    
    if args.test:
        print(f"\n🔍 Testing collect for {args.test}...")
        result = collector.collect_and_save(
            symbol=args.test,
            signal_type="TEST",
            confidence=50
        )
        print(f"   Result: {'✅ Saved' if result else '❌ Failed/Duplicate'}")
