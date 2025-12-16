"""
📊 Training Data Collector for ai_models/
==========================================
Thu thập dữ liệu training từ:
- analysis_results/ (signals)
- MT5 trade results (outcomes)
- indicator_output/ (features)
- pattern_signals/, pattern_price/ (patterns)
- trendline_sr/ (support/resistance)

Data được lưu vào: ai_training/data/
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Setup logging
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
PENDING_DIR = Path(__file__).parent / "pending"

# Source directories
ANALYSIS_DIR = BASE_DIR / "analysis_results"
INDICATOR_DIR = BASE_DIR / "indicator_output"
PATTERN_SIGNALS_DIR = BASE_DIR / "pattern_signals"
PATTERN_PRICE_DIR = BASE_DIR / "pattern_price"
TRENDLINE_SR_DIR = BASE_DIR / "trendline_sr"
OLD_PENDING_DIR = BASE_DIR / "ai_training_pending"


class TrainingDataCollector:
    """
    Thu thập và quản lý training data cho ai_models/
    """
    
    def __init__(self):
        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        
        # Database file
        self.db_file = DATA_DIR / "training_data.json"
        self.trade_results_file = DATA_DIR / "trade_results.json"
        
        # Load existing data
        self.signals = self._load_json(self.db_file, default=[])
        self.trade_results = self._load_json(self.trade_results_file, default=[])
        
        # Deduplication
        self._signal_hashes = set()
        self._init_hashes()
    
    def _load_json(self, path: Path, default=None) -> Any:
        """Load JSON file safely"""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
        return default if default is not None else {}
    
    def _save_json(self, path: Path, data: Any):
        """Save JSON file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Could not save {path}: {e}")
    
    def _init_hashes(self):
        """Initialize hash set for deduplication"""
        for sig in self.signals:
            h = self._hash_signal(sig)
            self._signal_hashes.add(h)
    
    def _hash_signal(self, sig: Dict) -> str:
        """Create hash for signal deduplication"""
        key = f"{sig.get('symbol')}_{sig.get('signal_type')}_{sig.get('timestamp', '')[:16]}"
        return key
    
    # ========================================
    # COLLECT FROM ANALYSIS RESULTS
    # ========================================
    
    def collect_from_analysis_results(self, days_back: int = 7) -> int:
        """Collect signals from analysis_results/*.json"""
        if not ANALYSIS_DIR.exists():
            return 0
        
        cutoff = datetime.now() - timedelta(days=days_back)
        collected = 0
        
        for signal_file in ANALYSIS_DIR.glob("*_signal_*.json"):
            try:
                # Check file date
                stat = signal_file.stat()
                file_date = datetime.fromtimestamp(stat.st_mtime)
                if file_date < cutoff:
                    continue
                
                with open(signal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                signal = self._extract_signal_from_analysis(data, signal_file.name)
                if signal and self._add_signal(signal):
                    collected += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {signal_file}: {e}")
        
        if collected > 0:
            self._save_json(self.db_file, self.signals)
            logger.info(f"✅ Collected {collected} signals from analysis_results/")
        
        return collected
    
    def _extract_signal_from_analysis(self, data: Dict, filename: str) -> Optional[Dict]:
        """Extract signal data from analysis result"""
        try:
            symbol = data.get('symbol', '')
            final_signal = data.get('final_signal', {})
            signal_type = final_signal.get('signal', 'HOLD')
            
            # Skip HOLD - not useful for training
            if signal_type in ['HOLD', 'NEUTRAL', 'NO_TRADE']:
                return None
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': final_signal.get('confidence', 50),
                'entry_price': final_signal.get('entry', 0),
                'stoploss': final_signal.get('stoploss', 0),
                'takeprofit': final_signal.get('takeprofit', 0),
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'source': 'analysis_results',
                'filename': filename,
                'indicators': self._load_indicators(symbol),
                'patterns': self._load_patterns(symbol),
                'trendline_sr': self._load_trendline_sr(symbol)
            }
        except Exception as e:
            logger.warning(f"Could not extract signal: {e}")
            return None
    
    def _add_signal(self, signal: Dict) -> bool:
        """Add signal if not duplicate"""
        h = self._hash_signal(signal)
        if h in self._signal_hashes:
            return False
        
        self._signal_hashes.add(h)
        self.signals.append(signal)
        return True
    
    # ========================================
    # LOAD SUPPORTING DATA
    # ========================================
    
    def _load_indicators(self, symbol: str) -> Dict:
        """Load indicators for symbol"""
        result = {}
        for tf in ['M15', 'M30', 'H1', 'H4']:
            ind_file = INDICATOR_DIR / f"{symbol}_{tf}.json"
            if ind_file.exists():
                try:
                    with open(ind_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        result[tf] = data[-1]  # Latest
                    elif isinstance(data, dict):
                        result[tf] = data
                except:
                    pass
        return result
    
    def _load_patterns(self, symbol: str) -> Dict:
        """Load patterns for symbol"""
        result = {'candle_patterns': [], 'price_patterns': []}
        
        # Candle patterns
        pattern_file = PATTERN_SIGNALS_DIR / f"{symbol}_patterns.json"
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    result['candle_patterns'] = json.load(f)
            except:
                pass
        
        # Price patterns
        price_file = PATTERN_PRICE_DIR / f"{symbol}_price_patterns.json"
        if price_file.exists():
            try:
                with open(price_file, 'r', encoding='utf-8') as f:
                    result['price_patterns'] = json.load(f)
            except:
                pass
        
        return result
    
    def _load_trendline_sr(self, symbol: str) -> Dict:
        """Load trendline/SR for symbol"""
        for tf in ['H1', 'M30', 'M15']:
            sr_file = TRENDLINE_SR_DIR / f"{symbol}_{tf}_trendline_sr.json"
            if sr_file.exists():
                try:
                    with open(sr_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    pass
        return {}
    
    # ========================================
    # COLLECT FROM MT5
    # ========================================
    
    def collect_from_mt5(self, days_back: int = 7) -> int:
        """Collect trade results from MT5"""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.warning("MetaTrader5 not available")
            return 0
        
        if not mt5.initialize():
            logger.warning("MT5 initialization failed")
            return 0
        
        from_date = datetime.now() - timedelta(days=days_back)
        to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        mt5.shutdown()
        
        if deals is None or len(deals) == 0:
            return 0
        
        # Group by position
        positions = {}
        for deal in deals:
            pos_id = deal.position_id
            if pos_id == 0:
                continue
            if pos_id not in positions:
                positions[pos_id] = []
            positions[pos_id].append(deal)
        
        collected = 0
        for pos_id, deals_list in positions.items():
            result = self._process_position(pos_id, deals_list)
            if result:
                self.trade_results.append(result)
                collected += 1
        
        if collected > 0:
            self._save_json(self.trade_results_file, self.trade_results)
            logger.info(f"✅ Collected {collected} trade results from MT5")
        
        return collected
    
    def _process_position(self, pos_id: int, deals: List) -> Optional[Dict]:
        """Process MT5 position deals"""
        entry_deal = None
        exit_deal = None
        
        for deal in deals:
            if deal.entry == 0:  # IN
                entry_deal = deal
            elif deal.entry == 1:  # OUT
                exit_deal = deal
        
        if not entry_deal or not exit_deal:
            return None
        
        symbol = entry_deal.symbol
        trade_type = "BUY" if entry_deal.type == 0 else "SELL"
        
        # Calculate pips
        if trade_type == "BUY":
            profit_pips = (exit_deal.price - entry_deal.price)
        else:
            profit_pips = (entry_deal.price - exit_deal.price)
        
        # Normalize pips
        if 'JPY' in symbol:
            profit_pips *= 100
        elif 'XAU' in symbol or 'GOLD' in symbol:
            profit_pips *= 10
        else:
            profit_pips *= 10000
        
        # Determine outcome
        if exit_deal.profit > 0:
            outcome = 'WIN'
        elif exit_deal.profit < 0:
            outcome = 'LOSS'
        else:
            outcome = 'BREAKEVEN'
        
        return {
            'ticket_id': pos_id,
            'symbol': symbol,
            'trade_type': trade_type,
            'entry_price': entry_deal.price,
            'exit_price': exit_deal.price,
            'profit_pips': round(profit_pips, 1),
            'profit_money': exit_deal.profit,
            'outcome': outcome,
            'entry_time': datetime.fromtimestamp(entry_deal.time).isoformat(),
            'exit_time': datetime.fromtimestamp(exit_deal.time).isoformat(),
            'duration_minutes': int((exit_deal.time - entry_deal.time) / 60)
        }
    
    # ========================================
    # MIGRATE OLD PENDING DATA
    # ========================================
    
    def migrate_old_pending(self) -> int:
        """Migrate from old ai_training_pending/ to new location"""
        if not OLD_PENDING_DIR.exists():
            return 0
        
        migrated = 0
        for old_file in OLD_PENDING_DIR.glob("*.json"):
            try:
                new_path = PENDING_DIR / old_file.name
                if not new_path.exists():
                    old_file.rename(new_path)
                    migrated += 1
            except Exception as e:
                logger.warning(f"Could not migrate {old_file}: {e}")
        
        logger.info(f"📦 Migrated {migrated} files from ai_training_pending/")
        return migrated
    
    # ========================================
    # GENERATE TRAINING EXAMPLES
    # ========================================
    
    def generate_training_examples(self) -> List[Dict]:
        """Generate training examples matching signals with outcomes"""
        examples = []
        
        # Index trade results by symbol + time
        result_index = {}
        for tr in self.trade_results:
            key = f"{tr['symbol']}_{tr['entry_time'][:13]}"  # Hour precision
            result_index[key] = tr
        
        for signal in self.signals:
            # Try to find matching trade result
            symbol = signal.get('symbol', '')
            timestamp = signal.get('timestamp', '')[:13]
            key = f"{symbol}_{timestamp}"
            
            trade_result = result_index.get(key)
            
            example = {
                'symbol': symbol,
                'signal_type': signal.get('signal_type'),
                'confidence': signal.get('confidence', 50),
                'indicators': signal.get('indicators', {}),
                'patterns': signal.get('patterns', {}),
                'trendline_sr': signal.get('trendline_sr', {}),
                'timestamp': signal.get('timestamp'),
                'has_outcome': trade_result is not None
            }
            
            if trade_result:
                example['outcome'] = trade_result.get('outcome')
                example['profit_pips'] = trade_result.get('profit_pips', 0)
                example['profit_money'] = trade_result.get('profit_money', 0)
            
            examples.append(example)
        
        return examples
    
    def export_for_training(self, output_file: Path = None) -> Path:
        """Export data for model training"""
        if output_file is None:
            output_file = DATA_DIR / "training_examples.json"
        
        examples = self.generate_training_examples()
        self._save_json(output_file, examples)
        
        logger.info(f"📁 Exported {len(examples)} examples to {output_file}")
        return output_file
    
    # ========================================
    # STATS
    # ========================================
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        examples = self.generate_training_examples()
        with_outcome = [e for e in examples if e.get('has_outcome')]
        
        wins = len([e for e in with_outcome if e.get('outcome') == 'WIN'])
        losses = len([e for e in with_outcome if e.get('outcome') == 'LOSS'])
        
        return {
            'total_signals': len(self.signals),
            'total_trade_results': len(self.trade_results),
            'training_examples': len(examples),
            'with_outcome': len(with_outcome),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            'pending_files': len(list(PENDING_DIR.glob("*.json")))
        }


# ========================================
# SINGLETON
# ========================================
_collector_instance = None


def get_collector() -> TrainingDataCollector:
    """Get singleton collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = TrainingDataCollector()
    return _collector_instance


# ========================================
# TEST
# ========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Training Data Collector Test")
    print("=" * 50)
    
    collector = get_collector()
    
    # Migrate old data
    collector.migrate_old_pending()
    
    # Collect from sources
    collector.collect_from_analysis_results(days_back=30)
    collector.collect_from_mt5(days_back=30)
    
    # Show stats
    stats = collector.get_stats()
    print(f"\n📈 Statistics:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Export
    collector.export_for_training()
    print("\n✅ Done!")
