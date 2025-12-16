"""
Auto Collect Training Data
Tự động thu thập data từ:
- comprehensive_aggregator signals
- MT5 trade results
- Tạo training examples

Chạy script này định kỳ (mỗi giờ hoặc cuối ngày)
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available")

from ai_server.training_data_collector import get_collector, print_training_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")
INDICATOR_DIR = os.path.join(BASE_DIR, "indicator_output")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def collect_today_signals():
    """Thu thập signals từ analysis_results được tạo hôm nay"""
    
    collector = get_collector()
    today = datetime.now().strftime("%Y%m%d")
    
    # Find signal files created today
    signal_files = glob.glob(os.path.join(ANALYSIS_DIR, f"*_signal_{today}*.json"))
    print(f"📁 Found {len(signal_files)} signal files from today")
    
    imported = 0
    for signal_file in signal_files:
        try:
            with open(signal_file, 'r', encoding='utf-8') as f:
                signal_data = json.load(f)
            
            symbol = signal_data.get('symbol', '')
            final_signal = signal_data.get('final_signal', {})
            signal_type = final_signal.get('signal', 'HOLD')
            
            # Skip HOLD/NEUTRAL - không có giá trị training
            if signal_type in ['HOLD', 'NEUTRAL', 'NO_TRADE']:
                continue
            
            entry = final_signal.get('entry', 0)
            sl = final_signal.get('stoploss', 0)
            tp = final_signal.get('takeprofit', 0)
            confidence = final_signal.get('confidence', 50)
            
            # Load indicators
            indicators = signal_data.get('indicator_summary', {})
            
            # Load full indicator data if available
            for tf in ['M15', 'M30', 'H1']:
                indicator_file = os.path.join(INDICATOR_DIR, f"{symbol}_{tf}.json")
                if os.path.exists(indicator_file):
                    with open(indicator_file, 'r', encoding='utf-8') as f:
                        ind_data = json.load(f)
                        if isinstance(ind_data, list) and len(ind_data) > 0:
                            indicators.update(ind_data[-1])
                        elif isinstance(ind_data, dict):
                            indicators.update(ind_data)
                    break
            
            # Get patterns
            patterns = {
                'candle_patterns': signal_data.get('candle_patterns', []),
                'price_patterns': signal_data.get('price_patterns', []),
                'support_levels': signal_data.get('support_levels', []),
                'resistance_levels': signal_data.get('resistance_levels', []),
                'trend_direction': signal_data.get('trend', 'UNKNOWN')
            }
            
            # Source
            source = signal_data.get('logic_type', 'comprehensive_aggregator')
            if 'AI' in str(source).upper():
                source = 'ai_server'
            else:
                source = 'comprehensive_aggregator'
            
            # Save signal
            signal_id = collector.save_signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry,
                stoploss=sl,
                takeprofit=tp,
                confidence=confidence,
                indicators=indicators,
                patterns=patterns,
                source=source
            )
            
            if signal_id > 0:
                imported += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {signal_file}: {e}")
    
    print(f"✅ Imported {imported} signals from today")
    return imported


def sync_mt5_trade_results(days_back: int = 1):
    """Đồng bộ kết quả trade từ MT5"""
    
    if not MT5_AVAILABLE:
        print("❌ MT5 not available, skipping trade sync")
        return 0
    
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return 0
    
    collector = get_collector()
    
    # Get closed trades
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()
    
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None or len(deals) == 0:
        print("📭 No closed deals found in the period")
        mt5.shutdown()
        return 0
    
    print(f"📊 Found {len(deals)} deals in last {days_back} day(s)")
    
    # Group deals by position
    positions = {}
    for deal in deals:
        pos_id = deal.position_id
        if pos_id == 0:
            continue
        if pos_id not in positions:
            positions[pos_id] = []
        positions[pos_id].append(deal)
    
    print(f"📦 Found {len(positions)} unique positions")
    
    synced = 0
    for pos_id, deals_list in positions.items():
        # Find entry and exit deals
        entry_deal = None
        exit_deal = None
        
        for deal in deals_list:
            if deal.entry == 0:  # Entry (IN)
                entry_deal = deal
            elif deal.entry == 1:  # Exit (OUT)
                exit_deal = deal
        
        if not entry_deal or not exit_deal:
            continue
        
        # Extract data
        symbol = entry_deal.symbol
        trade_type = "BUY" if entry_deal.type == 0 else "SELL"
        entry_price = entry_deal.price
        exit_price = exit_deal.price
        volume = entry_deal.volume
        profit_money = exit_deal.profit
        swap = exit_deal.swap
        commission = exit_deal.commission
        
        # Calculate pips
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            point = symbol_info.point
            digits = symbol_info.digits
            
            if trade_type == "BUY":
                profit_pips = (exit_price - entry_price) / point
            else:
                profit_pips = (entry_price - exit_price) / point
        else:
            # Fallback for unknown symbols
            if "JPY" in symbol:
                profit_pips = abs(exit_price - entry_price) * 100
            elif "USD" in symbol and ("XAU" in symbol or "GOLD" in symbol):
                profit_pips = abs(exit_price - entry_price) * 10
            else:
                profit_pips = abs(exit_price - entry_price) * 10000
            
            if (trade_type == "BUY" and exit_price < entry_price) or \
               (trade_type == "SELL" and exit_price > entry_price):
                profit_pips = -profit_pips
        
        # Times
        entry_time = datetime.fromtimestamp(entry_deal.time).strftime("%Y-%m-%d %H:%M:%S")
        exit_time = datetime.fromtimestamp(exit_deal.time).strftime("%Y-%m-%d %H:%M:%S")
        duration = int((exit_deal.time - entry_deal.time) / 60)
        
        # Determine exit reason
        if profit_pips > 0:
            if profit_pips > 100:
                exit_reason = "TP_HIT"
            else:
                exit_reason = "PROFIT_CLOSE"
        else:
            if profit_pips < -100:
                exit_reason = "SL_HIT"
            else:
                exit_reason = "LOSS_CLOSE"
        
        # Get indicators at entry time (if available)
        indicators_at_entry = None
        indicator_file = os.path.join(INDICATOR_DIR, f"{symbol}_M15.json")
        if os.path.exists(indicator_file):
            try:
                with open(indicator_file, 'r', encoding='utf-8') as f:
                    ind_data = json.load(f)
                    if isinstance(ind_data, list) and len(ind_data) > 0:
                        indicators_at_entry = ind_data[-1]
            except:
                pass
        
        # Save trade result
        result_id = collector.save_trade_result(
            ticket_id=pos_id,
            symbol=symbol,
            trade_type=trade_type,
            entry_time=entry_time,
            entry_price=entry_price,
            volume=volume,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            profit_money=profit_money,
            profit_pips=profit_pips,
            duration_minutes=duration,
            swap=swap,
            commission=commission,
            indicators_at_entry=indicators_at_entry
        )
        
        if result_id > 0:
            synced += 1
    
    mt5.shutdown()
    print(f"✅ Synced {synced} trade results from MT5")
    return synced


def generate_new_training_examples():
    """Tạo training examples từ data mới"""
    
    collector = get_collector()
    created = collector.generate_training_examples()
    print(f"✅ Generated {created} new training examples")
    return created


def export_training_data():
    """Export data cho fine-tuning - hỗ trợ cả PyTorch và future LLM"""
    
    collector = get_collector()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export cho PyTorch model (ai_models/training/pending/)
    pytorch_dir = os.path.join(BASE_DIR, "ai_models", "training", "pending")
    os.makedirs(pytorch_dir, exist_ok=True)
    pytorch_count = export_for_pytorch(collector, pytorch_dir, timestamp)
    
    # 2. Export standard JSON cho future LLM server
    llm_dir = os.path.join(BASE_DIR, "ai_server", "training_data")
    os.makedirs(llm_dir, exist_ok=True)
    standard_file = os.path.join(llm_dir, f"training_data_{timestamp}.json")
    data = collector.export_training_data(standard_file)
    
    print(f"📁 Exported training data:")
    print(f"   - PyTorch (ai_models/training/pending/): {pytorch_count} files")
    print(f"   - LLM Server (ai_server/training_data/): {len(data)} examples")
    
    return len(data)


def export_for_pytorch(collector, output_dir: str, timestamp: str = None) -> int:
    """
    Export data sang format cho PyTorch Neural Network model
    Mỗi sample là 1 file JSON chứa indicators, patterns, trendline_sr, signal
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get training examples from collector
    examples = collector.get_all_training_examples()
    
    exported = 0
    for ex in examples:
        try:
            symbol = ex.get('symbol', 'UNKNOWN')
            signal_type = ex.get('signal_type', 'HOLD')
            confidence = ex.get('confidence', 50)
            
            # Build data structure matching feature_extractor format
            data = {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'indicators': {
                    'M15': ex.get('indicators', {}),
                    'H1': ex.get('indicators', {})
                },
                'patterns': ex.get('patterns', {}),
                'trendline_sr': {
                    'supports': ex.get('patterns', {}).get('support_levels', []),
                    'resistances': ex.get('patterns', {}).get('resistance_levels', []),
                    'trend': ex.get('patterns', {}).get('trend_direction', 'Sideways'),
                },
                'news': [],
                'timestamp': ex.get('timestamp', datetime.now().isoformat()),
                'source': 'auto_collect'
            }
            
            # Save to file with unique name
            filename = f"{symbol}_{signal_type}_{timestamp}_{exported}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            exported += 1
            
        except Exception as e:
            print(f"⚠️ Error exporting example: {e}")
    
    return exported


def main():
    """Main function - Thu thập data tự động"""
    
    print("\n" + "="*60)
    print("🤖 AUTO COLLECT TRAINING DATA")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    collector = get_collector()
    
    # Step 1: Import existing data (chỉ chạy lần đầu)
    stats = collector.get_training_stats()
    if stats['total_signals'] == 0:
        print("\n📥 Step 1: First run - Importing existing data...")
        collector.import_from_trading_history()
        collector.import_from_execution_reports()
        collector.import_signals_from_analysis_results()
    else:
        print(f"\n📊 Database has {stats['total_signals']} signals, skipping full import")
    
    # Step 2: Collect today's signals
    print("\n📥 Step 2: Collecting today's signals...")
    collect_today_signals()
    
    # Step 3: Sync MT5 trade results
    print("\n📥 Step 3: Syncing trade results from MT5...")
    sync_mt5_trade_results(days_back=1)
    
    # Step 4: Generate training examples
    print("\n🔄 Step 4: Generating training examples...")
    generate_new_training_examples()
    
    # Step 5: Show report
    print("\n📊 Step 5: Training Data Report")
    stats = collector.get_training_stats()
    print_training_report(stats)
    
    # Step 6: Check if ready for fine-tuning
    ready = stats['training_examples'] - stats['used_in_training']
    if ready >= 100:
        print(f"\n🎉 You have {ready} examples ready for fine-tuning!")
        print("   Run: python ai_server/fine_tune_model.py")
    elif ready >= 50:
        print(f"\n📈 You have {ready} examples. Need 100+ for best results.")
        print("   Keep trading to collect more data!")
    else:
        print(f"\n📈 You have {ready} examples. Need more data for training.")
        print("   Continue using comprehensive_aggregator to generate signals.")
    
    print("\n✅ Auto collect completed!")
    return stats


if __name__ == "__main__":
    main()
