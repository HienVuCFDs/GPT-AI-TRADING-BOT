#!/usr/bin/env python3

import json
import os
from typing import List

def order_executor_atr_logic(symbol: str) -> float:
    """Copy exact logic from order_executor.py"""
    try:
        # Try multiple timeframes for ATR data
        timeframes = ['M5', 'M15', 'M30', 'H1']
        
        # Clean symbol - remove trailing dot if exists
        clean_symbol = symbol.rstrip('.')
        
        for tf in timeframes:
            # Try both original and clean symbol names
            for sym in [symbol, clean_symbol]:
                indicator_file = f"indicator_output/{sym}._{tf}_indicators.json"
                print(f"Checking ATR file: {indicator_file}")
                
                if os.path.exists(indicator_file):
                    try:
                        with open(indicator_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        if data and isinstance(data, list) and len(data) > 0:
                            # Get latest ATR value
                            latest_data = data[-1]
                            
                            for atr_key in ['ATR14', 'atr', 'ATR', 'ATR_14']:
                                if atr_key in latest_data and latest_data[atr_key] is not None:
                                    atr_value = float(latest_data[atr_key])
                                    print(f"✅ Found ATR {tf}: {atr_key}={atr_value} for {symbol}")
                                    return atr_value
                    except Exception as e:
                        print(f"Error reading ATR from {tf}: {e}")
                        continue
                        
        print(f"No ATR data found for {symbol} in any timeframe")
        return 0.0
        
    except Exception as e:
        print(f"Error getting ATR value: {e}")
        return 0.0

def comprehensive_aggregator_atr_logic(symbol: str) -> float:
    """Copy exact logic from comprehensive_aggregator.py"""
    def _last_num(ind_list, keys: List[str]):
        print(f"DEBUG _last_num called with keys: {keys}")
        if not ind_list:
            print("DEBUG: ind_list is empty")
            return None
        row = ind_list[-1] if isinstance(ind_list[-1], dict) else None
        print(f"DEBUG: row type: {type(row)}")
        if not isinstance(row, dict):
            print("DEBUG: row is not a dict")
            return None
        
        print(f"DEBUG: Available keys in row (ATR related): {[k for k in row.keys() if 'atr' in k.lower()]}")
        for k in keys:
            print(f"DEBUG: Checking key '{k}'")
            if k in row and row.get(k) is not None:
                print(f"DEBUG: Found key '{k}' with value: {row.get(k)}")
                try:
                    result = float(row.get(k))
                    print(f"DEBUG: Successfully converted to float: {result}")
                    return result
                except Exception as e:
                    print(f"DEBUG: Failed to convert to float: {e}")
                    return None
            else:
                print(f"DEBUG: Key '{k}' not found or is None")
        print("DEBUG: No keys found")
        return None

    def _ind_list(ind_any):
        if isinstance(ind_any, list):
            return ind_any
        if isinstance(ind_any, dict):
            # nếu là 1 dict single snapshot -> bọc lại
            return [ind_any]
        return []

    # Simulate comprehensive_aggregator loading - check H1 first
    try:
        indicator_file = f"indicator_output/{symbol}._H1_indicators.json"
        print(f"Comprehensive aggregator checking: {indicator_file}")
        
        if not os.path.exists(indicator_file):
            print(f"File does not exist: {indicator_file}")
            return 0.0
            
        with open(indicator_file, 'r') as f:
            raw_indicators = json.load(f)
        
        # Mock TimeframeData
        class MockTimeframeData:
            def __init__(self, indicators):
                self.indicators = indicators
                
        d = MockTimeframeData(raw_indicators)
        ind_list = _ind_list(d.indicators)
        
        print(f"ind_list length: {len(ind_list) if isinstance(ind_list, list) else 'N/A'}")
        
        atr = _last_num(ind_list, ["ATR14", "ATR", "atr14", "ATR_14"])
        
        if atr:
            print(f"✅ Comprehensive aggregator found ATR: {atr}")
            return float(atr)
        else:
            print("❌ Comprehensive aggregator could not find ATR")
            return 0.0
            
    except Exception as e:
        print(f"Error in comprehensive aggregator logic: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# Test both methods
print("="*60)
print("TESTING ATR EXTRACTION METHODS FOR USDJPY")
print("="*60)

print("\n1. ORDER_EXECUTOR.PY METHOD:")
print("-" * 40)
oe_atr = order_executor_atr_logic("USDJPY")

print("\n2. COMPREHENSIVE_AGGREGATOR.PY METHOD:")
print("-" * 40) 
ca_atr = comprehensive_aggregator_atr_logic("USDJPY")

print("\n" + "="*60)
print("RESULTS COMPARISON:")
print(f"Order Executor ATR:        {oe_atr}")
print(f"Comprehensive Aggregator ATR: {ca_atr}")
print(f"Match:                     {oe_atr == ca_atr}")
print("="*60)