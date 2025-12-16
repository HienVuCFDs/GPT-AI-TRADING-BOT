#!/usr/bin/env python3
"""Debug whitelist loading and normalization"""

import json
import os

# Test 1: Load whitelist file
print("=" * 60)
print("TEST 1: Load whitelist file")
print("=" * 60)

gui_wl_path = os.path.join('analysis_results', 'indicator_whitelist.json')
print(f"Looking for: {gui_wl_path}")
print(f"Exists: {os.path.exists(gui_wl_path)}")

if os.path.exists(gui_wl_path):
    with open(gui_wl_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    print(f"Raw whitelist: {raw}")
    print(f"Type: {type(raw)}")
    
    if isinstance(raw, list):
        print(f"Items: {len(raw)}")
        for i, item in enumerate(raw):
            print(f"  {i}: '{item}' (type: {type(item).__name__})")
        
        # Test 2: Normalize
        print("\n" + "=" * 60)
        print("TEST 2: Normalization")
        print("=" * 60)
        
        norm_map = {
            'rsi':'rsi','rsi14':'rsi',
            'macd':'macd',
            'adx':'adx','adx14':'adx',
            'stochrsi':'stochrsi','stochastic rsi':'stochrsi','stochastic_rsi':'stochrsi',
            'stochastic':'stochastic',
            'atr':'atr','atr14':'atr',
            'donchian':'donchian','donchian20':'donchian',
            'ema20':'ema20','ema_20':'ema20','ema50':'ema50','ema_50':'ema50','ema100':'ema100','ema200':'ema200',
            'sma20':'sma20','wma20':'wma20',
            'bollinger':'bollinger','bollinger bands':'bollinger',
            'keltner':'keltner','keltner channel':'keltner',
            'ichimoku':'ichimoku','ichimoku cloud':'ichimoku',
            'cci':'cci','commodity channel index':'cci',
            'williamsr':'williamsr','williams %r':'williamsr','williams % r':'williamsr','williams%r':'williamsr',
            'roc':'roc','rate of change':'roc',
            'obv':'obv','on balance volume':'obv',
            'chaikin':'chaikin','chaikin osc':'chaikin',
            'eom':'eom','ease of movement':'eom',
            'mfi':'mfi','money flow index':'mfi','mfi14':'mfi',
            'force':'force','force index':'force',
            'trix':'trix','dpo':'dpo','mass':'mass','mass index':'mass',
            'vortex':'vortex','kst':'kst','ultimate':'ultimate','ultimate osc':'ultimate',
            'envelopes':'envelopes','momentum':'momentum','momentum/cycle':'momentum',
            'fibonacci':'fibonacci',
            'psar':'psar',
            'candlestick':'candlestick',
            'pattern':'patterns','patterns':'patterns','price patterns':'patterns','price pattern':'patterns','price_patterns':'patterns'  # <-- KEY LINE
        }
        
        cleaned = []
        import re as _re
        for item in raw:
            if not isinstance(item, str):
                print(f"  Skipping non-string: {item}")
                continue
            key = item.strip().lower()
            print(f"  Processing: '{item}' -> lower: '{key}'", end="")
            key = norm_map.get(key, key)
            print(f" -> normalized: '{key}'")
            
            # Keep supported tokens
            if key in {
                'rsi','macd','adx','stochrsi','stochastic','atr','donchian',
                'bollinger','keltner','ichimoku','cci','williamsr','roc','obv','chaikin','eom','mfi','force','trix','dpo','mass','vortex','kst','ultimate','envelopes','momentum','psar','fibonacci','candlestick','patterns'
            }:
                if key not in cleaned:
                    cleaned.append(key)
                    print(f"    ✅ Added '{key}'")
                else:
                    print(f"    ⏭️  Already in list")
                continue
            
            if _re.fullmatch(r'(ema|sma|wma|tema)\d{1,3}', key or ''):
                if key not in cleaned:
                    cleaned.append(key)
                    print(f"    ✅ Added MA: '{key}'")
                else:
                    print(f"    ⏭️  MA already in list")
        
        print(f"\nFinal cleaned whitelist: {cleaned}")
        print(f"Count: {len(cleaned)}")
        
        # Test 3: Check if 'patterns' is in the list
        print("\n" + "=" * 60)
        print("TEST 3: Pattern check")
        print("=" * 60)
        
        wl_normalized = [str(x).lower() for x in cleaned]
        print(f"Normalized for checking: {wl_normalized}")
        
        test_tokens = ['patterns', 'pattern', 'price_pattern', 'price_patterns', 'price patterns']
        print(f"Testing tokens: {test_tokens}")
        
        for tok in test_tokens:
            found = tok in wl_normalized
            print(f"  '{tok}' in wl_normalized: {found}")
        
        patterns_enabled = any(tok in wl_normalized for tok in test_tokens)
        print(f"\n✓ patterns_enabled = {patterns_enabled}")
