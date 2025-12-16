#!/usr/bin/env python3
"""Debug price pattern loading for BNBUSD"""

import json
import os

sym = "BNBUSD_m"
timeframes = ["H1", "M30", "M15"]

print("=" * 60)
print("DEBUG: Price Pattern Loading for BNBUSD")
print("=" * 60)

for tf in timeframes:
    print(f"\nChecking {sym} {tf}:")
    
    # Try pattern_price folder
    pattern_files = [
        f"pattern_price/{sym}_{tf}_patterns.json",
        f"pattern_price/{sym}_{tf}_best.json",
        f"pattern_price/{sym}_m_{tf}_patterns.json",
    ]
    
    print(f"  Pattern files to try:")
    for pf in pattern_files:
        exists = os.path.exists(pf)
        print(f"    {pf}: {exists}")
        if exists:
            try:
                with open(pf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"      ✓ Loaded, type={type(data).__name__}, len={len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"      First item: {data[0]}")
            except Exception as e:
                print(f"      ✗ Error loading: {e}")
    
    # Try pattern_signals folder
    signal_files = [
        f"pattern_signals/{sym}_{tf}_priority_patterns.json",
        f"pattern_signals/{sym}_{tf}_patterns.json",
    ]
    
    print(f"  Signal files to try:")
    for sf in signal_files:
        exists = os.path.exists(sf)
        print(f"    {sf}: {exists}")
        if exists:
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"      ✓ Loaded, type={type(data).__name__}, len={len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"      First item keys: {list(data[0].keys())}")
            except Exception as e:
                print(f"      ✗ Error loading: {e}")
