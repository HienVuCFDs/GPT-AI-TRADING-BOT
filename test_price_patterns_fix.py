#!/usr/bin/env python3
"""Test that price patterns are now showing in reports after the fix"""

import json
import subprocess
import time
from pathlib import Path

# Run comprehensive_aggregator to generate new reports
print("=" * 60)
print("Running comprehensive_aggregator to generate new reports...")
print("=" * 60)

cmd = [
    "python",
    "comprehensive_aggregator.py",
    "--symbols", "BNBUSD_m",
    "--language", "vi"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print("STDOUT:", result.stdout[:1000] if result.stdout else "(empty)")
    print("STDERR:", result.stderr[:1000] if result.stderr else "(empty)")
    print("Return code:", result.returncode)
except subprocess.TimeoutExpired:
    print("❌ Command timed out")
except Exception as e:
    print(f"❌ Error: {e}")

# Now check if price patterns appear in the generated report
print("\n" + "=" * 60)
print("Checking generated reports...")
print("=" * 60)

report_dir = Path("analysis_results")
if not report_dir.exists():
    print("❌ analysis_results directory not found")
else:
    # Find the latest BNBUSD report files
    pattern_files = list(report_dir.glob("BNBUSD*report_vi_*.txt"))
    if not pattern_files:
        print("❌ No BNBUSD reports found")
    else:
        # Get the most recent one
        latest = max(pattern_files, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found latest report: {latest.name}")
        
        with open(latest, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for price patterns section
        has_price_patterns = "Mô hình giá:" in content
        has_candlestick_patterns = "Mô hình nến:" in content
        
        print(f"\n✓ Price patterns section ('Mô hình giá:'): {has_price_patterns}")
        print(f"✓ Candlestick patterns section ('Mô hình nến:'): {has_candlestick_patterns}")
        
        # Show relevant section
        if has_price_patterns:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "Mô hình giá:" in line:
                    print("\n" + "=" * 60)
                    print("PRICE PATTERNS SECTION:")
                    print("=" * 60)
                    # Print the price patterns section (next 6 lines)
                    for j in range(i, min(i+8, len(lines))):
                        print(lines[j])
                    break
        else:
            print("\n⚠️ Price patterns section not found in report!")
            print("Sample of report content:")
            print(content[:2000])
