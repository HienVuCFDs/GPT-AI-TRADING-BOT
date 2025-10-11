#!/usr/bin/env python3
"""
Smart Cleanup Trigger - Call this when user changes symbol selection
"""
import os
import pickle
import shutil
from datetime import datetime
from typing import Set, Dict, Any

def cleanup_indicator_data_for_symbols(active_symbols: Set[str]) -> Dict[str, int]:
    """
    Clean up indicator data that doesn't match active symbols
    
    Args:
        active_symbols: Set of currently active symbols
        
    Returns:
        Dict with cleanup stats
    """
    indicator_dir = "indicator_output"
    if not os.path.exists(indicator_dir):
        return {'files_deleted': 0, 'space_freed_mb': 0.0}
    
    files_deleted = 0
    space_freed = 0
    
    try:
        for filename in os.listdir(indicator_dir):
            if filename.endswith('_indicators.json'):
                # Extract symbol from filename (e.g., GBPUSD_m_H1_indicators.json -> GBPUSD_m)
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = f"{parts[0]}_m"  # Format: GBPUSD_m
                    
                    if symbol not in active_symbols:
                        file_path = os.path.join(indicator_dir, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            files_deleted += 1
                            space_freed += file_size
                        except Exception as e:
                            print(f"⚠️ Could not delete {filename}: {e}")
        
        space_freed_mb = space_freed / (1024 * 1024)
        
        if files_deleted > 0:
            print(f"🧹 Indicator cleanup: Removed {files_deleted} files, freed {space_freed_mb:.2f} MB")
        
        return {'files_deleted': files_deleted, 'space_freed_mb': space_freed_mb}
        
    except Exception as e:
        print(f"⚠️ Error cleaning indicator data: {e}")
        return {'files_deleted': 0, 'space_freed_mb': 0.0}

def cleanup_trendline_data_for_symbols(active_symbols: Set[str]) -> Dict[str, int]:
    """
    Clean up trendline data that doesn't match active symbols
    
    Args:
        active_symbols: Set of currently active symbols
        
    Returns:
        Dict with cleanup stats
    """
    trendline_dir = "trendline_sr"
    if not os.path.exists(trendline_dir):
        return {'files_deleted': 0, 'space_freed_mb': 0.0}
    
    files_deleted = 0
    space_freed = 0
    
    try:
        for filename in os.listdir(trendline_dir):
            if filename.endswith('_trendline_sr.json'):
                # Extract symbol from filename (e.g., GBPUSD_m_H1_trendline_sr.json -> GBPUSD_m)
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = f"{parts[0]}_m"  # Format: GBPUSD_m
                    
                    if symbol not in active_symbols:
                        file_path = os.path.join(trendline_dir, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            files_deleted += 1
                            space_freed += file_size
                        except Exception as e:
                            print(f"⚠️ Could not delete {filename}: {e}")
        
        space_freed_mb = space_freed / (1024 * 1024)
        
        if files_deleted > 0:
            print(f"🧹 Trendline cleanup: Removed {files_deleted} files, freed {space_freed_mb:.2f} MB")
        
        return {'files_deleted': files_deleted, 'space_freed_mb': space_freed_mb}
        
    except Exception as e:
        print(f"⚠️ Error cleaning trendline data: {e}")
        return {'files_deleted': 0, 'space_freed_mb': 0.0}

def trigger_smart_cleanup_on_symbol_change(new_symbols: Set[str]) -> Dict[str, Any]:
    """
    Trigger smart cleanup when user changes symbol selection
    
    Args:
        new_symbols: Set of newly selected symbols
        
    Returns:
        Dict with cleanup results
    """
    print(f"🔄 Symbol selection changed to: {list(new_symbols)}")
    
    # Update config immediately
    try:
        # Load existing config
        config = {}
        try:
            with open('user_config.pkl', 'rb') as f:
                config = pickle.load(f)
        except:
            pass
        
        # Update with new symbols
        config['checked_symbols'] = list(new_symbols)
        
        # Save config
        with open('user_config.pkl', 'wb') as f:
            pickle.dump(config, f)
            
        print(f"✅ Config updated with new symbols")
        
    except Exception as e:
        print(f"⚠️ Error updating config: {e}")
    
    # Force smart cleanup to run immediately
    cleanup_result = {'total_files_deleted': 0, 'total_space_freed_mb': 0.0}
    
    try:
        # Cleanup MT5 data
        from mt5_data_fetcher import EnhancedMT5DataFetcher
        fetcher = EnhancedMT5DataFetcher()
        mt5_result = fetcher.smart_cleanup_mt5_data()
        
        # Cleanup trendline data
        trendline_result = cleanup_trendline_data_for_symbols(new_symbols)
        
        # Cleanup indicator data
        indicator_result = cleanup_indicator_data_for_symbols(new_symbols)
        
        # Combine results
        cleanup_result = {
            'files_deleted': (mt5_result.get('files_deleted', 0) + 
                            trendline_result.get('files_deleted', 0) + 
                            indicator_result.get('files_deleted', 0)),
            'space_freed_mb': (mt5_result.get('space_freed_mb', 0.0) + 
                             trendline_result.get('space_freed_mb', 0.0) + 
                             indicator_result.get('space_freed_mb', 0.0))
        }
        
        # Force cleanup marker to be updated so it won't run again soon
        with open("last_smart_cleanup.txt", 'w') as f:
            f.write(str(datetime.now().timestamp()))
            
        print(f"🧹 Combined cleanup completed:")
        print(f"   MT5: {mt5_result.get('files_deleted', 0)} | Trendline: {trendline_result.get('files_deleted', 0)} | Indicator: {indicator_result.get('files_deleted', 0)} files")
        print(f"   Total space freed: {cleanup_result['space_freed_mb']:.2f} MB")
        
    except Exception as e:
        print(f"⚠️ Smart cleanup error: {e}")
    
    return cleanup_result

def should_auto_save_config() -> bool:
    """Check if config should be auto-saved (to avoid GUI lag)"""
    return True  # For now, always auto-save

if __name__ == "__main__":
    # Test the function
    test_symbols = {'XAUUSD_m', 'EURGBP_m'}
    result = trigger_smart_cleanup_on_symbol_change(test_symbols)
    print(f"✅ Test completed: {result}")
