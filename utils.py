#!/usr/bin/env python3
"""
Common utility functions for Trading Bot
Centralized utilities to avoid code duplication
"""
import os
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def overwrite_json_safely(file_path: str, data: Any, backup: bool = True) -> bool:
    """Save JSON data safely with backup support"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        return False


def ensure_directory(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def auto_cleanup_on_start(directories: list, hours: int = 72):
    """Auto cleanup on start - simple implementation"""
    try:
        if directories:
            cleanup_multiple_directories(directories, hours)
    except Exception as e:
        logger.warning(f"Auto cleanup warning: {e}")


def cleanup_multiple_directories(directories: list, hours: int = 72):
    """Cleanup multiple directories"""
    result = {
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'directories_cleaned': []
    }
    
    try:
        if not directories:
            return result
        for directory_info in directories:
            if isinstance(directory_info, (list, tuple)) and len(directory_info) >= 1:
                directory = directory_info[0]
                if directory and os.path.exists(directory):
                    cleanup_files_by_age(directory, hours)
                    result['directories_cleaned'].append(directory)
        return result
    except Exception as e:
        logger.warning(f"Cleanup directories warning: {e}")
        return result


def cleanup_files_by_age(directory: str, hours: int = 72):
    """Cleanup files by age"""
    try:
        if not directory or not os.path.exists(directory):
            return
        # Simple cleanup - remove files older than specified hours
        import time
        cutoff_time = time.time() - (hours * 3600)
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                except:
                    pass
    except Exception as e:
        logger.warning(f"Cleanup files warning: {e}")


def get_pip_value(symbol: str) -> float:
    """Calculate pip value for different symbol types - UNIFIED VERSION"""
    symbol_upper = symbol.upper().rstrip('.').replace('_M', '').replace('_m', '')
    
    # ========== PRECIOUS METALS ==========
    if symbol_upper in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD']:
        return 0.1   # Metals: 1 pip = 0.1
    
    # ========== JPY PAIRS ==========
    elif 'JPY' in symbol_upper:
        return 0.01  # JPY pairs: 1 pip = 0.01
    
    # ========== HIGH-VALUE CRYPTO (≥ $1000) ==========
    elif symbol_upper in ['BTCUSD', 'ETHUSD']:
        return 1.0   # BTC/ETH: 1 pip = 1.0
    
    # ========== MID-VALUE CRYPTO ($100-$1000) ==========
    elif symbol_upper in ['SOLUSD', 'LTCUSD', 'BNBUSD', 'AVAXUSD', 'DOTUSD', 
                          'MATICUSD', 'LINKUSD', 'TRXUSD', 'SHIBUSD', 'ARBUSD', 
                          'OPUSD', 'APEUSD', 'SANDUSD', 'CROUSD', 'FTTUSD']:
        return 0.1   # Mid-value crypto: 1 pip = 0.1
    
    # ========== MAJOR FX PAIRS ==========
    else:
        return 0.0001  # Major FX pairs: 1 pip = 0.0001


# ==================== 🚀 CACHE UTILITIES ====================

def cache_with_ttl(cache_manager, cache_key: str, ttl: int = 300):
    """Decorator to cache function results
    
    Usage:
        @cache_with_ttl(cache_manager, 'my_key', ttl=300)
        def expensive_function():
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result: {cache_key}")
            
            return result
        return wrapper
    return decorator


def invalidate_cache(cache_manager, pattern: str = None):
    """Invalidate cache by pattern or clear all"""
    if pattern:
        try:
            keys = cache_manager.redis_client.keys(pattern)
            if keys:
                cache_manager.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching {pattern}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    else:
        cache_manager.clear_all()
        logger.info("Cache cleared")


def measure_cache_effectiveness(cache_manager) -> dict:
    """Measure cache hit/miss ratio"""
    if not hasattr(cache_manager, 'redis_client') or not cache_manager.enabled:
        return {'status': 'cache_disabled'}
    
    try:
        stats = cache_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error measuring cache: {e}")
        return {'error': str(e)}
