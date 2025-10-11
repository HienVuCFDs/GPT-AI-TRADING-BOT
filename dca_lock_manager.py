#!/usr/bin/env python3
"""
DCA Lock Manager - Simple file-based locking system for DCA operations
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DCALockManager:
    """Simple file-based DCA lock manager"""
    
    def __init__(self, lock_dir: str = "dca_locks"):
        self.lock_dir = lock_dir
        self.locks_path = os.path.join(os.getcwd(), lock_dir)
        os.makedirs(self.locks_path, exist_ok=True)
        logger.info(f"🔒 DCA Lock Manager initialized")
    
    def _get_lock_file(self, symbol: str) -> str:
        """Get lock file path for symbol"""
        return os.path.join(self.locks_path, f"{symbol}_dca.lock")
    
    def acquire_lock(self, symbol: str, timeout: float = 3.0) -> bool:
        """Acquire DCA lock for symbol"""
        lock_file = self._get_lock_file(symbol)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if os.path.exists(lock_file):
                    # Check if lock is stale (>60 seconds)
                    try:
                        with open(lock_file, 'r') as f:
                            lock_data = json.load(f)
                        lock_age = time.time() - lock_data.get('timestamp', 0)
                        if lock_age > 60:
                            os.remove(lock_file)
                        else:
                            time.sleep(0.1)
                            continue
                    except:
                        os.remove(lock_file)
                
                # Create new lock
                lock_data = {
                    'symbol': symbol,
                    'timestamp': time.time(),
                    'process_id': os.getpid()
                }
                
                with open(lock_file, 'w') as f:
                    json.dump(lock_data, f)
                
                return True
                
            except Exception as e:
                logger.error(f"🔒 Lock error for {symbol}: {e}")
                time.sleep(0.1)
        
        return False
    
    def release_lock(self, symbol: str) -> bool:
        """Release DCA lock for symbol"""
        lock_file = self._get_lock_file(symbol)
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
            return True
        except Exception as e:
            logger.error(f"🔒 Release error for {symbol}: {e}")
            return False
    
    def cleanup_stale_locks(self, max_age_seconds: float = 300) -> int:
        """Clean up stale lock files"""
        if not os.path.exists(self.locks_path):
            return 0
        
        cleaned_count = 0
        current_time = time.time()
        
        try:
            for filename in os.listdir(self.locks_path):
                if filename.endswith('.lock'):
                    lock_file = os.path.join(self.locks_path, filename)
                    try:
                        file_mtime = os.path.getmtime(lock_file)
                        if current_time - file_mtime > max_age_seconds:
                            os.remove(lock_file)
                            cleaned_count += 1
                    except:
                        pass
        except:
            pass
        
        return cleaned_count