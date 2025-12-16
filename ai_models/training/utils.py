"""
🔧 Utility functions for ai_training/
======================================
- Sync pending data
- Cleanup old files
- Migration from old structure
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
PENDING_DIR = Path(__file__).parent / "pending"
DATA_DIR = Path(__file__).parent / "data"
OLD_PENDING_DIR = BASE_DIR / "ai_training_pending"


def sync_pending_data() -> int:
    """
    Process pending training data files.
    Moves them into the main training database.
    
    Returns:
        Number of files processed
    """
    from .data_collector import get_collector
    
    collector = get_collector()
    processed = 0
    
    # First migrate old location if exists
    if OLD_PENDING_DIR.exists():
        collector.migrate_old_pending()
    
    # Process pending files
    if not PENDING_DIR.exists():
        return 0
    
    for pending_file in PENDING_DIR.glob("*.json"):
        try:
            import json
            with open(pending_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add to collector
            if collector._add_signal(data):
                processed += 1
            
            # Remove processed file
            pending_file.unlink()
            
        except Exception as e:
            logger.warning(f"Error processing {pending_file}: {e}")
    
    if processed > 0:
        # Save updated database
        collector._save_json(collector.db_file, collector.signals)
        logger.info(f"✅ Processed {processed} pending files")
    
    return processed


def cleanup_old_files(days_old: int = 30) -> int:
    """
    Clean up old files from various directories.
    
    Args:
        days_old: Delete files older than this many days
        
    Returns:
        Number of files deleted
    """
    cutoff = datetime.now() - timedelta(days=days_old)
    deleted = 0
    
    # Directories to clean
    dirs_to_clean = [
        PENDING_DIR,
        OLD_PENDING_DIR,
        DATA_DIR / "backups"
    ]
    
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
        
        for file_path in dir_path.glob("*"):
            if not file_path.is_file():
                continue
            
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff:
                    file_path.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
    
    if deleted > 0:
        logger.info(f"🧹 Cleaned up {deleted} old files")
    
    return deleted


def migrate_from_old_structure() -> Dict:
    """
    Migrate data from old training structure to new ai_training/
    
    Old structure:
    - ai_training_pending/
    - ai_training_sender.py
    - ai_training_data_schema.py
    - sync_pending_data.py
    - sync_pending_data_v2.py
    - sync_training_data.py
    
    Returns:
        Dict with migration stats
    """
    stats = {
        'pending_files_moved': 0,
        'errors': []
    }
    
    # Ensure new directories exist
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Move pending files
    if OLD_PENDING_DIR.exists():
        for old_file in OLD_PENDING_DIR.glob("*.json"):
            try:
                new_path = PENDING_DIR / old_file.name
                if not new_path.exists():
                    shutil.move(str(old_file), str(new_path))
                    stats['pending_files_moved'] += 1
                else:
                    old_file.unlink()  # Duplicate, remove
            except Exception as e:
                stats['errors'].append(f"{old_file.name}: {e}")
        
        # Remove old directory if empty
        try:
            if not any(OLD_PENDING_DIR.iterdir()):
                OLD_PENDING_DIR.rmdir()
        except:
            pass
    
    logger.info(f"📦 Migration complete: {stats['pending_files_moved']} files moved")
    
    return stats


def get_training_status() -> Dict:
    """
    Get current status of training system.
    
    Returns:
        Dict with status information
    """
    from .data_collector import get_collector
    
    collector = get_collector()
    stats = collector.get_stats()
    
    # Add file system info
    stats['data_dir'] = str(DATA_DIR)
    stats['pending_dir'] = str(PENDING_DIR)
    stats['data_dir_exists'] = DATA_DIR.exists()
    stats['pending_dir_exists'] = PENDING_DIR.exists()
    
    # Check model
    model_path = BASE_DIR / "ai_models" / "saved" / "model_latest.pt"
    stats['model_exists'] = model_path.exists()
    
    if model_path.exists():
        stats['model_modified'] = datetime.fromtimestamp(
            model_path.stat().st_mtime
        ).isoformat()
    
    return stats


def print_status():
    """Print training system status"""
    status = get_training_status()
    
    print("\n" + "=" * 50)
    print("📊 AI TRAINING STATUS")
    print("=" * 50)
    print(f"  Signals collected:   {status.get('total_signals', 0)}")
    print(f"  Trade results:       {status.get('total_trade_results', 0)}")
    print(f"  Training examples:   {status.get('training_examples', 0)}")
    print(f"  With outcomes:       {status.get('with_outcome', 0)}")
    print(f"  Win rate:            {status.get('win_rate', 0):.1f}%")
    print(f"  Pending files:       {status.get('pending_files', 0)}")
    print(f"  Model exists:        {'✅' if status.get('model_exists') else '❌'}")
    if status.get('model_modified'):
        print(f"  Model updated:       {status.get('model_modified')}")
    print("=" * 50)


# ========================================
# CLI
# ========================================
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='AI Training Utilities')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--sync', action='store_true', help='Sync pending data')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Cleanup files older than N days')
    parser.add_argument('--migrate', action='store_true', help='Migrate from old structure')
    args = parser.parse_args()
    
    if args.status:
        print_status()
    
    if args.sync:
        processed = sync_pending_data()
        print(f"✅ Synced {processed} files")
    
    if args.cleanup:
        deleted = cleanup_old_files(args.cleanup)
        print(f"🧹 Deleted {deleted} old files")
    
    if args.migrate:
        stats = migrate_from_old_structure()
        print(f"📦 Migrated {stats['pending_files_moved']} files")
        if stats['errors']:
            print(f"⚠️ Errors: {len(stats['errors'])}")
    
    # Default: show status
    if not any([args.status, args.sync, args.cleanup, args.migrate]):
        print_status()
