"""
🤖 AI Training Main Script
===========================
Run this script to collect data and train the model.

Usage:
    python train_ai_model.py              # Full pipeline: collect + train
    python train_ai_model.py --collect    # Only collect data
    python train_ai_model.py --train      # Only train (use existing data)
    python train_ai_model.py --status     # Show status
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def show_status():
    """Show current training status"""
    from ai_models.training.utils import print_status
    print_status()


def collect_data(days_back: int = 30):
    """Collect training data from all sources"""
    from ai_models.training import get_collector
    from ai_models.training.utils import migrate_from_old_structure
    
    print("\n" + "=" * 60)
    print("📥 COLLECTING TRAINING DATA")
    print("=" * 60)
    
    # Migrate old data first
    print("\n📦 Checking for old data to migrate...")
    stats = migrate_from_old_structure()
    if stats['pending_files_moved'] > 0:
        print(f"   Migrated {stats['pending_files_moved']} files from ai_training_pending/")
    
    collector = get_collector()
    
    # Collect from analysis_results
    print(f"\n📊 Collecting signals from analysis_results/ (last {days_back} days)...")
    signals = collector.collect_from_analysis_results(days_back=days_back)
    print(f"   Found {signals} new signals")
    
    # Collect from MT5
    print(f"\n📈 Collecting trade results from MT5 (last {days_back} days)...")
    trades = collector.collect_from_mt5(days_back=days_back)
    print(f"   Found {trades} trade results")
    
    # Export for training
    print("\n💾 Exporting training examples...")
    output = collector.export_for_training()
    print(f"   Saved to: {output}")
    
    # Show stats
    stats = collector.get_stats()
    print("\n📊 Collection Summary:")
    print(f"   Total signals:      {stats['total_signals']}")
    print(f"   Trade results:      {stats['total_trade_results']}")
    print(f"   Training examples:  {stats['training_examples']}")
    print(f"   With outcomes:      {stats['with_outcome']}")
    print(f"   Win rate:           {stats['win_rate']:.1f}%")
    
    return stats


def train_model(epochs: int = 50):
    """Train the model"""
    from ai_models.training import train_model as do_train
    
    print("\n" + "=" * 60)
    print("🏋️ TRAINING AI MODEL")
    print("=" * 60)
    
    history = do_train(epochs=epochs)
    
    if 'error' in history:
        print(f"\n❌ Training failed: {history['error']}")
        return False
    
    print("\n✅ Training completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='AI Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ai_model.py              # Full pipeline
    python train_ai_model.py --collect    # Only collect data
    python train_ai_model.py --train 100  # Train for 100 epochs
    python train_ai_model.py --status     # Show status
    python train_ai_model.py --days 60    # Collect last 60 days
        """
    )
    
    parser.add_argument('--status', action='store_true', 
                        help='Show training status')
    parser.add_argument('--collect', action='store_true',
                        help='Only collect data (no training)')
    parser.add_argument('--train', type=int, nargs='?', const=50, metavar='EPOCHS',
                        help='Only train (use existing data). Default: 50 epochs')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of historical data to collect. Default: 30')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs. Default: 50')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🤖 AI TRAINING PIPELINE")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Status only
    if args.status:
        show_status()
        return
    
    # Train only
    if args.train is not None:
        train_model(epochs=args.train)
        return
    
    # Collect only
    if args.collect:
        collect_data(days_back=args.days)
        return
    
    # Full pipeline: collect + train
    stats = collect_data(days_back=args.days)
    
    # Check if enough data for training
    if stats['with_outcome'] < 10:
        print("\n⚠️ Not enough training examples with outcomes.")
        print("   Need at least 10 examples to train.")
        print("   Keep trading and collecting more data!")
        return
    
    train_model(epochs=args.epochs)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
