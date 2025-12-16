"""
Quick training script - no interactive input needed
"""
import sys
import os
import logging

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .deep_learning_model import train_model

if __name__ == "__main__":
    print("🚀 Starting Deep Learning Training...")
    model = train_model(epochs=100)
    print("✅ Training completed!")
