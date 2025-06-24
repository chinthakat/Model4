#!/usr/bin/env python3
"""
Test script to verify the --training option works correctly
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import argparse
from utils.config_loader import load_training_config

def test_training_option():
    """Test the --training option logic"""
    parser = argparse.ArgumentParser(description="Test Training Option")
    parser.add_argument("--config", type=str, default="config/config.json")
    parser.add_argument("--training", action="store_true")
    
    # Simulate the --training flag
    args = argparse.Namespace(config="config/config.json", training=True)
    
    print("🧪 Testing --training option...")
    
    # Load configuration
    config = load_training_config(args.config)
    print(f"📁 Original consolidated_file: {config['consolidated_file']}")
    
    # Handle --training flag to use synthetic balanced data
    if args.training:
        synthetic_data_path = "data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv"
        if Path(synthetic_data_path).exists():
            original_file = config['consolidated_file']
            config['consolidated_file'] = synthetic_data_path
            print("🎯 TRAINING MODE: Using synthetic balanced training data")
            print(f"   📁 Original data: {original_file}")
            print(f"   🔄 Training data: {synthetic_data_path}")
            print("   📈 This data contains balanced LONG, SHORT, and HOLD patterns")
            print("   🎯 Perfect for teaching your RL agent all three action types!")
            print(f"\n✅ Configuration updated successfully!")
            print(f"📊 New consolidated_file: {config['consolidated_file']}")
        else:
            print(f"❌ Warning: Synthetic training data not found at {synthetic_data_path}")
            print("   Please run generate_simple_synthetic_data.py first!")
            print("   Falling back to original data...")
    
    return config

if __name__ == "__main__":
    test_training_option()
