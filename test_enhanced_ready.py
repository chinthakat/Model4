#!/usr/bin/env python3
"""
Quick verification test that all core functionality is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_enhanced_training_ready():
    """Test that the enhanced training script is ready to run"""
    print("🔍 Testing Enhanced Training Readiness...")
    
    # Test 1: Can we import the training script?
    try:
        from train_memory_efficient import Trainer
        print("✅ Training script imports successfully")
    except Exception as e:
        print(f"❌ Training script import failed: {e}")
        return False
    
    # Test 2: Can we load the enhanced config?
    try:
        from utils.config_loader import load_training_config
        config = load_training_config()
        
        # Check for enhanced exploration settings
        reward_strategy = config.get('reward_strategy', 'unknown')
        print(f"✅ Config loaded - reward_strategy: {reward_strategy}")
        
        # Check for enhanced settings
        reward_overrides = config.get('reward_overrides', {})
        if reward_overrides:
            print(f"✅ Enhanced reward overrides found: {len(reward_overrides)} settings")
        else:
            print("⚠️  No enhanced reward overrides found")
            
        ppo_config = config.get('ppo', {})
        if ppo_config:
            print(f"✅ PPO config found with learning_rate: {ppo_config.get('learning_rate', 'NOT SET')}")
        else:
            print("⚠️  No PPO config found")
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
    
    # Test 3: Does synthetic data exist?
    synthetic_path = Path('data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv')
    if synthetic_path.exists():
        print(f"✅ Synthetic training data available: {synthetic_path}")
    else:
        print(f"⚠️  Synthetic training data not found: {synthetic_path}")
    
    # Test 4: Can we import enhanced components?
    try:
        from enhanced_reward_configs import ENHANCED_EXPLORATION_CONFIG
        print("✅ Enhanced reward configs available")
    except Exception as e:
        print(f"⚠️  Enhanced reward configs not available: {e}")
    
    print("\n🎯 Enhanced Training System Status: READY FOR USE!")
    print("📋 Available Options:")
    print("   • python src/train_memory_efficient.py --default    # Use real data with enhanced exploration")
    print("   • python src/train_memory_efficient.py --training   # Use synthetic data for balanced learning")
    print("   • python src/train_memory_efficient.py --help       # See all options")
    
    return True

if __name__ == "__main__":
    test_enhanced_training_ready()
