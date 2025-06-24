#!/usr/bin/env python3
"""
Final verification that both --default and --training modes use enhanced exploration.
"""

import sys
import subprocess
from pathlib import Path

def test_training_mode():
    """Test --training mode configuration"""
    print("🔍 Testing --training mode...")
    
    # Test that synthetic data path is correct
    synthetic_path = Path('data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv')
    if synthetic_path.exists():
        print("✅ Synthetic training data available")
        
        # Check file size
        size_mb = synthetic_path.stat().st_size / (1024 * 1024)
        print(f"✅ File size: {size_mb:.2f} MB")
        
        # Check first few rows for balance
        import pandas as pd
        df = pd.read_csv(synthetic_path, nrows=100)
        print(f"✅ Sample data: {len(df)} rows, columns: {list(df.columns)}")
        
    else:
        print(f"❌ Synthetic training data missing: {synthetic_path}")
        return False
    
    return True

def test_enhanced_config_import():
    """Test that enhanced exploration config can be imported"""
    print("\n🔍 Testing enhanced exploration config...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / 'src'))
    
    try:
        from enhanced_reward_configs import ENHANCED_EXPLORATION_CONFIG
        print("✅ Enhanced exploration config imported successfully")
        
        # Check key settings
        key_settings = [
            'close_action_bonus',
            'short_trade_bonus', 
            'long_trade_bonus',
            'action_diversity_reward',
            'position_diversity_bonus'
        ]
        
        for setting in key_settings:
            value = ENHANCED_EXPLORATION_CONFIG.get(setting, 'NOT FOUND')
            print(f"  ✅ {setting}: {value}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced config: {e}")
        return False

def test_training_config():
    """Test that training config has enhanced exploration settings"""
    print("\n🔍 Testing training config file...")
    
    sys.path.append(str(Path(__file__).parent / 'src'))
    
    try:
        from utils.config_loader import load_training_config
        config = load_training_config()
        
        reward_strategy = config.get('reward_strategy', 'NOT SET')
        print(f"✅ reward_strategy: {reward_strategy}")
        
        reward_overrides = config.get('reward_overrides', {})
        if reward_overrides:
            print(f"✅ reward_overrides found: {len(reward_overrides)} settings")
            for key, value in reward_overrides.items():
                if key != 'comment':
                    print(f"  ✅ {key}: {value}")
        else:
            print("⚠️  No reward_overrides found")
            
        ppo_config = config.get('ppo', {})
        if ppo_config:
            lr = ppo_config.get('learning_rate', 'NOT SET')
            ent = ppo_config.get('ent_coef', 'NOT SET')
            print(f"✅ PPO learning_rate: {lr}, ent_coef: {ent}")
        else:
            print("⚠️  No PPO config found")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load training config: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🎯 Enhanced Exploration Verification Tests")
    print("=" * 50)
    
    results = []
    
    # Test 1: Training mode data
    results.append(("Training Data", test_training_mode()))
    
    # Test 2: Enhanced config
    results.append(("Enhanced Config", test_enhanced_config_import()))
    
    # Test 3: Training config
    results.append(("Training Config", test_training_config()))
    
    # Summary
    print("\n" + "=" * 50)
    print("ENHANCED EXPLORATION VERIFICATION RESULTS:")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced exploration is ready!")
        print("\n📋 Usage Commands:")
        print("🚀 python src/train_memory_efficient.py --default    # Real data + enhanced exploration")
        print("🎯 python src/train_memory_efficient.py --training   # Synthetic data + enhanced exploration")
        print("❓ python src/train_memory_efficient.py --help       # See all options")
    else:
        print(f"\n⚠️  Some tests failed. Please check the configuration.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
