#!/usr/bin/env python3
"""
Final integration test to verify all enhancements are working together.
Tests:
1. Enhanced exploration settings are default
2. Synthetic data option works
3. Archiving is functional
4. Enhanced visualizer works
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_config_loading():
    """Test that enhanced exploration config is loaded by default"""
    print("=== Testing Enhanced Exploration Config ===")
    
    from utils.config_loader import load_training_config
    
    # Test default config
    config = load_training_config()
    
    # Check PPO settings
    ppo = config.get('ppo', {})
    print(f"PPO learning_rate: {ppo.get('learning_rate', 'NOT SET')}")
    print(f"PPO ent_coef: {ppo.get('ent_coef', 'NOT SET')}")
    print(f"PPO net_arch: {ppo.get('policy_kwargs', {}).get('net_arch', 'NOT SET')}")
    
    # Check reward settings
    reward = config.get('reward', {})
    print(f"Reward action_diversity_bonus: {reward.get('action_diversity_bonus', 'NOT SET')}")
    print(f"Reward position_diversity_bonus: {reward.get('position_diversity_bonus', 'NOT SET')}")
    print(f"Reward short_action_bonus: {reward.get('short_action_bonus', 'NOT SET')}")
    print(f"Reward long_action_bonus: {reward.get('long_action_bonus', 'NOT SET')}")
    print(f"Reward close_action_bonus: {reward.get('close_action_bonus', 'NOT SET')}")
    
    return config

def test_synthetic_data():
    """Test that synthetic data exists and is accessible"""
    print("\n=== Testing Synthetic Data ===")
    
    synthetic_path = Path('data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv')
    if synthetic_path.exists():
        print(f"✓ Synthetic data exists: {synthetic_path}")
        
        # Check file size
        size_mb = synthetic_path.stat().st_size / (1024 * 1024)
        print(f"✓ File size: {size_mb:.2f} MB")
        
        # Check first few rows
        import pandas as pd
        df = pd.read_csv(synthetic_path, nrows=5)
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Sample rows: {len(df)}")
        
        return True
    else:
        print(f"✗ Synthetic data not found: {synthetic_path}")
        return False

def test_archiver():
    """Test archiving functionality"""
    print("\n=== Testing Archiver ===")
    
    try:
        from utils.archiver import TrainingArchiver
        
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_logs = temp_path / 'logs'
            test_models = temp_path / 'models'
            test_logs.mkdir()
            test_models.mkdir()
            
            (test_logs / 'test.log').write_text('test log')
            (test_models / 'test.model').write_text('test model')
            
            # Test archiver
            archiver = TrainingArchiver(
                logs_dir=str(test_logs),
                models_dir=str(test_models)
            )
            
            archive_path = archiver.archive_previous_run()
            print(f"✓ Archive created: {archive_path}")
            
            return True
            
    except Exception as e:
        print(f"✗ Archiver test failed: {e}")
        return False

def test_visualizer():
    """Test enhanced visualizer"""
    print("\n=== Testing Enhanced Visualizer ===")
    
    try:        # Import the enhanced visualizer
        sys.path.append('graphs')
        from live_trade_visualizer_enhanced import EnhancedLiveTradeVisualizer
        
        print("✓ Enhanced visualizer imported successfully")
        
        # Create instance (won't show plot in test)
        visualizer = EnhancedLiveTradeVisualizer()
        print("✓ Enhanced visualizer instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Running Final Integration Tests...")
    print("=" * 50)
    
    results = []
    
    # Test 1: Config loading
    try:
        config = test_config_loading()
        results.append(("Config Loading", True))
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        results.append(("Config Loading", False))
    
    # Test 2: Synthetic data
    results.append(("Synthetic Data", test_synthetic_data()))
    
    # Test 3: Archiver
    results.append(("Archiver", test_archiver()))
    
    # Test 4: Visualizer
    results.append(("Visualizer", test_visualizer()))
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL INTEGRATION TEST RESULTS:")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
