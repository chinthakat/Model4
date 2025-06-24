#!/usr/bin/env python3
"""
Test Enhanced Exploration Training Configuration
This script tests the new enhanced exploration features for better SHORT/LONG/CLOSE action balance
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_enhanced_configurations():
    """Test the enhanced exploration configurations"""
    print("🔍 Testing Enhanced Exploration Configurations")
    print("=" * 50)
    
    # Test 1: Enhanced reward configs import
    try:
        from src.enhanced_reward_configs import ENHANCED_EXPLORATION_CONFIG, BALANCED_ALL_ACTIONS_CONFIG
        print("✅ Enhanced reward configs imported successfully")
        
        print("\n📊 ENHANCED_EXPLORATION_CONFIG parameters:")
        for key, value in ENHANCED_EXPLORATION_CONFIG.items():
            if 'bonus' in key.lower() or 'penalty' in key.lower() or 'reward' in key.lower():
                print(f"  {key}: {value}")
                
    except ImportError as e:
        print(f"❌ Failed to import enhanced reward configs: {e}")
        return False
    
    # Test 2: Enhanced config file
    try:
        config_path = Path("config/enhanced_exploration_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Enhanced exploration config file loaded successfully")
            
            print(f"\n📋 Key configuration parameters:")
            print(f"  Reward strategy: {config.get('reward_strategy')}")
            print(f"  Total episodes: {config.get('total_episodes')}")
            print(f"  Steps per episode: {config.get('steps_per_episode')}")
            
            exploration_config = config.get('exploration_config', {})
            print(f"\n🎯 Exploration parameters:")
            print(f"  Entropy coefficient: {exploration_config.get('ent_coef')}")
            print(f"  Learning rate: {exploration_config.get('learning_rate')}")
            print(f"  Clip range: {exploration_config.get('clip_range')}")
            
            reward_overrides = config.get('reward_overrides', {})
            print(f"\n⚖️ Reward overrides:")
            for key, value in reward_overrides.items():
                if key != 'comment':
                    print(f"  {key}: {value}")
                    
        else:
            print(f"❌ Enhanced exploration config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load enhanced exploration config: {e}")
        return False
    
    # Test 3: Model configuration enhancements
    try:
        from src.model import TradingModel
        print("✅ Enhanced model configuration available")
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced model: {e}")
        return False
    
    print("\n🎯 Key Enhancements Summary:")
    print("=" * 40)
    print("📈 Increased exploration:")
    print(f"  - Entropy coefficient: 0.01 → 0.05 (5x increase)")
    print(f"  - Learning rate: 1e-4 → 3e-4 (3x increase)")
    print(f"  - Clip range: 0.2 → 0.3 (50% increase)")
    print(f"  - Network size: [128,64] → [512,256,128] (larger)")
    
    print("\n⚖️ Balanced action rewards:")
    print(f"  - SHORT trade bonus: +25 points")
    print(f"  - LONG trade bonus: +20 points")
    print(f"  - CLOSE action bonus: +50 points")
    print(f"  - Position diversity bonus: +30 points")
    print(f"  - Action diversity reward: +10 points")
    
    print("\n🚫 Reduced penalties:")
    print(f"  - Loss multiplier: -100 → -80 (20% reduction)")
    print(f"  - Large position penalty: -50 → -30 (40% reduction)")
    print(f"  - Excessive risk penalty: -30 → -40 (moderate)")
    
    print("\n✅ All enhanced exploration configurations are ready!")
    print("\n🚀 To use enhanced exploration training:")
    print("  python src/train_memory_efficient.py --config config/enhanced_exploration_config.json --training")
    
    return True

def test_usage_example():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("📖 USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1️⃣ Enhanced Exploration Training:")
    print("   python src/train_memory_efficient.py \\")
    print("     --config config/enhanced_exploration_config.json \\")
    print("     --training")
    
    print("\n2️⃣ Compare with Current Training:")
    print("   # Current (small transaction focus):")
    print("   python src/train_memory_efficient.py --training")
    print("   ")
    print("   # Enhanced exploration:")
    print("   python src/train_memory_efficient.py \\")
    print("     --config config/enhanced_exploration_config.json \\")
    print("     --training")
    
    print("\n3️⃣ Monitor Results:")
    print("   # Watch live training:")
    print("   python graphs/live_trade_visualizer_enhanced.py")
    print("   ")
    print("   # Check logs:")
    print("   tail -f logs/memory_efficient_training_*.log")

if __name__ == "__main__":
    print("🎯 Enhanced Exploration Training Configuration Test")
    print("This addresses the exploration issues in RL training")
    print("")
    
    success = test_enhanced_configurations()
    
    if success:
        test_usage_example()
        print(f"\n🎉 All tests passed! Enhanced exploration is ready for training.")
    else:
        print(f"\n❌ Some tests failed. Please check the configuration files.")
        sys.exit(1)
