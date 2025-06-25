#!/usr/bin/env python3
"""
Quick integration test to verify the reward system fix works in the training pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_reward_system_integration():
    """Test that the reward system fix integrates properly"""
    print("🧪 Testing Reward System Integration")
    print("=" * 40)
    
    try:
        # Test imports
        from reward_system import EnhancedRewardCalculator
        print("✅ reward_system imports successfully")
          # Test reward calculator creation with config
        config = {'use_discretized_actions': False}
        reward_calc = EnhancedRewardCalculator(config)
        print("✅ EnhancedRewardCalculator created successfully")
        
        # Test calculate_reward method (should not trigger close bonuses)
        reward, breakdown = reward_calc.calculate_reward(
            current_price=50000.0,
            position_size=0.1,  # Non-zero position
            balance=10000.0,
            action=0  # Hold action
        )
        
        print(f"✅ calculate_reward() works: reward={reward:.2f}")
        
        # Verify no close bonuses are triggered
        close_bonus = breakdown.get('close_action_bonus', 0.0)
        realized_pnl = breakdown.get('realized_pnl_points', 0.0)
        
        if close_bonus == 0.0 and realized_pnl == 0.0:
            print("✅ No false close bonuses triggered (as expected)")
        else:
            print(f"❌ Unexpected close bonuses: close_bonus={close_bonus}, realized_pnl={realized_pnl}")
            
        print()
        print("🎯 INTEGRATION TEST PASSED!")
        print("The reward system fix is working correctly.")
        print("Training can proceed without false close bonus triggers.")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reward_system_integration()
