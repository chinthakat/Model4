#!/usr/bin/env python3
"""
Test the reward system fix for the close bonus bug.
This verifies that close bonuses are no longer incorrectly triggered.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_reward_system_fix():
    """Test that the reward system no longer triggers false close bonuses"""
    print("🔧 Testing Reward System Fix")
    print("=" * 50)
    
    try:
        from reward_system import EnhancedRewardCalculator, BALANCED_ENHANCED_CONFIG
        
        # Create reward calculator
        reward_calc = EnhancedRewardCalculator(BALANCED_ENHANCED_CONFIG)
        print("✅ Reward calculator initialized successfully")
        
        # Test scenario 1: Multiple trades open, net position size changes
        print("\n📊 SCENARIO 1: Net position size changes (should NOT trigger close bonus)")
        print("- Simulating multiple trades with changing net position")
        
        # First call - establish baseline
        reward1, breakdown1 = reward_calc.calculate_reward(
            current_price=50000.0,
            position_size=0.1,  # Some position open
            balance=10000.0,
            action=1  # Buy action
        )
        
        # Second call - net position becomes zero (but no actual trade closure)
        reward2, breakdown2 = reward_calc.calculate_reward(
            current_price=50100.0,
            position_size=0.0,  # Net position zero (e.g., offsetting trades)
            balance=10050.0,
            action=0  # Hold action
        )
        
        # Check if close bonuses were triggered
        close_bonus_triggered = (
            breakdown2.get('realized_pnl_points', 0) != 0 or
            breakdown2.get('close_action_bonus', 0) != 0
        )
        
        print(f"  Step 1 - Position: 0.1, Action: BUY")
        print(f"  Step 2 - Position: 0.0, Action: HOLD")
        print(f"  Realized PnL Points: {breakdown2.get('realized_pnl_points', 0)}")
        print(f"  Close Action Bonus: {breakdown2.get('close_action_bonus', 0)}")
        
        if close_bonus_triggered:
            print("  ❌ FAILED: Close bonuses incorrectly triggered!")
            return False
        else:
            print("  ✅ PASSED: No false close bonuses triggered")
        
        # Test scenario 2: Multiple calls with varying position sizes
        print("\n📊 SCENARIO 2: Various position size changes")
        print("- Testing multiple position size changes")
        
        false_triggers = 0
        test_cases = [
            (0.05, 1),  # Small long position
            (0.0, 0),   # Zero position
            (-0.03, 2), # Small short position
            (0.0, 0),   # Zero position again
            (0.08, 1),  # Another long position
        ]
        
        for i, (pos_size, action) in enumerate(test_cases):
            reward, breakdown = reward_calc.calculate_reward(
                current_price=50000.0 + i * 50,
                position_size=pos_size,
                balance=10000.0 + i * 10,
                action=action
            )
            
            if breakdown.get('realized_pnl_points', 0) != 0 or breakdown.get('close_action_bonus', 0) != 0:
                false_triggers += 1
                print(f"  Step {i+1}: Position {pos_size}, Action {action} - ❌ False close bonus!")
            else:
                print(f"  Step {i+1}: Position {pos_size}, Action {action} - ✅ No false bonus")
        
        if false_triggers > 0:
            print(f"  ❌ FAILED: {false_triggers} false close bonus triggers detected!")
            return False
        else:
            print("  ✅ PASSED: No false close bonuses in any test case")
        
        print("\n🎯 SUMMARY:")
        print("✅ Fix successfully implemented!")
        print("✅ Close bonuses (realized_pnl_points, close_action_bonus) are no longer")
        print("   incorrectly triggered by position size changes")
        print("✅ This resolves the mismatch between trade traces and reward logs")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to test reward system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reward_system_fix()
    if success:
        print("\n🎉 REWARD SYSTEM FIX VERIFIED!")
        print("The training script should now run without false close bonus triggers.")
    else:
        print("\n💥 REWARD SYSTEM FIX FAILED!")
        print("Further investigation needed.")
