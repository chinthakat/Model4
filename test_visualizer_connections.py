#!/usr/bin/env python3
"""
Test script to generate trade close events for visualizer testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment import TradingEnvironment
from src.utils.config_loader import load_training_config
import numpy as np
import time

def test_trade_closes():
    """Test generating some trade closes to see in visualizer"""
    print("🔄 Testing trade close generation for visualizer...")
    
    # Load config
    config = load_training_config()
    
    # Create environment
    env = TradingEnvironment(config)
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Initial balance: ${env.balance:.2f}")
    
    # Generate a few open and close actions
    actions_to_test = [
        (0, "BUY"),    # Open trade
        (1, "SELL"),   # Open trade  
        (0, "BUY"),    # Open trade
        (3, "CLOSE"),  # Close oldest
        (4, "CLOSE"),  # Close oldest
        (2, "SELL"),   # Open trade
        (5, "CLOSE"),  # Close oldest
    ]
    
    for i, (action, action_name) in enumerate(actions_to_test):
        print(f"\n--- Step {i+1}: {action_name} (action {action}) ---")
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.4f}")
        print(f"Balance: ${env.balance:.2f}")
        print(f"Open positions: {len(env.active_trades)}")
        
        if env.active_trades:
            print("Active trades:")
            for trade_id, trade in env.active_trades.items():
                print(f"  {trade_id}: {trade['side']} ${trade['entry_price']:.2f}")
        
        # Wait a bit to see updates in visualizer
        time.sleep(2)
        
        if done or truncated:
            print("Episode ended")
            break
    
    print(f"\n✅ Test completed. Final balance: ${env.balance:.2f}")
    print("Check the live visualizer to see trade connection lines!")

if __name__ == "__main__":
    test_trade_closes()
