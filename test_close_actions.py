#!/usr/bin/env python3
"""
Test script to specifically verify CLOSE actions are working properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import logging
from src.environment import TradingEnvironment
from src.reward_system import ULTRA_SMALL_TRANSACTION_CONFIG

# Configure logging to be more verbose for CLOSE actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_data():
    """Create test data for the environment"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    
    # Create simple trending data
    base_price = 50000
    price_change = np.random.randn(100).cumsum() * 10
    
    data = pd.DataFrame({
        'Open': base_price + price_change,
        'High': base_price + price_change + np.random.rand(100) * 100,
        'Low': base_price + price_change - np.random.rand(100) * 100,
        'Close': base_price + price_change + np.random.randn(100) * 50,
        'Volume': np.random.rand(100) * 1000000,
        'rsi': 50 + np.random.randn(100) * 20,
        'sma_20': base_price + price_change + np.random.randn(100) * 25,
        'ema_12': base_price + price_change + np.random.randn(100) * 20,
    }, index=dates)
    
    return data

def test_close_actions():
    """Test CLOSE actions specifically"""
    print("🔍 Testing CLOSE actions functionality...")
    
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=False
    )
    
    obs, info = env.reset()
    
    print("Step 1: Opening 3 trades...")
    for i in range(3):
        action = np.array([1, 0.05, 2.0])  # BUY action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Trade {i+1} opened, total open: {len(env.open_trades)}")
    
    print(f"\nAfter opening: {len(env.open_trades)} trades open")
    for i, trade in enumerate(env.open_trades):
        print(f"  Trade {i}: ID={trade['trade_id']}, Side={trade['side']}, Size={trade['size_btc']:.6f} BTC")
    
    print("\nStep 2: Testing CLOSE_TRADE_1 action (action type 3)...")
    action = np.array([3, 0, 1.0])  # CLOSE_TRADE_1
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  After CLOSE_TRADE_1: {len(env.open_trades)} trades open, reward: {reward:.2f}")
    
    print("\nStep 3: Testing CLOSE_TRADE_1 again (should close index 0 again)...")
    action = np.array([3, 0, 1.0])  # CLOSE_TRADE_1
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  After second CLOSE_TRADE_1: {len(env.open_trades)} trades open, reward: {reward:.2f}")
    
    print("\nStep 4: Testing CLOSE_TRADE_2 action (action type 4)...")
    action = np.array([4, 0, 1.0])  # CLOSE_TRADE_2
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  After CLOSE_TRADE_2: {len(env.open_trades)} trades open, reward: {reward:.2f}")
    
    print("\nStep 5: Testing invalid CLOSE action (no trades left)...")
    action = np.array([3, 0, 1.0])  # CLOSE_TRADE_1 when no trades
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  After invalid CLOSE: {len(env.open_trades)} trades open, reward: {reward:.2f}")
    
    print("\nStep 6: Open more trades and test higher close indices...")
    for i in range(5):
        action = np.array([np.random.choice([1, 2]), 0.05, 2.0])  # BUY or SELL
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  After opening more: {len(env.open_trades)} trades open")
    
    print("\nStep 7: Test CLOSE_TRADE_3, CLOSE_TRADE_4, CLOSE_TRADE_5...")
    for close_action in [5, 6, 7]:  # CLOSE_TRADE_3, 4, 5
        if len(env.open_trades) > 0:
            action = np.array([close_action, 0, 1.0])
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  After CLOSE_TRADE_{close_action-2}: {len(env.open_trades)} trades open, reward: {reward:.2f}")
        else:
            print(f"  Skipping CLOSE_TRADE_{close_action-2}: No trades left")
    
    print(f"\nFinal state: {len(env.open_trades)} trades open")

def test_action_mapping():
    """Test the action type mapping"""
    print("\n🗺️ Testing action type mapping...")
    
    print("Action mappings:")
    print("  Action 0: HOLD")
    print("  Action 1: BUY (open long)")
    print("  Action 2: SELL (open short)")
    print("  Action 3: CLOSE_TRADE_1 (close trade at index 0)")
    print("  Action 4: CLOSE_TRADE_2 (close trade at index 1)")
    print("  Action 5: CLOSE_TRADE_3 (close trade at index 2)")
    print("  ...")
    print("  Action 12: CLOSE_TRADE_10 (close trade at index 9)")
    
    # Test the mapping logic
    for action_type in range(13):
        if action_type == 0:
            action_name = "HOLD"
        elif action_type == 1:
            action_name = "BUY"
        elif action_type == 2:
            action_name = "SELL"
        elif action_type >= 3:
            close_index = action_type - 3
            action_name = f"CLOSE_TRADE_{close_index + 1} (index {close_index})"
        
        print(f"  Action {action_type}: {action_name}")

if __name__ == "__main__":
    print("🚀 Testing CLOSE Actions Specifically\n")
    
    test_action_mapping()
    test_close_actions()
    
    print("\n✅ CLOSE action test completed!")
