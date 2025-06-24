#!/usr/bin/env python3
"""
Test script to verify the new trade management logic:
1. Max 10 open trades limit
2. Negative reward for opening trades when >5 open
3. Positive reward for closing trades when >3 open
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import logging
from src.environment import TradingEnvironment
from src.reward_system import ULTRA_SMALL_TRANSACTION_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_data():
    """Create test data for the environment"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='15T')
    
    # Create simple trending data
    base_price = 50000
    price_change = np.random.randn(1000).cumsum() * 10
    
    data = pd.DataFrame({
        'Open': base_price + price_change,
        'High': base_price + price_change + np.random.rand(1000) * 100,
        'Low': base_price + price_change - np.random.rand(1000) * 100,
        'Close': base_price + price_change + np.random.randn(1000) * 50,
        'Volume': np.random.rand(1000) * 1000000,
        'rsi': 50 + np.random.randn(1000) * 20,
        'sma_20': base_price + price_change + np.random.randn(1000) * 25,
        'ema_12': base_price + price_change + np.random.randn(1000) * 20,
    }, index=dates)
    
    return data

def test_trade_limit():
    """Test the maximum trade limit functionality"""
    print("🔍 Testing trade limit functionality...")
    
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=False
    )
    
    obs, info = env.reset()
    
    # Try to open 12 trades (should be limited to 10)
    trades_opened = 0
    for i in range(12):
        # BUY action: [1, 0.05, 2.0] (small position, 2x leverage)
        action = np.array([1, 0.05, 2.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if len(env.open_trades) > trades_opened:
            trades_opened = len(env.open_trades)
            print(f"✅ Trade {trades_opened} opened successfully")
        else:
            print(f"❌ Trade {i+1} rejected - max limit reached")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Final Results:")
    print(f"   Total trades attempted: 12")
    print(f"   Total trades opened: {len(env.open_trades)}")
    print(f"   Max limit enforced: {len(env.open_trades) <= env.MAX_OPEN_TRADES}")
    
    return env

def test_reward_penalties():
    """Test the reward penalties and bonuses"""
    print("\n🎯 Testing reward penalties and bonuses...")
    
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=False
    )
    
    obs, info = env.reset()
    
    # Open 6 trades (above the limit of 5)
    print("Opening 6 trades to test penalty system...")
    rewards = []
    for i in range(6):
        action = np.array([1, 0.05, 2.0])  # BUY action
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"Trade {i+1}: reward = {reward:.2f}, open trades = {len(env.open_trades)}")
    
    # Check if penalties were applied for trades 6+ 
    penalty_applied = any(reward < -0.1 for reward in rewards[5:])  # Trade 6 should have penalty
    print(f"Penalty for exceeding 5 trades: {'✅ Applied' if penalty_applied else '❌ Not applied'}")
    
    # Now test close bonuses (we have >3 trades)
    print("\nTesting close bonuses...")
    close_rewards = []
    for i in range(3):  # Close 3 trades
        if len(env.open_trades) > 0:
            action = np.array([3, 0, 1.0])  # CLOSE_TRADE_1 action
            obs, reward, terminated, truncated, info = env.step(action)
            close_rewards.append(reward)
            print(f"Close {i+1}: reward = {reward:.2f}, open trades = {len(env.open_trades)}")
    
    # Check if bonuses were applied for closing trades
    bonus_applied = any(reward > 0.05 for reward in close_rewards)  # Should have bonus
    print(f"Bonus for closing when >3 trades: {'✅ Applied' if bonus_applied else '❌ Not applied'}")
    
    return env

def test_integration():
    """Test full integration of the trade management system"""
    print("\n🔗 Testing full integration...")
    
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=True
    )
    
    obs, info = env.reset()
    
    step_count = 0
    max_steps = 50
    terminated = False
    truncated = False
    
    print("Running simulation to test trade management in realistic scenario...")
    
    while step_count < max_steps and not (terminated or truncated):
        # Simulate realistic trading behavior
        if len(env.open_trades) < 3:
            # Open new trades when we have few open
            action = np.array([np.random.choice([1, 2]), 0.05, 2.0])
        elif len(env.open_trades) > 7:
            # Close trades when we have many open
            close_index = np.random.randint(0, min(len(env.open_trades), 5)) + 3
            action = np.array([close_index, 0, 1.0])
        else:
            # Random action
            action_type = np.random.choice([0, 1, 2, 3, 4, 5])
            if action_type <= 2:
                action = np.array([action_type, 0.05, 2.0])
            else:
                action = np.array([action_type, 0, 1.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: {len(env.open_trades)} trades open, reward = {reward:.2f}")
        
        step_count += 1
    
    print(f"\n📊 Integration Test Results:")
    print(f"   Steps completed: {step_count}")
    print(f"   Final open trades: {len(env.open_trades)}")
    print(f"   Final balance: ${env.balance:.2f}")
    print(f"   Total trades: {env.total_trades}")
    print(f"   Trade limit respected: {len(env.open_trades) <= env.MAX_OPEN_TRADES}")

if __name__ == "__main__":
    print("🚀 Testing New Trade Management System\n")
    
    # Test 1: Trade limit functionality
    env1 = test_trade_limit()
    
    # Test 2: Reward penalties and bonuses
    env2 = test_reward_penalties()
    
    # Test 3: Full integration
    test_integration()
    
    print("\n✅ All trade management tests completed!")
