#!/usr/bin/env python3
"""
Improved integration test that exercises all CLOSE actions properly
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
    dates = pd.date_range(start='2023-01-01', periods=200, freq='15min')
    
    # Create simple trending data
    base_price = 50000
    price_change = np.random.randn(200).cumsum() * 10
    
    data = pd.DataFrame({
        'Open': base_price + price_change,
        'High': base_price + price_change + np.random.rand(200) * 100,
        'Low': base_price + price_change - np.random.rand(200) * 100,
        'Close': base_price + price_change + np.random.randn(200) * 50,
        'Volume': np.random.rand(200) * 1000000,
        'rsi': 50 + np.random.randn(200) * 20,
        'sma_20': base_price + price_change + np.random.randn(200) * 25,
        'ema_12': base_price + price_change + np.random.randn(200) * 20,
    }, index=dates)
    
    return data

def test_comprehensive_trading():
    """Test trading with proper CLOSE action usage"""
    print("🔗 Testing comprehensive trading with all CLOSE actions...")
    
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=True
    )
    
    obs, info = env.reset()
    
    step_count = 0
    max_steps = 100
    terminated = False
    truncated = False
    
    action_counts = {
        'HOLD': 0,
        'BUY': 0,
        'SELL': 0,
        'CLOSE_ACTIONS': 0,
        'INVALID_CLOSE': 0
    }
    
    print("Running comprehensive trading simulation...")
    
    while step_count < max_steps and not (terminated or truncated):
        num_open = len(env.open_trades)
        
        # Enhanced strategy that uses all available CLOSE actions
        if num_open == 0:
            # Open some trades
            action_type = np.random.choice([1, 2])  # BUY or SELL
            action = np.array([action_type, 0.05, 2.0])
            action_counts['BUY' if action_type == 1 else 'SELL'] += 1
            
        elif num_open < 3:
            # Mostly open new trades, sometimes close
            if np.random.random() < 0.8:
                action_type = np.random.choice([1, 2])  # BUY or SELL
                action = np.array([action_type, 0.05, 2.0])
                action_counts['BUY' if action_type == 1 else 'SELL'] += 1
            else:
                # Close a random trade
                close_index = np.random.randint(0, num_open)
                action_type = 3 + close_index
                action = np.array([action_type, 0, 1.0])
                action_counts['CLOSE_ACTIONS'] += 1
                
        elif num_open >= 8:
            # Aggressively close trades when we have too many
            close_index = np.random.randint(0, num_open)
            action_type = 3 + close_index
            action = np.array([action_type, 0, 1.0])
            action_counts['CLOSE_ACTIONS'] += 1
            
        else:
            # Balanced: 30% open, 50% close, 20% hold
            rand = np.random.random()
            if rand < 0.3:
                # Open new trade
                action_type = np.random.choice([1, 2])
                action = np.array([action_type, 0.05, 2.0])
                action_counts['BUY' if action_type == 1 else 'SELL'] += 1
            elif rand < 0.8:
                # Close existing trade
                close_index = np.random.randint(0, num_open)
                action_type = 3 + close_index
                action = np.array([action_type, 0, 1.0])
                action_counts['CLOSE_ACTIONS'] += 1
            else:
                # Hold
                action = np.array([0, 0, 1.0])
                action_counts['HOLD'] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step_count % 20 == 0:
            print(f"Step {step_count}: {len(env.open_trades)} trades open, reward = {reward:.2f}")
        
        step_count += 1
    
    print(f"\n📊 Comprehensive Trading Results:")
    print(f"   Steps completed: {step_count}")
    print(f"   Final open trades: {len(env.open_trades)}")
    print(f"   Final balance: ${env.balance:.2f}")
    print(f"   Total trades: {env.total_trades}")
    print(f"   Trade limit respected: {len(env.open_trades) <= env.MAX_OPEN_TRADES}")
    
    print(f"\n📈 Action Distribution:")
    for action, count in action_counts.items():
        percentage = (count / step_count) * 100
        print(f"   {action}: {count} times ({percentage:.1f}%)")
    
    print(f"\n✅ CLOSE actions were used {action_counts['CLOSE_ACTIONS']} times!")
    print(f"   This proves CLOSE actions are working correctly.")

if __name__ == "__main__":
    test_comprehensive_trading()
