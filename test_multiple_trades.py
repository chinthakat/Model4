#!/usr/bin/env python3
"""
Quick test to verify the environment allows more than 5 trades
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment import TradingEnvironment
from reward_system import ULTRA_SMALL_TRANSACTION_CONFIG

def test_multiple_trades():
    """Test that environment allows opening more than 5 trades"""
    # Create dummy data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    data = pd.DataFrame({
        'Open': 50000 + np.random.randn(1000) * 100,
        'High': 50000 + np.random.randn(1000) * 100 + 50,
        'Low': 50000 + np.random.randn(1000) * 100 - 50,
        'Close': 50000 + np.random.randn(1000) * 100,
        'Volume': np.random.randn(1000) * 1000 + 5000
    }, index=dates)
    
    # Ensure High >= Low and Close between Low and High
    data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
    data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
      # Initialize environment
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        lookback_window=20,
        reward_config=ULTRA_SMALL_TRANSACTION_CONFIG,
        enable_trade_logging=False,
        logging_config={'enable_trade_tracing': False}
    )
    
    obs, info = env.reset()
    print(f"Environment initialized. Action space: {env.action_space}")
    
    trades_opened = 0
    
    # Try to open 10 small trades
    for i in range(15):  # Try more than the old limit
        # Action: [action_type=1 (BUY), size_index=1 (0.005), confidence=1.0]
        action = np.array([1, 1, 1.0], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if len(env.open_trades) > trades_opened:
            trades_opened = len(env.open_trades)
            print(f"Step {i+1}: Opened trade. Total open trades: {trades_opened}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {i+1}")
            break
    
    print(f"\nFinal result:")
    print(f"Maximum trades opened simultaneously: {trades_opened}")
    print(f"Current open trades: {len(env.open_trades)}")
    print(f"Environment balance: ${env.balance:.2f}")
    print(f"Environment equity: ${env.equity:.2f}")
    
    if trades_opened > 5:
        print("✅ SUCCESS: Environment allows more than 5 trades!")
        return True
    else:
        print("❌ FAILED: Environment still limited to 5 trades or less")
        return False

if __name__ == "__main__":
    test_multiple_trades()
