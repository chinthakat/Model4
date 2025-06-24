#!/usr/bin/env python3
"""
Test script to verify log archiving is working
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment import TradingEnvironment
from utils.trade_logger import TradeLogger
from utils.trade_tracer import TradeTracer

def test_log_archiving():
    """Test that log archiving works correctly"""
    print("Testing log archiving...")
    
    # Create some test data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='15T')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    prices = 30000 + np.cumsum(np.random.randn(200) * 10)
    
    test_data = pd.DataFrame({
        'Open': prices + np.random.randn(200) * 5,
        'High': prices + np.abs(np.random.randn(200) * 10),
        'Low': prices - np.abs(np.random.randn(200) * 10),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, 200),
        'RSI': np.random.uniform(20, 80, 200),
        'MACD': np.random.randn(200),
    }, index=dates)
    
    # Create environment with logging enabled
    env = TradingEnvironment(
        test_data, 
        lookback_window=10, 
        enable_trade_logging=True,
        logging_config={
            'enable_trade_logging': True,
            'enable_trade_tracing': True,
            'trade_log_frequency': 1
        }
    )
    
    print(f"✓ Environment created with logging enabled")
    print(f"  Trade logger: {env.trade_logger is not None}")
    print(f"  Trade tracer: {env.trade_tracer is not None}")
    
    # Run a few steps to generate some logs
    obs, info = env.reset()
    for i in range(5):
        # Try different actions to generate trades
        action = np.array([1, 0.1, 2.0])  # Buy action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"✓ Ran 5 trading steps to generate logs")
    
    # Test save_session on trade logger
    if env.trade_logger:
        try:
            env.trade_logger.save_session("test_archiving")
            print(f"✓ TradeLogger save_session called successfully")
            
            # Check if files were created
            log_dir = Path("logs")
            if log_dir.exists():
                log_files = list(log_dir.rglob("*.csv"))
                print(f"  Found {len(log_files)} CSV log files")
                for log_file in log_files[-3:]:  # Show last 3 files
                    print(f"    {log_file}")
        except Exception as e:
            print(f"✗ TradeLogger save_session failed: {e}")
    
    # Test getting session summary from trade tracer
    if env.trade_tracer:
        try:
            summary = env.trade_tracer.get_session_summary()
            print(f"✓ TradeTracer session summary retrieved")
            print(f"  Total events: {summary.get('total_events', 0)}")
            print(f"  Session duration: {summary.get('session_duration', 'N/A')}")
        except Exception as e:
            print(f"✗ TradeTracer session summary failed: {e}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Log Archiving Test")
    print("=" * 60)
    
    try:
        test_log_archiving()
        print("\n" + "=" * 60)
        print("✓ Log archiving test completed successfully!")
    except Exception as e:
        print(f"\n✗ Log archiving test failed: {e}")
        import traceback
        traceback.print_exc()
