#!/usr/bin/env python3
"""
Quick test of the model creation fix
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_model_creation():
    """Test that the model can be created without the duplicate parameter error"""
    print("Testing model creation...")
    
    try:
        from model import TradingModel
        from environment import TradingEnvironment
        import pandas as pd
        import numpy as np
        
        # Create a minimal test environment
        test_data = pd.DataFrame({
            'Open': np.random.random(100) * 100 + 50000,
            'High': np.random.random(100) * 100 + 50000,
            'Low': np.random.random(100) * 100 + 50000,
            'Close': np.random.random(100) * 100 + 50000,
            'Volume': np.random.random(100) * 1000000,
        })
        test_data.index = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Create test environment
        test_env = TradingEnvironment(
            df=test_data,
            initial_balance=10000,
            lookback_window=10
        )
        
        print("✅ Test environment created successfully")
        
        # Try to create the model
        model = TradingModel(
            env=test_env,
            model_name="test_model",
            device="cpu"
        )
        
        print("✅ TradingModel created successfully!")
        print("🎉 Model creation fix worked!")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("\n🚀 Model fix verified - ready for training!")
    else:
        print("\n⚠️  Model fix needs more work")
    sys.exit(0 if success else 1)
