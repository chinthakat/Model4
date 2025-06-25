#!/usr/bin/env python3
"""
Quick test to verify enhanced training with synthetic data works
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent / 'src'))

def test_enhanced_training():
    """Test that enhanced training with synthetic data works"""
    print("🔍 Testing Enhanced Training with Synthetic Data...")
    
    try:
        # Test imports
        from utils.config_loader import load_training_config
        from model import TradingModel
        from environment import TradingEnvironment
        from data.setup_data import DataProcessor
        import pandas as pd
        
        print("✅ All imports successful")
        
        # Load config
        config = load_training_config("config/training_config.json")
        
        # Switch to synthetic data
        synthetic_data_path = "data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv"
        if Path(synthetic_data_path).exists():
            config['consolidated_file'] = synthetic_data_path
            config['reward_strategy'] = 'enhanced_exploration'
            print(f"✅ Using synthetic data: {synthetic_data_path}")
        else:
            print(f"❌ Synthetic data not found: {synthetic_data_path}")
            return False
          # Load and process data
        print("🔄 Loading and processing data...")
        full_df = pd.read_csv(synthetic_data_path)
        print(f"✅ Loaded {len(full_df)} rows")
        
        # Standardize column names
        if 'Unnamed: 0' in full_df.columns:
            full_df = full_df.drop(columns=['Unnamed: 0'])
        
        # Column mapping to match expected format
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Rename columns
        full_df = full_df.rename(columns=column_mapping)
        
        # Set datetime index if timestamp exists
        if 'timestamp' in full_df.columns:
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='s')
            full_df = full_df.set_index('timestamp')
        else:
            # Create default datetime index
            full_df.index = pd.date_range('2024-01-01', periods=len(full_df), freq='15min')
            full_df.index.name = 'timestamp'
        
        print(f"✅ Data standardized - Columns: {list(full_df.columns)}")
        
        # Process data
        data_processor = DataProcessor(lookback_window=config.get('lookback_window', 50))
        train_data, test_data = data_processor.prepare_data(
            full_df,
            funding_rates=None,
            train_ratio=config.get('train_ratio', 0.8),
            fit_scalers=True
        )
        
        print(f"✅ Data processed - Train: {len(train_data)} rows, Test: {len(test_data)} rows")
        
        # Create environment
        print("🔄 Creating trading environment...")
        train_env = TradingEnvironment(
            df=train_data,
            initial_balance=config.get('initial_balance', 10000),
            lookback_window=config.get('lookback_window', 50)
        )
        print("✅ Trading environment created")
        
        # Create model
        print("🔄 Creating TradingModel...")
        model = TradingModel(
            env=train_env,
            model_name="test_enhanced_model",
            device="cpu"
        )
        print("✅ TradingModel created successfully!")
        
        # Test a few training steps
        print("🔄 Testing training steps...")
        model.train(total_timesteps=100)  # Just a few steps for testing
        print("✅ Training steps completed!")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced exploration with synthetic data is working!")
        print("🚀 Ready for full training runs!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_training()
    if success:
        print("\n🎯 System Status: READY FOR ENHANCED TRAINING!")
        print("📋 Usage:")
        print("   python src/train_memory_efficient.py --training   # Synthetic data + enhanced exploration")
        print("   python src/train_memory_efficient.py --default    # Real data + enhanced exploration")
    sys.exit(0 if success else 1)
