#!/usr/bin/env python3
"""
Test script to verify single model creation logic in training script
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.config_loader import load_training_config
from train_memory_efficient import Trainer

def test_single_model_creation():
    """Test that only one model is created throughout training"""
    print("🔬 Testing single model creation logic...")
    
    # Setup basic logging to capture model creation messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load config
        config = load_training_config()
        
        # Initialize trainer
        trainer = Trainer(config=config)
        
        print(f"✅ Trainer initialized")
        print(f"   Initial model state: {trainer.model}")
        
        # Test model creation logic multiple times
        print("\n🔄 Testing multiple model creation calls...")
        
        # First call should create model
        print("1. First call to _ensure_single_model_creation:")
        trainer.train_env = None  # Simulate no environment initially
        trainer._ensure_single_model_creation()
        first_model = trainer.model
        print(f"   Model after first call: {first_model}")
        
        # Second call should reuse existing model
        print("2. Second call to _ensure_single_model_creation:")
        trainer._ensure_single_model_creation()
        second_model = trainer.model
        print(f"   Model after second call: {second_model}")
        
        # Third call should still reuse the same model
        print("3. Third call to _ensure_single_model_creation:")
        trainer._ensure_single_model_creation()
        third_model = trainer.model
        print(f"   Model after third call: {third_model}")
        
        # Verify all references point to the same model object
        print(f"\n📊 Model Identity Verification:")
        print(f"   First model is second model: {first_model is second_model}")
        print(f"   Second model is third model: {second_model is third_model}")
        print(f"   All same object: {first_model is second_model is third_model}")
        
        if first_model is second_model is third_model:
            print("✅ SUCCESS: Single model creation working correctly!")
            print("   The same model object is reused throughout the session.")
        else:
            print("❌ FAILURE: Multiple models were created!")
            
        # Test session update logic
        print(f"\n🔄 Testing environment session updates...")
        trainer.current_episode = 5
        trainer._update_environment_sessions()
        print("✅ Environment session update completed")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if hasattr(trainer, 'cleanup'):
                trainer.cleanup()
        except:
            pass
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_single_model_creation()
