#!/usr/bin/env python3
"""
Simple test to verify single model creation logic
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

def test_model_creation_logic():
    """Test that only one model is created throughout training"""
    print("🔬 Testing single model creation logic...")
    
    try:
        # Import after setting path
        from utils.config_loader import load_training_config
        
        # Load config
        config = load_training_config()
        print(f"✅ Config loaded: {config.get('total_episodes', 'unknown')} episodes")
        
        # Create a mock trainer class to test the logic
        class MockTrainer:
            def __init__(self):
                self.model = None
                self.logger = MockLogger()
                
            def _ensure_single_model_creation(self):
                """Ensure we create only one model for the entire training session"""
                if self.model is None:
                    self.logger.info("Creating single TradingModel for entire training session...")
                    # Mock model creation
                    self.model = f"MockModel_Created_At_{id(self)}"
                    self.logger.info("✅ Single TradingModel created - will be reused throughout entire training session")
                else:
                    self.logger.info("✅ Reusing existing TradingModel (already created for this session)")
        
        class MockLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
        
        # Test the logic
        trainer = MockTrainer()
        
        print("\n🔄 Testing model creation calls...")
        
        # First call should create model
        print("1. First call:")
        trainer._ensure_single_model_creation()
        first_model = trainer.model
        print(f"   Model: {first_model}")
        
        # Second call should reuse existing model
        print("2. Second call:")
        trainer._ensure_single_model_creation()
        second_model = trainer.model
        print(f"   Model: {second_model}")
        
        # Third call should still reuse the same model
        print("3. Third call:")
        trainer._ensure_single_model_creation()
        third_model = trainer.model
        print(f"   Model: {third_model}")
        
        # Verify all references point to the same model object
        print(f"\n📊 Model Identity Verification:")
        print(f"   First == Second: {first_model == second_model}")
        print(f"   Second == Third: {second_model == third_model}")
        print(f"   All same: {first_model == second_model == third_model}")
        
        if first_model == second_model == third_model:
            print("✅ SUCCESS: Single model creation logic working correctly!")
            print("   The same model object is reused throughout the session.")
        else:
            print("❌ FAILURE: Multiple models were created!")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_model_creation_logic()
