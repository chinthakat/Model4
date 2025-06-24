#!/usr/bin/env python3
"""
Dry run test to verify training script structure
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

def test_training_script_structure():
    """Test the training script structure without running full training"""
    print("🔍 Testing training script structure...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from utils.config_loader import load_training_config
        from train_memory_efficient import Trainer
        print("   ✅ All imports successful")
        
        # Test config loading
        print("2. Testing config loading...")
        config = load_training_config()
        print(f"   ✅ Config loaded: {len(config)} keys")
        print(f"   - Episodes: {config.get('total_episodes', 'not set')}")
        print(f"   - Steps per episode: {config.get('steps_per_episode', 'not set')}")
        
        # Test trainer initialization (without full setup)
        print("3. Testing trainer initialization...")
        try:
            # Create a minimal trainer instance
            trainer = Trainer(config=config)
            print("   ✅ Trainer initialized successfully")
            
            # Test key attributes
            print("4. Testing trainer attributes...")
            print(f"   - Model: {trainer.model}")
            print(f"   - Train env: {trainer.train_env}")
            print(f"   - Test env: {trainer.test_env}")
            print(f"   - Total episodes: {trainer.total_episodes}")
            print("   ✅ All attributes properly initialized")
            
            # Test the single model creation method exists
            print("5. Testing single model creation method...")
            if hasattr(trainer, '_ensure_single_model_creation'):
                print("   ✅ _ensure_single_model_creation method exists")
            else:
                print("   ❌ _ensure_single_model_creation method missing")
            
            # Test other key methods exist
            methods_to_check = [
                'run_complete_training',
                '_process_batch_data', 
                '_create_or_update_environments',
                '_update_environment_sessions',
                'cleanup'
            ]
            
            print("6. Testing key methods exist...")
            for method in methods_to_check:
                if hasattr(trainer, method):
                    print(f"   ✅ {method} exists")
                else:
                    print(f"   ❌ {method} missing")
            
        except Exception as e:
            print(f"   ❌ Trainer initialization failed: {e}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Structure test completed!")

if __name__ == "__main__":
    test_training_script_structure()
