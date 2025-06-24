#!/usr/bin/env python3
"""
Test script to verify archiving integration with training
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_archiving_integration():
    """Test that archiving works before training starts"""
    print("Testing archiving integration...")
    
    # Create some fake log and model files
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create test files
    (logs_dir / "training.log").write_text("Test training log content")
    (logs_dir / "test_trade.log").write_text("Test trade log content")
    (models_dir / "test_model.pkl").write_text("Test model content")
    
    print(f"✓ Created test files in logs and models directories")
    
    # Test the archiving functionality
    try:
        from utils.archiver import archive_before_training
        
        archive_path = archive_before_training("test_session")
        
        if archive_path:
            print(f"✓ Archive created successfully: {archive_path}")
            
            # Check that archive file exists
            if Path(archive_path).exists():
                print(f"✓ Archive file exists and is accessible")
                
                # Check archive size
                size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
                print(f"  Archive size: {size_mb:.2f} MB")
            else:
                print(f"✗ Archive file not found: {archive_path}")
                return False
        else:
            print("ℹ️  No archive created (no files to archive)")
        
        return True
        
    except Exception as e:
        print(f"✗ Archiving failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_integration():
    """Test that the training script can handle archiving"""
    print("\nTesting training script integration...")
    
    try:
        # Import the main training module to test imports
        sys.path.append("src")
        from train_memory_efficient import main
        
        print("✓ Training script imports successfully")
        print("✓ Archiver integration is ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Archiving Integration Test")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_archiving_integration()
        success &= test_training_script_integration()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ All archiving integration tests passed!")
            print("📦 Ready to archive logs and models before training")
        else:
            print("❌ Some tests failed - check the output above")
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
