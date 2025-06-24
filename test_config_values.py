#!/usr/bin/env python3
"""
Test script to verify configuration values are being read correctly
"""

import json
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_training_config, get_logging_config

def test_config_loading():
    """Test configuration loading with --default flag"""
    print("Testing configuration loading...")
    
    # Test loading default config
    try:
        config = load_training_config()  # Load from default location
        print(f"✓ Default config loaded successfully")
        print(f"  steps_per_batch: {config.get('steps_per_batch', 'NOT SET')}")
        print(f"  total_episodes: {config.get('total_episodes', 'NOT SET')}")
        print(f"  logging enabled: {config.get('logging_config', {}).get('enable_trade_logging', 'NOT SET')}")
        
        # Test logging config
        logging_config = get_logging_config(config)
        print(f"  console_log_level: {logging_config.get('console_log_level', 'NOT SET')}")
        print(f"  enable_trade_logging: {logging_config.get('enable_trade_logging', 'NOT SET')}")
        
    except Exception as e:
        print(f"✗ Error loading default config: {e}")
        return False
    
    # Test loading from file directly
    try:
        config_path = Path("config/config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            print(f"✓ Config file loaded directly")
            print(f"  steps_per_batch: {file_config.get('steps_per_batch', 'NOT SET')}")
            print(f"  total_episodes: {file_config.get('total_episodes', 'NOT SET')}")
        else:
            print(f"✗ Config file not found at {config_path}")
    except Exception as e:
        print(f"✗ Error loading config file directly: {e}")
    
    return True

def test_argument_parsing():
    """Test argument parsing to see if --default is working"""
    print("\nTesting argument parsing...")
    
    # Simulate command line arguments
    test_args = ['--default']
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', action='store_true', help='Use default configuration')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args(test_args)
    print(f"✓ Parsed --default flag: {args.default}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Testing")
    print("=" * 60)
    
    test_config_loading()
    test_argument_parsing()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
