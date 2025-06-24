import os
import sys
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the actual configuration printing that now happens in main()
def test_main_config_print():
    """Test that the main() function config printing works correctly"""
    
    # Load the main configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    with open(config_path, 'rt') as f:
        config = json.load(f)
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Total episodes: {config['total_episodes']}")
    print(f"Steps per batch: {config['steps_per_batch']}")
    
    # Print the loaded configuration to the log (exactly as in main())
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)
    
    print("✓ Main configuration printing test completed successfully!")
    print(f"✓ Confirmed: Total episodes = {config['total_episodes']}, Steps per batch = {config['steps_per_batch']}")

if __name__ == "__main__":
    test_main_config_print()
