import os
import sys
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test config printing functionality
def test_config_print():
    """Test that configuration printing works as expected"""
    
    # Load configuration similar to training script
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    with open(config_path, 'rt') as f:
        config = json.load(f)
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Print the loaded configuration to the log (same as in training script)
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
    
    print("✓ Configuration printing test completed successfully!")

if __name__ == "__main__":
    test_config_print()
