#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.append('src')
from utils.config_loader import load_training_config

# Test the configuration
config = load_training_config('config/config.json')

print("=== UPDATED TRAINING CONFIGURATION ===")
print(f"Total Episodes: {config.get('total_episodes')}")
print(f"Steps per Batch: {config.get('steps_per_batch')}")
print(f"Expected Total Training Steps: {config.get('total_episodes', 1) * config.get('steps_per_batch', 0)}")
print(f"Initial Balance: {config.get('initial_balance')}")
print(f"Device: {config.get('device')}")
print(f"Trade Logging: {config.get('logging_config', {}).get('enable_trade_logging')}")
print("=======================================")

# Also show what was in the config file directly
print("\n=== RAW CONFIG FILE CONTENT ===")
with open('config/config.json', 'r') as f:
    raw_config = json.load(f)
    print(f"total_episodes: {raw_config.get('total_episodes')}")
    print(f"steps_per_batch: {raw_config.get('steps_per_batch')}")
print("===============================")
