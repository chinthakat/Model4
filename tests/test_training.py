
import unittest
import os
import json
from pathlib import Path
from src.train_memory_efficient import Trainer
from src.utils.config_loader import load_training_config

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.config_path = "config/config.json"
        self.logging_config_path = "config/logging_config.json"

        # Create dummy config files for testing
        self.config = {
            "consolidated_file": "data/processed/BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv",
            "initial_balance": 10000,
            "lookback_window": 50,
            "train_ratio": 0.8,
            "total_episodes": 1,
            "steps_per_batch": 100,
            "reward_strategy": "balanced",
            "encourage_small_trades": True,
            "ultra_aggressive_small_trades": True,
            "log_reward_details": False,
            "final_model_path": "models/test_model",
            "device": "cpu",
            "logging_config": {
                "console_log_level": "INFO",
                "file_log_level": "DEBUG",
                "trade_log_frequency": 10,
                "enable_trade_logging": False,
                "enable_trade_tracing": False
            }
        }

        self.logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO"
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO"
            }
        }

        os.makedirs("config", exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        with open(self.logging_config_path, 'w') as f:
            json.dump(self.logging_config, f)

    def test_trainer_initialization(self):
        config = load_training_config(self.config_path)
        trainer = Trainer(config)
        self.assertIsNotNone(trainer)

if __name__ == '__main__':
    unittest.main()
