#!/usr/bin/env python3
"""
Memory-Efficient Training Script for RL Trading Bot
Uses consolidated data files and streams data line by line for memory efficiency.
Each episode uses the entire dataset before starting a new episode.
"""

import os
import sys
import argparse
import logging
import logging.config
import traceback
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Iterator
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
import psutil
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration utilities
from utils.config_loader import load_training_config, get_logging_config, setup_console_logging, setup_file_logging, print_logging_config
from utils.archiver import TrainingArchiver, archive_before_training

# Import core components
from data.setup_data import DataProcessor
from environment import TradingEnvironment
from model import TradingModel, create_model_config
from reward_system import BALANCED_ENHANCED_CONFIG, CONSERVATIVE_ENHANCED_CONFIG, AGGRESSIVE_ENHANCED_CONFIG, SMALL_TRANSACTION_CONFIG, ULTRA_SMALL_TRANSACTION_CONFIG

class StreamingDataReader:
    """
    Streams data from a consolidated CSV file line by line for memory efficiency.
    """
    
    def __init__(self, file_path: str, lookback_window: int = 20, batch_size: int = 1000):
        """
        Initialize the streaming data reader.
        
        Args:
            file_path: Path to the consolidated CSV file
            lookback_window: Number of previous rows needed for feature calculation
            batch_size: Number of rows to read in each batch
        """
        self.file_path = Path(file_path)
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Consolidated file not found: {self.file_path}")
        
        # Read file info without loading all data
        self._analyze_file()
          # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"StreamingDataReader initialized for file: {self.file_path}")
        self.logger.info(f"File contains {self.total_rows} rows, {len(self.columns)} columns")
    
    def _analyze_file(self):
        """Analyze the file structure without loading all data"""
        # Read just the header and first few rows to understand structure
        # Use explicit column names based on known consolidated file format
        expected_columns = ['_temp_index', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        sample_df = pd.read_csv(
            self.file_path, 
            nrows=5, 
            header=0,
            names=expected_columns,
            skiprows=1  # Skip the original header
        )
        self.columns = list(sample_df.columns)
        
        # Count total rows efficiently
        with open(self.file_path, 'r') as f:
            self.total_rows = sum(1 for line in f) - 1  # Subtract 1 for header
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Analyzed file: {self.total_rows} rows, columns: {self.columns}")
    
    def get_episode_iterator(self) -> Iterator[pd.DataFrame]:
        """
        Returns an iterator that yields batches of data for training.
        Each complete iteration through this iterator constitutes one episode.
        """
        self.logger.info(f"Starting new episode iteration through {self.total_rows} rows")
          # Use pandas chunking to read the file in batches with explicit column names
        expected_columns = ['_temp_index', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        chunk_iter = pd.read_csv(
            self.file_path,
            header=0,
            names=expected_columns,
            skiprows=1,  # Skip the original header
            chunksize=self.batch_size,
            parse_dates=False  # We'll handle datetime conversion manually
        )
        
        accumulated_data = []
        rows_processed = 0
        
        for chunk in chunk_iter:
            # Standardize column names to match expected format
            chunk = self._standardize_column_names(chunk)
            
            # Accumulate data to ensure we have enough for lookback
            accumulated_data.append(chunk)
              # Dynamically calculate number of chunks to keep for lookback window
            # Keep enough chunks to ensure we always have sufficient lookback data
            max_chunks_needed = max(3, (self.lookback_window // self.batch_size) + 2)
            if len(accumulated_data) > max_chunks_needed:
                accumulated_data.pop(0)
            
            # Concatenate accumulated data
            if len(accumulated_data) == 1:
                batch_data = accumulated_data[0]
            else:
                batch_data = pd.concat(accumulated_data, ignore_index=False)
              # Only yield if we have enough data for lookback window
            if len(batch_data) >= self.lookback_window:
                rows_processed += len(chunk)
                self.logger.debug(f"Yielding batch: {len(chunk)} new rows, {len(batch_data)} total with lookback")
                yield batch_data
        
        self.logger.info(f"Episode completed: processed {rows_processed} rows")
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and convert timestamp to DatetimeIndex.
        Expects columns: ['_temp_index', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        """
        try:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Drop the temporary index column if it exists
            if '_temp_index' in df.columns:
                df = df.drop(columns=['_temp_index'])
            
            # Column mapping from lowercase to expected format
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'timestamp': 'timestamp'  # Keep timestamp as is for now
            }
            
            # Rename columns that exist
            existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_mappings)
            
            # Convert timestamp to datetime and set as index
            if 'timestamp' in df.columns:
                # Convert Unix timestamp to datetime
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    # Set timestamp as index
                    df = df.set_index('timestamp')
                except Exception as e:
                    self.logger.warning(f"Failed to convert timestamp column: {e}. Creating default datetime index.")
                    # Remove problematic timestamp column and create default index
                    df = df.drop(columns=['timestamp'])
                    df.index = pd.date_range(
                        start='2024-01-01', 
                        periods=len(df), 
                        freq='15min'
                    )
            elif df.index.name != 'timestamp' and not isinstance(df.index, pd.DatetimeIndex):
                # If there's no timestamp column but we have a numeric index,
                # assume the index contains timestamps or try to convert index
                if pd.api.types.is_numeric_dtype(df.index):
                    # Try to convert index to datetime assuming it's Unix timestamp
                    try:
                        df.index = pd.to_datetime(df.index, unit='s')
                    except:
                        # If conversion fails, create a simple datetime index
                        df.index = pd.date_range(
                            start='2024-01-01', 
                            periods=len(df), 
                            freq='15min'
                        )
                else:
                    # Create a default datetime index
                    df.index = pd.date_range(
                        start='2024-01-01', 
                        periods=len(df), 
                        freq='15min'
                    )
            
            # Ensure index name is set properly
            if df.index.name is None:
                df.index.name = 'timestamp'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Critical error in _standardize_column_names: {e}")
            # Last resort: create a minimal valid dataframe with datetime index
            df_clean = df.copy()
            if '_temp_index' in df_clean.columns:
                df_clean = df_clean.drop(columns=['_temp_index'])
            if 'timestamp' in df_clean.columns:
                df_clean = df_clean.drop(columns=['timestamp'])
            
            # Create datetime index
            df_clean.index = pd.date_range(
                start='2024-01-01', 
                periods=len(df_clean), 
                freq='15min'
            )
            df_clean.index.name = 'timestamp'
            
            return df_clean
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data file"""
        return {
            'file_path': str(self.file_path),
            'total_rows': self.total_rows,
            'columns': self.columns,
            'lookback_window': self.lookback_window,
            'batch_size': self.batch_size
        }

class Trainer:
    """
    Memory-efficient training manager that streams data from consolidated files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logging_config = get_logging_config(config)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized MemoryEfficientTrainingManager")
        print_logging_config(self.logging_config)
        
        # Print the loaded configuration to the log file
        self.logger.info("=== LOADED TRAINING CONFIGURATION ===")
        for key, value in self.config.items():
            # Format sensitive or complex values appropriately
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("=== END CONFIGURATION ===")
        self.logger.info("")
        
        # Initialize data processor (for feature engineering only)
        self.data_processor = DataProcessor(lookback_window=self.config.get('lookback_window', 20))
          # Training components - Initialize model and environments once
        self.model = None
        self.train_env = None
        self.test_env = None
        self.streaming_reader = None
        
        # Episode tracking
        self.current_episode = 0
        self.current_batch = 0
        self.total_episodes = self.config.get('total_episodes', 10)
        self.steps_per_episode = 0
        self.total_steps_trained = 0
          # Generic session names for consistent logging
        self.train_session_name = "training_session"
        self.test_session_name = "testing_session"
        
        self._log_memory_usage("Initialization")
        self.logger.info("MemoryEfficientTrainingManager initialized.")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"memory_efficient_training_{timestamp}.log"
        
        console_level = self.logging_config.get('console_log_level', 'INFO')
        setup_console_logging(console_level)
        
        file_level = self.logging_config.get('file_log_level', 'DEBUG')
        if file_level.upper() != 'DISABLED':
            setup_file_logging(str(log_file), file_level)
    
    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            self.logger.info(f"Memory usage {context}: {memory_mb:.1f} MB ({memory_percent:.1f}%)")            
            if memory_mb > 4000:
                self.logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
        except Exception as e:
            self.logger.debug(f"Could not get memory info: {e}")
    
    def _get_reward_config(self) -> Dict[str, Any]:
        """Get reward configuration based on strategy with enhanced exploration as default"""
        reward_strategy = self.config.get('reward_strategy', 'enhanced_exploration').lower()  # Default to enhanced exploration
        encourage_small_trades = self.config.get('encourage_small_trades', False)
        ultra_aggressive_small = self.config.get('ultra_aggressive_small_trades', False)

        # Enhanced exploration strategy (now default)
        if reward_strategy == 'enhanced_exploration' or reward_strategy == 'balanced':
            try:
                from enhanced_reward_configs import ENHANCED_EXPLORATION_CONFIG
                base_config = ENHANCED_EXPLORATION_CONFIG.copy()
                self.logger.info("🎯 Using ENHANCED_EXPLORATION_CONFIG for improved SHORT/LONG/CLOSE action balance.")
            except ImportError:
                self.logger.warning("ENHANCED_EXPLORATION_CONFIG not found, falling back to BALANCED_ENHANCED_CONFIG")
                base_config = BALANCED_ENHANCED_CONFIG.copy()
        elif ultra_aggressive_small:
            base_config = ULTRA_SMALL_TRANSACTION_CONFIG.copy()
            self.logger.info("Using ULTRA_SMALL_TRANSACTION_CONFIG for reward system.")
        elif encourage_small_trades:
            base_config = SMALL_TRANSACTION_CONFIG.copy()
            self.logger.info("Using SMALL_TRANSACTION_CONFIG for reward system.")
        elif reward_strategy == 'conservative':
            base_config = CONSERVATIVE_ENHANCED_CONFIG.copy()
            self.logger.info("Using CONSERVATIVE_ENHANCED_CONFIG for reward system.")
        elif reward_strategy == 'aggressive':
            base_config = AGGRESSIVE_ENHANCED_CONFIG.copy()
            self.logger.info("Using AGGRESSIVE_ENHANCED_CONFIG for reward system.")
        else:
            # Fallback - also use enhanced exploration
            try:
                from enhanced_reward_configs import ENHANCED_EXPLORATION_CONFIG
                base_config = ENHANCED_EXPLORATION_CONFIG.copy()
                self.logger.info("🎯 Using ENHANCED_EXPLORATION_CONFIG as fallback for improved action diversity.")
            except ImportError:
                base_config = BALANCED_ENHANCED_CONFIG.copy()
                self.logger.info("Using BALANCED_ENHANCED_CONFIG as final fallback.")
        
        # Apply reward detail logging and overrides
        base_config['log_rewards'] = self.config.get('log_reward_details', True)  # Enable by default for analysis
        reward_overrides = self.config.get('reward_overrides', {})
        base_config.update(reward_overrides)
        
        return base_config
    
    def initialize_streaming_reader(self, consolidated_file_path: str):
        """Initialize the streaming data reader"""
        self.logger.info(f"Initializing streaming reader for: {consolidated_file_path}")
        
        self.streaming_reader = StreamingDataReader(
            file_path=consolidated_file_path,
            lookback_window=self.config.get('lookback_window', 20),
            batch_size=self.config.get('streaming_batch_size', 1000)
        )
          # Log data information
        data_info = self.streaming_reader.get_data_info()
        self.logger.info(f"Data file info: {data_info['total_rows']} rows, {len(data_info['columns'])} columns")
        
        return self.streaming_reader
    def create_environments(self, train_data: pd.DataFrame, test_data: pd.DataFrame, episode_num: int, batch_num: int):
        """Create or update training and test environments from processed data with episode/batch tracking"""
        reward_config = self._get_reward_config()
        
        try:
            # Create training environment session with episode/batch info
            train_session = f"{self.train_session_name}_ep{episode_num:03d}_batch{batch_num:03d}"
            
            # Only create environments once, then update their data
            if self.train_env is None:
                self.logger.info("Creating training environment for the first time...")
                self.train_env = TradingEnvironment(
                    df=train_data,
                    initial_balance=self.config.get('initial_balance', 10000),
                    lookback_window=self.config.get('lookback_window', 20),
                    reward_config=reward_config,
                    trade_logger_session=train_session,
                    enable_trade_logging=True,
                    use_discretized_actions=True,  # Enable discrete actions for better learning
                    episode_num=episode_num,
                    batch_num=batch_num
                )
                self.logger.info("Training environment created successfully.")
            else:
                self.logger.debug(f"Updating training environment data for episode {episode_num}, batch {batch_num}")
                # Use the new update_data method to maintain model continuity
                self.train_env.update_data(train_data, episode_num, batch_num)
                # Update trade logger session
                if self.train_env.trade_logger:
                    self.train_env.trade_logger.session_name = train_session
                self.logger.debug("Training environment data updated.")
            
            # Create test environment session with episode/batch info
            test_session = f"{self.test_session_name}_ep{episode_num:03d}_batch{batch_num:03d}"
            
            if self.test_env is None:
                self.logger.info("Creating test environment for the first time...")
                self.test_env = TradingEnvironment(
                    df=test_data,
                    initial_balance=self.config.get('initial_balance', 10000),
                    lookback_window=self.config.get('lookback_window', 20),
                    reward_config=reward_config,
                    trade_logger_session=test_session,
                    enable_trade_logging=True,
                    use_discretized_actions=True,  # Enable discrete actions for better learning
                    episode_num=episode_num,
                    batch_num=batch_num
                )
                self.logger.info("Test environment created successfully.")
            else:
                self.logger.debug(f"Updating test environment data for episode {episode_num}, batch {batch_num}")
                # Use the new update_data method to maintain model continuity
                self.test_env.update_data(test_data, episode_num, batch_num)                # Update trade logger session
                if self.test_env.trade_logger:
                    self.test_env.trade_logger.session_name = test_session
                self.logger.debug("Test environment data updated.")
            
            self.logger.debug(f"Environments ready - Train: {len(train_data)} steps, Test: {len(test_data)} steps")
            
        except Exception as e:
            self.logger.error(f"Failed to create/update environments: {e}")
            raise
    
    def run_complete_training(self, consolidated_file_path: str) -> Dict[str, Any]:
        """
        Run the complete memory-efficient training pipeline.
        
        Args:
            consolidated_file_path: Path to the consolidated data file
            
        Returns:
            Dictionary with final training metrics
        """
        self.logger.info("Starting memory-efficient training pipeline")

        # Initialize streaming reader once for all episodes
        self.initialize_streaming_reader(consolidated_file_path)

        # Load the entire dataset once
        self.logger.info(f"Loading entire dataset from {consolidated_file_path}...")
        full_df = pd.read_csv(consolidated_file_path)
        self.logger.info(f"Loaded {len(full_df)} rows.")

        # Process the full dataset once (fit scalers on first processing)
        self.logger.info("Processing full dataset for all episodes...")
        train_df, test_df = self._process_batch_data(full_df, is_first_batch=True)
        
        if train_df is None or test_df is None:
            self.logger.error("Failed to process dataset. Aborting training.")
            return {"success": False, "error": "Data processing failed"}

        # Create environments once at the start
        self.logger.info("Creating environments for training session...")
        self._create_or_update_environments(train_df, test_df, is_first_batch=True)

        # Create model ONCE for the entire training session
        self.logger.info("Creating model for the entire training session...")
        self._ensure_single_model_creation()

        # Loop through episodes - reusing the same model and environments
        for episode_num in range(1, self.total_episodes + 1):
            self.current_episode = episode_num
            self.logger.info(f"Starting Episode {self.current_episode}/{self.total_episodes}")

            # Update environment session tracking (but reuse same environment objects)
            self._update_environment_sessions()

            # Train the model on the dataset for this episode
            steps_for_episode = self.config.get('steps_per_episode', 500000)  # Use config value
            self.logger.info(f"Training model for episode {episode_num} with {steps_for_episode} steps...")
            
            # Train the model (it will use the existing environment data)
            self.model.train(total_timesteps=steps_for_episode)
            self.total_steps_trained += steps_for_episode

            # Evaluate the model at the end of the episode
            self._evaluate_model()

            self.logger.info(f"Episode {self.current_episode} completed. Total steps trained: {self.total_steps_trained}")

        self.logger.info("Training pipeline finished.")
        if self.model:
            final_model_path = self.config.get('final_model_path', 'models/memory_efficient_model_final')
            self.model.save_model(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")

        return {"success": True, "total_steps_trained": self.total_steps_trained}

    def _process_batch_data(self, batch_data: pd.DataFrame, is_first_batch: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process a batch of raw data into features and split into train/test.
        
        Args:
            batch_data: Raw OHLCV data batch
            is_first_batch: Whether this is the first batch (for fitting scalers)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            # Standardize column names before processing
            if not all(col in batch_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                self.logger.debug("Standardizing column names for the batch.")
                # Ensure streaming_reader is initialized
                if not hasattr(self, 'streaming_reader') or self.streaming_reader is None:
                    self.initialize_streaming_reader(self.config['consolidated_file'])
                batch_data = self.streaming_reader._standardize_column_names(batch_data)
            
            # Process features using existing data processor
            train_data, test_data = self.data_processor.prepare_data(
                batch_data,
                funding_rates=None,
                train_ratio=self.config.get('train_ratio', 0.8),
                fit_scalers=is_first_batch  # Fit scalers only on the first batch
            )
            
            if train_data is None or test_data is None or train_data.empty or test_data.empty:
                self.logger.warning("Data processing returned empty result")
                return None, None
            
            self.logger.debug(f"Processed batch: {len(batch_data)} -> train: {len(train_data)}, test: {len(test_data)}")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error processing data batch: {e}")
            self.logger.debug(f"Batch info - Shape: {batch_data.shape}, Columns: {batch_data.columns.tolist()}")
            return None, None

    def _create_or_update_environments(self, train_data: pd.DataFrame, test_data: pd.DataFrame, is_first_batch: bool = False):
        """
        Create or update the training and testing environments with the provided data.
        
        Args:
            train_data: Processed training data
            test_data: Processed testing data
            is_first_batch: Flag indicating if this is the first batch (for initialization)
        """
        reward_config = self._get_reward_config()
        
        try:
            # Create training environment session
            train_session = f"{self.train_session_name}_ep{self.current_episode:03d}_batch{self.current_batch:03d}"
            
            if self.train_env is None:
                self.logger.info("Creating training environment for the first time...")
                self.train_env = TradingEnvironment(
                    df=train_data,
                    initial_balance=self.config.get('initial_balance', 10000),
                    lookback_window=self.config.get('lookback_window', 20),
                    reward_config=reward_config,
                    trade_logger_session=train_session,
                    enable_trade_logging=True,
                    use_discretized_actions=True,  # Enable discrete actions for better learning
                    episode_num=self.current_episode,
                    batch_num=self.current_batch
                )
                self.logger.info("Training environment created successfully.")
            else:
                self.logger.debug(f"Updating training environment data for episode {self.current_episode}, batch {self.current_batch}")
                # Use the new update_data method to maintain model continuity
                self.train_env.update_data(train_data, self.current_episode, self.current_batch)
                # Update trade logger session
                if self.train_env.trade_logger:
                    self.train_env.trade_logger.session_name = train_session
                self.logger.debug("Training environment data updated.")
            
            # Create test environment session
            test_session = f"{self.test_session_name}_ep{self.current_episode:03d}_batch{self.current_batch:03d}"
            
            if self.test_env is None:
                self.logger.info("Creating test environment for the first time...")
                self.test_env = TradingEnvironment(
                    df=test_data,
                    initial_balance=self.config.get('initial_balance', 10000),
                    lookback_window=self.config.get('lookback_window', 20),
                    reward_config=reward_config,
                    trade_logger_session=test_session,
                    enable_trade_logging=True,
                    use_discretized_actions=True,  # Enable discrete actions for better learning
                    episode_num=self.current_episode,
                    batch_num=self.current_batch
                )
                self.logger.info("Test environment created successfully.")
            else:
                self.logger.debug(f"Updating test environment data for episode {self.current_episode}, batch {self.current_batch}")
                # Use the new update_data method to maintain model continuity
                self.test_env.update_data(test_data, self.current_episode, self.current_batch)
                # Update trade logger session
                if self.test_env.trade_logger:
                    self.test_env.trade_logger.session_name = test_session
                self.logger.debug("Test environment data updated.")            
            self.logger.debug(f"Environments ready - Train: {len(train_data)} steps, Test: {len(test_data)} steps")
            
        except Exception as e:
            self.logger.error(f"Failed to create/update environments: {e}")
            raise
    
    def _train_on_batch(self):
        """Train the model on the current batch of data"""
        if self.model is None:
            self.logger.error("Model is not initialized, cannot train")
            return
        
        try:
            # Number of steps to train on this batch/episode
            train_steps = self.config.get('steps_per_episode', 500000)  # Use steps_per_episode from config
            
            self.logger.info(f"Training model on current batch for {train_steps} steps...")
            self.model.train(total_timesteps=train_steps)
            
            self.total_steps_trained += train_steps
            self.logger.info(f"Model trained on {train_steps} steps. Total steps trained: {self.total_steps_trained}")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise
    
    def _evaluate_model(self):
        """Evaluate the model on the test environment"""
        if self.model is None or self.test_env is None:
            self.logger.warning("Model or test environment not available for evaluation")
            return
        
        try:
            self.logger.info("Evaluating model on the test set...")
            eval_metrics = self.model.evaluate(
                eval_env=self.test_env,
                n_eval_episodes=1,
                deterministic=True            )            
            mean_reward = eval_metrics.get('mean_reward', 0.0)
            self.logger.info(f"Evaluation completed. Mean reward: {mean_reward:.4f}")
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
            
    def cleanup(self):
        """
        Clean up resources and close file handles properly.
        """
        self.logger.info("Starting cleanup process...")
        
        # Save final session logs and archives
        try:
            self.logger.info("Archiving final training session logs...")
            session_name = f"final_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save and archive model if it exists
            if hasattr(self, 'model') and self.model is not None:
                self.logger.info("Saving final model with archiving...")
                final_model_path = Path(self.config.get('final_model_path', 'models/final_model'))
                self.model.archive_and_save_model(final_model_path, create_archive=True)
                
        except Exception as e:
            self.logger.warning(f"Failed to archive final session: {e}")
        
        # Close environments if they exist
        if hasattr(self, 'train_env') and self.train_env:
            try:
                # Save and close trade loggers
                if hasattr(self.train_env, 'trade_logger') and self.train_env.trade_logger:
                    self.logger.info("Saving and closing train environment trade logger...")
                    self.train_env.trade_logger.save_session("training_session")
                    
                # Close any trade tracers  
                if hasattr(self.train_env, 'trade_tracer') and self.train_env.trade_tracer:
                    self.logger.info("Closing train environment trade tracer...")
                    # TradeTracer uses append mode, no explicit close needed
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up train environment: {e}")
        
        if hasattr(self, 'test_env') and self.test_env:
            try:
                # Save and close trade loggers
                if hasattr(self.test_env, 'trade_logger') and self.test_env.trade_logger:
                    self.logger.info("Saving and closing test environment trade logger...")
                    self.test_env.trade_logger.save_session("testing_session")
                    
                # Close any trade tracers
                if hasattr(self.test_env, 'trade_tracer') and self.test_env.trade_tracer:
                    self.logger.info("Closing test environment trade tracer...")
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up test environment: {e}")
        
        self.logger.info("Cleanup process finished.")
    
    def _ensure_single_model_creation(self):
        """Ensure we create only one model for the entire training session with enhanced exploration"""
        if self.model is None:
            self.logger.info("Creating single TradingModel with enhanced exploration for entire training session...")
            model_name = f"enhanced_exploration_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.model = TradingModel(
                env=self.train_env,
                model_name=model_name,
                logging_config=self.logging_config,
                device=self.config.get('device', 'auto')
            )
            
            self.logger.info("✅ Enhanced Exploration TradingModel created - optimized for SHORT/LONG/CLOSE balance")
        else:
            self.logger.info("✅ Reusing existing Enhanced TradingModel (already created for this session)")

    def _update_environment_sessions(self):
        """Update environment session names for current episode without recreating environments"""
        try:
            # Update training environment session
            if self.train_env and hasattr(self.train_env, 'trade_logger') and self.train_env.trade_logger:
                train_session = f"{self.train_session_name}_ep{self.current_episode:03d}"
                self.train_env.trade_logger.session_name = train_session
                self.logger.debug(f"Updated train environment session to: {train_session}")
            
            # Update test environment session
            if self.test_env and hasattr(self.test_env, 'trade_logger') and self.test_env.trade_logger:
                test_session = f"{self.test_session_name}_ep{self.current_episode:03d}"
                self.test_env.trade_logger.session_name = test_session
                self.logger.debug(f"Updated test environment session to: {test_session}")
                
        except Exception as e:
            self.logger.warning(f"Failed to update environment sessions: {e}")
def main():
    """Main function to run the training process"""
    parser = argparse.ArgumentParser(description="Memory-Efficient RL Trading Bot Trainer")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the configuration file")
    parser.add_argument("--logging-config", type=str, default="config/logging_config.json", help="Path to the logging configuration file")
    parser.add_argument("--default", action="store_true", help="Use real market data with enhanced exploration and automatic archiving.")
    parser.add_argument("--training", action="store_true", help="Use synthetic balanced training data with enhanced exploration for initial RL learning.")
    parser.add_argument("--no-archive", action="store_true", help="Skip archiving previous logs and models")
    args = parser.parse_args()

    # Archive previous training session and clean workspace before starting new one (unless disabled)
    # When using --default, always enable archiving unless explicitly disabled with --no-archive
    should_archive = not args.no_archive
    if args.default:
        print("🔧 Using default settings with automatic archiving enabled")
        should_archive = True  # Force archiving when using --default
        
    if should_archive:
        try:
            print("🗄️  Archiving previous training session and cleaning workspace...")
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create archiver and clean workspace
            from utils.archiver import TrainingArchiver
            archiver = TrainingArchiver()
            archive_path = archiver.prepare_clean_workspace(backup_first=True)
            
            if archive_path:
                print(f"✅ Previous session archived to: {archive_path}")
                print("✅ Workspace cleaned - fresh logs and models directories created")
            else:
                print("ℹ️  No previous session found to archive")
                print("✅ Fresh logs and models directories ensured")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to archive previous session: {e}")
            print("Continuing with training...")

    # Setup logging
    logging_config_path = Path(args.logging_config)
    if logging_config_path.exists():
        with open(logging_config_path, 'rt') as f:
            log_config = json.load(f)
        logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    trainer = None
    try:
        logger.info("Initializing Trainer...")        # Load configuration
        config = load_training_config(args.config)
          # Handle --training flag to use synthetic balanced data
        if args.training:
            synthetic_data_path = "data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv"
            if Path(synthetic_data_path).exists():
                original_file = config['consolidated_file']
                config['consolidated_file'] = synthetic_data_path
                # Ensure enhanced exploration is used with training data
                config['reward_strategy'] = 'enhanced_exploration'
                print("🎯 TRAINING MODE: Using synthetic balanced training data + enhanced exploration")
                print(f"   📁 Original data: {original_file}")
                print(f"   🔄 Training data: {synthetic_data_path}")
                print("   📈 This data contains balanced LONG, SHORT, and HOLD patterns")
                print("   🚀 Enhanced exploration enabled for improved action diversity!")
                print("   🎯 Perfect for teaching your RL agent all three action types!")
            else:
                print(f"❌ Warning: Synthetic training data not found at {synthetic_data_path}")
                print("   Please run generate_simple_synthetic_data.py first!")
                print("   Falling back to original data with enhanced exploration...")
                config['reward_strategy'] = 'enhanced_exploration'
        
        # Handle --default flag settings
        if args.default:
            # Ensure enhanced exploration is used with default data
            config['reward_strategy'] = 'enhanced_exploration'
            print("🔧 DEFAULT MODE: Using real data + enhanced exploration + archiving")
            print("   🚀 Enhanced exploration enabled for improved SHORT/LONG/CLOSE balance!")
        
        # Print the loaded configuration to the log
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
        
        trainer = Trainer(config=config)
        logger.info("Trainer initialized. Starting training run...")
        trainer.run_complete_training(config['consolidated_file'])
        logger.info("Training run completed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        if trainer:
            logger.info("Cleaning up resources...")
            trainer.cleanup()
            logger.info("Resources cleaned up.")

if __name__ == '__main__':
    main()
