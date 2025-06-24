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
import traceback
import time
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
        
        # Setup logging        self.logger = logging.getLogger(__name__)
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

class MemoryEfficientTrainingManager:
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
        """Get reward configuration based on strategy"""
        reward_strategy = self.config.get('reward_strategy', 'balanced').lower()
        encourage_small_trades = self.config.get('encourage_small_trades', False)
        ultra_aggressive_small = self.config.get('ultra_aggressive_small_trades', False)

        if ultra_aggressive_small:
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
            base_config = BALANCED_ENHANCED_CONFIG.copy()
            self.logger.info("Using BALANCED_ENHANCED_CONFIG for reward system.")
        
        base_config['log_rewards'] = self.config.get('log_reward_details', False)
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
    
    def process_data_batch(self, raw_batch: pd.DataFrame, fit_scalers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process a batch of raw data into features and split into train/test.
        
        Args:
            raw_batch: Raw OHLCV data batch
            fit_scalers: Whether to fit scalers (only for first batch)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            # Standardize column names before processing
            if not all(col in raw_batch.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                self.logger.debug("Standardizing column names for the batch.")
                # Ensure streaming_reader is initialized
                if not hasattr(self, 'streaming_reader') or self.streaming_reader is None:
                    self.initialize_streaming_reader(self.config['consolidated_file'])
                raw_batch = self.streaming_reader._standardize_column_names(raw_batch)
            
            # Process features using existing data processor
            train_data, test_data = self.data_processor.prepare_data(
                raw_batch,
                funding_rates=None,
                train_ratio=self.config.get('train_ratio', 0.8)
            )
            
            if train_data is None or test_data is None or train_data.empty or test_data.empty:
                self.logger.warning("Data processing returned empty result")
                return None, None
            
            self.logger.debug(f"Processed batch: {len(raw_batch)} -> train: {len(train_data)}, test: {len(test_data)}")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error processing data batch: {e}")
            self.logger.debug(f"Batch info - Shape: {raw_batch.shape}, Columns: {raw_batch.columns.tolist()}")
            return None, None

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
                self.test_env.update_data(test_data, episode_num, batch_num)
                # Update trade logger session
                if self.test_env.trade_logger:
                    self.test_env.trade_logger.session_name = test_session
                self.logger.debug("Test environment data updated.")
            
            self.logger.debug(f"Environments ready - Train: {len(train_data)} steps, Test: {len(test_data)} steps")
            
        except Exception as e:
            self.logger.error(f"Failed to create/update environments: {e}")
            raise
    def initialize_model(self, env):
        """Initialize the trading model once and reuse it throughout training"""
        if self.model is None:
            self.logger.info("Creating new TradingModel (will be reused for all episodes and batches)...")
            model_name = f"memory_efficient_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.model = TradingModel(
                env=env,
                model_name=model_name,
                logging_config=self.logging_config,
                device=self.config.get('device', 'auto')
            )
            
            self.logger.info("TradingModel created successfully. This model will be reused throughout training.")
        else:
            # Model already exists, just ensure it's using the current environment
            # The environment data has been updated, but the model doesn't need to be recreated
            self.logger.debug("Reusing existing model (environment data has been updated)")
            # Only update the environment reference if it's a different environment object
            if self.model.model.get_env() != env:
                self.model.model.set_env(env)
                self.logger.debug("Model environment reference updated.")
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Run one complete episode through the entire dataset.
        
        Args:
            episode_num: Current episode number
            
        Returns:
            Dictionary with episode metrics
        """
        self.logger.info(f"Starting Episode {episode_num}/{self.total_episodes}")
        episode_start_time = time.time()
        
        episode_metrics = {
            'episode': episode_num,
            'total_steps': 0,
            'training_steps': 0,
            'mean_reward': 0.0,
            'success': False
        }
        
        try:
            # 1. Collect all data for the episode
            self.logger.info("Loading all data for the episode...")
            episode_iterator = self.streaming_reader.get_episode_iterator()
            all_batches = [raw_batch for raw_batch in episode_iterator]
            
            if not all_batches:
                self.logger.warning("No data loaded for this episode. Skipping.")
                return episode_metrics

            # Concatenate all batches into a single DataFrame
            full_episode_data = pd.concat(all_batches, ignore_index=False)
            self.logger.info(f"Loaded {len(full_episode_data)} total rows for the episode.")
            self._log_memory_usage("After loading full episode data")

            # 2. Process the entire dataset
            self.logger.info("Processing full episode data...")
            train_data, test_data = self.process_data_batch(full_episode_data, fit_scalers=True)
            
            if train_data is None or test_data is None or train_data.empty:
                self.logger.error("Failed to process episode data. Skipping episode.")
                return episode_metrics

            # 3. Create environments once for the entire episode
            self.logger.info("Creating environments for the full episode...")
            self.create_environments(train_data, test_data, episode_num, batch_num=0) # batch_num=0 for the whole episode

            # 4. Initialize or update model
            self.initialize_model(self.train_env)
            
            # 5. Train on the entire episode data
            training_steps = len(train_data) # Train for the full length of the training data
            self.logger.info(f"Training model on full episode for {training_steps} steps...")
            
            self.model.train(total_timesteps=training_steps)
            
            # 6. Evaluate on the test set for the episode
            self.logger.info("Evaluating model on the episode's test set...")
            eval_metrics = self.model.evaluate(
                eval_env=self.test_env,
                n_eval_episodes=1,
                deterministic=True
            )
            
            mean_reward = eval_metrics.get('mean_reward', 0.0)
            
            episode_metrics['total_steps'] = len(full_episode_data)
            episode_metrics['training_steps'] = training_steps
            episode_metrics['mean_reward'] = mean_reward
            episode_metrics['success'] = True
            
            self.logger.info(f"Episode {episode_num} completed successfully")
            self.logger.info(f"  Total steps: {episode_metrics['total_steps']}")
            self.logger.info(f"  Training steps: {episode_metrics['training_steps']}")
            self.logger.info(f"  Mean reward: {episode_metrics['mean_reward']:.4f}")

        except Exception as e:
            self.logger.error(f"Episode {episode_num} failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        episode_duration = time.time() - episode_start_time
        episode_metrics['duration_seconds'] = episode_duration
        self.logger.info(f"Episode {episode_num} finished in {episode_duration:.1f} seconds")
        
        return episode_metrics
    
    def run_complete_training(self, consolidated_file_path: str) -> Dict[str, Any]:
        """
        Run the complete memory-efficient training pipeline.
        
        Args:
            consolidated_file_path: Path to the consolidated data file
            
        Returns:
            Dictionary with final training metrics
        """
        self.logger.info("Starting memory-efficient training pipeline")

        # Load the entire dataset for the episode
        self.logger.info(f"Loading entire dataset from {consolidated_file_path} for the episode...")
        full_df = pd.read_csv(consolidated_file_path)
        self.logger.info(f"Loaded {len(full_df)} rows.")

        all_episode_metrics = []
        successful_episodes = 0

        # Loop through episodes
        for episode_num in range(1, self.total_episodes + 1):
            self.current_episode = episode_num
            self.logger.info(f"Starting Episode {self.current_episode}/{self.total_episodes}")

            # Process the full dataset
            train_df, test_df = self._process_batch_data(full_df, is_first_batch=True)

            if train_df is None or test_df is None:
                self.logger.error(f"Skipping episode {episode_num} due to data processing error.")
                continue

            # Create environments once per episode
            self._create_or_update_environments(train_df, test_df, is_first_batch=True)

            # Create model if it doesn't exist
            if self.model is None:
                self._create_model()
            else:
                # Ensure the model is using the latest environment
                self.model.set_environment(self.train_env)

            # Train the model on the entire dataset for a specified number of steps
            self._train_on_batch()

            # Evaluate the model at the end of the episode
            eval_metrics = self._evaluate_model()

            if eval_metrics:
                all_episode_metrics.append(eval_metrics)
                successful_episodes += 1

            self.logger.info(f"Episode {self.current_episode} completed. Total steps trained: {self.total_steps_trained}")

        self.logger.info("Training pipeline finished.")
        if self.model:
            final_model_path = self.config.get('final_model_path', 'models/memory_efficient_model_final')
            self.model.save_model(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")

        # Aggregate final metrics
        final_metrics = {
            'total_episodes': self.total_episodes,
            'successful_episodes': successful_episodes,
            'all_episode_metrics': all_episode_metrics,
            'trade_statistics': {}
        }

        if all_episode_metrics:
            rewards = [m.get('mean_reward', 0.0) for m in all_episode_metrics if m]
            if rewards:
                final_metrics['mean_episode_reward'] = np.mean(rewards)
                final_metrics['best_episode_reward'] = np.max(rewards)
            final_metrics['final_evaluation'] = all_episode_metrics[-1] if all_episode_metrics else {}

        # Get trade statistics from the last evaluation
        if self.test_env:
            final_metrics['trade_statistics'] = self.test_env.get_episode_statistics()

        return final_metrics

    def _evaluate_model(self) -> Optional[Dict[str, Any]]:
        """Evaluate the model on the test environment"""
        if self.model is None or self.test_env is None:
            self.logger.warning("Model or test environment not available for evaluation")
            return None
        
        try:
            self.logger.info("Evaluating model on the test set...")
            eval_metrics = self.model.evaluate(
                eval_env=self.test_env,
                n_eval_episodes=1,
                deterministic=True
            )
            
            mean_reward = eval_metrics.get('mean_reward', 0.0)
            self.logger.info(f"Evaluation completed. Mean reward: {mean_reward:.4f}")
            return eval_metrics
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            return None

    def cleanup(self):
        """
        Clean up resources and close file handles properly.
        """
        self.logger.info("Starting cleanup process...")
        
        # Close environments if they exist
        if hasattr(self, 'train_env') and self.train_env:
            try:
                # Close any trade loggers
                if hasattr(self.train_env, 'trade_logger') and self.train_env.trade_logger:
                    self.logger.info("Closing train environment trade logger...")
                    # TradeLogger uses append mode, no explicit close needed
                    
                # Close any trade tracers  
                if hasattr(self.train_env, 'trade_tracer') and self.train_env.trade_tracer:
                    self.logger.info("Closing train environment trade tracer...")
                    # TradeTracer uses append mode, no explicit close needed
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up train environment: {e}")
        
        if hasattr(self, 'test_env') and self.test_env:
            try:
                # Close any trade loggers
                if hasattr(self.test_env, 'trade_logger') and self.test_env.trade_logger:
                    self.logger.info("Closing test environment trade logger...")
                    
                # Close any trade tracers
                if hasattr(self.test_env, 'trade_tracer') and self.test_env.trade_tracer:
                    self.logger.info("Closing test environment trade tracer...")
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up test environment: {e}")
        
        # Close all logging handlers
        self.logger.info("Closing logging handlers...")
        close_all_logging_handlers()
        
        self.logger.info("Cleanup completed successfully")

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all relevant libraries.
    
    Args:
        seed: Random seed value to use
    """
    import random
    import numpy as np
    import torch
    
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed} for reproducibility")


def configure_device(device_preference: str = 'auto') -> str:
    """
    Configure and return the device to use for training.
    
    Args:
        device_preference: User preference - 'auto', 'cpu', 'cuda', or 'gpu'
        
    Returns:
        Device string to use ('cpu' or 'cuda')
    """
    import torch
    
    # Normalize device preference
    if device_preference.lower() in ['gpu', 'cuda']:
        device_preference = 'cuda'
    elif device_preference.lower() == 'cpu':
        device_preference = 'cpu'
    else:  # 'auto'
        device_preference = 'auto'
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if device_preference == 'auto':
        # Auto-detect best device
        if cuda_available:
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown GPU'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            print(f"🚀 Auto-detected CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            print("💻 Auto-detected CPU (CUDA not available)")
    elif device_preference == 'cuda':
        # User requested CUDA
        if cuda_available:
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown GPU'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            print(f"🚀 Using requested CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    else:
        # User requested CPU
        device = 'cpu'
        print("💻 Using requested CPU device")
    
    # Set PyTorch device
    torch.set_default_device(device)
    
    # Additional CUDA optimizations if using GPU
    if device == 'cuda':
        # Enable optimizations for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache to start fresh
        torch.cuda.empty_cache()
        
        print(f"✅ Device configured: {device.upper()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
    else:
        print(f"✅ Device configured: {device.upper()}")
        cpu_count = torch.get_num_threads()
        print(f"   CPU Threads: {cpu_count}")
    
    return device


def close_all_logging_handlers():
    """
    Close all logging handlers to release file locks before archiving.
    """
    logger = logging.getLogger()
    handlers_to_remove = []
    
    # Collect file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)
    
    # Close and remove file handlers
    for handler in handlers_to_remove:
        handler.close()
        logger.removeHandler(handler)
    
    # Also check and close handlers on specific loggers
    for name in logging.Logger.manager.loggerDict:
        specific_logger = logging.getLogger(name)
        if hasattr(specific_logger, 'handlers'):
            handlers_to_remove = []
            for handler in specific_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                handler.close()
                specific_logger.removeHandler(handler)

def archive_existing_logs_and_models():
    """
    Archive existing logs and models before starting new training.
    Creates timestamped archive folders to preserve previous runs.
    """
    # Close all logging handlers first to release file locks
    print("🔒 Closing logging handlers to prepare for archiving...")
    close_all_logging_handlers()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = Path("archive") / f"training_run_{timestamp}"
    
    # Directories to archive
    dirs_to_archive = [
        ("logs", "logs"),
        ("models", "models"),
        ("tensorboard", "logs/tensorboard")
    ]
    
    archived_something = False
    
    for dir_name, source_path in dirs_to_archive:
        source = Path(source_path)
        if source.exists() and any(source.iterdir()):  # Directory exists and is not empty
            archive_dir = archive_base / dir_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Move contents to archive with error handling
            for item in source.iterdir():
                if item.is_file() or item.is_dir():
                    try:
                        shutil.move(str(item), str(archive_dir / item.name))
                        archived_something = True
                    except PermissionError as e:
                        print(f"⚠️  Could not move {item}: {e}")
                        print(f"   File may still be in use. Skipping: {item}")
                    except Exception as e:
                        print(f"⚠️  Error moving {item}: {e}")
    
    if archived_something:
        print(f"📦 Archived previous training data to: {archive_base}")
    else:
        print("📦 No previous training data found to archive")
    
    return archive_base if archived_something else None

def main():
    """Main function to run the memory-efficient training pipeline"""
    load_dotenv()
    
    # Archive existing logs and models before starting
    print("🗃️  Archiving previous training data...")
    archive_existing_logs_and_models()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    parser = argparse.ArgumentParser(
        description='Memory-Efficient RL Trading Bot Training',
        epilog='''
Examples:
  # Quick test run with default settings (CPU, 1 episode, ultra-aggressive small trades):
  python train_memory_efficient.py --consolidated-file data.csv --default
  
  # Full training run with GPU:
  python train_memory_efficient.py --consolidated-file data.csv --total-episodes 10 --device cuda
  
  # CPU training with custom settings:
  python train_memory_efficient.py --consolidated-file data.csv --total-episodes 5 --device cpu --verbose        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--consolidated-file', type=str, required=False,
                       help='Path to consolidated data file (optional with --default)')
    parser.add_argument('--total-episodes', type=int, default=10, 
                       help='Total number of episodes (full dataset passes)')
    parser.add_argument('--steps-per-batch', type=int, default=1000, 
                       help='Training steps per data batch')
    parser.add_argument('--streaming-batch-size', type=int, default=1000, 
                       help='Number of rows to read per streaming batch')
    parser.add_argument('--initial-balance', type=float, default=10000, 
                       help='Initial trading balance')
    parser.add_argument('--lookback-window', type=int, default=20, 
                       help='Lookback window for features')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Train/test split ratio for each batch')
    parser.add_argument('--reward-strategy', type=str, default='balanced', 
                       choices=['balanced', 'conservative', 'aggressive'],
                       help='Reward strategy')
    parser.add_argument('--encourage-small-trades', action='store_true', 
                       help='Use small transaction reward config')
    parser.add_argument('--ultra-aggressive-small-trades', action='store_true', 
                       help='Use ultra-aggressive small transaction reward config')
    parser.add_argument('--log-reward-details', action='store_true', 
                       help='Enable detailed reward logging')
    parser.add_argument('--save-model-every-episodes', type=int, default=5, 
                       help='Save model after every N episodes')
    parser.add_argument('--final-eval-episodes', type=int, default=10, 
                       help='Number of episodes for final evaluation')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'gpu'],
                       help='Device to use for training: auto (detect best), cpu, cuda/gpu')
    parser.add_argument('--default', action='store_true',
                       help='Run with default settings optimized for quick testing (forces CPU, 1 episode, ultra-aggressive small trades)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Handle --default option with predefined settings
    if args.default:
        print("🚀 Using --default mode with optimized settings for quick testing")
        
        # Check if consolidated file is provided, if not, use the specified default
        if not args.consolidated_file:
            args.consolidated_file = r".\data\processed\BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv"
            print(f"   Using default consolidated file: {args.consolidated_file}")
        
        # Override specific settings for default mode
        args.total_episodes = 1
        args.steps_per_batch = 10000
        args.random_seed = 123
        args.verbose = True
        args.ultra_aggressive_small_trades = True
        args.device = 'cpu'  # Force CPU usage
        
        print("   Default settings applied:")
        print(f"     - Device: CPU (forced)")
        print(f"     - Episodes: 1")
        print(f"     - Steps per batch: 10,000")
        print(f"     - Random seed: 123")
        print(f"     - Verbose: Enabled")
        print(f"     - Ultra-aggressive small trades: Enabled")
        print()
    
    # Validate that consolidated file is provided (either via argument or default)
    if not args.consolidated_file:
        parser.error("--consolidated-file is required unless using --default mode")
    
    # Convert argparse namespace directly into a dictionary
    config = vars(args)    # Configure device based on user preference
    actual_device = configure_device(config['device'])
    config['device'] = actual_device  # Store the actual device used
    
    # Override default seed with user-provided seed
    if config['random_seed'] != 42:
        set_random_seeds(config['random_seed'])
    
    # Configure device for training
    device = configure_device(config['device'])
    config['device'] = device  # Store the actual device used
    
    # Verify consolidated file exists
    if not Path(config['consolidated_file']).exists():
        print(f"❌ Consolidated file not found: {config['consolidated_file']}")
        print("Please run consolidate_data.py first to create a consolidated file.")
        sys.exit(1)
    
    print("🧠 Memory-Efficient RL Trading Bot Training")
    print("=" * 50)
    if config.get('default', False):
        print("🚀 RUNNING IN DEFAULT MODE (Quick Testing)")
        print("=" * 50)
    print(f"Consolidated File: {config['consolidated_file']}")
    print(f"Device: {config['device'].upper()}")
    print(f"Total Episodes: {config['total_episodes']}")
    print(f"Steps per Batch: {config['steps_per_batch']}")
    print(f"Streaming Batch Size: {config['streaming_batch_size']}")
    print(f"Initial Balance: ${config['initial_balance']:,.2f}")
    print(f"Lookback Window: {config['lookback_window']}")
    print(f"Train Ratio: {config['train_ratio']}")
    print(f"Reward Strategy: {config['reward_strategy']}")
    if config['encourage_small_trades']:
        print("Small Trades: Encouraged")
    if config['ultra_aggressive_small_trades']:
        print("Ultra-Aggressive Small Trades: Enabled")
    print("=" * 50)
    
    # Add additional configuration for logging
    config.update({
        'console_log_level': 'DEBUG' if config['verbose'] else 'INFO',
        'file_log_level': 'DEBUG',
        'trade_logging': True,
        'trade_tracing': True,
        'tensorboard_logging': True,
        'max_eval_batches': 5  # For improved final evaluation
    })
    
    try:
        # Archive existing logs and models
        archive_existing_logs_and_models()
        
        # Initialize training manager
        print("🔧 Initializing memory-efficient training manager...")
        trainer = MemoryEfficientTrainingManager(config)
        
        # Run training
        print("🚀 Starting training pipeline...")
        start_time = time.time()
        
        final_metrics = trainer.run_complete_training(config['consolidated_file'])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Clean up resources before reporting results
        print("🧹 Cleaning up resources...")
        trainer.cleanup()
        
        # Print results
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Episodes: {final_metrics.get('successful_episodes', 0)}/{final_metrics.get('total_episodes', 0)}")
        
        if 'mean_episode_reward' in final_metrics:
            print(f"Mean Episode Reward: {final_metrics['mean_episode_reward']:.4f}")
            print(f"Best Episode Reward: {final_metrics['best_episode_reward']:.4f}")
        
        if 'final_evaluation' in final_metrics and 'mean_reward' in final_metrics['final_evaluation']:
            print(f"Final Evaluation Reward: {final_metrics['final_evaluation']['mean_reward']:.4f}")
        
        # Display trade statistics
        if 'trade_statistics' in final_metrics and final_metrics['trade_statistics']:
            trade_stats = final_metrics['trade_statistics']
            print("\n📊 TRADE STATISTICS:")
            
            if 'final_balance' in trade_stats:
                print(f"Final Balance: ${trade_stats['final_balance']:.2f}")
            if 'total_return' in trade_stats:
                print(f"Total Return: {trade_stats['total_return']:.2%}")
        try:
            self.logger.info("Evaluating model on the test set...")
            eval_metrics = self.model.evaluate(
                eval_env=self.test_env,
                n_eval_episodes=1,
                deterministic=True
            )
            
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
        
        # Close environments if they exist
        if hasattr(self, 'train_env') and self.train_env:
            try:
                # Close any trade loggers
                if hasattr(self.train_env, 'trade_logger') and self.train_env.trade_logger:
                    self.logger.info("Closing train environment trade logger...")
                    # TradeLogger uses append mode, no explicit close needed
                    
                # Close any trade tracers  
                if hasattr(self.train_env, 'trade_tracer') and self.train_env.trade_tracer:
                    self.logger.info("Closing train environment trade tracer...")
                    # TradeTracer uses append mode, no explicit close needed
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up train environment: {e}")
        
        if hasattr(self, 'test_env') and self.test_env:
            try:
                # Close any trade loggers
                if hasattr(self.test_env, 'trade_logger') and self.test_env.trade_logger:
                    self.logger.info("Closing test environment trade logger...")
                    
                # Close any trade tracers
                if hasattr(self.test_env, 'trade_tracer') and self.test_env.trade_tracer:
                    self.logger.info("Closing test environment trade tracer...")
                    
            except Exception as e:
                self.logger.warning(f"Error cleaning up test environment: {e}")
        
        # Close all logging handlers
        self.logger.info("Closing logging handlers...")
        close_all_logging_handlers()
        
        self.logger.info("Cleanup completed successfully")

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all relevant libraries.
    
    Args:
        seed: Random seed value to use
    """
    import random
    import numpy as np
    import torch
    
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed} for reproducibility")


def configure_device(device_preference: str = 'auto') -> str:
    """
    Configure and return the device to use for training.
    
    Args:
        device_preference: User preference - 'auto', 'cpu', 'cuda', or 'gpu'
        
    Returns:
        Device string to use ('cpu' or 'cuda')
    """
    import torch
    
    # Normalize device preference
    if device_preference.lower() in ['gpu', 'cuda']:
        device_preference = 'cuda'
    elif device_preference.lower() == 'cpu':
        device_preference = 'cpu'
    else:  # 'auto'
        device_preference = 'auto'
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if device_preference == 'auto':
        # Auto-detect best device
        if cuda_available:
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown GPU'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            print(f"🚀 Auto-detected CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            print("💻 Auto-detected CPU (CUDA not available)")
    elif device_preference == 'cuda':
        # User requested CUDA
        if cuda_available:
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown GPU'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            print(f"🚀 Using requested CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    else:
        # User requested CPU
        device = 'cpu'
        print("💻 Using requested CPU device")
    
    # Set PyTorch device
    torch.set_default_device(device)
    
    # Additional CUDA optimizations if using GPU
    if device == 'cuda':
        # Enable optimizations for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache to start fresh
        torch.cuda.empty_cache()
        
        print(f"✅ Device configured: {device.upper()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
    else:
        print(f"✅ Device configured: {device.upper()}")
        cpu_count = torch.get_num_threads()
        print(f"   CPU Threads: {cpu_count}")
    
    return device


def close_all_logging_handlers():
    """
    Close all logging handlers to release file locks before archiving.
    """
    logger = logging.getLogger()
    handlers_to_remove = []
    
    # Collect file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)
    
    # Close and remove file handlers
    for handler in handlers_to_remove:
        handler.close()
        logger.removeHandler(handler)
    
    # Also check and close handlers on specific loggers
    for name in logging.Logger.manager.loggerDict:
        specific_logger = logging.getLogger(name)
        if hasattr(specific_logger, 'handlers'):
            handlers_to_remove = []
            for handler in specific_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                handler.close()
                specific_logger.removeHandler(handler)

def archive_existing_logs_and_models():
    """
    Archive existing logs and models before starting new training.
    Creates timestamped archive folders to preserve previous runs.
    """
    # Close all logging handlers first to release file locks
    print("🔒 Closing logging handlers to prepare for archiving...")
    close_all_logging_handlers()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = Path("archive") / f"training_run_{timestamp}"
    
    # Directories to archive
    dirs_to_archive = [
        ("logs", "logs"),
        ("models", "models"),
        ("tensorboard", "logs/tensorboard")
    ]
    
    archived_something = False
    
    for dir_name, source_path in dirs_to_archive:
        source = Path(source_path)
        if source.exists() and any(source.iterdir()):  # Directory exists and is not empty
            archive_dir = archive_base / dir_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Move contents to archive with error handling
            for item in source.iterdir():
                if item.is_file() or item.is_dir():
                    try:
                        shutil.move(str(item), str(archive_dir / item.name))
                        archived_something = True
                    except PermissionError as e:
                        print(f"⚠️  Could not move {item}: {e}")
                        print(f"   File may still be in use. Skipping: {item}")
                    except Exception as e:
                        print(f"⚠️  Error moving {item}: {e}")
    
    if archived_something:
        print(f"📦 Archived previous training data to: {archive_base}")
    else:
        print("📦 No previous training data found to archive")
    
    return archive_base if archived_something else None

def main():
    """Main function to run the memory-efficient training pipeline"""
    load_dotenv()
    
    # Archive existing logs and models before starting
    print("🗃️  Archiving previous training data...")
    archive_existing_logs_and_models()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    parser = argparse.ArgumentParser(
        description='Memory-Efficient RL Trading Bot Training',
        epilog='''
Examples:
  # Quick test run with default settings (CPU, 1 episode, ultra-aggressive small trades):
  python train_memory_efficient.py --consolidated-file data.csv --default
  
  # Full training run with GPU:
  python train_memory_efficient.py --consolidated-file data.csv --total-episodes 10 --device cuda
  
  # CPU training with custom settings:
  python train_memory_efficient.py --consolidated-file data.csv --total-episodes 5 --device cpu --verbose        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--consolidated-file', type=str, required=False,
                       help='Path to consolidated data file (optional with --default)')
    parser.add_argument('--total-episodes', type=int, default=10, 
                       help='Total number of episodes (full dataset passes)')
    parser.add_argument('--steps-per-batch', type=int, default=1000, 
                       help='Training steps per data batch')
    parser.add_argument('--streaming-batch-size', type=int, default=1000, 
                       help='Number of rows to read per streaming batch')
    parser.add_argument('--initial-balance', type=float, default=10000, 
                       help='Initial trading balance')
    parser.add_argument('--lookback-window', type=int, default=20, 
                       help='Lookback window for features')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Train/test split ratio for each batch')
    parser.add_argument('--reward-strategy', type=str, default='balanced', 
                       choices=['balanced', 'conservative', 'aggressive'],
                       help='Reward strategy')
    parser.add_argument('--encourage-small-trades', action='store_true', 
                       help='Use small transaction reward config')
    parser.add_argument('--ultra-aggressive-small-trades', action='store_true', 
                       help='Use ultra-aggressive small transaction reward config')
    parser.add_argument('--log-reward-details', action='store_true', 
                       help='Enable detailed reward logging')
    parser.add_argument('--save-model-every-episodes', type=int, default=5, 
                       help='Save model after every N episodes')
    parser.add_argument('--final-eval-episodes', type=int, default=10, 
                       help='Number of episodes for final evaluation')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'gpu'],
                       help='Device to use for training: auto (detect best), cpu, cuda/gpu')
    parser.add_argument('--default', action='store_true',
                       help='Run with default settings optimized for quick testing (forces CPU, 1 episode, ultra-aggressive small trades)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Handle --default option with predefined settings
    if args.default:
        print("🚀 Using --default mode with optimized settings for quick testing")
        
        # Check if consolidated file is provided, if not, use the specified default
        if not args.consolidated_file:
            args.consolidated_file = r".\data\processed\BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv"
            print(f"   Using default consolidated file: {args.consolidated_file}")
        
        # Override specific settings for default mode
        args.total_episodes = 1
        args.steps_per_batch = 10000
        args.random_seed = 123
        args.verbose = True
        args.ultra_aggressive_small_trades = True
        args.device = 'cpu'  # Force CPU usage
        
        print("   Default settings applied:")
        print(f"     - Device: CPU (forced)")
        print(f"     - Episodes: 1")
        print(f"     - Steps per batch: 10,000")
        print(f"     - Random seed: 123")
        print(f"     - Verbose: Enabled")
        print(f"     - Ultra-aggressive small trades: Enabled")
        print()
    
    # Validate that consolidated file is provided (either via argument or default)
    if not args.consolidated_file:
        parser.error("--consolidated-file is required unless using --default mode")
    
    # Convert argparse namespace directly into a dictionary
    config = vars(args)    # Configure device based on user preference
    actual_device = configure_device(config['device'])
    config['device'] = actual_device  # Store the actual device used
    
    # Override default seed with user-provided seed
    if config['random_seed'] != 42:
        set_random_seeds(config['random_seed'])
    
    # Configure device for training
    device = configure_device(config['device'])
    config['device'] = device  # Store the actual device used
    
    # Verify consolidated file exists
    if not Path(config['consolidated_file']).exists():
        print(f"❌ Consolidated file not found: {config['consolidated_file']}")
        print("Please run consolidate_data.py first to create a consolidated file.")
        sys.exit(1)
    
    print("🧠 Memory-Efficient RL Trading Bot Training")
    print("=" * 50)
    if config.get('default', False):
        print("🚀 RUNNING IN DEFAULT MODE (Quick Testing)")
        print("=" * 50)
    print(f"Consolidated File: {config['consolidated_file']}")
    print(f"Device: {config['device'].upper()}")
    print(f"Total Episodes: {config['total_episodes']}")
    print(f"Steps per Batch: {config['steps_per_batch']}")
    print(f"Streaming Batch Size: {config['streaming_batch_size']}")
    print(f"Initial Balance: ${config['initial_balance']:,.2f}")
    print(f"Lookback Window: {config['lookback_window']}")
    print(f"Train Ratio: {config['train_ratio']}")
    print(f"Reward Strategy: {config['reward_strategy']}")
    if config['encourage_small_trades']:
        print("Small Trades: Encouraged")
    if config['ultra_aggressive_small_trades']:
        print("Ultra-Aggressive Small Trades: Enabled")
    print("=" * 50)
    
    # Add additional configuration for logging
    config.update({
        'console_log_level': 'DEBUG' if config['verbose'] else 'INFO',
        'file_log_level': 'DEBUG',
        'trade_logging': True,
        'trade_tracing': True,
        'tensorboard_logging': True,
        'max_eval_batches': 5  # For improved final evaluation
    })
    
    try:
        # Archive existing logs and models
        archive_existing_logs_and_models()
        
        # Initialize training manager
        print("🔧 Initializing memory-efficient training manager...")
        trainer = MemoryEfficientTrainingManager(config)
        
        # Run training
        print("🚀 Starting training pipeline...")
        start_time = time.time()
        
        final_metrics = trainer.run_complete_training(config['consolidated_file'])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Clean up resources before reporting results
        print("🧹 Cleaning up resources...")
        trainer.cleanup()
        
        # Print results
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Episodes: {final_metrics.get('successful_episodes', 0)}/{final_metrics.get('total_episodes', 0)}")
        
        if 'mean_episode_reward' in final_metrics:
            print(f"Mean Episode Reward: {final_metrics['mean_episode_reward']:.4f}")
            print(f"Best Episode Reward: {final_metrics['best_episode_reward']:.4f}")
        
        if 'final_evaluation' in final_metrics and 'mean_reward' in final_metrics['final_evaluation']:
            print(f"Final Evaluation Reward: {final_metrics['final_evaluation']['mean_reward']:.4f}")
        
        # Display trade statistics
        if 'trade_statistics' in final_metrics and final_metrics['trade_statistics']:
            trade_stats = final_metrics['trade_statistics']
            print("\n📊 TRADE STATISTICS:")
            
            if 'final_balance' in trade_stats:
                print(f"Final Balance: ${trade_stats['final_balance']:.2f}")
            if 'total_return' in trade_stats:
                print(f"Total Return: {trade_stats['total_return']:.2%}")
            if 'total_trades' in trade_stats:
                print(f"Total Trades: {trade_stats['total_trades']}")
            if 'win_rate' in trade_stats:
                print(f"Win Rate: {trade_stats['win_rate']:.2%}")
            if 'profit_factor' in trade_stats:
                if trade_stats['profit_factor'] == float('inf'):
                    print(f"Profit Factor: ∞ (no losses)")
                else:
                    print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
            if 'max_drawdown' in trade_stats:
                print(f"Max Drawdown: {trade_stats['max_drawdown']:.2%}")
            if 'total_commission' in trade_stats:
                print(f"Total Commission: ${trade_stats['total_commission']:.2f}")
        
        print("=" * 50)
        
        # Save training report
        report_path = Path("logs") / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("Memory-Efficient Training Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Consolidated File: {config['consolidated_file']}\n")
            f.write(f"Training Time: {total_time:.1f} seconds\n")
            f.write(f"Episodes: {final_metrics.get('successful_episodes', 0)}/{final_metrics.get('total_episodes', 0)}\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"Final Metrics: {final_metrics}\n")
        
        print(f"📄 Training report saved to: {report_path}")
        
    except Exception as e:
        # Try to cleanup even if training failed
        try:
            if 'trainer' in locals():
                print("🧹 Cleaning up resources after error...")
                trainer.cleanup()
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup failed: {cleanup_error}")
        
        print(f"❌ Training failed: {e}")
        print("Check logs in logs/ directory for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
