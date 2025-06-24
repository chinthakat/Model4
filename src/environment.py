#!/usr/bin/env python3
"""
RL Trading Environment for Binance Futures
Custom Gymnasium environment for cryptocurrency futures trading with PPO
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any
import math
import sys
from pathlib import Path

# Import the enhanced corrected reward system
try:
    from reward_system import EnhancedRewardCalculator, BALANCED_ENHANCED_CONFIG, SMALL_TRANSACTION_CONFIG, ULTRA_SMALL_TRANSACTION_CONFIG
except ImportError:
    # Handle case where reward_system is in parent directory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from reward_system import EnhancedRewardCalculator, BALANCED_ENHANCED_CONFIG, SMALL_TRANSACTION_CONFIG, ULTRA_SMALL_TRANSACTION_CONFIG

# Import trade tracer for detailed trade analysis
try:
    from utils.trade_tracer import TradeTracer
except ImportError:
    # Handle case where trade_tracer is not available
    TradeTracer = None
    logging.getLogger(__name__).warning("TradeTracer not available - trade traces disabled")

# Import the liquidation tracker
try:
    from utils.liquidation_tracker import LiquidationTracker
except ImportError:
    LiquidationTracker = None
    logging.getLogger(__name__).warning("LiquidationTracker not available - liquidation risk management disabled")

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents
    Simulates futures trading with proper risk management
    """
    
    metadata = {'render_modes': ['human']}
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        commission_rate: float = 0.0005,
        max_leverage: float = 10.0,
        max_risk_per_trade: float = 0.02,
        lookback_window: int = 50,
        max_position_size: float = 1.0,
        enable_trade_logging: bool = True,
        trade_logger_session: Optional[str] = None,
        reward_config: Optional[Dict] = None,
        encourage_small_trades: bool = True,
        ultra_aggressive_small_trades: bool = True,
        use_discretized_actions: bool = False,
        logging_config: Optional[Dict] = None,
        episode_num: int = 0,
        batch_num: int = 0
    ):
        """
        Initialize trading environment
        
        Args:
            df: DataFrame with OHLCV data and features
            initial_balance: Starting account balance
            commission_rate: Trading commission rate
            max_leverage: Maximum leverage allowed
            max_risk_per_trade: Maximum risk per trade as fraction of balance
            lookback_window: Number of time steps to look back for observations
            max_position_size: Maximum position size relative to balance
            trade_logger_session: Optional session name to reuse existing trade logger
            reward_config: Optional configuration dict for reward system
            encourage_small_trades: If True, uses SMALL_TRANSACTION_CONFIG; if False, uses BALANCED_ENHANCED_CONFIG
            ultra_aggressive_small_trades: If True (DEFAULT), uses ULTRA_SMALL_TRANSACTION_CONFIG (overrides encourage_small_trades)
            use_discretized_actions: If True, uses discretized position sizes for easier learning
            episode_num: Episode number for logging
            batch_num: Batch number for logging
        """
        # Store episode and batch info for logging
        self.episode_num = episode_num
        self.batch_num = batch_num
        
        # Store the data
        self.data = df.copy()
        self.lookback_window = lookback_window
        
        # Validate data format
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
          # Add timestamp column if not present
        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = self.data.index.astype('int64') // 10**9
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_leverage = max_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.symbol = "BINANCEFTS_PERP_BTC_USDT"  # Default symbol for tracking
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size
        
        # Initialize liquidation tracker for risk management
        if LiquidationTracker:
            self.liquidation_tracker = LiquidationTracker(
                initial_balance=self.initial_balance,
                margin_rate=1.0 / self.max_leverage,
                liquidation_threshold=1.1  # 110% margin level
            )
        else:
            self.liquidation_tracker = None
        
        # Initialize comprehensive reward system        # Configure reward system
        if reward_config is None:
            if ultra_aggressive_small_trades:
                self.reward_config = ULTRA_SMALL_TRANSACTION_CONFIG
            elif encourage_small_trades:
                self.reward_config = SMALL_TRANSACTION_CONFIG
            else:
                self.reward_config = BALANCED_ENHANCED_CONFIG
        else:
            self.reward_config = reward_config
          # Store action space configuration
        self.use_discretized_actions = use_discretized_actions
        
        # Individual trade management system (MUST be before observation space calculation)
        self.MAX_OBSERVED_TRADES = 10  # Maximum trades agent can observe simultaneously for observation space
        self.open_trades = []  # List of individual trade dictionaries
        self.trade_id_counter = 0  # Unique ID generator for trades        # Define discretized position sizes for small transaction focus
        # These are optimized based on the reward analysis
        if ultra_aggressive_small_trades:
            # Ultra-aggressive: Micro position sizes for maximum frequency and minimal risk
            self.discretized_sizes = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # 0% to 6%
        elif encourage_small_trades:
            # Aggressive small transactions: Smaller position sizes 
            self.discretized_sizes = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]  # 0% to 15%
        else:
            # Standard: Broader range of position sizes
            self.discretized_sizes = [0.0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]  # 0% to 30%
        
        # BTC minimum size (futures trading minimum is usually 0.001 BTC)
        self.btc_minimum_size = 0.001
        
        self.reward_calculator = EnhancedRewardCalculator(self.reward_config)
        
        # Set max_steps based on data length
        self.max_steps = len(df) - self.lookback_window
        if self.max_steps <= 0:
            self.max_steps = 100  # Fallback minimum
          # Logging
        self.logger = logging.getLogger(__name__)
        
        # Apply logging configuration
        self.logging_config = logging_config or {}
        self.enable_trade_logging = self.logging_config.get('enable_trade_logging', enable_trade_logging)
        self.enable_trade_tracing = self.logging_config.get('enable_trade_tracing', True)
        trade_log_frequency = self.logging_config.get('trade_log_frequency', 10)
        enable_console_logging = self.logging_config.get('console_log_level', 'INFO') != 'DISABLED'
        
        # Trade logging with consistent session name
        if self.enable_trade_logging:
            try:
                # Import here to avoid circular imports
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent / "utils"))
                from utils.trade_logger import TradeLogger
                
                # Use provided session name or create new one
                if trade_logger_session:
                    session_name = trade_logger_session
                else:
                    session_name = f"env_session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.trade_logger = TradeLogger(
                    session_name=session_name,
                    enable_logging=self.enable_trade_logging,
                    log_frequency=trade_log_frequency,
                    enable_console_logging=enable_console_logging
                )
                # Initialize the trade logger with the starting balance
                self.trade_logger.initialize_session(initial_balance)
                self.logger.info(f"Trade logger initialized with session: {session_name}")
            except ImportError:
                self.logger.warning("Trade logger not available, disabling trade logging")
                self.trade_logger = None
                self.enable_trade_logging = False
        else:
            self.trade_logger = None
        
        # Trade tracer for detailed trade analysis
        if self.enable_trade_tracing and TradeTracer is not None:
            try:
                # Use same session name as trade logger if available
                if trade_logger_session:
                    tracer_session_name = f"{trade_logger_session}_traces"
                else:
                    tracer_session_name = f"env_traces_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.trade_tracer = TradeTracer(
                    session_name=tracer_session_name,
                    enable_tracing=self.enable_trade_tracing,
                    episode_num=self.episode_num,
                    batch_num=self.batch_num
                )
                self.logger.info(f"Trade tracer initialized: {tracer_session_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize trade tracer: {e}")
                self.trade_tracer = None
                self.enable_trade_tracing = False
        else:
            self.trade_tracer = None
            self.enable_trade_tracing = False
        
        # Track open trades for tracer
        self.current_trade_id = None
        self.trade_counter = 0
          # Identify feature columns (exclude datetime if present)
        datetime_cols = ['datetime', 'timestamp', 'date', 'time']
        self.feature_columns = [col for col in df.columns if isinstance(col, str) and col.lower() not in datetime_cols]
        
        # Log detailed information
        self.logger.info(f"DataFrame shape: {df.shape}")
        self.logger.info(f"DataFrame columns: {list(df.columns)}")
        self.logger.info(f"Feature columns: {self.feature_columns}")
        self.logger.info(f"Number of feature columns: {len(self.feature_columns)}")
        self.logger.info(f"Lookback window: {self.lookback_window}")
        self.logger.info(f"Max steps: {self.max_steps}")        # Calculate observation space size consistently
        market_features_size = len(self.feature_columns) * self.lookback_window
        account_features_size = 7  # Basic account features
        individual_trades_features_size = self.MAX_OBSERVED_TRADES * 4  # 4 features per trade slot
        self.obs_size = market_features_size + account_features_size + individual_trades_features_size
        
        self.logger.info(f"Market features size: {market_features_size} ({len(self.feature_columns)} x {self.lookback_window})")
        self.logger.info(f"Account features size: {account_features_size}")
        self.logger.info(f"Individual trades features size: {individual_trades_features_size} ({self.MAX_OBSERVED_TRADES} trades x 4 features)")
        self.logger.info(f"Total observation size: {self.obs_size}")        # Define action and observation spaces BEFORE calling reset
        if self.use_discretized_actions:
            # Enhanced discrete action space for individual trade management:
            # 0: HOLD
            # 1: BUY (open new long position)
            # 2: SELL (open new short position)
            # 3-12: CLOSE_TRADE_1 through CLOSE_TRADE_10 (close specific trades)
            # Total actions: 3 + MAX_OBSERVED_TRADES = 13
            # Also include size_index and leverage as before
            self.action_space = spaces.Box(
                low=np.array([0, 0, 1.0]),
                high=np.array([2 + self.MAX_OBSERVED_TRADES, len(self.discretized_sizes)-1, self.max_leverage]),
                dtype=np.float32
            )
            self.logger.info(f"Using discrete action space with individual trade management:")
            self.logger.info(f"  - Actions 0-2: HOLD, BUY, SELL")
            self.logger.info(f"  - Actions 3-{2+self.MAX_OBSERVED_TRADES}: CLOSE_TRADE_1 to CLOSE_TRADE_{self.MAX_OBSERVED_TRADES}")
            self.logger.info(f"  - {len(self.discretized_sizes)} position sizes: {self.discretized_sizes}")
        else:
            # CONTINUOUS ACTION SPACE WITH INDIVIDUAL TRADE MANAGEMENT
            # action[0]: action_type (0=hold, 1=buy, 2=sell, 3-12=close_trade_1-10)
            # action[1]: position_size (0.0 to 0.5)
            # action[2]: confidence/leverage (0.1 to max_leverage)
            self.action_space = spaces.Box(
                low=np.array([0, 0.0, 0.1]),
                high=np.array([2 + self.MAX_OBSERVED_TRADES, 0.5, self.max_leverage]),
                dtype=np.float32
            )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.closed_trades = []
        
        # Reset individual trades system
        if self.liquidation_tracker:
            self.liquidation_tracker.open_trades.clear()
        self.open_trades = []
        self.trade_id_counter = 0# Reset additional tracking attributes
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_profit = 0.0
        self.current_trade_id = None
        self.trade_counter = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0        }
        
        self.logger.info(f"Environment initialized with observation space: {self.observation_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Reset environment to initial state"""
        # Trade log is automatically saved via append mode, no need for save_session
        if self.trade_logger and hasattr(self, 'current_step') and self.current_step > 0:
            self.logger.info(f"Episode reset: {self.current_step} steps completed")
        
        # Close any open trades in tracer before reset
        if self.enable_trade_tracing and self.trade_tracer and hasattr(self, 'current_trade_id') and self.current_trade_id:
            try:
                self.trade_tracer.close_all_open_positions("Episode reset", current_episode_step=self.current_step)
                self.current_trade_id = None
                self.trade_counter = 0
                self.logger.info("Closed open trades for episode reset")
            except Exception as e:
                self.logger.warning(f"Failed to close open trades on reset: {e}")
        
        if seed is not None:
            np.random.seed(seed)
          # Reset to beginning of data
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.closed_trades = []
        
        # Reset individual trades system
        if self.liquidation_tracker:
            self.liquidation_tracker.open_trades.clear()
        self.open_trades = []
        self.trade_id_counter = 0# Reset additional tracking attributes
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_profit = 0.0
        self.current_trade_id = None
        self.trade_counter = 0
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Episode statistics
        self.episode_stats = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_trade_return': 0.0
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment"""
        # CRITICAL FIX: Validate and clamp invalid model actions before processing
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Replace NaN/inf values with safe defaults
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure action has correct shape
        if len(action) < 3:
            action = np.pad(action, (0, 3 - len(action)), constant_values=0.0)
        elif len(action) > 3:
            action = action[:3]
          # Clip action to valid ranges
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Validate action[1] based on whether we're using discretized actions
        if self.use_discretized_actions:
            # For discretized actions, action[1] should be a valid index into discretized_sizes
            if action[1] < 0 or action[1] >= len(self.discretized_sizes):
                self.logger.warning(f"Invalid size index detected: {action[1]:.0f}, clamping to [0, {len(self.discretized_sizes)-1}]")
                action[1] = np.clip(action[1], 0, len(self.discretized_sizes) - 1)
        else:
            # For continuous actions, action[1] should be between -1.0 and 1.0
            if action[1] < -1.0 or action[1] > 1.0:
                self.logger.warning(f"Invalid position size detected: {action[1]:.4f}, clamping to [-1.0, 1.0]")
                action[1] = np.clip(action[1], -1.0, 1.0)
        
        # Validate leverage/confidence
        if action[2] < 0.1:
            action[2] = 0.1  # Minimum confidence/leverage
        elif action[2] > self.max_leverage:
            action[2] = self.max_leverage
          # Check if episode should end
        if self.current_step >= self.max_steps:
            terminated = True
            truncated = False
            # Trade log is automatically saved via append mode
            if self.trade_logger:
                self.logger.info(f"Episode ended: {self.current_step} steps")
        else:
            terminated = False
            truncated = False
        
        if terminated or truncated:
            # Close any open trades in tracer when episode ends
            if self.enable_trade_tracing and self.trade_tracer and hasattr(self, 'current_trade_id') and self.current_trade_id:
                try:
                    self.trade_tracer.close_all_open_positions("Episode ended", current_episode_step=self.current_step)
                    self.current_trade_id = None
                    self.logger.info("Closed open trades for episode end")
                except Exception as e:
                    self.logger.warning(f"Failed to close open trades on episode end: {e}")
            
            # Return final state
            obs = self._get_observation()
            reward = 0.0
            info = self._get_info()
            return obs, reward, terminated, truncated, info        # Process action with individual trade management
        action_type = int(round(action[0]))
        
        if self.use_discretized_actions:
            # Map size index to actual size percentage
            size_index = int(np.clip(action[1], 0, len(self.discretized_sizes) - 1))
            size_pct = self.discretized_sizes[size_index]
            
            # Log the discrete action selection for debugging
            if action_type <= 2:
                self.logger.debug(f"Discrete action: type={action_type}, size_index={size_index}, size_pct={size_pct:.3f} ({size_pct*100:.1f}%)")
            else:
                close_trade_index = action_type - 3
                self.logger.debug(f"Discrete action: CLOSE_TRADE_{close_trade_index + 1} (action_type={action_type})")
        else:
            # Original continuous action processing
            size_pct = np.clip(action[1], -1.0, 1.0)
        
        confidence = np.clip(action[2], 0.1, self.max_leverage)
        
        # Execute trade
        reward = self._execute_action(action_type, size_pct, confidence)
        
        # Move to next step
        self.current_step += 1
          # Check if episode should end after this step
        if self.current_step >= self.max_steps:
            terminated = True
        elif self.equity <= self.initial_balance * 0.1:  # Stop loss at 90% loss
            terminated = True
        
        # Check for liquidation
        if self.liquidation_tracker:
            current_prices = {trade_id: self._get_current_price() for trade_id in self.liquidation_tracker.open_trades}
            if self.liquidation_tracker.is_liquidation_imminent(current_prices):
                terminated = True
                self.logger.warning("Liquidation event! Episode terminated.")
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward_with_breakdown(self, realized_pnl_change: float, commission: float = 0.0, action_type: int = 0) -> tuple:
        """
        Calculate comprehensive reward using the EnhancedRewardCalculator and return breakdown
        
        Args:
            realized_pnl_change: Change in realized P&L from this action
            commission: Commission paid for this action
            action_type: Action taken (0=hold, 1=buy, 2=sell, 3=close)
            
        Returns:
            Tuple of (reward, reward_breakdown) using comprehensive reward system
        """
        try:
            # Clean inputs to prevent calculation errors
            realized_pnl_change = float(np.nan_to_num(realized_pnl_change, nan=0.0))
            commission = float(np.nan_to_num(commission, nan=0.0))
            action_type = int(action_type) if not np.isnan(action_type) else 0
            
            # Get current market data for timing analysis
            market_data = self._extract_market_data()
            current_price = self._get_current_price()
            
            # Ensure current_price is valid
            if np.isnan(current_price) or current_price <= 0:
                current_price = self.data['Close'].iloc[self.current_step] if self.current_step < len(self.data) else 50000.0
            
            # Calculate current leverage safely
            leverage = 1.0
            if self.position_size != 0 and self.balance > 0:
                position_value = abs(self.position_size) * current_price
                leverage = min(position_value / self.balance, self.max_leverage)
            
            # Use comprehensive reward system
            total_reward, reward_breakdown = self.reward_calculator.calculate_reward(
                current_price=current_price,
                position_size=self.position_size,
                balance=max(self.balance, 1.0),  # Ensure positive balance
                action=action_type,
                leverage=leverage,
                commission_cost=commission,
                market_data=market_data
            )
              # Log reward breakdown for debugging if enabled (less verbose)
            if self.reward_config.get('log_rewards', False) and (
                abs(total_reward) > 10.0 or  # Only log significant rewards
                self.reward_config.get('log_all_rewards', False)  # Or if explicitly enabled
            ):
                significant_components = {k: v for k, v in reward_breakdown.items() if abs(v) > 0.1}
                if significant_components:
                    components_str = ", ".join([f"{k}: {v:.1f}" for k, v in significant_components.items()])
                    self.logger.info(f"Reward: {components_str} | Total: {total_reward:.1f}")
                else:
                    self.logger.info(f"Total Reward: {total_reward:.1f}")
            
            # Scale reward to reasonable range for RL training
            # Use a more conservative scaling to avoid clipping
            scaled_reward = np.clip(total_reward / 100.0, -1.0, 1.0)
            
            # Final safety check
            if np.isnan(scaled_reward) or np.isinf(scaled_reward):
                self.logger.warning(f"Invalid reward calculated: {scaled_reward}, using fallback")
                return self._calculate_simple_reward(realized_pnl_change, commission)
                
            return float(scaled_reward), reward_breakdown
            
        except Exception as e:
            self.logger.warning(f"Error in comprehensive reward calculation: {e}, using fallback")
            # Fallback to simple reward calculation
            return self._calculate_simple_reward(realized_pnl_change, commission)

    def _calculate_simple_reward(self, realized_pnl_change: float, commission: float) -> tuple:
        """
        Fallback simple reward calculation (original method)
        
        Args:
            realized_pnl_change: Change in realized P&L from this action
            commission: Commission paid for this action
            
        Returns:
            Simple calculated reward and empty breakdown dictionary
        """
        # Clean inputs
        realized_pnl_change = float(np.nan_to_num(realized_pnl_change, nan=0.0, posinf=100.0, neginf=-100.0))
        commission = float(np.nan_to_num(commission, nan=0.0, posinf=10.0, neginf=0.0))
        
        # Simple reward calculation
        reward = realized_pnl_change - commission
        
        # Add small unrealized P&L component
        unrealized_component = np.nan_to_num(self.unrealized_pnl * 0.01, nan=0.0)
        reward += unrealized_component
        
        # Normalize by initial balance
        reward = reward / self.initial_balance
        
        # Strict bounds
        reward = np.clip(reward, -1.0, 1.0)
          # Final safety check
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        return float(reward), {}

    def _get_current_timestamp(self):
        """Get current market timestamp as a pandas Timestamp object"""
        try:
            if self.current_step < len(self.data):
                # Get timestamp from the data
                if 'timestamp' in self.data.columns:
                    # Convert integer timestamp to pandas Timestamp
                    ts_value = self.data.iloc[self.current_step]['timestamp']
                    return pd.Timestamp.fromtimestamp(ts_value)
                else:
                    # Use index directly if it's a DatetimeIndex
                    return self.data.index[self.current_step]
            else:
                # If we're at the end, use the last timestamp
                if 'timestamp' in self.data.columns:
                    ts_value = self.data.iloc[-1]['timestamp']
                    return pd.Timestamp.fromtimestamp(ts_value)
                else:
                    return self.data.index[-1]
        except Exception as e:
            # Fallback to current time
            self.logger.warning(f"Error getting timestamp: {e}, using current time")
            return pd.Timestamp.now()

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            Normalized observation array
        """
        try:
            # Get the required lookback window
            start_idx = max(0, self.current_step - self.lookback_window + 1)
            end_idx = self.current_step + 1
            
            # Extract price and volume data
            window_data = self.data.iloc[start_idx:end_idx]
            
            # Basic OHLCV features
            prices = window_data[['Open', 'High', 'Low', 'Close']].values
            volumes = window_data['Volume'].values.reshape(-1, 1)
            
            # Normalize prices relative to the first close price in the window
            if len(prices) > 0:
                base_price = prices[0, 3]  # First close price
                if base_price > 0:
                    normalized_prices = prices / base_price - 1.0
                else:
                    normalized_prices = np.zeros_like(prices)
            else:
                normalized_prices = np.zeros((1, 4))
            
            # Normalize volumes (log transform and standardize)
            if len(volumes) > 0 and volumes.max() > 0:
                log_volumes = np.log1p(volumes)
                volume_mean = log_volumes.mean()
                volume_std = log_volumes.std() + 1e-8
                normalized_volumes = (log_volumes - volume_mean) / volume_std
            else:
                normalized_volumes = np.zeros_like(volumes)
            
            # Combine price and volume data
            features = np.hstack([normalized_prices, normalized_volumes])
            
            # Add technical indicators if available
            tech_cols = [col for col in window_data.columns 
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']]
            
            if tech_cols:
                tech_data = window_data[tech_cols].values
                # Simple normalization for technical indicators
                tech_data = np.nan_to_num(tech_data, nan=0.0, posinf=0.0, neginf=0.0)
                if tech_data.std() > 0:
                    tech_data = (tech_data - tech_data.mean()) / (tech_data.std() + 1e-8)
                features = np.hstack([features, tech_data])
            
            # Pad with zeros if we don't have enough history
            target_length = self.lookback_window
            current_length = features.shape[0]
            
            if current_length < target_length:
                padding = np.zeros((target_length - current_length, features.shape[1]))
                features = np.vstack([padding, features])            # Add portfolio state features with individual trade information
            current_price = self._get_current_price()
            
            # Update unrealized P&L for all trades
            self._update_individual_trades_pnl(current_price)
            
            # Calculate aggregate metrics for compatibility
            total_unrealized_pnl = self._get_total_unrealized_pnl()
            net_position_size = self._get_net_position_size()
            
            # Calculate normalized trade frequency (trades per step)
            current_frequency = (self.reward_calculator.trade_frequency_counter / 
                               max(self.current_step, 1)) if self.current_step > 0 else 0.0
            
            # Normalize steps since last trade (0-1 scale, capped at 50 steps)
            normalized_steps_since_trade = min(self.reward_calculator.steps_since_last_trade / 50.0, 1.0)
            
            # Basic portfolio features (7 features)
            portfolio_features = np.array([
                self.balance / self.initial_balance - 1.0,  # Normalized balance change
                net_position_size * current_price / self.initial_balance,  # Net position value as fraction of initial balance
                float(len(self.open_trades) > 0),  # Whether we have any open positions
                total_unrealized_pnl / self.initial_balance,  # Total unrealized P&L normalized
                current_frequency,  # Current trade frequency (trades per step)
                normalized_steps_since_trade,  # Steps since last trade (normalized 0-1)
                float(len(self.open_trades))  # Number of open trades
            ])
            
            # Individual trades features (4 features per trade slot)
            individual_trades_features = []
            for i in range(self.MAX_OBSERVED_TRADES):
                if i < len(self.open_trades):
                    trade = self.open_trades[i]
                    # Features for this trade slot:
                    # 1. Unrealized P&L percentage
                    unrealized_pnl_pct = trade["unrealized_pnl"] / self.initial_balance
                    # 2. Position size in BTC
                    size_btc = trade["size_btc"]
                    # 3. Steps held (normalized to 0-1, capped at 100 steps)
                    steps_held_normalized = min(trade["steps_held"] / 100.0, 1.0)
                    # 4. Side indicator: +1 for LONG, -1 for SHORT
                    side_indicator = 1.0 if trade["side"] == "LONG" else -1.0
                    
                    trade_features = [unrealized_pnl_pct, size_btc, steps_held_normalized, side_indicator]
                else:
                    # Empty slot - all zeros
                    trade_features = [0.0, 0.0, 0.0, 0.0]
                
                individual_trades_features.extend(trade_features)
            
            individual_trades_features = np.array(individual_trades_features)
            
            # Flatten and concatenate everything
            observation = np.concatenate([
                features.flatten(),
                portfolio_features,
                individual_trades_features
            ])
            
            # Ensure finite values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error creating observation: {e}")
            # Return zero observation as fallback
            fallback_size = self.obs_size
            return np.zeros(fallback_size, dtype=np.float32)

    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return final episode statistics.
        """
        # Update final stats before returning
        self.episode_stats['total_return'] = (self.equity / self.initial_balance) - 1.0 if self.initial_balance > 0 else 0.0
        self.episode_stats['total_trades'] = self.total_trades
        if self.total_trades > 0:
            self.episode_stats['win_rate'] = self.winning_trades / self.total_trades
        else:
            self.episode_stats['win_rate'] = 0.0
        self.episode_stats['max_drawdown'] = self.max_drawdown
        
        # A simple profit factor calculation
        total_profit = sum(trade.get('realized_pnl', 0) for trade in self.closed_trades if trade.get('realized_pnl', 0) > 0)
        total_loss = abs(sum(trade.get('realized_pnl', 0) for trade in self.closed_trades if trade.get('realized_pnl', 0) < 0))
        if total_loss > 0:
            self.episode_stats['profit_factor'] = total_profit / total_loss
        else:
            self.episode_stats['profit_factor'] = float('inf') if total_profit > 0 else 1.0
            
        self.episode_stats['final_balance'] = self.equity
        self.episode_stats['total_commission'] = self.total_commission

        return self.episode_stats

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary with individual trade support"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'net_position_size': self._get_net_position_size(),  # Compatibility
            'total_unrealized_pnl': self._get_total_unrealized_pnl(),
            'total_margin_used': self._get_total_margin_used(),
            'realized_pnl': self.realized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'open_trades_count': len(self.open_trades),
            'max_drawdown': self.max_drawdown,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'current_price': self._get_current_price(),
            'open_trades': [trade.copy() for trade in self.open_trades]  # Copy for safety
        }

    def _execute_action(self, action_type: int, size_pct: float, confidence: float) -> float:
        """Execute trading action with individual trade management"""
        current_price = self._get_current_price()
        commission = 0.0
        realized_pnl_change = 0.0
        trade_reason = ""
        
        # Update unrealized P&L for all open trades
        self._update_individual_trades_pnl(current_price)
        
        # Validate inputs to prevent data quality issues
        if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
            self.logger.warning(f"Invalid price detected: {current_price}, skipping trade")
            return 0.0
        
        if np.isnan(size_pct) or np.isinf(size_pct):
            size_pct = 0.0
        
        if np.isnan(confidence) or np.isinf(confidence):
            confidence = 1.0
        
        # Store previous state for P&L calculation
        prev_balance = self.balance
        prev_total_unrealized_pnl = self._get_total_unrealized_pnl()
        
        # Extract market data for trade context
        market_data = self._extract_market_data()
        
        # Initialize tracking variables
        action_str = "UNKNOWN"
        trade_reason = "Unknown action"
        is_valid_action = False
        commission = 0.0
        reward_breakdown = {}
        
        # Execute the action based on type
        if action_type == 0:  # HOLD
            action_str = "HOLD"
            trade_reason = "Hold position"
            is_valid_action = True
            
        elif action_type == 1:  # BUY (LONG)
            if size_pct > 0:
                # Calculate trade size more conservatively
                # size_pct is percentage of balance to risk, confidence is leverage
                margin_amount = self.balance * size_pct  # Amount of balance to use as margin
                position_value = margin_amount * confidence  # Total position value with leverage
                shares_to_buy = position_value / current_price
                
                # Try to open new long trade (pass leverage separately)
                new_trade = self._open_individual_trade("LONG", shares_to_buy, confidence, current_price, market_data)
                if new_trade:
                    action_str = "BUY"
                    trade_reason = f"Opened long trade: {new_trade['size_btc']:.6f} BTC (ID: {new_trade['trade_id']})"
                    is_valid_action = True
                    commission = new_trade["entry_commission"]
                    
                    self.logger.info(f"BUY executed: {new_trade['size_btc']:.6f} BTC at ${current_price:.2f}")
                else:
                    action_str = "BUY_REJECTED"
                    trade_reason = "Failed to open long trade (insufficient balance, liquidation risk, or size too small)"
                    is_valid_action = False
                    self.logger.warning(f"BUY rejected: {trade_reason}")
            else:
                action_str = "HOLD"
                trade_reason = "Buy signal with zero size treated as Hold"
                is_valid_action = True
                
        elif action_type == 2:  # SELL (SHORT)
            if size_pct > 0:
                # Calculate trade size
                margin_amount = self.balance * size_pct  # Amount of balance to use as margin
                position_value = margin_amount * confidence  # Total position value with leverage
                shares_to_sell = position_value / current_price
                
                # Try to open new short trade
                new_trade = self._open_individual_trade("SHORT", shares_to_sell, confidence, current_price, market_data)
                
                if new_trade:
                    action_str = "SELL"
                    trade_reason = f"Opened short trade: {new_trade['size_btc']:.6f} BTC (ID: {new_trade['trade_id']})"
                    is_valid_action = True
                    commission = new_trade["entry_commission"]
                    
                    self.logger.info(f"SELL executed: {new_trade['size_btc']:.6f} BTC at ${current_price:.2f}")
                else:
                    action_str = "SELL_REJECTED"
                    trade_reason = "Failed to open short trade (insufficient balance, liquidation risk, or size too small)"
                    is_valid_action = False
                    self.logger.warning(f"SELL rejected: {trade_reason}")
            else:
                action_str = "HOLD"
                trade_reason = "Sell signal with zero size treated as Hold"
                is_valid_action = True
                
        elif action_type >= 3:  # CLOSE_TRADE_N actions
            close_trade_index = action_type - 3  # Convert to 0-based index
            
            if close_trade_index < len(self.open_trades):
                # Close the specific trade
                close_info = self._close_individual_trade(close_trade_index, current_price, market_data)
                
                if close_info:
                    action_str = f"CLOSE_TRADE_{close_trade_index + 1}"
                    trade_reason = f"Closed {close_info['side']} trade {close_info['trade_id']}: {close_info['realized_pnl']:+.2f} P&L"
                    is_valid_action = True
                    realized_pnl_change = close_info["realized_pnl"]
                    commission = close_info["total_commission"] - close_info.get("entry_commission", 0)  # Only exit commission
                    
                    self.logger.info(f"CLOSE executed: {close_info['trade_id']} with {close_info['realized_pnl']:+.2f} P&L")
                else:
                    action_str = f"CLOSE_TRADE_{close_trade_index + 1}_FAILED"
                    trade_reason = "Failed to close trade"
                    is_valid_action = False
            else:
                action_str = f"CLOSE_TRADE_{close_trade_index + 1}_INVALID"
                trade_reason = f"No trade in slot {close_trade_index + 1} to close"
                is_valid_action = False
                self.logger.debug(f"Cannot close trade slot {close_trade_index + 1}: only {len(self.open_trades)} trades open")
        else:
            # Handle invalid action types
            action_str = "INVALID_ACTION"
            trade_reason = f"Invalid action type: {action_type}"
            is_valid_action = False
        
        # Update equity calculation with individual trades
        total_unrealized_pnl = self._get_total_unrealized_pnl()
        total_margin_used = self._get_total_margin_used()
        self.equity = self.balance + total_margin_used + total_unrealized_pnl
        
        # Update total commission and max profit tracking (only for valid actions)
        if is_valid_action and commission > 0:
            # Commission already included in balance changes above
            pass
            
        total_pnl = self.realized_pnl + total_unrealized_pnl
        if total_pnl > self.max_profit:
            self.max_profit = total_pnl
        
        # Update peak equity and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Calculate reward
        reward, reward_breakdown = self._calculate_reward_with_breakdown(
            realized_pnl_change=realized_pnl_change,
            commission=commission,
            action_type=action_type
        )
        
        # Log invalid actions with tracer for analysis
        if not is_valid_action and self.trade_tracer:
            self.trade_tracer.log_invalid_action(
                action_type=action_type,
                action_str=action_str,
                reason=trade_reason,
                size_pct=size_pct,
                current_price=current_price,
                balance=self.balance,
                position_size=self._get_net_position_size(),
                market_timestamp=self._get_current_timestamp(),
                market_data=market_data,
                observation_space=self._get_observation_dict(),
                confidence=confidence,
                episode_step=self.current_step,
                reward_info=reward_breakdown
            )
            
        return reward

    def _open_individual_trade(self, side: str, size_btc: float, leverage: float, current_price: float, market_data: Optional[Dict] = None) -> Optional[dict]:
        """
        Open a new individual trade and add it to the open_trades list.
        
        Args:
            side: "LONG" or "SHORT"
            size_btc: Size in BTC
            leverage: Leverage for this trade
            current_price: Current market price
            
        Returns:
            Dictionary representing the new trade, or None if failed
        """
        # Apply BTC minimum size rounding
        size_btc = self._round_to_btc_minimum(size_btc)
        
        if size_btc < self.btc_minimum_size:
            self.logger.warning(f"Trade size {size_btc:.6f} BTC below minimum {self.btc_minimum_size:.6f} BTC")
            return None
        
        # Calculate position value and margin requirement
        position_value = size_btc * current_price
        margin_required = position_value / leverage if leverage > 0 else position_value
        commission = position_value * self.commission_rate
        
        # Check if we have enough balance for margin + commission
        if self.balance < margin_required + commission:
            self.logger.warning(f"Insufficient balance for trade: need ${margin_required + commission:.2f}, have ${self.balance:.2f}")
            return None

        # Check for liquidation risk BEFORE opening the trade
        if self.liquidation_tracker:
            # Simulate adding the new trade to check margin level
            trade_id_sim = f"SIM_{self.trade_id_counter}"
            self.liquidation_tracker.open_trade(trade_id_sim, size_btc, current_price)
            
            current_prices = {trade["trade_id"]: current_price for trade in self.open_trades}
            current_prices[trade_id_sim] = current_price

            if self.liquidation_tracker.is_liquidation_imminent(current_prices):
                self.logger.warning(f"Liquidation risk detected. Cannot open new {side} trade.")
                # Remove the simulated trade
                self.liquidation_tracker.close_trade(trade_id_sim, current_price) 
                return None
            else:
                # The trade is safe, so remove the simulation to add the real one
                self.liquidation_tracker.close_trade(trade_id_sim, current_price)

        # Create new trade
        trade_id = f"TRADE_{self.trade_id_counter:05d}"
        self.trade_id_counter += 1
        
        new_trade = {
            "trade_id": trade_id,
            "entry_price": current_price,
            "size_btc": size_btc,
            "side": side,  # "LONG" or "SHORT"
            "leverage": leverage,
            "entry_step": self.current_step,
            "unrealized_pnl": 0.0,
            "steps_held": 0,
            "margin_used": margin_required,
            "entry_commission": commission
        }
        
        # Update balance (deduct margin and commission)
        self.balance -= margin_required + commission
        self.total_commission += commission
        
        # Add to open trades and liquidation tracker
        self.open_trades.append(new_trade)
        if self.liquidation_tracker:
            self.liquidation_tracker.open_trade(trade_id, size_btc, current_price)
        self.total_trades += 1

        self.logger.info(f"Opened {side} trade: {size_btc:.6f} BTC at ${current_price:.2f} (ID: {trade_id})")

        # Log trade opening for tracing
        if self.enable_trade_tracing and self.trade_tracer:
            self.trade_tracer.log_trade_open(
                trade_id=trade_id,
                action="BUY" if side == "LONG" else "SELL",
                symbol=self.symbol,
                entry_price=current_price,
                position_size=size_btc if side == "LONG" else -size_btc,
                leverage=leverage,
                commission=commission,
                balance_before=self.balance + margin_required + commission,  # Before deducting
                balance_after=self.balance,  # After deducting
                market_timestamp=self._get_current_timestamp(),
                market_data=market_data,
                confidence=leverage,  # Using leverage as confidence for now
                trade_reason=f"Individual {side} trade",
                episode_step=self.current_step,
                observation_space=self._get_observation_dict() # Add observation space
            )

        return new_trade
    
    def _close_individual_trade(self, trade_index: int, current_price: float, market_data: Optional[Dict] = None) -> dict:
        """
        Close an individual trade by its index in the open_trades list.
        
        Args:
            trade_index: Index of trade in self.open_trades list
            current_price: Current market price
            
        Returns:
            Dictionary with trade close information, or None if failed
        """
        if trade_index >= len(self.open_trades):
            self.logger.warning(f"Cannot close trade index {trade_index}: only {len(self.open_trades)} trades open")
            return None
        
        trade = self.open_trades[trade_index]
        
        # Calculate realized P&L
        position_value = trade["size_btc"] * current_price
        exit_commission = position_value * self.commission_rate
        
        if trade["side"] == "LONG":
            # For long: profit when price goes up
            pnl = trade["size_btc"] * (current_price - trade["entry_price"])
        else:  # SHORT
            # For short: profit when price goes down
            pnl = trade["size_btc"] * (trade["entry_price"] - current_price)
        
        # Total realized P&L after commissions
        realized_pnl = pnl - trade["entry_commission"] - exit_commission
        
        # Return margin to balance and add realized P&L
        self.balance += trade["margin_used"] + realized_pnl
        self.total_commission += exit_commission
        self.realized_pnl += realized_pnl
        
        # Track winning/losing trades
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Create close information
        close_info = {
            "trade_id": trade["trade_id"],
            "side": trade["side"],
            "size_btc": trade["size_btc"],
            "entry_price": trade["entry_price"],
            "exit_price": current_price,
            "steps_held": self.current_step - trade["entry_step"],
            "realized_pnl": realized_pnl,
            "price_change_pnl": pnl,
            "total_commission": trade["entry_commission"] + exit_commission,
            "margin_returned": trade["margin_used"]
        }
        
        self.logger.info(f"Closed {trade['side']} trade {trade['trade_id']}: {realized_pnl:+.2f} P&L (held {close_info['steps_held']} steps)")
        
        self.closed_trades.append(close_info)
        
        # Log trade closing for tracing
        if self.enable_trade_tracing and self.trade_tracer:
            self.trade_tracer.log_trade_close(
                trade_id=trade["trade_id"],
                exit_action="CLOSE",
                exit_price=current_price,
                commission=exit_commission,
                balance_before=self.balance - trade["margin_used"] - realized_pnl,  # Before returning margin and P&L
                balance_after=self.balance,  # After returning margin and P&L
                market_timestamp=self._get_current_timestamp(),
                market_data=market_data,
                episode_step=self.current_step,
                observation_space=self._get_observation_dict() # Add observation space
            )
        
        # Remove from open trades and liquidation tracker
        if self.liquidation_tracker:
            self.liquidation_tracker.close_trade(trade["trade_id"], current_price)
        self.open_trades.pop(trade_index)
        
        return close_info
    
    def _update_individual_trades_pnl(self, current_price: float):
        """Update unrealized P&L and steps held for all open trades"""
        for trade in self.open_trades:
            # Update steps held
            trade["steps_held"] = self.current_step - trade["entry_step"]
            
            # Calculate unrealized P&L
            if trade["side"] == "LONG":
                # For long: profit when price goes up
                trade["unrealized_pnl"] = trade["size_btc"] * (current_price - trade["entry_price"])
            else:  # SHORT
                # For short: profit when price goes down
                trade["unrealized_pnl"] = trade["size_btc"] * (trade["entry_price"] - current_price)
    
    def _round_to_btc_minimum(self, size_btc: float) -> float:
        """Rounds the trade size to the nearest minimum BTC size."""
        if self.btc_minimum_size > 0:
            return round(size_btc / self.btc_minimum_size) * self.btc_minimum_size
        return size_btc

    def _get_current_price(self) -> float:
        """Get current close price - handle both normalized and actual price data"""
        if self.current_step + self.lookback_window < len(self.data):
            price = self.data.iloc[self.current_step + self.lookback_window]['Close']
        else:
            price = self.data.iloc[-1]['Close']
        
        # Check if data appears to be normalized (values typically between -3 and 3)
        # If not, assume it's actual price data
        if abs(price) <= 10:  # Likely normalized data
            # Use stored price range for denormalization
            if hasattr(self, '_price_min') and hasattr(self, '_price_max'):
                actual_price = price * (self._price_max - self._price_min) + self._price_min
            else:
                # Fallback: assume BTC price range and denormalize
                # Map normalized range [-1, 1] to reasonable BTC price range [20000, 80000]
                actual_price = (price + 1) * 30000 + 20000  # Simple linear mapping
        else:
            # Data appears to be actual prices, use as-is
            actual_price = price
        
        return max(actual_price, 1000.0)  # Ensure minimum reasonable price
    
    def update_data(self, new_df: pd.DataFrame, episode_num: int = 0, batch_num: int = 0):
        """
        Update environment with new data while preserving model state
        
        Args:
            new_df: New DataFrame with market data
            episode_num: Episode number for logging
            batch_num: Batch number for logging
        """
        self.data = new_df
        self.max_steps = len(new_df) - self.lookback_window - 1
        self.episode_num = episode_num
        self.batch_num = batch_num
        
        # Update trade tracer if it exists
        if self.trade_tracer:
            self.trade_tracer.episode_num = episode_num
            self.trade_tracer.batch_num = batch_num
        
        # Reset environment state but keep model configuration
        self.reset()
        
        self.logger.info(f"Environment data updated - Episode {episode_num}, Batch {batch_num}, {len(new_df)} steps")

    def _get_observation_dict(self) -> Dict[str, Any]:
        """
        Get current observation as a dictionary for logging, with filtering for relevance.
        
        Returns:
            A dictionary containing the structured and filtered observation space.
        """
        # Market Features (only the most recent step for conciseness)
        if self.current_step < len(self.data):
            market_features = self.data[self.feature_columns].iloc[self.current_step].to_dict()
            # Clean for JSON
            for k, v in market_features.items():
                if pd.isna(v):
                    market_features[k] = None
        else:
            market_features = {}

        # Portfolio Features (all are generally relevant)
        current_price = self._get_current_price()
        total_unrealized_pnl = self._get_total_unrealized_pnl()
        net_position_size = self._get_net_position_size()
        current_frequency = (self.reward_calculator.trade_frequency_counter *
                           max(self.current_step, 1)) if self.current_step > 0 else 0.0
        normalized_steps_since_trade = min(self.reward_calculator.steps_since_last_trade / 50.0, 1.0)

        portfolio_features = {
            "normalized_balance": self.balance / self.initial_balance,
            "equity": self.equity,
            "net_position_value_fraction": (net_position_size * current_price) / self.initial_balance if self.initial_balance > 0 else 0,
            "has_open_positions": float(len(self.open_trades) > 0),
            "total_unrealized_pnl_normalized": total_unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0,
            "trade_frequency": current_frequency,
            "normalized_steps_since_trade": normalized_steps_since_trade,
            "open_trades_count": len(self.open_trades)
        }

        # Individual Trades Features (all are relevant)
        individual_trades_features = []
        for trade in self.open_trades:
            trade_features = {
                "trade_id": trade.get("trade_id"),
                "side": trade.get("side"),
                "unrealized_pnl_pct": (trade.get("unrealized_pnl", 0) / self.initial_balance) if self.initial_balance > 0 else 0,
                "size_btc": trade.get("size_btc"),
                "steps_held": trade.get("steps_held"),
                "leverage": trade.get("leverage")
            }
            individual_trades_features.append(trade_features)

        # Filtered observation space for logging
        observation_dict = {
            "market_context": market_features,
            "portfolio_overview": portfolio_features,
            "open_trades_details": individual_trades_features
        }
        
        return observation_dict

    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(i) for i in obj]
        elif isinstance(obj, (np.ndarray,)):
            return self._convert_to_json_serializable(obj.tolist())
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return obj

    def _get_net_position_size(self) -> float:
        """Calculate the net position size from all open trades."""
        return sum(trade['size_btc'] if trade['side'] == 'LONG' else -trade['size_btc'] for trade in self.open_trades)

    def _get_total_unrealized_pnl(self) -> float:
        """Calculate the total unrealized P&L from all open trades."""
        return sum(trade['unrealized_pnl'] for trade in self.open_trades)

    def _get_total_margin_used(self) -> float:
        """Calculate the total margin used for all open trades."""
        return sum(trade['margin_used'] for trade in self.open_trades)

    # ======================== OBSERVATION AND INFO METHODS ========================
    def _extract_market_data(self) -> Dict[str, float]:
        """Extract market data for the current step."""
        if self.current_step < len(self.data):
            return self.data.iloc[self.current_step].to_dict()
        return {}
