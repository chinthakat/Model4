"""
Enhanced Point-Based Reward System for RL Trading Bot - CORRECTED VERSION

This module implements the improved reward system with CRITICAL CORRECTIONS:
1. Fixed realized P&L calculation using stored position size and action
2. Corrected risk management position value calculation  
3. Fixed trade history P&L amount calculations

Key Improvements:
1. Risk-adjusted profit/loss scaling
2. Separate realized vs unrealized P&L rewards
3. Enhanced stop-loss and drawdown management
4. Better trade history tracking with actual P&L amounts
5. Configurable hyperparameters for easy tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

class EnhancedRewardCalculator:
    """
    CORRECTED Enhanced point-based reward system for trading decisions
    
    Reward Components:
    1. Risk-Adjusted Profit/Loss Points: Normalized by account balance percentage
    2. Realized vs Unrealized P&L Distinction: Higher rewards for closed trades
    3. Advanced Risk Management: Stop-loss rewards, drawdown penalties
    4. Market Timing Points: Trend following with multiple indicators
    5. Consistency & Performance Tracking: Enhanced trade history
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize enhanced reward calculator
        
        Args:
            config: Dictionary with reward configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
          # === CORE REWARD PARAMETERS ===
        
        # 1. Profit/Loss Points (Risk-Adjusted)
        self.PROFIT_MULTIPLIER = self.config.get('profit_multiplier', 100.0)
        self.LOSS_MULTIPLIER = self.config.get('loss_multiplier', -100.0)
        self.REALIZED_PNL_BONUS = self.config.get('realized_pnl_bonus', 1.5)  # Extra reward for realized trades
        self.USE_RISK_ADJUSTED_SCALING = self.config.get('use_risk_adjusted_scaling', True)
        self.POSITION_SIZE_SCALING_CAP = self.config.get('position_size_scaling_cap', 2.0)
        
        # 2. Position Management
        self.POSITION_OPEN_COST = self.config.get('position_open_cost', -0.5) # Less costly to open
        self.POSITION_CLOSE_COST = self.config.get('position_close_cost', -0.1) # Less costly to close
        self.HOLD_POSITION_REWARD = self.config.get('hold_position_reward', 0.01) # Lower reward for holding
        self.NO_POSITION_HOLD_PENALTY = self.config.get('no_position_hold_penalty', -1.0) # More penalty for inaction

        # === NEW: Penalty for zero-size actions ===
        self.ZERO_SIZE_ACTION_PENALTY = self.config.get('zero_size_action_penalty', -10.0) # Penalty for BUY/SELL actions with zero size

        # === NEW: Bonus for closing a trade ===
        self.CLOSE_ACTION_BONUS = self.config.get('close_action_bonus', 25.0) # A significant bonus for closing a trade

        # === NEW: SMALL TRANSACTION INCENTIVES ===
        
        # 3. Position Size Management (Encouraging Small Trades)
        self.SMALL_POSITION_BONUS = self.config.get('small_position_bonus', 15.0)  # More bonus for small positions
        self.SMALL_POSITION_THRESHOLD = self.config.get('small_position_threshold', 0.1)  # 10% of balance
        self.LARGE_POSITION_PENALTY = self.config.get('large_position_penalty', -50.0)  # Harsher penalty for large positions  
        self.LARGE_POSITION_THRESHOLD = self.config.get('large_position_threshold', 0.5)  # 50% of balance
        self.POSITION_SIZE_PENALTY_EXPONENT = self.config.get('position_size_penalty_exponent', 2.0)  # Exponential penalty
        
        # 4. Transaction Frequency Incentives
        self.FREQUENT_TRADING_BONUS = self.config.get('frequent_trading_bonus', 10.0)  # More bonus for frequent trades
        self.GRADUAL_POSITION_BONUS = self.config.get('gradual_position_bonus', 8.0)  # Bonus for incremental building
        self.OPTIMAL_TRADE_FREQUENCY = self.config.get('optimal_trade_frequency', 10)  # Trades per 100 steps
        
        # 5. Volume-Based Rewards
        self.CONSISTENT_VOLUME_BONUS = self.config.get('consistent_volume_bonus', 3.0)  # Consistent trading volume
        self.PREFERRED_VOLUME_RANGE = self.config.get('preferred_volume_range', (0.05, 0.2))  # 5-20% of balance          # 6. Enhanced Risk Management 
        self.GOOD_RISK_REWARD = self.config.get('good_risk_reward', 5.0)
        self.EXCESSIVE_RISK_PENALTY = self.config.get('excessive_risk_penalty', -30.0)  # Increased penalty
        self.STOP_LOSS_HIT_REWARD = self.config.get('stop_loss_hit_reward', -20.0)  # FIXED: Now negative penalty
        self.DRAWDOWN_PENALTY_MULTIPLIER = self.config.get('drawdown_penalty_multiplier', -50.0)
        self.MAX_DRAWDOWN_THRESHOLD = self.config.get('max_drawdown_threshold', 0.05)  # 5% drawdown threshold
        
        # 7. Market Timing
        self.TREND_FOLLOW_REWARD = self.config.get('trend_follow_reward', 3.0)
        self.COUNTER_TREND_PENALTY = self.config.get('counter_trend_penalty', -5.0)
        self.RSI_TIMING_REWARD = self.config.get('rsi_timing_reward', 2.0)
        
        # 8. Consistency & Performance
        self.WIN_STREAK_BONUS = self.config.get('win_streak_bonus', 2.0)
        self.CONSISTENCY_BONUS = self.config.get('consistency_bonus', 10.0)
        self.WIN_RATE_THRESHOLD = self.config.get('win_rate_threshold', 0.6)  # 60% for bonus        # 9. Commission & Fees
        self.COMMISSION_PENALTY_MULTIPLIER = self.config.get('commission_penalty_multiplier', -5.0)
        
        # 10. CRITICAL FIX: Time-based holding penalty to prevent indefinite holding
        self.HOLDING_TIME_PENALTY = self.config.get('holding_time_penalty', -0.2)  # Stronger penalty per step
        self.MAX_HOLDING_STEPS = self.config.get('max_holding_steps', 48)  # 12 hours for 15min bars
        self.EXCESSIVE_HOLDING_PENALTY = self.config.get('excessive_holding_penalty', -5.0)  # Stronger penalty for long holds
        
        # Store action space configuration
        self.use_discretized_actions = config.get('use_discretized_actions', False)
        
        # === ENHANCED: Action-specific bonuses for exploration ===
        self.SHORT_TRADE_BONUS = self.config.get('short_trade_bonus', 0.0)
        self.LONG_TRADE_BONUS = self.config.get('long_trade_bonus', 0.0)
        self.POSITION_DIVERSITY_BONUS = self.config.get('position_diversity_bonus', 0.0)
        self.EXPLORATION_BONUS = self.config.get('exploration_bonus', 0.0)
        self.ACTION_DIVERSITY_REWARD = self.config.get('action_diversity_reward', 0.0)
        self.RECENT_ACTIONS_WINDOW = self.config.get('recent_actions_window', 20)
        
        # === Position management enhancements ===
        self.MULTIPLE_POSITIONS_PENALTY = self.config.get('multiple_positions_penalty', 0.0)
        self.POSITION_LIMIT_THRESHOLD = self.config.get('position_limit_threshold', 5)
        self.CLOSE_PROFITABLE_BONUS = self.config.get('close_profitable_bonus', 0.0)
        self.CLOSE_LOSING_PENALTY = self.config.get('close_losing_penalty', 0.0)
        
        # === TRACKING & STATE VARIABLES ===
        self.trade_history: List[Dict] = []
        self.current_streak = 0
        self.total_reward_points = 0.0
        self.last_action = 0  # 0=hold, 1=buy, 2=sell
          # NEW: Small transaction tracking
        self.recent_trade_sizes = []  # Track recent trade sizes
        self.trade_frequency_counter = 0
        self.steps_since_last_trade = 0
        self.position_build_steps = []  # Track gradual position building
        self.last_position_size = 0.0  # Track previous position size
        
        # CRITICAL FIX: Enhanced position tracking
        self.position_entry_price = None
        self.position_entry_balance = None
        self.position_entry_size = None  # CRITICAL FIX: Track position size when opened
        self.position_entry_step = None
        self.position_entry_action = None  # CRITICAL FIX: Store position type (1=long, 2=short)
        self.step_count = 0
        
        # Balance & Drawdown Tracking
        self.initial_balance = None
        self.peak_balance = 0.0
        self.balance_history = []
        
        # Performance metrics
        self.episode_start_balance = None
        
    def calculate_reward(self, 
                        current_price: float,
                        position_size: float,
                        balance: float,
                        action: int,
                        leverage: float = 1.0,
                        commission_cost: float = 0.0,
                        market_data: Dict = None,
                        stop_loss_hit: bool = False,
                        trade_closed_this_step: bool = False,
                        size_pct: float = 0.0) -> Tuple[float, Dict]:
        """
        Calculate enhanced point-based reward
        
        Args:
            current_price: Current asset price
            position_size: Current position size (positive=long, negative=short, 0=no position)
            balance: Current account balance
            action: Action taken (0=hold, 1=buy/long, 2=sell/short)
            leverage: Current leverage used
            commission_cost: Commission paid for this action
            market_data: Dictionary with market indicators (sma_short, sma_long, rsi, etc.)
            stop_loss_hit: Whether a stop loss was triggered this step
            trade_closed_this_step: Whether a trade was actually closed in this step
            size_pct: The percentage of balance for the action size
            
        Returns:
            Tuple of (total_reward_points, detailed_breakdown_dict)
        """
        # Initialize tracking on first call
        if self.initial_balance is None:
            self.initial_balance = balance
            self.episode_start_balance = balance
            self.peak_balance = balance        
        # Update balance tracking
        self.balance_history.append(balance)
        self.peak_balance = max(self.peak_balance, balance)
        
        # TEMPORARY FIX: Disable faulty position closing detection
        # The original logic incorrectly triggers close bonuses when net position size changes
        # This causes realized_pnl_points and close_action_bonus to be triggered without actual trade closures
        # TODO: Implement proper trade closure tracking via environment communication
        was_position_open = self.position_entry_price is not None and self.position_entry_size is not None
        
        # PROPER FIX: Use explicit trade closure indication from the environment
        is_position_closed = trade_closed_this_step

        # Original buggy logic (commented out):
        # is_position_closed = (
        #     (was_position_open and position_size == 0) or  # Position externally closed
        #     (self.last_action != 0 and action == 0 and position_size == 0)  # Explicit close action
        # )
        
        # Initialize reward breakdown
        reward_breakdown = {
            'unrealized_pnl_points': 0.0,
            'realized_pnl_points': 0.0,
            'realized_pnl_raw': 0.0,
            'close_action_bonus': 0.0,
            'position_management_points': 0.0,
            'risk_management_points': 0.0,
            'market_timing_points': 0.0,
            'consistency_points': 0.0,
            'commission_penalty': 0.0,
            'drawdown_penalty': 0.0,
            'stop_loss_bonus': 0.0,
            # NEW: Small transaction incentives
            'small_position_bonus': 0.0,
            'position_size_penalty': 0.0,
            'frequency_bonus': 0.0,
            'gradual_building_bonus': 0.0,
            'volume_consistency_bonus': 0.0,
            # CRITICAL FIX: Time-based holding penalty
            'holding_time_penalty': 0.0,
            'zero_size_action_penalty': 0.0,
            'total_points': 0.0
        }
        
        # === NEW: PENALTY FOR ZERO-SIZE ACTIONS ===
        zero_size_penalty = 0.0
        if action in [1, 2] and size_pct == 0.0:
            zero_size_penalty = self.ZERO_SIZE_ACTION_PENALTY
        reward_breakdown['zero_size_action_penalty'] = zero_size_penalty

        # === 1. UNREALIZED P&L POINTS (Risk-Adjusted) ===
        unrealized_points = self._calculate_unrealized_pnl_points(
            current_price, position_size, balance
        )
        reward_breakdown['unrealized_pnl_points'] = unrealized_points
        
        # === 2. REALIZED P&L POINTS (When position closes) ===
        realized_points = 0.0
        if is_position_closed:
            realized_points, pnl_component = self._calculate_realized_pnl_points(
                current_price, position_size, balance
            )
            # Add a bonus for the action of closing a trade
            realized_points += self.CLOSE_ACTION_BONUS
            reward_breakdown['close_action_bonus'] = self.CLOSE_ACTION_BONUS
            reward_breakdown['realized_pnl_raw'] = pnl_component

        reward_breakdown['realized_pnl_points'] = realized_points
        
        # === 3. POSITION MANAGEMENT POINTS ===
        position_mgmt_points = self._calculate_position_management_points(action, position_size)
        reward_breakdown['position_management_points'] = position_mgmt_points
        
        # === 4. RISK MANAGEMENT POINTS (Enhanced) ===
        risk_mgmt_points = self._calculate_risk_management_points(
            position_size, balance, leverage, stop_loss_hit, current_price
        )
        reward_breakdown['risk_management_points'] = risk_mgmt_points
        
        # === 5. DRAWDOWN PENALTY ===
        drawdown_penalty = self._calculate_drawdown_penalty(balance)
        reward_breakdown['drawdown_penalty'] = drawdown_penalty
        
        # === 6. STOP LOSS BONUS ===
        stop_loss_bonus = self.STOP_LOSS_HIT_REWARD if stop_loss_hit else 0.0
        reward_breakdown['stop_loss_bonus'] = stop_loss_bonus
        
        # === 7. MARKET TIMING POINTS ===
        market_timing_points = self._calculate_market_timing_points(action, current_price, market_data)
        reward_breakdown['market_timing_points'] = market_timing_points
        
        # === 8. CONSISTENCY POINTS ===
        is_profitable_step = (unrealized_points + realized_points) > 0
        consistency_points = self._calculate_consistency_points(is_profitable_step)
        reward_breakdown['consistency_points'] = consistency_points
          # === 9. COMMISSION PENALTY ===
        commission_penalty = commission_cost * self.COMMISSION_PENALTY_MULTIPLIER
        reward_breakdown['commission_penalty'] = commission_penalty
          # === 10. SMALL TRANSACTION INCENTIVES ===
        small_transaction_rewards = self._calculate_small_transaction_incentives(
            position_size, balance, action, current_price
        )
        reward_breakdown['small_position_bonus'] = small_transaction_rewards['small_position_bonus']
        reward_breakdown['position_size_penalty'] = small_transaction_rewards['position_size_penalty']
        reward_breakdown['frequency_bonus'] = small_transaction_rewards['frequency_bonus']
        reward_breakdown['gradual_building_bonus'] = small_transaction_rewards['gradual_building_bonus']
        reward_breakdown['volume_consistency_bonus'] = small_transaction_rewards['volume_consistency_bonus']
          # === 11. CRITICAL FIX: HOLDING TIME PENALTY ===
        holding_time_penalty = self._calculate_holding_time_penalty(position_size)
        reward_breakdown['holding_time_penalty'] = holding_time_penalty
        
        # === 12. NEW: ENHANCED EXPLORATION BONUSES ===
        exploration_bonuses = self._calculate_exploration_bonuses(action, position_size, balance)
        reward_breakdown['action_specific_bonus'] = exploration_bonuses['action_specific_bonus']
        reward_breakdown['diversity_bonus'] = exploration_bonuses['diversity_bonus']
        reward_breakdown['exploration_bonus'] = exploration_bonuses['exploration_bonus']
        reward_breakdown['close_quality_bonus'] = exploration_bonuses['close_quality_bonus']        # === CALCULATE TOTAL REWARD ===
        total_reward = (
            unrealized_points +
            realized_points +
            position_mgmt_points +
            risk_mgmt_points +
            drawdown_penalty +
            stop_loss_bonus +
            market_timing_points +
            consistency_points +
            commission_penalty +
            holding_time_penalty +
            # Add small transaction incentives
            small_transaction_rewards['small_position_bonus'] +
            small_transaction_rewards['position_size_penalty'] +
            small_transaction_rewards['frequency_bonus'] +
            small_transaction_rewards['gradual_building_bonus'] +
            small_transaction_rewards['volume_consistency_bonus'] +
            # NEW: Add exploration bonuses
            exploration_bonuses['action_specific_bonus'] +
            exploration_bonuses['diversity_bonus'] +
            exploration_bonuses['exploration_bonus'] +
            exploration_bonuses['close_quality_bonus'] +
            zero_size_penalty
        )
        
        reward_breakdown['total_points'] = total_reward
        
        # Update state
        self.total_reward_points += total_reward
        self.step_count += 1
        self._update_tracking_variables(action, current_price, is_profitable_step, balance, position_size)
        
        # Intelligent logging
        if self._should_log_reward(total_reward):
            self._log_reward_breakdown(reward_breakdown)
        
        return total_reward, reward_breakdown
    
    def _calculate_unrealized_pnl_points(self, current_price: float, position_size: float, balance: float) -> float:
        """Calculate risk-adjusted unrealized P&L points"""
        if position_size == 0 or self.position_entry_price is None:
            return 0.0
        
        # Calculate profit/loss percentage
        if position_size > 0:  # Long position
            profit_pct = (current_price - self.position_entry_price) / self.position_entry_price
        else:  # Short position
            profit_pct = (self.position_entry_price - current_price) / self.position_entry_price
        
        # Convert to base points
        if profit_pct > 0:
            points = profit_pct * self.PROFIT_MULTIPLIER
        else:
            points = profit_pct * abs(self.LOSS_MULTIPLIER)
          # Apply risk-adjusted scaling
        if self.USE_RISK_ADJUSTED_SCALING:
            # Scale by percentage of balance invested, not raw position size
            position_value = abs(position_size) * current_price
            balance_percentage = position_value / balance if balance > 0 else 1.0
            # Cap the scaling to prevent excessive rewards for large positions
            scaling_factor = min(balance_percentage, self.POSITION_SIZE_SCALING_CAP)
            points *= scaling_factor
        else:
            # Original scaling by absolute position size
            points *= abs(position_size)
        
        return points
    
    def _calculate_realized_pnl_points(self, current_price: float, position_size: float, balance: float) -> Tuple[float, float]:
        """CORRECTED: Calculate enhanced points for realized P&L when position closes"""
        # CRITICAL FIX: Check for all required tracking variables
        if (self.position_entry_price is None or 
            self.position_entry_balance is None or 
            self.position_entry_size is None or
            self.position_entry_action is None):
            return 0.0, 0.0
        
        actual_pnl_dollar_amount = 0.0
        # CRITICAL FIX: Calculate PnL using the stored position action and size
        if self.position_entry_action == 1:  # Was long position
            actual_pnl_dollar_amount = self.position_entry_size * (current_price - self.position_entry_price)
        elif self.position_entry_action == 2:  # Was short position
            actual_pnl_dollar_amount = self.position_entry_size * (self.position_entry_price - current_price)
        else:
            return 0.0, 0.0  # Should not happen
        
        # Convert to percentage of entry balance for normalization
        pnl_percentage = actual_pnl_dollar_amount / self.position_entry_balance if self.position_entry_balance > 0 else 0.0
        
        # Calculate base points
        if pnl_percentage > 0:
            points = pnl_percentage * self.PROFIT_MULTIPLIER
            # CRITICAL FIX #2: Make profitable exits the most significant positive reward
            # Add extra multiplier for profitable realized trades
            profit_exit_multiplier = self.config.get('profit_exit_multiplier', 100.0)  # Large multiplier for profitable exits
            points *= profit_exit_multiplier
        else:
            points = pnl_percentage * abs(self.LOSS_MULTIPLIER)
            # Apply loss penalty multiplier for closing losing trades
            loss_exit_multiplier = self.config.get('loss_exit_multiplier', 50.0)  # Penalty for closing losing trades
            points *= loss_exit_multiplier
        
        # Apply realized P&L bonus
        points *= self.REALIZED_PNL_BONUS
        
        return points, actual_pnl_dollar_amount
    
    def _calculate_position_management_points(self, action: int, position_size: float) -> float:
        """Calculate points for opening, closing, or holding positions"""
        points = 0.0
        
        # Penalize opening new positions to ensure they are high-quality
        if self.last_action == 0 and action != 0:  # From hold to trade
            points += self.POSITION_OPEN_COST
            
        # Penalize closing positions
        elif self.last_action != 0 and action == 0:  # From trade to hold
            points += self.POSITION_CLOSE_COST
            
        # Reward for holding a position (if profitable)
        elif self.last_action != 0 and action != 0 and position_size != 0:
            points += self.HOLD_POSITION_REWARD
        
        # NEW: Penalize for holding when no position is open
        elif position_size == 0 and action == 0:
            points += self.NO_POSITION_HOLD_PENALTY
            
        return points

    def _calculate_risk_management_points(self, position_size: float, balance: float, 
                                        leverage: float, stop_loss_hit: bool, current_price: float) -> float:
        """CORRECTED: Enhanced risk management point calculation"""
        points = 0.0
        
        if position_size == 0:
            return points        # CRITICAL FIX: Calculate position risk using market value, not leveraged value
        position_value = abs(position_size) * current_price  # Market value of the position
        risk_percentage = position_value / balance if balance > 0 else 1.0
        
        # Reward good risk management (position <= 20% of balance)
        if risk_percentage <= 0.2:
            points += self.GOOD_RISK_REWARD
        # Moderate risk (20-50% of balance) - neutral
        elif risk_percentage <= 0.5:
            pass  # No penalty or reward
        # Penalize excessive risk (> 50% of balance)
        else:
            excess_risk = risk_percentage - 0.5
            penalty = self.EXCESSIVE_RISK_PENALTY * excess_risk
            points += penalty
          # Penalize excessive leverage
        if leverage > 20.0:  # Very high leverage threshold
            leverage_penalty = -5.0 * (leverage - 20.0)
            points += leverage_penalty
        elif leverage > 15.0:  # High leverage threshold
            leverage_penalty = -2.0 * (leverage - 15.0)
            points += leverage_penalty
        
        return points
    
    def _calculate_drawdown_penalty(self, current_balance: float) -> float:
        """Calculate penalty for significant drawdowns"""
        if self.peak_balance <= 0:
            return 0.0
        
        # Calculate current drawdown
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Apply penalty if drawdown exceeds threshold
        if drawdown > self.MAX_DRAWDOWN_THRESHOLD:
            excess_drawdown = drawdown - self.MAX_DRAWDOWN_THRESHOLD
            penalty = excess_drawdown * self.DRAWDOWN_PENALTY_MULTIPLIER
            return penalty
        
        return 0.0
    
    def _calculate_market_timing_points(self, action: int, current_price: float, market_data: Dict) -> float:
        """Enhanced market timing points with multiple indicators"""
        if market_data is None:
            return 0.0
        
        points = 0.0
        
        # Get indicators with defaults
        sma_short = market_data.get('sma_short', current_price)
        sma_long = market_data.get('sma_long', current_price)
        rsi = market_data.get('rsi', 50.0)
        volume_trend = market_data.get('volume_trend', 1.0)  # 1.0 = neutral
        
        # Calculate trend strength
        if sma_long > 0:
            trend_strength = abs(sma_short - sma_long) / sma_long
        else:
            trend_strength = 0.0
        
        # Determine trend direction and strength
        is_strong_uptrend = current_price > sma_short > sma_long and trend_strength > 0.02
        is_strong_downtrend = current_price < sma_short < sma_long and trend_strength > 0.02
        is_uptrend = current_price > sma_short > sma_long
        is_downtrend = current_price < sma_short < sma_long
        
        # Reward trend following
        if action == 1:  # Buy action
            if is_strong_uptrend:
                points += self.TREND_FOLLOW_REWARD * (1 + trend_strength)
            elif is_uptrend:
                points += self.TREND_FOLLOW_REWARD
            elif is_downtrend:
                points += self.COUNTER_TREND_PENALTY
        
        elif action == 2:  # Sell action
            if is_strong_downtrend:
                points += self.TREND_FOLLOW_REWARD * (1 + trend_strength)
            elif is_downtrend:
                points += self.TREND_FOLLOW_REWARD
            elif is_uptrend:
                points += self.COUNTER_TREND_PENALTY
        
        # RSI-based timing points
        if action == 1 and rsi < 30:  # Buy when oversold
            points += self.RSI_TIMING_REWARD
        elif action == 2 and rsi > 70:  # Sell when overbought
            points += self.RSI_TIMING_REWARD
        
        # Volume confirmation bonus
        if volume_trend > 1.2 and (is_strong_uptrend or is_strong_downtrend):
            points += 1.0
        
        return points
    
    def _calculate_consistency_points(self, is_profitable_action: bool) -> float:
        """Calculate points for consistent performance"""
        points = 0.0
        
        # Update win streak
        if is_profitable_action:
            self.current_streak += 1
            # Bonus for consecutive profitable actions (capped at 5)
            if self.current_streak > 1:
                points += self.WIN_STREAK_BONUS * min(self.current_streak - 1, 5)
        else:
            self.current_streak = 0
        
        # Consistency bonus for good win rate over recent trades
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            profitable_count = sum(1 for trade in recent_trades if trade['profitable'])
            win_rate = profitable_count / len(recent_trades)
            
            if win_rate >= self.WIN_RATE_THRESHOLD:
                # Bonus proportional to win rate above threshold
                excess_win_rate = win_rate - 0.5  # Above 50%
                points += self.CONSISTENCY_BONUS * excess_win_rate
        
        return points
    
    def _calculate_small_transaction_incentives(self, position_size: float, balance: float, 
                                               action: int, current_price: float) -> Dict:
        """
        Calculate incentives to encourage smaller, more frequent transactions
        
        Args:
            position_size: Current position size
            balance: Current account balance
            action: Action taken (0=hold, 1=buy, 2=sell)
            current_price: Current asset price
            
        Returns:
            Dictionary with small transaction reward components
        """
        incentives = {
            'small_position_bonus': 0.0,
            'position_size_penalty': 0.0,
            'frequency_bonus': 0.0,
            'gradual_building_bonus': 0.0,
            'volume_consistency_bonus': 0.0
        }
        
        # Calculate position value as percentage of balance
        position_value = abs(position_size) * current_price
        position_pct = position_value / balance if balance > 0 else 0
        
        # === 1. SMALL POSITION BONUS ===
        # Reward positions under the small threshold
        if 0 < position_pct <= self.SMALL_POSITION_THRESHOLD:
            # More bonus for smaller positions
            size_factor = (self.SMALL_POSITION_THRESHOLD - position_pct) / self.SMALL_POSITION_THRESHOLD
            incentives['small_position_bonus'] = self.SMALL_POSITION_BONUS * size_factor
          # === 2. POSITION SIZE PENALTY (FIXED: Clamped to prevent explosion) ===
        # Exponentially penalize large positions with hard cap
        if position_pct > self.LARGE_POSITION_THRESHOLD:
            excess_ratio = (position_pct - self.LARGE_POSITION_THRESHOLD) / self.LARGE_POSITION_THRESHOLD
            # FIXED: Clamp the penalty multiplier to prevent exponential explosion
            penalty_multiplier = min(excess_ratio ** self.POSITION_SIZE_PENALTY_EXPONENT, 50.0)  # Cap at 50x
            raw_penalty = self.LARGE_POSITION_PENALTY * penalty_multiplier
            # FIXED: Hard cap the total penalty to prevent destabilizing training
            incentives['position_size_penalty'] = max(raw_penalty, -1000.0)  # Never more than -1000
        
        # === 3. TRANSACTION FREQUENCY BONUS ===
        # Update tracking variables
        self.steps_since_last_trade += 1
        if action != 0:  # If action is not hold
            self.trade_frequency_counter += 1
            self.steps_since_last_trade = 0
            
            # Track trade size for volume consistency
            self.recent_trade_sizes.append(position_pct)
            if len(self.recent_trade_sizes) > 20:  # Keep last 20 trades
                self.recent_trade_sizes.pop(0)
        
        # Reward optimal trading frequency
        if self.step_count > 0:
            current_frequency = (self.trade_frequency_counter / self.step_count) * 100
            optimal_freq = self.OPTIMAL_TRADE_FREQUENCY
            
            # Bonus if close to optimal frequency
            if abs(current_frequency - optimal_freq) < optimal_freq * 0.3:  # Within 30% of optimal
                frequency_factor = 1.0 - abs(current_frequency - optimal_freq) / optimal_freq
                incentives['frequency_bonus'] = self.FREQUENT_TRADING_BONUS * frequency_factor
        
        # === 4. GRADUAL POSITION BUILDING BONUS ===
        # Reward incremental position increases rather than large jumps
        if action != 0 and hasattr(self, 'last_position_size'):
            position_change = abs(position_size) - abs(self.last_position_size)
            if position_change > 0:  # Position increased
                change_pct = position_change * current_price / balance
                
                # Bonus for small incremental increases
                if 0 < change_pct <= self.SMALL_POSITION_THRESHOLD / 2:
                    self.position_build_steps.append(change_pct)
                    if len(self.position_build_steps) > 10:
                        self.position_build_steps.pop(0)
                    
                    # More bonus for consistent small increases
                    if len(self.position_build_steps) >= 3:
                        avg_increase = np.mean(self.position_build_steps)
                        if avg_increase <= self.SMALL_POSITION_THRESHOLD / 2:
                            incentives['gradual_building_bonus'] = self.GRADUAL_POSITION_BONUS
        
        # === 5. VOLUME CONSISTENCY BONUS ===
        # Reward consistent trading volumes within preferred range
        if len(self.recent_trade_sizes) >= 5:
            avg_size = np.mean(self.recent_trade_sizes)
            size_std = np.std(self.recent_trade_sizes)
            
            min_pref, max_pref = self.PREFERRED_VOLUME_RANGE
            
            # Bonus if average size is in preferred range and consistent
            if min_pref <= avg_size <= max_pref and size_std < avg_size * 0.5:  # Low variance
                consistency_factor = 1.0 - (size_std / avg_size)  # Higher bonus for lower variance
                incentives['volume_consistency_bonus'] = self.CONSISTENT_VOLUME_BONUS * consistency_factor
        
        # Store for next calculation
        self.last_position_size = position_size
        
        return incentives

    def _update_tracking_variables(self, action: int, current_price: float, 
                                 is_profitable: bool, balance: float, position_size: float):
        """CORRECTED: Enhanced tracking with detailed trade records"""
        # Track position entry
        if self.last_action == 0 and action != 0:  # Opening new position
            self.position_entry_price = current_price
            self.position_entry_balance = balance
            self.position_entry_size = abs(position_size)  # CRITICAL FIX: Store position size
            self.position_entry_step = self.step_count
            self.position_entry_action = action  # CRITICAL FIX: Store the position type        # Track position exit with enhanced data
        # CORRECTED: Detect position closing by position size change, not just action change
        was_position_open_before = self.position_entry_price is not None and self.position_entry_size is not None
        if was_position_open_before and position_size == 0:  # Position closed (any way)
            if (self.position_entry_price is not None and 
                self.position_entry_size is not None):
                
                # CRITICAL FIX: Calculate actual P&L using stored position action and size
                if self.position_entry_action == 1:  # Was long
                    pnl_dollar_amount = self.position_entry_size * (current_price - self.position_entry_price)
                else:  # Was short (self.position_entry_action == 2)
                    pnl_dollar_amount = self.position_entry_size * (self.position_entry_price - current_price)
                
                # Calculate percentage of initial balance for consistency
                pnl_amount_pct_of_balance = pnl_dollar_amount / self.position_entry_balance if self.position_entry_balance > 0 else 0.0
                
                trade_record = {
                    'entry_price': self.position_entry_price,
                    'exit_price': current_price,
                    'entry_balance': self.position_entry_balance,
                    'exit_balance': balance,
                    'position_size': self.position_entry_size,
                    'pnl_amount': pnl_amount_pct_of_balance,  # As percentage of initial trade capital
                    'pnl_dollar_amount': pnl_dollar_amount,   # CORRECTED: Actual dollar P&L
                    'profitable': is_profitable,
                    'duration': self.step_count - self.position_entry_step if self.position_entry_step else 0,
                    'action_type': 'long' if self.position_entry_action == 1 else 'short',
                    'step': self.step_count
                }
                self.trade_history.append(trade_record)
            
            # Reset position tracking
            self.position_entry_price = None
            self.position_entry_balance = None
            self.position_entry_size = None  # CRITICAL FIX: Reset position size
            self.position_entry_step = None
            self.position_entry_action = None  # CRITICAL FIX: Reset position action
        
        # CRITICAL FIX: Update last_action at the very end, after all calculations
        self.last_action = action
    
    def _should_log_reward(self, total_reward: float) -> bool:
        """Intelligent logging decision"""
        log_config = self.config.get('log_rewards', False)
        if not log_config:
            return False
        
        # Log conditions
        return (
            self.step_count % 100 == 0 or  # Every 100 steps
            abs(total_reward) > 10.0 or    # Significant rewards
            self.config.get('log_all_rewards', False)  # Override for detailed analysis
        )
    
    def _log_reward_breakdown(self, breakdown: Dict):
        """Log detailed reward breakdown for debugging"""
        significant_components = {k: v for k, v in breakdown.items() 
                                if abs(v) > 0.1 and k != 'total_points'}
        
        if significant_components:
            components_str = ", ".join([f"{k}: {v:.1f}" for k, v in significant_components.items()])
            self.logger.info(f"Step {self.step_count}: {components_str} | Total: {breakdown['total_points']:.1f}")
        else:
            self.logger.info(f"Step {self.step_count}: Total: {breakdown['total_points']:.1f}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary with corrected calculations"""
        if not self.trade_history:
            return {'no_trades': True}
        
        profitable_trades = [t for t in self.trade_history if t['profitable']]
        losing_trades = [t for t in self.trade_history if not t['profitable']]
        
        # Calculate metrics
        total_pnl = sum(trade['pnl_amount'] for trade in self.trade_history)
        total_dollar_pnl = sum(trade['pnl_dollar_amount'] for trade in self.trade_history)
        
        avg_win = np.mean([t['pnl_amount'] for t in profitable_trades]) if profitable_trades else 0.0
        avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0.0
        
        return {
            'total_trades': len(self.trade_history),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(self.trade_history),
            'total_pnl_percentage': total_pnl,
            'total_dollar_pnl': total_dollar_pnl,
            'average_win_percentage': avg_win,
            'average_loss_percentage': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_consecutive_wins': self._calculate_max_streak(True),
            'max_consecutive_losses': self._calculate_max_streak(False),
            'current_streak': self.current_streak,
            'total_reward_points': self.total_reward_points,
            'avg_trade_duration': np.mean([t['duration'] for t in self.trade_history]),
            'max_drawdown': self._calculate_max_drawdown(),
            'total_steps': self.step_count
        }
    
    def _calculate_max_streak(self, profitable: bool) -> int:
        """Calculate maximum consecutive winning or losing streak"""
        if not self.trade_history:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in self.trade_history:
            if trade['profitable'] == profitable:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from balance history"""
        if len(self.balance_history) < 2:
            return 0.0
        
        peak = self.balance_history[0]
        max_dd = 0.0
        
        for balance in self.balance_history:
            if balance > peak:
                peak = balance
            else:
                drawdown = (peak - balance) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def reset(self):
        """Reset the reward calculator for new episode"""
        self.trade_history = []
        self.current_streak = 0
        self.total_reward_points = 0.0
        self.last_action = 0
        self.position_entry_price = None
        self.position_entry_balance = None
        self.position_entry_size = None  # CRITICAL FIX: Reset position size
        self.position_entry_step = None
        self.position_entry_action = None  # CRITICAL FIX: Reset position action
        self.step_count = 0
        self.initial_balance = None
        self.peak_balance = 0.0
        self.balance_history = []
        self.episode_start_balance = None

    def _calculate_holding_time_penalty(self, position_size: float) -> float:
        """
        CRITICAL FIX: Calculate time-based penalty for holding positions too long
        This addresses Issue #4: Holding Losing Trades Indefinitely
        
        Args:
            position_size: Current position size (0 = no position)
            
        Returns:
            Penalty points (negative) for holding positions over time
        """
        if position_size == 0:
            # No position, no penalty
            return 0.0
        
        # Calculate how long the position has been held
        if self.position_entry_step is not None:
            holding_steps = self.step_count - self.position_entry_step
        else:
            # Fallback if position entry step not tracked
            holding_steps = 0
        
        penalty = 0.0
        
        # Apply small continuous penalty for holding any position
        if holding_steps > 0:
            penalty += self.HOLDING_TIME_PENALTY * holding_steps
        
        # Apply larger penalty for excessively long holds
        if holding_steps > self.MAX_HOLDING_STEPS:
            excessive_steps = holding_steps - self.MAX_HOLDING_STEPS
            penalty += self.EXCESSIVE_HOLDING_PENALTY * excessive_steps
        
        return penalty

    def _calculate_exploration_bonuses(self, action: int, position_size: float, balance: float) -> Dict[str, float]:
        """
        Calculate exploration bonuses for balanced action diversity
        
        Args:
            action: Current action taken (0=hold, 1=long, 2=short, 3+=close)
            position_size: Current position size
            balance: Current account balance
            
        Returns:
            Dictionary with different exploration bonus components
        """
        bonuses = {
            'action_specific_bonus': 0.0,
            'diversity_bonus': 0.0, 
            'exploration_bonus': 0.0,
            'close_quality_bonus': 0.0
        }
        
        # === 1. ACTION-SPECIFIC BONUSES ===
        if action == 1:  # LONG/BUY action
            bonuses['action_specific_bonus'] = self.LONG_TRADE_BONUS
        elif action == 2:  # SHORT/SELL action  
            bonuses['action_specific_bonus'] = self.SHORT_TRADE_BONUS
        elif action >= 3:  # CLOSE actions (3-12)
            bonuses['action_specific_bonus'] = self.CLOSE_ACTION_BONUS
            
            # === CLOSE QUALITY BONUS ===
            # Check if this is a profitable or losing close
            if hasattr(self, 'position_entry_price') and self.position_entry_price is not None:
                # Estimate if close would be profitable (simplified)
                # This would need actual price data, so we'll use a placeholder
                bonuses['close_quality_bonus'] = self.CLOSE_PROFITABLE_BONUS * 0.5  # Average bonus
        
        # === 2. ACTION DIVERSITY TRACKING ===
        # Track recent actions for diversity calculation
        if not hasattr(self, 'recent_actions'):
            self.recent_actions = []
        
        self.recent_actions.append(action)
        
        # Keep only recent actions within window
        if len(self.recent_actions) > self.RECENT_ACTIONS_WINDOW:
            self.recent_actions = self.recent_actions[-self.RECENT_ACTIONS_WINDOW:]
        
        # Calculate diversity bonus
        if len(self.recent_actions) >= 5:  # Need minimum history
            unique_actions = len(set(self.recent_actions))
            if unique_actions >= 3:  # Used at least 3 different action types
                diversity_ratio = unique_actions / len(set(self.recent_actions[-10:]))  # Last 10 actions
                bonuses['diversity_bonus'] = self.ACTION_DIVERSITY_REWARD * diversity_ratio
        
        # === 3. POSITION DIVERSITY BONUS ===
        # Track if agent uses both LONG and SHORT strategies
        if not hasattr(self, 'used_long_trades'):
            self.used_long_trades = False
            self.used_short_trades = False
        
        if action == 1:  # LONG
            self.used_long_trades = True
        elif action == 2:  # SHORT
            self.used_short_trades = True
            
        # Bonus for using both strategies
        if self.used_long_trades and self.used_short_trades:
            bonuses['diversity_bonus'] += self.POSITION_DIVERSITY_BONUS * 0.1  # Small ongoing bonus
        
        # === 4. GENERAL EXPLORATION BONUS ===
        # Small bonus for any non-hold action
        if action != 0:  # Not holding
            bonuses['exploration_bonus'] = self.EXPLORATION_BONUS * 0.5
            
        return bonuses
    
# === CONFIGURATION CONSTANTS ===

# CORRECTED Configuration Presets
CONSERVATIVE_ENHANCED_CONFIG = {
    'profit_multiplier': 50.0,           # Lower profit emphasis for stability
    'loss_multiplier': -150.0,          # Higher loss penalty for risk aversion
    'realized_pnl_bonus': 2.0,          # Higher bonus for completing trades
    'good_risk_reward': 10.0,           # Higher reward for good risk management
    'excessive_risk_penalty': -50.0,    # Higher penalty for excessive risk
    'max_drawdown_threshold': 0.03,     # Stricter 3% drawdown threshold
    'stop_loss_hit_reward': -25.0,       # FIXED: Penalty for hitting stop-loss
    'commission_penalty_multiplier': -2.0,  # Lower commission penalty for conservative approach
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 1.5,   # Lower scaling cap
    # CRITICAL FIX: Time-based holding penalty parameters
    'holding_time_penalty': -0.05,       # Small penalty per step (-0.05 per step)
    'max_holding_steps': 480,            # 40 hours for 5min bars (conservative)
    'excessive_holding_penalty': -1.0,   # Moderate penalty for excessive holding
    # CRITICAL FIX #2: Conservative incentives for closing trades
    'profit_exit_multiplier': 150.0,     # Strong multiplier for profitable exits
    'loss_exit_multiplier': 50.0,        # Moderate penalty for closing losing trades
    'rejected_action_penalty': -25.0,    # CRITICAL FIX #3: Penalty for rejected trades
    'log_rewards': False
}

AGGRESSIVE_ENHANCED_CONFIG = {
    'profit_multiplier': 200.0,         # Higher profit emphasis
    'loss_multiplier': -100.0,          # Lower loss penalty
    'realized_pnl_bonus': 1.2,          # Lower bonus (more focus on unrealized)
    'trend_follow_reward': 5.0,         # Higher trend following reward
    'win_streak_bonus': 5.0,            # Higher streak bonus
    'good_risk_reward': 3.0,            # Lower risk management reward
    'excessive_risk_penalty': -15.0,    # Lower risk penalty
    'max_drawdown_threshold': 0.10,     # Relaxed 10% drawdown threshold
    'commission_penalty_multiplier': -10.0,  # Higher commission penalty to encourage efficiency
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 3.0,   # Higher scaling cap
    # CRITICAL FIX: Time-based holding penalty parameters
    'holding_time_penalty': -0.1,        # Slightly higher penalty per step for aggressive style
    'max_holding_steps': 288,            # 24 hours for 5min bars (aggressive)
    'excessive_holding_penalty': -2.0,   # Higher penalty for excessive holding
    'log_rewards': False
}

BALANCED_ENHANCED_CONFIG = {
    'profit_multiplier': 100.0,         # Balanced profit emphasis
    'loss_multiplier': -100.0,          # Balanced loss penalty
    'realized_pnl_bonus': 1.5,          # Moderate realized bonus
    'good_risk_reward': 5.0,            # Moderate risk management reward
    'excessive_risk_penalty': -30.0,    # Increased risk penalty to discourage large positions
    'trend_follow_reward': 3.0,         # Moderate trend following
    'win_streak_bonus': 2.0,            # Moderate streak bonus
    'max_drawdown_threshold': 0.05,     # Standard 5% drawdown threshold
    'stop_loss_hit_reward': -20.0,       # FIXED: Penalty for hitting stop-loss    'commission_penalty_multiplier': -5.0,  # Balanced commission penalty
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 2.0,   # Standard scaling cap
    
    # CRITICAL FIX: Time-based holding penalty parameters
    'holding_time_penalty': -0.1,        # Standard penalty per step
    'max_holding_steps': 360,            # 30 hours for 5min bars (balanced)
    'excessive_holding_penalty': -1.5,   # Balanced penalty for excessive holding
    
    # CRITICAL FIX #2: Massive incentives for closing trades
    'profit_exit_multiplier': 200.0,     # HUGE multiplier for profitable exits
    'loss_exit_multiplier': 75.0,        # Strong penalty for closing losing trades (encourages better exits)
    'rejected_action_penalty': -50.0,    # CRITICAL FIX #3: Strong penalty for rejected trades
    
    # NEW: Small transaction incentives
    'small_position_bonus': 15.0,       # Strong bonus for small positions
    'small_position_threshold': 0.15,   # 15% of balance threshold for small position
    'large_position_penalty': -40.0,    # Heavy penalty for large positions
    'large_position_threshold': 0.4,    # 40% of balance threshold for large position
    'position_size_penalty_exponent': 2.5,  # Exponential penalty for oversized positions
    
    'frequent_trading_bonus': 8.0,      # Bonus for optimal trading frequency
    'gradual_position_bonus': 12.0,     # Bonus for building positions gradually
    'optimal_trade_frequency': 12,      # 12 trades per 100 steps
      'consistent_volume_bonus': 6.0,     # Bonus for consistent trade volumes
    'preferred_volume_range': (0.05, 0.20),  # 5-20% of balance per trade
    
    # NEW: Trade management rewards/penalties
    'trade_limit_penalty_base': -10.0,   # Base penalty for opening trades over limit
    'trade_limit_penalty_scale': 2.0,    # Scale factor as trades approach max
    'close_bonus_base': 5.0,             # Base bonus for closing trades when too many open
    'close_bonus_scale': 1.0,            # Scale factor for close bonus
    
    'log_rewards': False
}

# AGGRESSIVELY TUNED: Small transaction focused configuration with maximum incentives
SMALL_TRANSACTION_CONFIG = {
    'profit_multiplier': 70.0,          # Lower profit emphasis to focus on frequency
    'loss_multiplier': -70.0,           # Lower loss penalty to encourage small risk-taking
    'realized_pnl_bonus': 4.0,          # Much higher realized bonus
    'good_risk_reward': 12.0,           # Much higher risk management reward
    'excessive_risk_penalty': -70.0,    # Heavy risk penalty for large positions
    'trend_follow_reward': 1.5,         # Lower trend following (encourage frequent reversals)
    'win_streak_bonus': 4.0,            # Higher streak bonus
    'max_drawdown_threshold': 0.025,    # Stricter 2.5% drawdown threshold
    'stop_loss_hit_reward': -30.0,       # FIXED: Strong penalty for hitting stop-loss    'commission_penalty_multiplier': -2.5,  # Much lower commission penalty (encourage high frequency)
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 1.3,   # Lower scaling cap
    
    # CRITICAL FIX: Time-based holding penalty parameters  
    'holding_time_penalty': -0.15,       # Higher penalty per step for small transaction focus
    'max_holding_steps': 240,            # 20 hours for 5min bars (aggressive)
    'excessive_holding_penalty': -3.0,   # Strong penalty for excessive holding
    
    # MAXIMUM small transaction incentives
    'small_position_bonus': 40.0,       # Massive bonus for small positions
    'small_position_threshold': 0.10,   # 10% of balance threshold (stricter)
    'large_position_penalty': -100.0,   # Massive penalty for large positions
    'large_position_threshold': 0.25,   # 25% of balance threshold (stricter)
    'position_size_penalty_exponent': 3.5,  # Very aggressive exponential penalty
    
    'frequent_trading_bonus': 25.0,     # Massive bonus for frequent trading
    'gradual_position_bonus': 30.0,     # Massive bonus for gradual building
    'optimal_trade_frequency': 20,      # 20 trades per 100 steps (very frequent)
      'consistent_volume_bonus': 15.0,    # Very strong bonus for consistency
    'preferred_volume_range': (0.02, 0.10),  # 2-10% of balance per trade (much smaller)
    
    # NEW: Trade management rewards/penalties
    'trade_limit_penalty_base': -15.0,   # Higher penalty for opening trades over limit
    'trade_limit_penalty_scale': 2.5,    # Higher scale factor as trades approach max
    'close_bonus_base': 8.0,             # Higher bonus for closing trades when too many open
    'close_bonus_scale': 1.5,            # Higher scale factor for close bonus
    
    'log_rewards': False
}

# ULTRA-AGGRESSIVE: Small transaction focused configuration with EXTREME incentives
ULTRA_SMALL_TRANSACTION_CONFIG = {
    'profit_multiplier': 50.0,          # Much lower profit emphasis to focus on frequency over profit
    'loss_multiplier': -50.0,           # Much lower loss penalty to encourage micro risk-taking
    'realized_pnl_bonus': 5.0,          # Very high realized bonus
    'good_risk_reward': 20.0,           # Extremely high risk management reward
    'excessive_risk_penalty': -120.0,   # Extreme risk penalty for large positions
    'trend_follow_reward': 0.5,         # Very low trend following (encourage micro reversals)
    'win_streak_bonus': 8.0,            # Extremely high streak bonus
    'max_drawdown_threshold': 0.015,    # Ultra-strict 1.5% drawdown threshold
    'stop_loss_hit_reward': -40.0,       # FIXED: Maximum penalty for hitting stop-loss    'commission_penalty_multiplier': -1.5,  # Minimal commission penalty (encourage ultra-high frequency)
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 1.1,   # Very low scaling cap
    
    # CRITICAL FIX: Time-based holding penalty parameters
    'holding_time_penalty': -0.2,        # Maximum penalty per step for ultra-small transaction focus
    'max_holding_steps': 144,            # 12 hours for 5min bars (ultra-aggressive)
    'excessive_holding_penalty': -5.0,   # Maximum penalty for excessive holding
    
    # CRITICAL FIX #2: EXTREME incentives for closing trades
    'profit_exit_multiplier': 500.0,     # MASSIVE multiplier for profitable exits
    'loss_exit_multiplier': 200.0,       # Very strong penalty for closing losing trades
    'rejected_action_penalty': -100.0,   # CRITICAL FIX #3: EXTREME penalty for rejected trades
    
    # EXTREME small transaction incentives
    'small_position_bonus': 80.0,       # MASSIVE bonus for small positions
    'small_position_threshold': 0.06,   # 6% of balance threshold (ultra-small)
    'large_position_penalty': -200.0,   # EXTREME penalty for large positions
    'large_position_threshold': 0.15,   # 15% of balance threshold (ultra-strict)
    'position_size_penalty_exponent': 5.0,  # Ultra-aggressive exponential penalty
    
    'frequent_trading_bonus': 50.0,     # MASSIVE bonus for frequent trading
    'gradual_position_bonus': 60.0,     # MASSIVE bonus for gradual building
    'optimal_trade_frequency': 35,      # 35 trades per 100 steps (ultra-frequent)
      'consistent_volume_bonus': 30.0,    # Extremely strong bonus for consistency
    'preferred_volume_range': (0.005, 0.06),  # 0.5-6% of balance per trade (ultra-small)
    
    # NEW: Trade management rewards/penalties
    'trade_limit_penalty_base': -20.0,   # Maximum penalty for opening trades over limit
    'trade_limit_penalty_scale': 3.0,    # Maximum scale factor as trades approach max
    'close_bonus_base': 10.0,            # Maximum bonus for closing trades when too many open
    'close_bonus_scale': 2.0,            # Maximum scale factor for close bonus
    
    'log_rewards': False
}
