#!/usr/bin/env python3
"""
Enhanced Reward Configuration for Better Exploration
Addresses the issues with SHORT/LONG/CLOSE action exploration
"""

# Enhanced Exploration Configuration
ENHANCED_EXPLORATION_CONFIG = {
    # Base rewards - more balanced between profit and exploration
    'profit_multiplier': 120.0,          # Good profit reward but not dominant
    'loss_multiplier': -80.0,            # Reduced loss penalty to encourage exploration
    'realized_pnl_bonus': 2.5,           # Good bonus for completing trades
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 1.8,    # Allow some scaling but not excessive
    
    # === CRITICAL: Enhanced action-specific bonuses ===
    'close_action_bonus': 50.0,          # MAJOR bonus for any close action
    'short_trade_bonus': 25.0,           # NEW: Specific bonus for SHORT trades
    'long_trade_bonus': 20.0,            # NEW: Slightly lower bonus for LONG trades
    'position_diversity_bonus': 30.0,    # NEW: Bonus for using both LONG and SHORT
    'exploration_bonus': 15.0,           # NEW: General exploration reward
    
    # === Position management with less aggressive penalties ===
    'position_open_cost': -1.0,          # Small cost to open
    'position_close_cost': 0.0,          # NO cost to close (encourage closing)
    'hold_position_reward': 0.05,        # Small reward for holding profitable positions
    'no_position_hold_penalty': -2.0,    # Moderate penalty for doing nothing
    
    # === Reduced time-based penalties to allow longer-term strategies ===
    'holding_time_penalty': -0.3,        # Increased penalty to encourage action
    'max_holding_steps': 96,             # 24 hours for 15min bars (reasonable)
    'excessive_holding_penalty': -10.0,  # Strong penalty for excessive holding
    
    # === Position size management - encourage medium sizes ===
    'small_position_bonus': 10.0,        # Modest bonus for small positions
    'small_position_threshold': 0.15,    # 15% of balance
    'large_position_penalty': -30.0,     # Moderate penalty for large positions
    'large_position_threshold': 0.4,     # 40% of balance (more permissive)
    'position_size_penalty_exponent': 1.8,  # Less aggressive penalty curve
    
    # === NEW: Multiple position penalties ===
    'multiple_positions_penalty': -20.0, # Penalty for having too many open positions
    'position_limit_threshold': 3,       # Start penalizing after 3 open positions
    
    # === Enhanced trade completion rewards ===
    'close_profitable_bonus': 40.0,      # NEW: Extra bonus for closing profitable trades
    'close_losing_penalty': -15.0,       # NEW: Reduced penalty for closing losing trades
    
    # === Action diversity rewards ===
    'action_diversity_reward': 10.0,     # NEW: Reward for using different actions
    'recent_actions_window': 20,         # NEW: Window to track action diversity
    
    # === Risk management - more permissive ===
    'good_risk_reward': 8.0,             # Moderate risk management reward
    'excessive_risk_penalty': -40.0,     # Moderate excessive risk penalty
    'stop_loss_hit_reward': -25.0,       # Moderate stop loss penalty
    'drawdown_penalty_multiplier': -30.0, # Moderate drawdown penalty
    'max_drawdown_threshold': 0.08,      # 8% drawdown threshold (more permissive)
    
    # === Market timing - balanced ===
    'trend_follow_reward': 5.0,          # Good trend following reward
    'counter_trend_penalty': -8.0,       # Moderate counter-trend penalty
    'rsi_timing_reward': 3.0,            # Moderate RSI timing reward
    
    # === Consistency - encourage but don't dominate ===
    'win_streak_bonus': 3.0,             # Moderate win streak bonus
    'consistency_bonus': 15.0,           # Good consistency bonus
    'win_rate_threshold': 0.55,          # 55% win rate threshold
    
    # === Commission - minimal impact ===
    'commission_penalty_multiplier': -2.0, # Minimal commission penalty
    
    # === Transaction frequency - encourage moderate activity ===
    'frequent_trading_bonus': 8.0,       # Moderate frequent trading bonus
    'gradual_position_bonus': 12.0,      # Good gradual building bonus
    'optimal_trade_frequency': 8,        # 8 trades per 100 steps (moderate)
    'consistent_volume_bonus': 5.0,      # Moderate volume consistency bonus
    'preferred_volume_range': (0.1, 0.3), # 10-30% of balance per trade
    
    'log_rewards': True  # Enable detailed logging for analysis
}

# Balanced configuration that encourages all three action types
BALANCED_ALL_ACTIONS_CONFIG = {
    # Base rewards
    'profit_multiplier': 100.0,
    'loss_multiplier': -70.0,
    'realized_pnl_bonus': 2.0,
    'use_risk_adjusted_scaling': True,
    'position_size_scaling_cap': 2.0,
    
    # Action-specific rewards
    'close_action_bonus': 35.0,          # Strong bonus for closing
    'short_trade_bonus': 15.0,           # Bonus for SHORT trades
    'long_trade_bonus': 12.0,            # Bonus for LONG trades
    'hold_reward': 0.1,                  # Small reward for holding when appropriate
    
    # Position management
    'position_open_cost': -0.8,
    'position_close_cost': 0.0,
    'hold_position_reward': 0.08,
    'no_position_hold_penalty': -1.5,
    
    # Time management
    'holding_time_penalty': -0.2,
    'max_holding_steps': 120,
    'excessive_holding_penalty': -8.0,
    
    # Position sizing
    'small_position_bonus': 8.0,
    'small_position_threshold': 0.2,
    'large_position_penalty': -25.0,
    'large_position_threshold': 0.35,
    
    # Risk management
    'good_risk_reward': 6.0,
    'excessive_risk_penalty': -35.0,
    'stop_loss_hit_reward': -20.0,
    'max_drawdown_threshold': 0.06,
    
    'log_rewards': True
}
