#!/usr/bin/env python3
"""
Synthetic Training Data Generator for Balanced RL Trading
Creates market data with clear LONG, SHORT, and HOLD signals for effective RL training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

class SyntheticMarketDataGenerator:
    """Generate synthetic market data with clear trading patterns"""
    
    def __init__(self, start_price=50000, volatility=0.01):
        self.start_price = start_price
        self.volatility = volatility
        self.timestamp_start = datetime(2024, 1, 1)
        
    def generate_trend_pattern(self, length, trend_strength, noise_level=0.005):
        """Generate a trending pattern (up or down)"""
        trend = np.linspace(0, trend_strength, length)
        noise = np.random.normal(0, noise_level, length)
        return trend + noise
    
    def generate_sideways_pattern(self, length, noise_level=0.003):
        """Generate sideways/ranging pattern"""
        return np.random.normal(0, noise_level, length)
    
    def generate_reversal_pattern(self, length, peak_position=0.5, amplitude=0.03):
        """Generate reversal pattern (good for HOLD scenarios)"""
        x = np.linspace(0, 1, length)
        peak_idx = int(length * peak_position)
        
        # Create a pattern that goes up then down (or down then up)
        pattern = np.zeros(length)
        for i in range(length):
            if i <= peak_idx:
                pattern[i] = amplitude * (i / peak_idx)
            else:
                pattern[i] = amplitude * (1 - (i - peak_idx) / (length - peak_idx))
        
        return pattern
    
    def create_ohlcv_from_returns(self, returns, base_price, volume_base=1000000):
        """Convert returns to OHLCV data"""
        prices = [base_price]
        for ret in returns:
            # Cap extreme returns to prevent overflow
            capped_ret = np.clip(ret, -0.05, 0.05)  # Cap at ±5% per period
            new_price = prices[-1] * (1 + capped_ret)
            # Ensure price doesn't go below $1000 or above $200,000
            new_price = np.clip(new_price, 1000, 200000)
            prices.append(new_price)
        
        ohlcv_data = []
        for i in range(len(prices) - 1):
            open_price = prices[i]
            close_price = prices[i + 1]
            
            # Generate realistic high/low based on volatility
            volatility = min(abs(returns[i]) * 2, 0.02)  # Cap volatility at 2%
            high_price = max(open_price, close_price) * (1 + volatility * np.random.uniform(0.2, 0.8))
            low_price = min(open_price, close_price) * (1 - volatility * np.random.uniform(0.2, 0.8))
            
            # Generate volume with some correlation to price movement
            volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on bigger moves
            volume = int(volume_base * volume_multiplier * np.random.uniform(0.5, 1.5))
            
            ohlcv_data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        return ohlcv_data
    
    def generate_balanced_training_data(self, total_periods=10000):
        """Generate balanced training data with clear LONG, SHORT, and HOLD patterns"""
        
        print("🏗️ Generating balanced training data for RL...")
        
        all_returns = []
        pattern_labels = []  # Track what pattern each segment represents
        
        # Split into roughly equal segments for each pattern type
        segment_length = 100  # Each pattern lasts 100 periods (25 hours at 15min intervals)
        num_segments = total_periods // segment_length
        
        patterns_per_type = num_segments // 3
        
        # 1. LONG-favorable patterns (clear uptrends)
        print(f"📈 Creating {patterns_per_type} LONG-favorable patterns...")
        for i in range(patterns_per_type):
            # Moderate uptrend 
            trend_strength = np.random.uniform(0.04, 0.08)  # 4-8% gain over 100 periods
            returns_per_period = trend_strength / segment_length  # Distribute over periods
            pattern = np.random.normal(returns_per_period, 0.003, segment_length)
            
            # Add small corrections to make it realistic
            correction_points = np.random.choice(segment_length, size=3, replace=False)
            for cp in correction_points:
                if cp < segment_length - 5:
                    correction = np.random.uniform(-0.01, -0.003)  # Small correction
                    pattern[cp:cp+3] += correction
            
            all_returns.extend(pattern)
            pattern_labels.extend(['LONG'] * segment_length)
        
        # 2. SHORT-favorable patterns (clear downtrends)
        print(f"📉 Creating {patterns_per_type} SHORT-favorable patterns...")
        for i in range(patterns_per_type):
            # Moderate downtrend
            trend_strength = np.random.uniform(-0.08, -0.04)  # 4-8% decline over 100 periods
            returns_per_period = trend_strength / segment_length  # Distribute over periods
            pattern = np.random.normal(returns_per_period, 0.003, segment_length)
            
            # Add small bounces to make it realistic
            bounce_points = np.random.choice(segment_length, size=3, replace=False)
            for bp in bounce_points:
                if bp < segment_length - 5:
                    bounce = np.random.uniform(0.003, 0.01)  # Small bounce
                    pattern[bp:bp+3] += bounce
            
            all_returns.extend(pattern)
            pattern_labels.extend(['SHORT'] * segment_length)
        
        # 3. HOLD-favorable patterns (sideways, reversals, choppy markets)
        print(f"🔄 Creating {patterns_per_type} HOLD-favorable patterns...")
        for i in range(patterns_per_type):
            pattern_type = np.random.choice(['sideways', 'reversal', 'choppy'])
            
            if pattern_type == 'sideways':
                # Tight range-bound market
                pattern = self.generate_sideways_pattern(segment_length, noise_level=0.002)
                
            elif pattern_type == 'reversal':
                # Price goes up then down (or down then up) - bad for trend following
                direction = np.random.choice([-1, 1])
                base_pattern = self.generate_reversal_pattern(segment_length, 
                                                            peak_position=np.random.uniform(0.3, 0.7),
                                                            amplitude=np.random.uniform(0.01, 0.02))
                pattern = direction * base_pattern / segment_length  # Convert to per-period returns
                
            else:  # choppy
                # High volatility with no clear direction
                pattern = np.random.normal(0, 0.004, segment_length)
                # Add some fake breakouts that fail
                breakout_points = np.random.choice(segment_length, size=2, replace=False)
                for bp in breakout_points:
                    if bp < segment_length - 10:
                        direction = np.random.choice([-1, 1])
                        # Initial breakout
                        pattern[bp:bp+3] += direction * 0.008
                        # Then reversal
                        pattern[bp+3:bp+8] -= direction * 0.01
            
            all_returns.extend(pattern)
            pattern_labels.extend(['HOLD'] * segment_length)
        
        # Fill remaining periods with mixed patterns
        remaining_periods = total_periods - len(all_returns)
        if remaining_periods > 0:
            print(f"🎯 Adding {remaining_periods} mixed pattern periods...")
            mixed_pattern = np.random.normal(0, self.volatility, remaining_periods)
            all_returns.extend(mixed_pattern)
            pattern_labels.extend(['MIXED'] * remaining_periods)
        
        # Convert to OHLCV data
        print("💱 Converting to OHLCV format...")
        ohlcv_data = self.create_ohlcv_from_returns(all_returns, self.start_price)
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Add timestamps (15-minute intervals)
        timestamps = []
        current_time = self.timestamp_start
        for i in range(len(df)):
            timestamps.append(current_time)
            current_time += timedelta(minutes=15)
        
        df.insert(0, 'timestamp', timestamps)
        
        # Add technical indicators for RL features
        print("📊 Adding technical indicators...")
        df = self.add_technical_indicators(df)
        
        # Add pattern labels for analysis (optional)
        df['pattern_type'] = pattern_labels[:len(df)]
        
        print(f"✅ Generated {len(df)} periods of balanced training data")
        print(f"📈 LONG patterns: {pattern_labels.count('LONG')} periods")
        print(f"📉 SHORT patterns: {pattern_labels.count('SHORT')} periods") 
        print(f"🔄 HOLD patterns: {pattern_labels.count('HOLD')} periods")
        print(f"🎯 MIXED patterns: {pattern_labels.count('MIXED')} periods")
        
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators similar to the original dataset"""
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD_12_26_9'] = exp1 - exp2
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['BBM_20_2.0'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + (bb_std * 2)
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - (bb_std * 2)
        df['BBB_20_2.0'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        df['BBP_20_2.0'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['STOCHk_14_3_3'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(window=3).mean()
        
        # Williams %R
        df['WillR'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_SMA'] = df['Price_Change'].rolling(window=20).mean()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Support/Resistance levels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        # Gap indicators
        df['Gap_Up'] = (df['Open'] > df['Close'].shift()).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Close'].shift()).astype(int)
        
        # Candlestick patterns (simplified)
        df['Doji'] = (np.abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
        df['Hammer'] = ((df['Low'] < df[['Open', 'Close']].min(axis=1)) & 
                       (df['High'] - df[['Open', 'Close']].max(axis=1) < df[['Open', 'Close']].min(axis=1) - df['Low'])).astype(int)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Trading session indicators
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add timestamp as float for compatibility
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        
        # Fill NaN values forward and backward
        df = df.bfill().ffill()
        
        return df

def main():
    """Generate synthetic training dataset"""
    
    print("🎯 Synthetic RL Training Data Generator")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SyntheticMarketDataGenerator(start_price=50000, volatility=0.01)
    
    # Generate balanced training data
    # 10,000 periods = ~104 days of 15-minute data
    df = generator.generate_balanced_training_data(total_periods=10000)
    
    # Save to CSV in the same format as original data
    output_file = output_dir / "SYNTHETIC_BALANCED_BTC_15m_training.csv"
    
    # Reorder columns to match original format (remove pattern_type for final export)
    analysis_df = df.copy()  # Keep for analysis
    
    # Remove pattern_type column for training
    df = df.drop('pattern_type', axis=1)
    
    # Add unnamed index column like original
    df.insert(0, 'Unnamed: 0', range(len(df)))
    
    # Save training data
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved training dataset to: {output_file}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Time range: {pd.to_datetime(df['timestamp'].iloc[0], unit='s')} to {pd.to_datetime(df['timestamp'].iloc[-1], unit='s')}")
    
    # Save analysis version with pattern labels
    analysis_file = output_dir / "SYNTHETIC_BALANCED_BTC_15m_analysis.csv"
    analysis_df.insert(0, 'Unnamed: 0', range(len(analysis_df)))
    analysis_df.to_csv(analysis_file, index=False)
    print(f"📋 Saved analysis dataset (with pattern labels) to: {analysis_file}")
    
    # Generate summary statistics
    print(f"\n📈 Pattern Distribution:")
    pattern_counts = analysis_df['pattern_type'].value_counts()
    for pattern, count in pattern_counts.items():
        percentage = (count / len(analysis_df)) * 100
        print(f"   {pattern}: {count} periods ({percentage:.1f}%)")
    
    # Price statistics
    price_start = df['Close'].iloc[0]
    price_end = df['Close'].iloc[-1]
    price_min = df['Low'].min()
    price_max = df['High'].max()
    total_return = ((price_end - price_start) / price_start) * 100
    
    print(f"\n💰 Price Statistics:")
    print(f"   Start Price: ${price_start:,.2f}")
    print(f"   End Price: ${price_end:,.2f}")
    print(f"   Min Price: ${price_min:,.2f}")
    print(f"   Max Price: ${price_max:,.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    
    # Volatility statistics by pattern
    print(f"\n📊 Volatility by Pattern Type:")
    for pattern in analysis_df['pattern_type'].unique():
        pattern_data = analysis_df[analysis_df['pattern_type'] == pattern]
        volatility = pattern_data['Price_Change'].std() * 100
        avg_return = pattern_data['Price_Change'].mean() * 100
        print(f"   {pattern}: {volatility:.3f}% volatility, {avg_return:.3f}% avg return")
    
    print(f"\n✅ Synthetic training data generation complete!")
    print(f"🎯 This dataset is designed to help your RL agent learn:")
    print(f"   📈 LONG strategies during clear uptrends")
    print(f"   📉 SHORT strategies during clear downtrends") 
    print(f"   🔄 HOLD strategies during choppy/sideways markets")
    print(f"\n🚀 Use this file with your training script for balanced learning!")

if __name__ == "__main__":
    main()
