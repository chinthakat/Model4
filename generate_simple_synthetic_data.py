"""
Simple synthetic data generator that matches the original BINANCE data format exactly.
This creates balanced LONG, SHORT, and HOLD patterns for RL training.
"""

import pandas as pd
import numpy as np
import datetime
import os

class SimpleSyntheticGenerator:
    def __init__(self, base_price=42000, volatility=0.02):
        self.base_price = base_price
        self.volatility = volatility
        
    def generate_balanced_data(self, num_candles=20000, start_timestamp=1704067200):
        """Generate synthetic data with the exact same format as original BINANCE data."""
        data = []
        current_price = self.base_price
        current_timestamp = start_timestamp
        
        # Create pattern segments
        patterns = ['uptrend', 'downtrend', 'sideways'] * (num_candles // 3 + 1)
        np.random.shuffle(patterns)
        
        for i in range(num_candles):
            pattern = patterns[i % len(patterns)]
            
            # Generate price movement based on pattern
            if pattern == 'uptrend':
                # Favor upward movement (good for LONG)
                price_change = np.random.uniform(0.0005, 0.003)
                volume_multiplier = np.random.uniform(1.2, 2.0)
            elif pattern == 'downtrend':
                # Favor downward movement (good for SHORT)
                price_change = np.random.uniform(-0.003, -0.0005)
                volume_multiplier = np.random.uniform(1.2, 2.0)
            else:  # sideways
                # Small movements (good for HOLD)
                price_change = np.random.uniform(-0.0008, 0.0008)
                volume_multiplier = np.random.uniform(0.8, 1.2)
            
            # Add noise
            noise = np.random.normal(0, self.volatility * 0.3)
            total_change = price_change + noise
            
            # Clamp to prevent extreme moves
            total_change = np.clip(total_change, -0.05, 0.05)
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + total_change)
            
            # Generate realistic high/low
            volatility = abs(total_change) + np.random.uniform(0.001, 0.005)
            high_price = max(open_price, close_price) * (1 + volatility * 0.4)
            low_price = min(open_price, close_price) * (1 - volatility * 0.4)
            
            # Generate volume
            base_volume = 2000
            volume = base_volume * volume_multiplier * np.random.uniform(0.7, 1.3)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'timestamp': current_timestamp
            })
            
            current_price = close_price
            current_timestamp += 900  # 15 minutes
            
            # Prevent price from drifting too far
            if current_price < self.base_price * 0.7:
                current_price = self.base_price * 0.75
            elif current_price > self.base_price * 1.4:
                current_price = self.base_price * 1.35
        
        return pd.DataFrame(data)

def main():
    print("🎯 Generating simple synthetic BTC data matching original format...")
    
    generator = SimpleSyntheticGenerator(base_price=42000, volatility=0.015)
    df = generator.generate_balanced_data(num_candles=20000)
    
    # Reset index and add it as unnamed column (to match original format)
    df = df.reset_index(drop=True)
    
    # Create output directory
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with exact same format as original
    output_path = os.path.join(output_dir, "SYNTHETIC_SIMPLE_BTC_15m_training.csv")
    df.to_csv(output_path, index=True)
    
    # Print statistics
    print(f"\n✅ Synthetic dataset created successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Dataset Statistics:")
    print(f"   Total candles: {len(df)}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Start price: ${df['open'].iloc[0]:.2f}")
    print(f"   End price: ${df['close'].iloc[-1]:.2f}")
    print(f"   Total return: {((df['close'].iloc[-1] / df['open'].iloc[0]) - 1) * 100:.2f}%")
    
    # Check header format matches original
    print(f"\n📋 Column structure:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    
    # Show sample of data
    print(f"\n📄 Sample data (first 3 rows):")
    print(df.head(3).to_string())
    
    print(f"\n🎯 This dataset has balanced patterns suitable for:")
    print(f"   📈 LONG strategies (uptrend periods)")
    print(f"   📉 SHORT strategies (downtrend periods)")
    print(f"   🔄 HOLD strategies (sideways periods)")
    print(f"\n🚀 Use this file in your RL training pipeline!")

if __name__ == "__main__":
    main()
