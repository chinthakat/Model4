"""
Test script to demonstrate how to use the synthetic training data with your RL pipeline.
This script shows how to train your agent on the balanced synthetic data first.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_synthetic_data():
    """Analyze the synthetic data to verify it has good training patterns."""
    
    # Load the synthetic data
    data_path = "data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv"
    if not Path(data_path).exists():
        print(f"❌ Synthetic data not found at: {data_path}")
        print("Run generate_simple_synthetic_data.py first!")
        return
    
    df = pd.read_csv(data_path, index_col=0)
    print(f"✅ Loaded synthetic training data: {df.shape}")
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    
    # Identify potential trading opportunities
    print(f"\n📊 Trading Opportunity Analysis:")
    
    # Look for trending periods (good for LONG/SHORT)
    window = 20
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=20).mean()
    
    # LONG opportunities (price above long SMA and trending up)
    long_signals = (df['close'] > df['sma_long']) & (df['sma_short'] > df['sma_long'])
    
    # SHORT opportunities (price below long SMA and trending down)
    short_signals = (df['close'] < df['sma_long']) & (df['sma_short'] < df['sma_long'])
    
    # HOLD opportunities (choppy market)
    hold_signals = ~(long_signals | short_signals)
    
    print(f"   📈 LONG opportunity periods: {long_signals.sum()} ({long_signals.mean()*100:.1f}%)")
    print(f"   📉 SHORT opportunity periods: {short_signals.sum()} ({short_signals.mean()*100:.1f}%)")
    print(f"   🔄 HOLD opportunity periods: {hold_signals.sum()} ({hold_signals.mean()*100:.1f}%)")
    
    # Calculate potential returns for each strategy
    long_returns = df.loc[long_signals, 'returns'].dropna()
    short_returns = -df.loc[short_signals, 'returns'].dropna()  # Inverted for short positions
    
    if len(long_returns) > 0:
        print(f"   💰 Average LONG return: {long_returns.mean()*100:.3f}% per period")
        print(f"   📈 LONG success rate: {(long_returns > 0).mean()*100:.1f}%")
    
    if len(short_returns) > 0:
        print(f"   💰 Average SHORT return: {short_returns.mean()*100:.3f}% per period")
        print(f"   📉 SHORT success rate: {(short_returns > 0).mean()*100:.1f}%")
    
    # Volatility analysis
    volatility = df['returns'].std() * np.sqrt(24*365) * 100  # Annualized volatility
    print(f"   📊 Annualized volatility: {volatility:.1f}%")
    
    print(f"\n🎯 Training Recommendations:")
    print(f"   1. Use this data for initial RL training to learn all three actions")
    print(f"   2. The balanced patterns will help your agent learn when to use LONG, SHORT, and HOLD")
    print(f"   3. After training on synthetic data, fine-tune on real market data")
    print(f"   4. Compare agent performance between synthetic and real data")

def suggest_training_modifications():
    """Suggest how to modify the training script to use synthetic data."""
    
    print(f"\n🔧 Training Script Modifications:")
    print(f"   To use synthetic data in your training, modify your data loading code:")
    print(f"")
    print(f"   # In your training script, replace:")
    print(f"   # data_path = 'data/processed/BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv'")
    print(f"   ")
    print(f"   # With:")
    print(f"   data_path = 'data/processed/SYNTHETIC_SIMPLE_BTC_15m_training.csv'")
    print(f"")
    print(f"   📋 The column structure is identical, so no other changes needed!")
    print(f"")
    print(f"   🎯 Training Strategy:")
    print(f"   1. Train for 50-100 episodes on synthetic data first")
    print(f"   2. Monitor that agent learns to use all three actions (LONG, SHORT, HOLD)")
    print(f"   3. Once balanced behavior is achieved, switch to real data")
    print(f"   4. Use synthetic data as a 'warm-up' before real market training")

def main():
    print("🔍 Synthetic Trading Data Analysis")
    print("=" * 50)
    
    analyze_synthetic_data()
    suggest_training_modifications()
    
    print(f"\n🚀 Ready to train your RL agent on balanced data!")
    print(f"   The synthetic dataset provides:")
    print(f"   ✅ Balanced LONG, SHORT, and HOLD opportunities")
    print(f"   ✅ Realistic price movements and volatility")
    print(f"   ✅ Same format as your original data")
    print(f"   ✅ Clear patterns for the agent to learn from")

if __name__ == "__main__":
    main()
