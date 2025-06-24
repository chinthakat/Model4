#!/usr/bin/env python3
"""
Data preprocessing and feature engineering module
Processes raw OHLCV data and creates technical indicators for RL training
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, Dict, List
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Processes raw market data and creates features for RL training
    Handles technical indicators, normalization, and data splitting
    """
    
    def __init__(self, lookback_window: int = 50):
        """
        Initialize data processor
        
        Args:
            lookback_window: Number of periods to look back for features
        """
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        self.fitted_scalers = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DataProcessor with lookback_window={lookback_window}")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        self.logger.info("Adding technical indicators...")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # 1. Moving Average Convergence/Divergence (MACD)
        macd_data = ta.macd(data['Close'])
        data = data.join(macd_data)
        
        # 2. Relative Strength Index (RSI)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # 3. Simple Moving Averages
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        
        # 4. Exponential Moving Averages
        data['EMA_12'] = ta.ema(data['Close'], length=12)
        data['EMA_26'] = ta.ema(data['Close'], length=26)
        
        # 5. Bollinger Bands
        bb_data = ta.bbands(data['Close'], length=20)
        data = data.join(bb_data)
        
        # 6. Average True Range (ATR) for volatility
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        
        # 7. Stochastic Oscillator
        stoch_data = ta.stoch(data['High'], data['Low'], data['Close'])
        data = data.join(stoch_data)
        
        # 8. Williams %R
        data['WillR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)
        
        # 9. Commodity Channel Index (CCI)
        data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'], length=20)
        
        # 10. Volume indicators
        data['Volume_SMA'] = ta.sma(data['Volume'], length=20)
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # 11. Price change indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_SMA'] = ta.sma(data['Price_Change'], length=10)
        
        # 12. Volatility indicators
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        
        # 13. Support and Resistance levels (simplified)
        data['High_20'] = data['High'].rolling(window=20).max()
        data['Low_20'] = data['Low'].rolling(window=20).min()
        
        # 14. Gap detection
        data['Gap_Up'] = (data['Open'] > data['Close'].shift(1)).astype(int)
        data['Gap_Down'] = (data['Open'] < data['Close'].shift(1)).astype(int)
        
        # 15. Candlestick patterns (basic)
        data['Doji'] = (abs(data['Close'] - data['Open']) <= (data['High'] - data['Low']) * 0.1).astype(int)
        data['Hammer'] = ((data['Close'] > data['Open']) & 
                         ((data['Open'] - data['Low']) > 2 * (data['Close'] - data['Open'])) &
                         ((data['High'] - data['Close']) < (data['Close'] - data['Open']))).astype(int)
        
        self.logger.info(f"Added technical indicators. Shape: {data.shape}")
        return data
    
    def add_futures_specific_features(self, df: pd.DataFrame, funding_rates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add futures-specific features
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            funding_rates: Optional funding rates data
            
        Returns:
            DataFrame with futures-specific features
        """
        self.logger.info("Adding futures-specific features...")
        
        data = df.copy()
        
        # Add funding rates if available
        if funding_rates is not None and not funding_rates.empty:
            # Resample funding rates to match data frequency
            funding_resampled = funding_rates.resample('15T').ffill()
            
            # Merge with main data
            data = data.join(funding_resampled['funding_rate'], how='left')
            data['funding_rate'].fillna(method='ffill', inplace=True)
            data['funding_rate'].fillna(0, inplace=True)
            
            # Funding rate features
            data['funding_rate_sma'] = ta.sma(data['funding_rate'], length=24)  # 24 periods (8 hours for 15min data)
            data['funding_rate_change'] = data['funding_rate'].diff()
            
            self.logger.info("Added real funding rate features")
        else:
            # Skip funding rate features entirely if not available
            self.logger.info("No funding rate data available - skipping funding rate features")
        
        # Remove mock open interest and long/short ratio - these should come from real APIs
        # Focus on price-based features only
        
        self.logger.info("Added futures-specific features")
        return data
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        data = df.copy()
        
        # Extract time components
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Market session indicators
        data['is_asian_session'] = ((data['hour'] >= 0) & (data['hour'] < 8)).astype(int)
        data['is_european_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(int)
        data['is_american_session'] = ((data['hour'] >= 16) & (data['hour'] < 24)).astype(int)
        
        # Weekend indicator
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        self.logger.info("Added time-based features")
        return data
    
    def normalize_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize data using MinMaxScaler
        
        Args:
            df: DataFrame to normalize
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized DataFrame
        """
        self.logger.info(f"Normalizing data (fit_scaler={fit_scaler})...")
        
        data = df.copy()
        
        # Separate price columns for different scaling
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_cols = ['Volume']
        
        # Get all numeric columns except price columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in price_cols]
        
        if fit_scaler:
            # Fit and transform price columns
            if price_cols[0] in data.columns:
                data[price_cols] = self.price_scaler.fit_transform(data[price_cols])
                self.fitted_scalers['price'] = self.price_scaler
            
            # Fit and transform other features
            if feature_cols:
                data[feature_cols] = self.scaler.fit_transform(data[feature_cols])
                self.fitted_scalers['features'] = self.scaler
        else:
            # Only transform using fitted scalers
            if 'price' in self.fitted_scalers and price_cols[0] in data.columns:
                data[price_cols] = self.fitted_scalers['price'].transform(data[price_cols])
            
            if 'features' in self.fitted_scalers and feature_cols:
                data[feature_cols] = self.fitted_scalers['features'].transform(data[feature_cols])
        
        self.logger.info("Data normalization completed")
        return data
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for RL training
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        self.logger.info(f"Creating sequences with lookback_window={self.lookback_window}")
        
        # Select feature columns (exclude non-numeric or target columns)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any remaining NaN values
        df_clean = df[feature_cols].dropna()
        
        if len(df_clean) < self.lookback_window:
            raise ValueError(f"Not enough data points. Need at least {self.lookback_window}, got {len(df_clean)}")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.lookback_window, len(df_clean)):
            # Features: lookback_window previous observations
            seq = df_clean.iloc[i-self.lookback_window:i].values
            sequences.append(seq)
            
            # Target: next Close price (for reward calculation)
            target = df_clean.iloc[i]['Close'] if 'Close' in df_clean.columns else 0
            targets.append(target)
        
        features = np.array(sequences)
        targets = np.array(targets)
        
        self.logger.info(f"Created {len(sequences)} sequences with shape {features.shape}")
        return features, targets
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets (time-based split)
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_index = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        self.logger.info(f"Split data: Train {len(train_df)} rows, Test {len(test_df)} rows")
        return train_df, test_df
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        funding_rates: Optional[pd.DataFrame] = None,
        train_ratio: float = 0.8,
        fit_scalers: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline
        
        Args:
            df: Raw OHLCV DataFrame
            funding_rates: Optional funding rates data
            train_ratio: Proportion for training data
            fit_scalers: Whether to fit scalers on the training data.
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        self.logger.info("Starting complete data preparation pipeline...")
        
        # 1. Add technical indicators
        df_with_indicators = self.add_technical_indicators(df)
        
        # 2. Add futures-specific features
        df_with_futures = self.add_futures_specific_features(df_with_indicators, funding_rates)
        
        # 3. Add time features
        df_with_time = self.create_time_features(df_with_futures)
        
        # 4. Split data (before normalization to avoid data leakage)
        train_df, test_df = self.split_data(df_with_time, train_ratio)
        
        # 5. Normalize data (fit on training data only)
        train_normalized = self.normalize_data(train_df, fit_scaler=fit_scalers)
        test_normalized = self.normalize_data(test_df, fit_scaler=False)
        
        # 6. Drop remaining NaN values
        train_clean = train_normalized.dropna()
        test_clean = test_normalized.dropna()
        
        self.logger.info(f"Data preparation completed. Train: {train_clean.shape}, Test: {test_clean.shape}")
        
        return train_clean, test_clean
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers for later use"""
        import joblib
        joblib.dump(self.fitted_scalers, filepath)
        self.logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load previously fitted scalers"""
        import joblib
        self.fitted_scalers = joblib.load(filepath)
        self.scaler = self.fitted_scalers.get('features')
        self.price_scaler = self.fitted_scalers.get('price')
        self.logger.info(f"Scalers loaded from {filepath}")

def main():
    """Example usage of DataProcessor"""
    from download_binance import BinanceDataDownloader
    
    # Download sample data
    try:
        downloader = BinanceDataDownloader(testnet=True)
        df = downloader.download_recent_data(symbol="BTCUSDT", days=30)
        
        # Process data
        processor = DataProcessor(lookback_window=50)
        train_df, test_df = processor.prepare_data(df)
        
        print(f"Original data shape: {df.shape}")
        print(f"Processed train data shape: {train_df.shape}")
        print(f"Processed test data shape: {test_df.shape}")
        print(f"Feature columns: {len(processor.get_feature_names(train_df))}")
        
        # Show sample of processed data
        print("\nSample of processed training data:")
        print(train_df.head())
        
    except Exception as e:
        print(f"Error in data processing example: {e}")
        print("This may be due to missing API keys or network issues")

if __name__ == "__main__":
    main()
