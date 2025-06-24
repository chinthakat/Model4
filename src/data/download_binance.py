#!/usr/bin/env python3
"""
Binance data acquisition module
Downloads historical OHLCV data from Binance Futures API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import os
from pathlib import Path
import logging
from typing import Optional, List, Tuple, Dict
import time
import json

class BinanceDataDownloader:
    """
    Downloads and manages historical data from Binance Futures API
    Supports both testnet and mainnet with proper rate limiting
    """
    
    def __init__(self, testnet: bool = True, data_dir: str = "data/binance"):
        """
        Initialize Binance data downloader for futures trading
        
        Args:
            testnet: Whether to use testnet (default: True for safer testing)
        """
        # Get API credentials from environment
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
        
        # Initialize Binance client for futures
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            # Set testnet base URL for futures
            self.client.API_URL = 'https://testnet.binancefuture.com'
            self.logger.info("Initialized Binance testnet futures client")
        else:
            self.client = Client(api_key, api_secret)
            self.logger.info("Initialized Binance production futures client")
        
        self.testnet = testnet
        
        # Test connection
        try:
            # Test futures account access
            account_info = self.client.futures_account()
            self.logger.info("Successfully connected to Binance futures API")
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance futures API: {e}")
            raise

        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_minute = 1200  # Binance limit
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized Binance client (testnet: {testnet})")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Wait if approaching rate limit
        if self.request_count >= self.max_requests_per_minute - 10:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                self.logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    def get_exchange_info(self) -> dict:
        """Get exchange information including available symbols"""
        self._rate_limit()
        try:
            info = self.client.futures_exchange_info()
            self.logger.info("Retrieved exchange information")
            return info
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        exchange_info = self.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        self.logger.info(f"Found {len(symbols)} available symbols")
        return symbols
    
    def _map_symbol_to_binance_futures(self, symbol: str) -> str:
        """Map symbol to Binance futures format"""
        symbol_mappings = {
            'BTCUSDT': 'BTCUSDT',  # Futures uses same format
            'BTCUSD_PERP': 'BTCUSDT',  # Map perpetual to USDT futures
            'BTC': 'BTCUSDT',
            'BINANCE_SPOT_PERP_BTC': 'BTCUSDT',  # Map CoinAPI format to Binance
            'ETHUSDT': 'ETHUSDT',
            'ETHUSD_PERP': 'ETHUSDT',
            'ETH': 'ETHUSDT',
            'BINANCE_SPOT_PERP_ETH': 'ETHUSDT',  # Map CoinAPI format to Binance
        }
        
        mapped = symbol_mappings.get(symbol.upper(), symbol.upper())
        if mapped != symbol.upper():
            self.logger.info(f"Mapped {symbol} to Binance futures symbol: {mapped}")
        
        return mapped

    def download_historical_data(self, 
                               symbol: str, 
                               interval: str, 
                               start_date: str, 
                               end_date: str) -> Optional[pd.DataFrame]:
        """
        Download historical futures data from Binance
        
        Args:
            symbol: Trading symbol (will be mapped to futures format)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Map symbol to Binance futures format
            futures_symbol = self._map_symbol_to_binance_futures(symbol)
            
            self.logger.info(f"Downloading futures data for {futures_symbol} from {start_date} to {end_date}")
            
            # Use futures klines endpoint
            klines = self.client.futures_historical_klines(
                symbol=futures_symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date
            )
            
            if not klines:
                self.logger.warning(f"No futures data found for {futures_symbol}")
                return None
            
            # Convert to DataFrame
            df = self._process_klines_data(klines)
            
            self.logger.info(f"Downloaded {len(df)} futures records for {futures_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download futures data for {symbol}: {e}")
            return None

    def download_recent_data(self, 
                           symbol: str, 
                           interval: str, 
                           days: int = 30) -> Optional[pd.DataFrame]:
        """
        Download recent futures data from Binance
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            days: Number of recent days to download
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Map symbol to Binance futures format
            futures_symbol = self._map_symbol_to_binance_futures(symbol)
            
            # Calculate start time (days ago)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            self.logger.info(f"Downloading {days} days of recent futures data for {futures_symbol}")
            
            # Use futures klines endpoint
            klines = self.client.futures_historical_klines(
                symbol=futures_symbol,
                interval=interval,
                start_str=start_time.strftime('%Y-%m-%d'),
                end_str=end_time.strftime('%Y-%m-%d')
            )
            
            if not klines:
                self.logger.warning(f"No recent futures data found for {futures_symbol}")
                return None
            
            # Convert to DataFrame
            df = self._process_klines_data(klines)
            
            self.logger.info(f"Downloaded {len(df)} recent futures records for {futures_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download recent futures data for {symbol}: {e}")
            return None

    def get_futures_account_info(self) -> Dict:
        """Get futures account information"""
        try:
            account_info = self.client.futures_account()
            self.logger.info("Retrieved futures account information")
            return account_info
        except Exception as e:
            self.logger.error(f"Failed to get futures account info: {e}")
            return {}

    def get_futures_position_info(self, symbol: str = None) -> List[Dict]:
        """Get futures position information"""
        try:
            futures_symbol = self._map_symbol_to_binance_futures(symbol) if symbol else None
            positions = self.client.futures_position_information(symbol=futures_symbol)
            self.logger.info(f"Retrieved futures position info for {futures_symbol or 'all symbols'}")
            return positions
        except Exception as e:
            self.logger.error(f"Failed to get futures position info: {e}")
            return []

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price for a symbol"""
        self._rate_limit()
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            self.logger.info(f"Current {symbol} price: {price}")
            return price
        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            raise
    
    def get_24hr_ticker(self, symbol: str = "BTCUSDT") -> dict:
        """Get 24hr ticker statistics"""
        self._rate_limit()
        try:
            ticker = self.client.futures_24hr_ticker(symbol=symbol)
            self.logger.info(f"Retrieved 24hr ticker for {symbol}")
            return ticker
        except Exception as e:
            self.logger.error(f"Failed to get 24hr ticker: {e}")
            raise
    
    def load_saved_data(self, filename: str) -> pd.DataFrame:
        """Load previously saved data from CSV file"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        self.logger.info(f"Loaded data from {filepath}: {len(df)} rows")
        return df
    
    def list_saved_data(self) -> List[str]:
        """List all saved data files"""
        csv_files = list(self.data_dir.glob("*.csv"))
        filenames = [f.name for f in csv_files]
        self.logger.info(f"Found {len(filenames)} saved data files")
        return filenames
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data integrity
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check OHLC relationships
        invalid_ohlc = df[(df['High'] < df['Low']) | 
                         (df['High'] < df['Open']) | 
                         (df['High'] < df['Close']) |
                         (df['Low'] > df['Open']) | 
                         (df['Low'] > df['Close'])]
        if len(invalid_ohlc) > 0:
            issues.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")
        
        # Check for negative values
        negative_vals = df[required_cols] < 0
        if negative_vals.any().any():
            issues.append("Negative values found in price/volume data")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            issues.append("Duplicate timestamps found")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _process_klines(self, klines: list) -> pd.DataFrame:
        """
        Process klines data into DataFrame with timestamp validation
        
        Args:
            klines: List of kline data from Binance API
            
        Returns:
            Processed DataFrame
        """
        if not klines:
            raise ValueError("No klines data provided")
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp from milliseconds to seconds
        df['timestamp'] = df['timestamp'].astype(float) / 1000
        
        # Validate timestamps are not in future
        current_timestamp = datetime.now().timestamp()
        future_mask = df['timestamp'] > current_timestamp
        
        if future_mask.any():
            future_count = future_mask.sum()
            self.logger.warning(f"Found {future_count} records with future timestamps")
            
            # Option 1: Remove future records
            df = df[~future_mask].copy()
            self.logger.info(f"Removed {future_count} future records, remaining: {len(df)}")
            
            # Option 2: Alternative - shift all timestamps to be historical
            # if len(df) == 0:  # If all records were future
            #     df = pd.DataFrame(klines, columns=[...])  # Reload original
            #     df['timestamp'] = df['timestamp'].astype(float) / 1000
            #     max_timestamp = df['timestamp'].max()
            #     shift_seconds = current_timestamp - max_timestamp - (86400 * 30)  # 30 days ago
            #     df['timestamp'] = df['timestamp'] + shift_seconds
            #     self.logger.info(f"Shifted all timestamps by {shift_seconds/86400:.1f} days")
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate we have data after timestamp filtering
        if len(df) == 0:
            raise ValueError("No valid data after timestamp validation")
        
        # Keep only necessary columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Log timestamp range
        min_date = pd.to_datetime(df['timestamp'].min(), unit='s')
        max_date = pd.to_datetime(df['timestamp'].max(), unit='s')
        self.logger.info(f"Processed data timestamp range: {min_date} to {max_date}")
        
        return df

def main():
    """Example usage of BinanceDataDownloader"""
    # Initialize downloader (testnet for safety)
    downloader = BinanceDataDownloader(testnet=True)
    
    # Download BTCUSDT 15-minute data
    df = downloader.download_historical_data(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_15MINUTE,
        start_date="1 Jan, 2023",
        end_date="1 Mar, 2023"
    )
    
    print(f"Downloaded data shape: {df.shape}")
    print(f"Data preview:\n{df.head()}")
    
    # Validate data
    is_valid, issues = downloader.validate_data(df)
    if is_valid:
        print("✓ Data validation passed")
    else:
        print(f"⚠ Data validation issues: {issues}")

if __name__ == "__main__":
    main()
