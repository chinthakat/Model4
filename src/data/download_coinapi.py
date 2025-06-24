#!/usr/bin/env python3
"""
CoinAPI data acquisition module
Downloads additional market data like funding rates and sentiment data
"""

import pandas as pd
import numpy as np  # Add missing numpy import
import requests
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

class CoinAPIDataDownloader:
    """
    Enhanced CoinAPI data downloader for comprehensive futures market data
    """
    
    def __init__(self, data_dir: str = "data/coinapi"):
        """
        Initialize CoinAPI downloader
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get API key
        self.api_key = os.getenv("COINAPI_API_KEY", "d7ca471c-8bee-4177-9a0f-bfe24d099ba3")
        if not self.api_key:
            logging.warning("CoinAPI key not found. Some features will be limited.")
        
        self.base_url = "https://rest.coinapi.io/v1"
        self.headers = {
            "X-CoinAPI-Key": self.api_key,
            "Accept": "application/json"  # Changed from text/plain to application/json
        }
        
        # Rate limiting
        self.requests_per_day = 100000  # Free tier limit
        self.request_count = 0
        self.last_request_time = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized CoinAPI downloader")
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        
        # Ensure minimum time between requests (to avoid hitting rate limits)
        min_interval = 0.1  # 100ms between requests
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count % 100 == 0:
            self.logger.info(f"Made {self.request_count} API requests")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with error handling"""
        if not self.api_key:
            raise ValueError("CoinAPI key required for this operation")
        
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            self.logger.info(f"Successfully fetched data from {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_available_exchanges(self) -> List[Dict]:
        """Get list of available exchanges that support futures"""
        try:
            response = requests.get(
                f"{self.base_url}/exchanges",
                headers=self.headers
            )
            response.raise_for_status()
            
            exchanges = response.json()
            
            # Filter for exchanges that likely have futures
            futures_exchanges = []
            futures_keywords = ['futures', 'future', 'perp', 'swap', 'derivat']
            
            for exchange in exchanges:
                exchange_id = exchange.get('exchange_id', '').lower()
                name = exchange.get('name', '').lower()
                
                # Check if exchange supports futures
                if any(keyword in exchange_id or keyword in name for keyword in futures_keywords):
                    futures_exchanges.append(exchange)
                
                # Known major exchanges with futures
                if exchange_id in ['binance', 'bybit', 'okex', 'ftx', 'bitmex', 'deribit']:
                    futures_exchanges.append(exchange)
            
            self.logger.info(f"Found {len(futures_exchanges)} exchanges with futures support")
            return futures_exchanges
            
        except Exception as e:
            self.logger.error(f"Failed to get exchanges: {e}")
            return []
    
    def get_available_symbols(self, exchange: str = 'BINANCE', limit: int = 100) -> list:
        """Get list of available symbols from CoinAPI - DISABLED to save quota"""
        # Disable API call to save quota - return empty list
        self.logger.info(f"get_available_symbols disabled to save API quota for {exchange}")
        return []
        
        # Original code commented out to save API quota
        # try:
        #     url = "https://rest.coinapi.io/v1/symbols"
        #     headers = {
        #         'X-CoinAPI-Key': self.api_key,
        #         'Accept': 'application/json'
        #     }
        #     
        #     params = {
        #         'filter_exchange_id': exchange
        #     }
        #     
        #     self.logger.info(f"Fetching available symbols for {exchange}")
        #     response = requests.get(url, headers=headers, params=params, timeout=30)
        #     response.raise_for_status()
        #     
        #     symbols_data = response.json()
        #     
        #     # Get all symbols first
        #     all_symbols = [symbol['symbol_id'] for symbol in symbols_data]
        #     
        #     # Filter for different types
        #     spot_symbols = [s for s in all_symbols if 'SPOT' in s and ('USDT' in s or 'USD' in s)]
        #     perp_symbols = [s for s in all_symbols if 'PERP' in s and ('USDT' in s or 'USD' in s)]
        #     futures_symbols = [s for s in all_symbols if any(keyword in s for keyword in ['FUT', 'FUTURE']) and ('USDT' in s or 'USD' in s)]
        #     btc_symbols = [s for s in all_symbols if 'BTC' in s]
        #     
        #     self.logger.info(f"Found {len(all_symbols)} total symbols for {exchange}")
        #     self.logger.info(f"  Spot symbols: {len(spot_symbols)}")
        #     self.logger.info(f"  Perpetual symbols: {len(perp_symbols)}")
        #     self.logger.info(f"  Futures symbols: {len(futures_symbols)}")
        #     self.logger.info(f"  BTC symbols: {len(btc_symbols)}")
        #     
        #     return all_symbols[:limit]
        #     
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch available symbols: {e}")
        #     return []

    def print_symbol_analysis(self, exchange: str = 'BINANCE'):
        """Print detailed analysis of available symbols - DISABLED to save quota"""
        print(f"\n=== SYMBOL ANALYSIS FOR {exchange} ===")
        print("Symbol analysis disabled to save API quota")
        print("Using hardcoded symbol mappings instead")
        print("\nKnown working symbols:")
        print("  BINANCE_SPOT_PERP_BTC (for BTC perpetual)")
        print("  BINANCE_SPOT_PERP_ETH (for ETH perpetual)")
        return

        # try:
        #     symbols = self.get_available_symbols(exchange, limit=1000)  # Get more symbols
        #     
        #     if not symbols:
        #         print("No symbols found or API key not configured")
        #         return
        #     
        #     print(f"\n=== SYMBOL ANALYSIS FOR {exchange} ===")
        #     print(f"Total symbols found: {len(symbols)}")
        #     
        #     # Categorize symbols
        #     categories = {
        #         'SPOT BTC': [s for s in symbols if 'SPOT' in s and 'BTC' in s],
        #         'PERP BTC': [s for s in symbols if 'PERP' in s and 'BTC' in s],
        #         'FUTURES BTC': [s for s in symbols if any(keyword in s for keyword in ['FUT', 'FUTURE']) and 'BTC' in s],
        #         'SPOT USDT': [s for s in symbols if 'SPOT' in s and 'USDT' in s],
        #         'PERP USDT': [s for s in symbols if 'PERP' in s and 'USDT' in s],
        #         'SPOT USD': [s for s in symbols if 'SPOT' in s and 'USD' in s and 'USDT' not in s],
        #         'PERP USD': [s for s in symbols if 'PERP' in s and 'USD' in s and 'USDT' not in s],
        #     }
        #     
        #     for category, symbol_list in categories.items():
        #         print(f"\n{category} ({len(symbol_list)} symbols):")
        #         for symbol in symbol_list[:10]:  # Show first 10
        #             print(f"  {symbol}")
        #         if len(symbol_list) > 10:
        #             print(f"  ... and {len(symbol_list) - 10} more")
        #     
        #     # Find the best BTC symbol for our use
        #     btc_candidates = [s for s in symbols if 'BTC' in s and ('USD' in s or 'USDT' in s)]
        #     print(f"\nBTC CANDIDATES ({len(btc_candidates)} total):")
        #     for symbol in btc_candidates[:20]:  # Show first 20
        #         print(f"  {symbol}")
        #     
        # except Exception as e:
        #     print(f"Error analyzing symbols: {e}")
    
    def _map_symbol_to_coinapi(self, symbol: str, exchange: str = 'BINANCE') -> str:
        """Map trading symbol to CoinAPI format with correct BINANCEFTS symbol discovery"""
        # Normalize symbol
        original_symbol = symbol.upper()
        
        # If the symbol is already in CoinAPI format, return it as-is
        if original_symbol.startswith('BINANCEFTS_'):
            self.logger.info(f"Symbol already in CoinAPI format: {original_symbol}")
            return original_symbol
        
        # Updated mappings based on actual CoinAPI symbol format from curl command
        symbol_mappings = {
            'BINANCE': {
                'BTCUSDT': 'BINANCEFTS_PERP_BTC_USDT',
                'BTCUSD_PERP': 'BINANCEFTS_PERP_BTC_USDT',
                'BTC': 'BINANCEFTS_PERP_BTC_USDT',
                'ETHUSDT': 'BINANCEFTS_PERP_ETH_USDT',
                'ETHUSD_PERP': 'BINANCEFTS_PERP_ETH_USDT',
                'ETH': 'BINANCEFTS_PERP_ETH_USDT',
                'ADAUSDT': 'BINANCEFTS_PERP_ADA_USDT',
                'SOLUSDT': 'BINANCEFTS_PERP_SOL_USDT',
                'XRPUSDT': 'BINANCEFTS_PERP_XRP_USDT',
                'DOGEUSDT': 'BINANCEFTS_PERP_DOGE_USDT',
                'AVAXUSDT': 'BINANCEFTS_PERP_AVAX_USDT',
                'LINKUSDT': 'BINANCEFTS_PERP_LINK_USDT',
                'MATICUSDT': 'BINANCEFTS_PERP_MATIC_USDT',
                'DOTUSDT': 'BINANCEFTS_PERP_DOT_USDT',
                'LTCUSDT': 'BINANCEFTS_PERP_LTC_USDT'
            }
        }
        
        # Try exact match first
        if exchange in symbol_mappings and original_symbol in symbol_mappings[exchange]:
            mapped_symbol = symbol_mappings[exchange][original_symbol]
            self.logger.info(f"Mapped {original_symbol} to {mapped_symbol}")
            return mapped_symbol
        
        # Try to extract base currency and map to BINANCEFTS_PERP format
        if original_symbol.endswith('USDT'):
            base = original_symbol[:-4]  # Remove USDT
            mapped_symbol = f"BINANCEFTS_PERP_{base}_USDT"
            self.logger.info(f"Generated mapping {original_symbol} -> {mapped_symbol}")
            return mapped_symbol
        elif original_symbol.endswith('USD'):
            base = original_symbol[:-3]  # Remove USD
            mapped_symbol = f"BINANCEFTS_PERP_{base}_USDT"
            self.logger.info(f"Generated mapping {original_symbol} -> {mapped_symbol}")
            return mapped_symbol
        elif original_symbol.endswith('_PERP'):
            base = original_symbol.replace('_PERP', '').replace('USD', '').replace('USDT', '')
            mapped_symbol = f"BINANCEFTS_PERP_{base}_USDT"
            self.logger.info(f"Generated PERP mapping {original_symbol} -> {mapped_symbol}")
            return mapped_symbol
        
        # Default fallback to BTC
        self.logger.warning(f"No mapping found for {original_symbol}, using BTC fallback")
        return "BINANCEFTS_PERP_BTC_USDT"

    def find_best_symbol_match(self, target_symbol: str, exchange: str = 'BINANCE') -> str:
        """Find the best matching symbol from available symbols"""
        # For common symbols, return direct mapping
        if target_symbol.upper() in ['BTCUSDT', 'BTCUSD_PERP', 'BTC']:
            return "BINANCEFTS_PERP_BTC_USDT"
        
        try:
            available_symbols = self.get_available_symbols(exchange, limit=1000)
            
            if not available_symbols:
                self.logger.warning("No available symbols found, using BTC fallback")
                return "BINANCEFTS_PERP_BTC_USDT"
            
            # Try to find exact match first
            if target_symbol in available_symbols:
                self.logger.info(f"Found exact match: {target_symbol}")
                return target_symbol
            
            # Extract base currency from target symbol
            target_base = target_symbol.upper().replace('USDT', '').replace('USD', '').replace('_PERP', '')
            
            # Look for SPOT_PERP format first
            spot_perp_pattern = f"BINANCEFTS_PERP_{target_base}_USDT"
            if spot_perp_pattern in available_symbols:
                self.logger.info(f"Found SPOT_PERP match: {spot_perp_pattern}")
                return spot_perp_pattern
            
            # Look for any matching base currency
            for symbol in available_symbols:
                if target_base in symbol and 'PERP' in symbol:
                    self.logger.info(f"Found PERP matching symbol: {symbol} for {target_symbol}")
                    return symbol
            
            # If no match found, use BTC as fallback
            btc_symbols = [s for s in available_symbols if 'BTC' in s and 'PERP' in s]
            if btc_symbols:
                self.logger.info(f"Using BTC fallback: {btc_symbols[0]} for {target_symbol}")
                return btc_symbols[0]
            
            # Final fallback
            self.logger.warning(f"No suitable symbol found, using generic BTC fallback")
            return "BINANCEFTS_PERP_BTC_USDT"
            
        except Exception as e:
            self.logger.error(f"Symbol matching failed: {e}, using BTC fallback")
            return "BINANCEFTS_PERP_BTC_USDT"

    def download_recent_data(self, symbol: str, interval: str = '15m', days: int = 30, 
                           exchange: str = 'BINANCE') -> Optional[pd.DataFrame]:
        """Download recent OHLCV data with default 15m interval"""
        # Calculate dates ensuring they're historical
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days)
        
        self.logger.info(f"Downloading recent {days} days of {interval} data for {symbol}")
        
        return self.download_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            exchange=exchange
        )

    def download_historical_data(self, symbol: str, interval: str = '15m', start_date: str = None, 
                               end_date: str = None, exchange: str = 'BINANCE') -> Optional[pd.DataFrame]:
        """Download historical OHLCV data with correct BINANCEFTS symbol mapping"""
        # Store original symbol for better mapping
        self.symbol = symbol
        
        # Use the symbol mapping function to get correct BINANCEFTS format
        symbol_id = self._map_symbol_to_coinapi(symbol, exchange)
        
        period_id = self._map_interval_to_coinapi(interval)
        
        self.logger.info(f"Downloading {interval} OHLCV data for {symbol_id} from {start_date} to {end_date}")
        
        try:
            url = f"{self.base_url}/ohlcv/{symbol_id}/history"
            
            params = {
                'period_id': period_id,
                'limit': 10000,  # Max limit
            }
            
            # Add date parameters if provided
            if start_date:
                params['time_start'] = f"{start_date}T00:00:00"
            if end_date:
                params['time_end'] = f"{end_date}T23:59:59"
            
            self.logger.info(f"API Request: URL={url}, Period={period_id}, Symbol={symbol_id}")
            print(f"🌐 Making CoinAPI request to: {url}")
            print(f"📊 Symbol: {symbol_id}, Period: {period_id}")
            print(f"📅 Date range: {start_date} to {end_date}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            # Enhanced error logging with HTTP status and response content
            self.logger.info(f"CoinAPI Response Status: {response.status_code}")
            self.logger.info(f"CoinAPI Response Headers: {dict(response.headers)}")
            
            # Print HTTP status code and error details
            print(f"🌐 HTTP Status Code: {response.status_code}")
            
            if response.status_code == 400:
                print(f"❌ Bad Request (400) - Invalid parameters")
                self.logger.error(f"Bad Request (400) Details:")
                self.logger.error(f"  URL: {url}")
                self.logger.error(f"  Params: {params}")
                try:
                    error_detail = response.json()
                    print(f"❌ API Error Response: {error_detail}")
                    self.logger.error(f"  Error Response: {error_detail}")
                except:
                    error_text = response.text
                    print(f"❌ Raw Error Response: {error_text}")
                    self.logger.error(f"  Raw Response: {error_text}")
            elif response.status_code == 401:
                print(f"❌ Unauthorized (401) - Check API key")
                print(f"   Current API key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else ''}")
                self.logger.error(f"Unauthorized (401) - Check API key")
            elif response.status_code == 403:
                print(f"❌ Forbidden (403) - API key doesn't have permission")
                self.logger.error(f"Forbidden (403) - API key doesn't have permission")
            elif response.status_code == 404:
                print(f"❌ Not Found (404) - Symbol or endpoint doesn't exist")
                print(f"   Symbol used: {symbol_id}")
                print(f"   Original symbol: {symbol}")
                print(f"   Try these common symbols:")
                print(f"     BINANCEFTS_PERP_BTC_USDT")
                print(f"     BINANCEFTS_PERP_ETH_USDT")
                print(f"     BINANCEFTS_PERP_ADA_USDT")
                self.logger.error(f"Not Found (404) - Symbol {symbol_id} not found")
            elif response.status_code == 429:
                print(f"⚠️  Rate Limit Exceeded (429) - Too many requests")
                self.logger.error(f"Rate limit exceeded (429)")
                try:
                    retry_after = response.headers.get('Retry-After', 'unknown')
                    print(f"   Retry after: {retry_after} seconds")
                except:
                    pass
            elif response.status_code >= 500:
                print(f"❌ Server Error ({response.status_code}) - CoinAPI service issue")
                self.logger.error(f"Server error ({response.status_code})")
            elif response.status_code == 200:
                print(f"✅ Request successful (200)")
            else:
                print(f"⚠️  Unexpected status code: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Print response data info
            if not data:
                print(f"⚠️  API returned empty data array")
                print(f"   Date range: {start_date} to {end_date}")
                print(f"   Symbol: {symbol_id}")
                print(f"   Period: {period_id}")
                print(f"   This could mean:")
                print(f"     - No trading data for this symbol in the date range")
                print(f"     - Symbol is not active yet")
                print(f"     - Date range is outside available data")
                self.logger.warning(f"No data found for {symbol_id} in the given date range")
                return None
            else:
                print(f"📊 Received {len(data)} data points from API")
                if len(data) > 0:
                    first_record = data[0]
                    last_record = data[-1]
                    print(f"   First record: {first_record.get('time_period_start', 'N/A')}")
                    print(f"   Last record: {last_record.get('time_period_start', 'N/A')}")
            
            df = pd.DataFrame(data)
            
            # Standardize the data format
            df = self._standardize_coinapi_data(df)
            
            self.logger.info(f"Successfully downloaded {len(df)} records for {symbol_id}")
            return df
        
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error occurred: {e}")
            self.logger.error(f"HTTP Error: {e}")
            if 'response' in locals():
                print(f"   Status Code: {response.status_code}")
                print(f"   Response Headers: {dict(response.headers)}")
                try:
                    error_content = response.json()
                    print(f"   Error Content: {error_content}")
                except:
                    print(f"   Raw Content: {response.text[:500]}...")
                self.logger.error(f"Response Status: {response.status_code}")
                self.logger.error(f"Response Text: {response.text}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: {e}")
            print("   Check your internet connection")
            self.logger.error(f"Connection Error: {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout Error: {e}")
            print("   Request took too long - try again")
            self.logger.error(f"Timeout Error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error downloading data for {symbol}: {e}")
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return None

    def _map_interval_to_coinapi(self, interval: str) -> str:
        """
        Map our interval format to CoinAPI period ID
        
        Args:
            interval: Interval string (e.g., '15m', '1h', '1d')
            
        Returns:
            CoinAPI period ID
        """
        interval_mapping = {
            '1m': '1MIN',
            '5m': '5MIN',
            '15m': '15MIN',  # Fixed: was 'c', now correct '15MIN'
            '30m': '30MIN',
            '1h': '1HRS',
            '4h': '4HRS',
            '1d': '1DAY',
            '1w': '7DAY'
        }
        
        mapped_interval = interval_mapping.get(interval, '15MIN')
        self.logger.info(f"Mapped interval '{interval}' to CoinAPI period '{mapped_interval}'")
        return mapped_interval
    
    def _standardize_coinapi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize CoinAPI data to match our expected format
        
        Args:
            df: Raw CoinAPI DataFrame
            
        Returns:
            Standardized DataFrame with proper DatetimeIndex
        """
        try:
            # Map CoinAPI columns to our standard format with uppercase names
            column_mapping = {
                'time_period_start': 'Timestamp',
                'time_period_end': 'Close_time',
                'price_open': 'Open',
                'price_high': 'High',
                'price_low': 'Low',
                'price_close': 'Close',
                'volume_traded': 'Volume',
                'trades_count': 'Count'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to proper datetime and set as index
            if 'Timestamp' in df.columns:
                # Convert to datetime first, then set as index
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = df.set_index('Timestamp')
                
                # Also add timestamp as Unix seconds for compatibility
                df['timestamp'] = df.index.astype('int64') // 10**9
            
            # Ensure numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by index (timestamp)
            df = df.sort_index()
            
            # Add any missing required columns with uppercase names
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Volume':
                        df[col] = 0.0  # Default volume if missing
                    elif col == 'timestamp':
                        df[col] = df.index.astype('int64') // 10**9
                    else:
                        self.logger.warning(f"Missing required column: {col}")
            
            return df[required_columns]
            
        except Exception as e:
            self.logger.error(f"Failed to standardize CoinAPI data: {e}")
            return df

    def validate_data(self, df: pd.DataFrame) -> tuple:
        """
        Validate downloaded data with proper DatetimeIndex
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, issues_list)
        """
        # Check for proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return (False, ["DataFrame must have DatetimeIndex"])
        
        # Check for missing columns (uppercase)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return (False, [f"Missing columns: {', '.join(missing_columns)}"])
        
        # Check for empty DataFrame
        if df.empty:
            return (False, ["DataFrame is empty"])
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            return (False, ["Duplicate timestamps found"])
        
        # Check for negative or zero volume
        if 'Volume' in df.columns and (df['Volume'] <= 0).any():
            return (False, ["Negative or zero volume found"])
        
        # Check for valid numeric types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return (False, [f"Column {col} is not numeric"])
        
        return (True, [])
    
    def test_api_connection(self) -> bool:
        """Test if API connection is working"""
        try:
            response = requests.get(
                f"{self.base_url}/exchangerate/USD/EUR",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info("CoinAPI connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"CoinAPI connection test failed: {e}")
            return False
    
    def get_symbol_info(self, symbol_id: str) -> Optional[Dict]:
        """Get detailed information about a symbol"""
        try:
            # This would require the symbol details endpoint
            # For now, return basic info
            return {
                'symbol_id': symbol_id,
                'base_asset': symbol_id.split('_')[-2] if '_' in symbol_id else 'BTC',
                'quote_asset': symbol_id.split('_')[-1] if '_' in symbol_id else 'USDT',
                'status': 'TRADING'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get symbol info: {e}")
            return None
    
    def get_funding_rates(
        self,
        symbol_id: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get funding rates data (conceptual - actual endpoint may vary)
        
        Args:
            symbol_id: Futures symbol identifier
            time_start: Start time
            time_end: End time
            
        Returns:
            DataFrame with funding rates
        """
        # Note: This is a conceptual implementation
        # Actual CoinAPI funding rates endpoint may be different
        params = {}
        if time_start:
            params["time_start"] = time_start
        if time_end:
            params["time_end"] = time_end
        
        try:
            # This endpoint may not exist in CoinAPI - check documentation
            data = self._make_request(f"futures/funding-rates/{symbol_id}", params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Retrieved funding rates for {symbol_id}")
            return df
            
        except Exception as e:
            self.logger.warning(f"Funding rates not available: {e}")
            return pd.DataFrame()
    
    def get_market_data_summary(self, symbol_ids: List[str]) -> pd.DataFrame:
        """
        Get market data summary for multiple symbols
        
        Args:
            symbol_ids: List of symbol identifiers
            
        Returns:
            DataFrame with market summary
        """
        summaries = []
        
        for symbol_id in symbol_ids:
            try:
                data = self._make_request(f"quotes/{symbol_id}/current")
                if data:
                    summaries.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to get summary for {symbol_id}: {e}")
                continue
        
        if summaries:
            df = pd.DataFrame(summaries)
            self.logger.info(f"Retrieved market summaries for {len(summaries)} symbols")
            return df
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file"""
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        self.logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded data from {filepath}")
        return df
    
    def get_mock_funding_rates(self, symbol: str = "BTCUSD_PERP") -> pd.DataFrame:
        """
        DEPRECATED: Generate mock funding rates
        This method should not be used in production - kept for testing only
        """
        self.logger.warning("get_mock_funding_rates is deprecated and should not be used")
        
        # Return empty DataFrame instead of mock data
        return pd.DataFrame()

    def get_mock_sentiment_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        DEPRECATED: Generate mock sentiment data
        This method should not be used in production - kept for testing only
        """
        self.logger.warning("get_mock_sentiment_data is deprecated and should not be used")
        
        # Return empty DataFrame instead of mock data
        return pd.DataFrame()

    def get_real_funding_rates(
        self,
        symbol_id: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get real funding rates data from CoinAPI
        
        Args:
            symbol_id: Futures symbol identifier
            time_start: Start time
            time_end: End time
            
        Returns:
            DataFrame with funding rates or empty DataFrame if not available
        """
        if not self.api_key:
            self.logger.info("No CoinAPI key - funding rates not available")
            return pd.DataFrame()
          # This would require finding the correct CoinAPI endpoint for funding rates
        # For now, return empty DataFrame until proper endpoint is implemented
        self.logger.info("Real funding rates endpoint not yet implemented")
        return pd.DataFrame()

    def _check_daily_file_exists(self, symbol: str, interval: str, date_str: str, 
                                min_records: int = 50) -> bool:
        """
        Check if daily data file exists and is valid with enhanced validation
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            date_str: Date string (YYYY-MM-DD)
            min_records: Minimum number of records expected for a full day
            
        Returns:
            bool: True if valid file exists with sufficient data
        """
        try:
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            filename = f"{symbol_clean}_{interval}_{date_str}.csv"
            filepath = symbol_dir / filename
            
            # Check if file exists
            if not filepath.exists():
                self.logger.debug(f"File does not exist: {filepath}")
                return False
            
            # Check if file is not empty
            file_size = filepath.stat().st_size
            if file_size == 0:
                self.logger.warning(f"Empty file found: {filepath}, will re-download")
                return False
            
            # Check if file is suspiciously small (< 1KB likely indicates incomplete data)
            if file_size < 1024:
                self.logger.warning(f"Suspiciously small file ({file_size} bytes): {filepath}, will re-download")
                return False
            
            # Quick validation - try to read the file
            try:
                df = pd.read_csv(filepath)
                
                # Check for empty dataframe
                if len(df) == 0:
                    self.logger.warning(f"File contains no data: {filepath}, will re-download")
                    return False
                
                # Check for minimum record count (15m interval should have ~96 records per day)
                expected_records = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}
                min_expected = expected_records.get(interval, min_records)
                
                if len(df) < min_expected * 0.8:  # Allow 20% tolerance for partial days
                    self.logger.warning(f"Insufficient data in file: {filepath} ({len(df)} records, expected ~{min_expected}), will re-download")
                    return False
                
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"File missing columns {missing_cols}: {filepath}, will re-download")
                    return False
                
                # Check for data quality - no negative prices
                price_cols = ['Open', 'High', 'Low', 'Close']
                for col in price_cols:
                    if col in df.columns and (df[col] <= 0).any():
                        self.logger.warning(f"Invalid negative/zero prices in {col}: {filepath}, will re-download")
                        return False
                  # Check for reasonable price ranges (basic sanity check)
                if 'Close' in df.columns:
                    close_prices = df['Close']
                    if close_prices.max() / close_prices.min() > 100:  # Price moved more than 100x in one day
                        self.logger.warning(f"Unrealistic price range in file: {filepath}, will re-download")
                        return False
                
                self.logger.debug(f"Valid file exists: {filepath} ({len(df)} records, {file_size} bytes)")
                return True
                
            except Exception as e:
                self.logger.warning(f"Cannot read or validate file {filepath}: {e}, will re-download")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking file existence for {date_str}: {e}")
            return False

    def download_daily_data_batch(self, symbol: str, interval: str = '15m', 
                                 start_date: str = None, end_date: str = None,
                                 exchange: str = 'BINANCE', force_redownload: bool = False,
                                 max_concurrent: int = 1) -> bool:
        """
        Download data day by day and save to individual files, skip existing files
        Enhanced with better progress reporting and validation
        
        Args:
            symbol: Trading symbol
            interval: Data interval (15m recommended)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Exchange name
            force_redownload: Force re-download even if files exist
            max_concurrent: Maximum concurrent downloads (currently unused, for future enhancement)
            
        Returns:
            bool: Success status
        """
        try:
            # Parse dates
            from datetime import datetime, timedelta
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            self.logger.info(f"Starting daily batch download for {symbol} from {start_date} to {end_date}")
            
            # Create symbol-specific directory
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Print the directory location at the start
            print(f"📂 Saving daily files to: {symbol_dir.absolute()}")
            self.logger.info(f"Daily files directory: {symbol_dir.absolute()}")
            
            current_date = start
            success_count = 0
            skipped_count = 0
            failed_count = 0
            total_days = (end - start).days + 1
            
            # Progress tracking
            downloaded_files = []
            failed_dates = []
            
            while current_date <= end:
                date_str = current_date.strftime('%Y-%m-%d')
                filename = f"{symbol_clean}_{interval}_{date_str}.csv"
                filepath = symbol_dir / filename
                
                # Check if file already exists and is valid (unless force redownload)
                if not force_redownload and self._check_daily_file_exists(symbol, interval, date_str):
                    try:
                        # Quick file size check for display
                        file_records = len(pd.read_csv(filepath))
                        print(f"⏭️  Skipping existing file: {date_str} ({file_records} records)")
                        self.logger.info(f"Skipping existing file: {filename}")
                        skipped_count += 1
                        success_count += 1  # Count as success since we have the data
                    except Exception:
                        print(f"⏭️  Skipping existing file: {date_str}")
                        skipped_count += 1
                        success_count += 1
                    
                    current_date += timedelta(days=1)
                    continue
                
                # Download single day
                progress = skipped_count + success_count + failed_count + 1
                print(f"🔄 Downloading {date_str} ({progress}/{total_days})...")
                self.logger.info(f"Downloading data for {date_str} ({progress}/{total_days})")
                
                try:
                    df = self.download_historical_data(
                        symbol=symbol,
                        interval=interval,
                        start_date=date_str,
                        end_date=date_str,
                        exchange=exchange
                    )
                    
                    if df is not None and len(df) > 0:
                        # Validate data
                        is_valid, issues = self.validate_data(df)
                        if is_valid:
                            # Save to file
                            df.to_csv(filepath, index=False)
                            print(f"💾 Saved {len(df)} records to: {date_str}")
                            self.logger.info(f"Saved {len(df)} records to {filename}")
                            downloaded_files.append(date_str)
                            success_count += 1
                        else:
                            print(f"❌ Invalid data for {date_str}: {issues}")
                            self.logger.warning(f"Invalid data for {date_str}: {issues}")
                            failed_dates.append((date_str, f"Invalid data: {issues}"))
                            failed_count += 1
                    else:
                        print(f"⚠️  No data received for {date_str}")
                        self.logger.warning(f"No data received for {date_str}")
                        failed_dates.append((date_str, "No data received"))
                        failed_count += 1
                        
                except Exception as e:
                    print(f"❌ Download failed for {date_str}: {str(e)[:100]}...")
                    self.logger.error(f"Download failed for {date_str}: {e}")
                    failed_dates.append((date_str, str(e)))
                    failed_count += 1
                
                # Rate limiting - wait between requests
                time.sleep(0.5)
                current_date += timedelta(days=1)
            
            # Summary
            total_needed = total_days
            total_success = success_count
            success_rate = (total_success / total_needed) * 100 if total_needed > 0 else 0
            
            print(f"\n📊 Batch download summary:")
            print(f"   Total days requested: {total_needed}")
            print(f"   Successfully downloaded: {success_count - skipped_count}")
            print(f"   Skipped (already exist): {skipped_count}")
            print(f"   Failed downloads: {failed_count}")
            print(f"   Total available: {total_success}")
            print(f"   Success rate: {success_rate:.1f}%")
            
            # Show failed dates if any
            if failed_dates:
                print(f"\n⚠️  Failed dates ({len(failed_dates)}):")
                for date_str, reason in failed_dates[:5]:  # Show first 5
                    print(f"   {date_str}: {reason[:60]}...")
                if len(failed_dates) > 5:
                    print(f"   ... and {len(failed_dates) - 5} more")
            
            self.logger.info(f"Batch download completed: {total_success}/{total_needed} days ({success_rate:.1f}%)")
            self.logger.info(f"Downloaded: {success_count - skipped_count}, Skipped: {skipped_count}, Failed: {failed_count}")
            
            # Return True if we have at least some data
            return total_success > 0
            
        except Exception as e:
            self.logger.error(f"Batch download failed: {e}")
            print(f"❌ Batch download failed: {e}")
            return False

    def download_recent_batch(self, symbol: str, interval: str = '15m', days: int = 30,
                             exchange: str = 'BINANCE', force_redownload: bool = False) -> bool:
        """
        Download recent data in daily batches, skip existing files
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            days: Number of recent days to download
            exchange: Exchange name
            force_redownload: Force re-download even if files exist
            
        Returns:
            Success status
        """
        # Calculate date range
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days)
        
        return self.download_daily_data_batch(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            exchange=exchange,
            force_redownload=force_redownload
        )

    def get_daily_file_status(self, symbol: str, interval: str = '15m', 
                             start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Get status of daily files for a date range
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with file status information
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            
            status = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'directory': str(symbol_dir),
                'existing_files': [],
                'missing_files': [],
                'total_days': 0,
                'existing_days': 0,
                'missing_days': 0,
                'coverage_percent': 0.0
            }
            
            current_date = start
            while current_date <= end:
                date_str = current_date.strftime('%Y-%m-%d')
                status['total_days'] += 1
                
                if self._check_daily_file_exists(symbol, interval, date_str):
                    status['existing_files'].append(date_str)
                    status['existing_days'] += 1
                else:
                    status['missing_files'].append(date_str)
                    status['missing_days'] += 1
                
                current_date += timedelta(days=1)
            
            if status['total_days'] > 0:
                status['coverage_percent'] = (status['existing_days'] / status['total_days']) * 100
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get file status: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'interval': interval
            }

    def print_file_status(self, symbol: str, interval: str = '15m', 
                         start_date: str = None, end_date: str = None):
        """Print file status for a date range"""
        status = self.get_daily_file_status(symbol, interval, start_date, end_date)
        
        if 'error' in status:
            print(f"❌ Error getting file status: {status['error']}")
            return
        
        print(f"\n📁 File Status for {symbol} ({interval})")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Directory: {status['directory']}")
        print(f"   Total days: {status['total_days']}")
        print(f"   Existing files: {status['existing_days']}")
        print(f"   Missing files: {status['missing_days']}")
        print(f"   Coverage: {status['coverage_percent']:.1f}%")
        
        if status['missing_files']:
            print(f"   Missing dates: {status['missing_files'][:10]}")
            if len(status['missing_files']) > 10:
                print(f"   ... and {len(status['missing_files']) - 10} more")

    def get_available_dates(self, symbol: str, interval: str = '15m') -> List[str]:
        """
        Get list of available dates for a symbol by checking existing files
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            List of available date strings (YYYY-MM-DD)
        """
        try:
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            
            if not symbol_dir.exists():
                self.logger.info(f"No data directory found for {symbol} {interval}")
                return []
            
            available_dates = []
            for csv_file in symbol_dir.glob(f"{symbol_clean}_{interval}_*.csv"):
                # Extract date from filename
                try:
                    filename = csv_file.stem
                    # Look for date pattern: YYYY-MM-DD
                    import re
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                    if date_match:
                        date_str = date_match.group(1)
                        # Verify file is valid
                        if self._check_daily_file_exists(symbol, interval, date_str):
                            available_dates.append(date_str)
                except Exception as e:
                    self.logger.debug(f"Could not parse date from {csv_file}: {e}")
                    continue
            
            available_dates.sort()
            self.logger.info(f"Found {len(available_dates)} available dates for {symbol} {interval}")
            return available_dates
            
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            return []

    def get_data_coverage(self, symbol: str, interval: str = '15m', 
                         start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Analyze data coverage for a symbol and date range
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            
        Returns:
            Dictionary with coverage analysis
        """
        try:
            available_dates = self.get_available_dates(symbol, interval)
            
            if not available_dates:
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'total_available': 0,
                    'coverage_ratio': 0.0,
                    'missing_dates': [],
                    'available_dates': [],
                    'date_range': 'No data available'
                }
            
            # If date range specified, analyze coverage within that range
            if start_date and end_date:
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                # Generate expected dates
                expected_dates = []
                current_date = start_dt
                while current_date <= end_dt:
                    expected_dates.append(current_date.strftime('%Y-%m-%d'))
                    current_date += timedelta(days=1)
                
                # Find missing dates
                missing_dates = [date for date in expected_dates if date not in available_dates]
                coverage_ratio = (len(expected_dates) - len(missing_dates)) / len(expected_dates) if expected_dates else 0.0
                
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'requested_range': f"{start_date} to {end_date}",
                    'total_expected': len(expected_dates),
                    'total_available': len(expected_dates) - len(missing_dates),
                    'coverage_ratio': coverage_ratio,
                    'missing_dates': missing_dates,
                    'available_in_range': [date for date in available_dates if start_date <= date <= end_date]
                }
            else:
                # Overall coverage analysis
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'total_available': len(available_dates),
                    'date_range': f"{available_dates[0]} to {available_dates[-1]}" if available_dates else "No data",
                    'available_dates': available_dates,
                    'coverage_ratio': 1.0  # All available dates are 100% covered
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing data coverage: {e}")
            return {'error': str(e)}

    def cleanup_corrupted_files(self, symbol: str, interval: str = '15m') -> int:
        """
        Clean up corrupted or incomplete data files
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Number of files cleaned up
        """
        try:
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            
            if not symbol_dir.exists():
                self.logger.info(f"No data directory found for {symbol} {interval}")
                return 0
            
            cleaned_count = 0
            for csv_file in symbol_dir.glob(f"{symbol_clean}_{interval}_*.csv"):
                try:
                    # Extract date from filename
                    import re
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_file.stem)
                    if date_match:
                        date_str = date_match.group(1)
                        
                        # Check if file is valid
                        if not self._check_daily_file_exists(symbol, interval, date_str):
                            self.logger.info(f"Removing corrupted file: {csv_file}")
                            csv_file.unlink()  # Delete the file
                            cleaned_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error processing file {csv_file}: {e}")
                    continue
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} corrupted files for {symbol} {interval}")
            else:
                self.logger.info(f"No corrupted files found for {symbol} {interval}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0

    def get_download_statistics(self, symbol: str, interval: str = '15m') -> Dict[str, Any]:
        """
        Get statistics about downloaded data files
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Dictionary with download statistics
        """
        try:
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            symbol_dir = self.data_dir / symbol_clean / interval
            
            if not symbol_dir.exists():
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'total_files': 0,
                    'total_size_mb': 0.0,
                    'date_range': 'No data',
                    'avg_records_per_file': 0,
                    'total_records': 0
                }
            
            files = list(symbol_dir.glob(f"{symbol_clean}_{interval}_*.csv"))
            if not files:
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'total_files': 0,
                    'total_size_mb': 0.0,
                    'date_range': 'No data',
                    'avg_records_per_file': 0,
                    'total_records': 0
                }
            
            total_size = sum(f.stat().st_size for f in files)
            total_records = 0
            dates = []
            
            for csv_file in files:
                try:
                    # Extract date
                    import re
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_file.stem)
                    if date_match:
                        dates.append(date_match.group(1))
                    
                    # Count records
                    df = pd.read_csv(csv_file)
                    total_records += len(df)
                    
                except Exception:
                    continue
            
            dates.sort()
            date_range = f"{dates[0]} to {dates[-1]}" if dates else "No valid dates"
            avg_records = total_records / len(files) if files else 0
            
            return {
                'symbol': symbol,
                'interval': interval,
                'total_files': len(files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'date_range': date_range,
                'avg_records_per_file': round(avg_records, 1),
                'total_records': total_records,
                'earliest_date': dates[0] if dates else None,
                'latest_date': dates[-1] if dates else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting download statistics: {e}")
            return {'error': str(e)}

def main():
    """Command line interface for CoinAPI data downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download cryptocurrency data from CoinAPI')
    
    parser.add_argument('--symbol', type=str, default='BTCUSD_PERP',
                       help='Trading symbol (default: BTCUSD_PERP)')
    parser.add_argument('--interval', type=str, default='15m',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Data interval (default: 15m)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of recent days to download (default: 30)')
    parser.add_argument('--start-date', type=str,
                       help='Start date in YYYY-MM-DD format (alternative to --days)')
    parser.add_argument('--end-date', type=str,
                       help='End date in YYYY-MM-DD format (default: yesterday)')
    parser.add_argument('--exchange', type=str, default='BINANCE',
                       help='Exchange name (default: BINANCE)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--batch-download', action='store_true',
                       help='Use batch download (recommended for large date ranges)')
    parser.add_argument('--check-status', action='store_true',
                       help='Check file status without downloading')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up corrupted files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize downloader
    downloader = CoinAPIDataDownloader()
    
    try:
        # Calculate date range
        if args.start_date and args.end_date:
            start_date = args.start_date
            end_date = args.end_date
        elif args.start_date:
            start_date = args.start_date
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        
        print(f"📊 CoinAPI Data Downloader")
        print(f"Symbol: {args.symbol}")
        print(f"Interval: {args.interval}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Exchange: {args.exchange}")
        
        # Check status if requested
        if args.check_status:
            print("\n=== FILE STATUS CHECK ===")
            downloader.print_file_status(args.symbol, args.interval, start_date, end_date)
            
            # Show statistics
            stats = downloader.get_download_statistics(args.symbol, args.interval)
            if 'error' not in stats:
                print(f"\n📈 Download Statistics:")
                print(f"   Total files: {stats['total_files']}")
                print(f"   Total size: {stats['total_size_mb']} MB")
                print(f"   Total records: {stats['total_records']:,}")
                print(f"   Date range: {stats['date_range']}")
            return
        
        # Cleanup if requested
        if args.cleanup:
            print("\n=== CLEANING UP CORRUPTED FILES ===")
            cleaned = downloader.cleanup_corrupted_files(args.symbol, args.interval)
            print(f"Cleaned up {cleaned} corrupted files")
            return
        
        # Download data
        if args.batch_download:
            print("\n=== STARTING BATCH DOWNLOAD ===")
            success = downloader.download_daily_data_batch(
                symbol=args.symbol,
                interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                exchange=args.exchange,
                force_redownload=args.force_redownload
            )
        else:
            print("\n=== STARTING SINGLE DOWNLOAD ===")
            df = downloader.download_historical_data(
                symbol=args.symbol,
                interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                exchange=args.exchange
            )
            success = df is not None and len(df) > 0
            
            if success:
                # Save to file
                filename = f"{args.symbol}_{args.interval}_{start_date}_to_{end_date}.csv"
                downloader.save_data(df, filename)
                print(f"💾 Saved {len(df)} records to {filename}")
        
        if success:
            print("✅ Download completed successfully")
            
            # Show final status
            print("\n=== FINAL STATUS ===")
            downloader.print_file_status(args.symbol, args.interval, start_date, end_date)
        else:
            print("❌ Download failed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
