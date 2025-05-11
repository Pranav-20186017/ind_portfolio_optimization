import logging
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta
import requests
import hashlib
import time
import json
import os
from pathlib import Path
import webbrowser
from urllib.parse import urlparse, parse_qs
import yfinance as yf

logging.basicConfig(level=logging.DEBUG)

# Your API credentials from Kite developer account
API_KEY = ""
API_SECRET = ""

# Token cache file path
TOKEN_CACHE_FILE = "kite_token_cache.json"
TOKEN_EXPIRY_BUFFER = 300  # 5 minutes buffer before token expiry

class TokenManager:
    def __init__(self, cache_file=TOKEN_CACHE_FILE):
        self.cache_file = cache_file
        self.token_data = self._load_token_cache()
    
    def _load_token_cache(self):
        """Load token data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading token cache: {e}")
        return None
    
    def _save_token_cache(self, token_data):
        """Save token data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(token_data, f)
        except Exception as e:
            logging.warning(f"Error saving token cache: {e}")
    
    def get_valid_token(self):
        """Get a valid token, generating new one if needed"""
        if self.token_data:
            # Check if token is still valid (with buffer time)
            if time.time() < (self.token_data['timestamp'] + self.token_data['expires_in'] - TOKEN_EXPIRY_BUFFER):
                return self.token_data['access_token']
        
        # Generate new token
        new_token = generate_session_token()
        if new_token:
            self.token_data = {
                'access_token': new_token,
                'timestamp': time.time(),
                'expires_in': 86400  # 24 hours in seconds
            }
            self._save_token_cache(self.token_data)
            return new_token
        return None

# Initialize KiteConnect with your API key
kite = KiteConnect(api_key=API_KEY)
token_manager = TokenManager()

def get_request_token():
    """Get request token from login URL"""
    try:
        # Get the login URL
        login_url = kite.login_url()
        print(f"\nPlease visit this URL to login: {login_url}")
        print("After logging in, you'll be redirected to a URL. Copy the 'request_token' parameter from that URL.")
        
        # Open the login URL in browser
        webbrowser.open(login_url)
        
        # Wait for user to input the request token
        request_token = input("\nEnter the request token from the redirect URL: ")
        return request_token
    except Exception as e:
        logging.error(f"Error getting request token: {str(e)}")
        return None

def generate_session_token():
    """Generate session token using KiteConnect's built-in method"""
    try:
        # Get request token first
        request_token = get_request_token()
        if not request_token:
            logging.error("Failed to get request token")
            return None
            
        # Generate session using KiteConnect with request token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        logging.info("Successfully generated access token")
        return access_token
    except Exception as e:
        logging.error(f"Error generating access token: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"Response content: {e.response.text}")
        return None

def validate_token():
    """Validate the access token by making a simple API call"""
    try:
        # Try to get user profile as a test
        profile = kite.margins()
        logging.info("Token validation successful")
        return True
    except Exception as e:
        logging.error(f"Token validation failed: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"Response content: {e.response.text}")
        return False

def get_instrument_details(symbol, exchange, isin=None):
    """Get instrument details including ISIN for a given symbol and exchange"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get all instruments
            instruments = kite.instruments()
            
            # First try to find by symbol and exchange
            instrument = next(
                (item for item in instruments 
                 if item['tradingsymbol'] == symbol and item['exchange'] == exchange),
                None
            )
            
            # If not found and ISIN is provided, try to find by ISIN
            if not instrument and isin:
                instrument = next(
                    (item for item in instruments 
                     if item.get('isin') == isin and item['exchange'] == exchange),
                    None
                )
                if instrument:
                    logging.info(f"Found instrument by ISIN: {isin}")
            
            if instrument:
                return {
                    'tradingsymbol': instrument['tradingsymbol'],
                    'exchange': instrument['exchange'],
                    'isin': instrument.get('isin', ''),
                    'instrument_token': instrument['instrument_token']
                }
            else:
                logging.error(f"Instrument not found for {symbol} on {exchange}")
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Error getting instrument details after {max_retries} attempts: {e}")
                return None

def validate_ohlc_data(data, instrument_key):
    """Validate OHLC data structure and content"""
    try:
        if not isinstance(data, dict):
            logging.error(f"Invalid data type for {instrument_key}: expected dict, got {type(data)}")
            return False
            
        required_fields = ['timestamp', 'ohlc', 'volume']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing required field '{field}' in data for {instrument_key}")
                return False
        
        # Validate timestamp
        if not isinstance(data['timestamp'], list):
            logging.error(f"Invalid timestamp type for {instrument_key}: expected list, got {type(data['timestamp'])}")
            return False
            
        # Validate OHLC data
        ohlc = data['ohlc']
        if not isinstance(ohlc, dict):
            logging.error(f"Invalid OHLC type for {instrument_key}: expected dict, got {type(ohlc)}")
            return False
            
        ohlc_fields = ['open', 'high', 'low', 'close']
        for field in ohlc_fields:
            if field not in ohlc:
                logging.error(f"Missing OHLC field '{field}' for {instrument_key}")
                return False
            if not isinstance(ohlc[field], list):
                logging.error(f"Invalid OHLC {field} type for {instrument_key}: expected list, got {type(ohlc[field])}")
                return False
        
        # Validate volume
        if not isinstance(data['volume'], list):
            logging.error(f"Invalid volume type for {instrument_key}: expected list, got {type(data['volume'])}")
            return False
            
        # Validate data lengths
        expected_length = len(data['timestamp'])
        for field in ohlc_fields:
            if len(ohlc[field]) != expected_length:
                logging.error(f"Length mismatch for {field} in {instrument_key}: expected {expected_length}, got {len(ohlc[field])}")
                return False
        if len(data['volume']) != expected_length:
            logging.error(f"Length mismatch for volume in {instrument_key}: expected {expected_length}, got {len(data['volume'])}")
            return False
            
        # Validate data values
        for field in ohlc_fields:
            if not all(isinstance(x, (int, float)) for x in ohlc[field]):
                logging.error(f"Invalid values in {field} for {instrument_key}: expected numeric values")
                return False
        if not all(isinstance(x, (int, float)) for x in data['volume']):
            logging.error(f"Invalid values in volume for {instrument_key}: expected numeric values")
            return False
            
        return True
    except Exception as e:
        logging.error(f"Error validating data for {instrument_key}: {str(e)}")
        return False

def get_ohlc_data_batch(instruments, access_token):
    """Get OHLC data for multiple instruments in a single API call using the /quote/ohlc endpoint"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Prepare the query parameters
            params = {'i': []}
            for instrument in instruments:
                params['i'].append(f"{instrument['exchange']}:{instrument['tradingsymbol']}")
            
            # Make the API request
            url = "https://api.kite.trade/quote/ohlc"
            headers = {
                "X-Kite-Version": "3",
                "Authorization": f"token {API_KEY}:{access_token}"
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') != 'success':
                logging.error(f"API returned error status: {data}")
                return None
                
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logging.warning(f"Network error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Network error fetching OHLC data after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Error fetching OHLC data after {max_retries} attempts: {e}")
                return None

def get_historical_data(instrument_token, from_date, to_date, access_token, interval='day'):
    """Get historical data for a single instrument"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    # Validate date range
    current_date = datetime.now()
    if from_date > current_date or to_date > current_date:
        logging.error(f"Date range cannot be in the future. Current date: {current_date}")
        return None
    
    for attempt in range(max_retries):
        try:
            # Format dates for the API
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            
            # Make the API request
            url = f"https://api.kite.trade/instruments/historical/{instrument_token}/{interval}"
            params = {
                'from': from_date_str,
                'to': to_date_str
            }
            headers = {
                "X-Kite-Version": "3",
                "Authorization": f"token {API_KEY}:{access_token}"
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            # Get the error message if any
            if response.status_code != 200:
                error_msg = response.text
                logging.error(f"API Error for token {instrument_token}: {error_msg}")
                response.raise_for_status()
            
            data = response.json()
            if data.get('status') != 'success':
                logging.error(f"API returned error status: {data}")
                return None
                
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logging.warning(f"Network error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Network error fetching historical data after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Error fetching historical data after {max_retries} attempts: {e}")
                return None

def fetch_multiple_instruments(instrument_tokens, from_date, to_date, interval='day'):
    """
    Fetch historical data for multiple instruments using their tokens directly
    
    Parameters:
    instrument_tokens (list): List of instrument tokens
    from_date (datetime): Start date
    to_date (datetime): End date
    interval (str): Data interval (minute, day, etc.)
    
    Returns:
    pd.DataFrame: Multi-level DataFrame with OHLCV data for all instruments
    """
    # Get a valid token
    access_token = token_manager.get_valid_token()
    if not access_token:
        logging.error("Failed to get valid access token")
        return None
    
    # Set the token for KiteConnect
    kite.set_access_token(access_token)
    
    try:
        # Process the data into DataFrames
        dfs = []
        for token in instrument_tokens:
            try:
                # Get historical data for each instrument
                data = get_historical_data(
                    token,
                    from_date,
                    to_date,
                    access_token,
                    interval
                )
                
                if not data or 'candles' not in data:
                    logging.warning(f"No historical data found for token {token}")
                    continue
                    
                # Convert the historical data to a DataFrame
                candles = data['candles']
                df = pd.DataFrame(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert date string to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Validate DataFrame
                if df.empty:
                    logging.warning(f"Empty DataFrame for token {token}")
                    continue
                    
                if df.isnull().any().any():
                    logging.warning(f"Found null values in data for token {token}")
                    df = df.fillna(method='ffill')  # Forward fill null values
                
                df['instrument_token'] = token
                dfs.append(df)
                
            except Exception as e:
                logging.error(f"Error processing data for token {token}: {str(e)}")
                continue
        
        if dfs:
            try:
                # Combine all DataFrames
                combined_df = pd.concat(dfs, axis=0)
                
                # Create multi-level index
                combined_df.set_index(['instrument_token', 'date'], inplace=True)
                combined_df = combined_df.reorder_levels(['instrument_token', 'date'])
                
                # Final validation
                if combined_df.empty:
                    logging.error("Final DataFrame is empty")
                    return None
                    
                # Check for any remaining null values
                if combined_df.isnull().any().any():
                    logging.warning("Found null values in final DataFrame, filling with forward fill")
                    combined_df = combined_df.fillna(method='ffill')
                
                return combined_df
                
            except Exception as e:
                logging.error(f"Error creating final DataFrame: {str(e)}")
                return None
        else:
            logging.error("No data found for any instruments")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        return None

def compare_with_yfinance(kite_df, from_date, to_date):
    """
    Compare Kite data with yfinance data
    
    Parameters:
    kite_df (pd.DataFrame): DataFrame from Kite
    from_date (datetime): Start date
    to_date (datetime): End date
    
    Returns:
    pd.DataFrame: Comparison results
    """
    import yfinance as yf
    from datetime import timezone
    
    # Define yfinance symbols
    yf_symbols = {
        256265: '^NSEI',  # NIFTY 50
        424961: 'ABB.BO',  # ABB
        408065: 'INFY.NS',  # INFY
        2953217: 'TCS.NS'  # TCS
    }
    
    # Get yfinance data with multi_level_index=False
    yf_data = yf.download(
        list(yf_symbols.values()),
        start=from_date,
        end=to_date,
        multi_level_index=False
    )
    
    # Process Kite data
    kite_data = {}
    for token in yf_symbols.keys():
        if token in kite_df.index.get_level_values('instrument_token'):
            df = kite_df.xs(token, level='instrument_token')
            df = df[['open', 'high', 'low', 'close']]  # Only OHLC
            
            # Convert timezone-aware index to timezone-naive
            df.index = df.index.tz_localize(None)
            
            kite_data[yf_symbols[token]] = df
    
    # Compare data
    comparison_results = []
    for symbol in yf_symbols.values():
        if symbol in kite_data:
            kite_df = kite_data[symbol]
            
            # Get yfinance data for this symbol
            yf_cols = [(col, symbol) for col in ['Open', 'High', 'Low', 'Close']]
            yf_df = yf_data[yf_cols].copy()
            yf_df.columns = [col[0].lower() for col in yf_cols]  # Convert to lowercase
            
            # Align dates
            common_dates = kite_df.index.intersection(yf_df.index)
            kite_df = kite_df.loc[common_dates]
            yf_df = yf_df.loc[common_dates]
            
            # Calculate differences
            for col in ['open', 'high', 'low', 'close']:
                diff = (kite_df[col] - yf_df[col]).abs()
                max_diff = diff.max()
                mean_diff = diff.mean()
                comparison_results.append({
                    'Symbol': symbol,
                    'Column': col.upper(),
                    'Max Difference': max_diff,
                    'Mean Difference': mean_diff,
                    'Kite Sample': kite_df[col].iloc[0],
                    'YFinance Sample': yf_df[col].iloc[0],
                    'Kite Data Points': len(kite_df),
                    'YFinance Data Points': len(yf_df),
                    'Common Data Points': len(common_dates)
                })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Add summary statistics
    summary = comparison_df.groupby('Symbol').agg({
        'Kite Data Points': 'first',
        'YFinance Data Points': 'first',
        'Common Data Points': 'first',
        'Max Difference': 'max',
        'Mean Difference': 'mean'
    }).round(4)
    
    print("\nData Points Summary:")
    print(summary)
    
    return comparison_df

def main():
    # Example usage with direct instrument tokens
    instrument_tokens = [
        256265,  # NIFTY 50
        424961,  # ABB
        408065,  # INFY
        2953217,  # TCS
    ]
    
    # Set date range - use past dates only
    to_date = datetime.now()
    from_date = datetime(2023, 1, 1)
    
    # Fetch data for all instruments
    df = fetch_multiple_instruments(instrument_tokens, from_date, to_date)
    
    if df is not None:
        # Print the shape and first few rows
        print("\nKite DataFrame Shape:", df.shape)
        print("\nFirst few rows of Kite data:")
        print(df.head())
        
        # Compare with yfinance
        print("\nComparing with yfinance data...")
        comparison_df = compare_with_yfinance(df, from_date, to_date)
        print("\nComparison Results:")
        print(comparison_df)
        
        # Save to CSV
        filename = "historical_data_multiple.csv"
        df.to_csv(filename)
        logging.info(f"Data saved to {filename}")
        
        # Save comparison results
        comparison_filename = "data_comparison.csv"
        comparison_df.to_csv(comparison_filename)
        logging.info(f"Comparison results saved to {comparison_filename}")
    else:
        logging.error("Failed to fetch historical data")

if __name__ == "__main__":
    main()

